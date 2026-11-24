"""Evaluate a frozen AI6 snapshot once per event, with fixed decoding rules."""
import argparse
from collections import Counter
import json
import time
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from common import ROOT, MODEL_ROOT, DATA_ROOT, JointInteractionNetwork, FocalLoss, sha, save_json
from server_runtime import dataset as runtime_dataset, loader as runtime_loader, DevicePrefetch


def candidate_counts(g, selected):
    """Truth is used only after components have been reconstructed."""
    a, b = g.edge_index.cpu().numpy()
    adj = coo_matrix((np.ones(selected.sum()), (a[selected], b[selected])), shape=(g.num_nodes, g.num_nodes))
    _, labels = connected_components(adj, directed=False)
    truth = g.truth.cpu().numpy()
    stereo = g.stereo.cpu().numpy()
    axial = ~stereo
    sizes = {t: int(((truth == t) & axial).sum()) for t in np.unique(truth)}
    eligible = {t: n for t, n in sizes.items() if t >= 0 and n >= 5}
    found = Counter()
    counts = Counter(truth_tracks=len(eligible))
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        aa = indices[axial[indices]]
        if len(aa) < 5:
            continue
        counts['output_candidates'] += 1
        tids, ns = np.unique(truth[aa], return_counts=True)
        t = tids[ns.argmax()]
        n = int(ns.max())
        if t not in eligible or n / len(aa) < .7 or n / eligible[t] < .7:
            counts['fake_candidates'] += 1
            continue
        found[t] += 1
        counts['matched_candidates'] += 1
        counts['axial_completeness_sum'] += n / eligible[t]
        ss = indices[stereo[indices]]
        counts['correct_stereo'] += int((truth[ss] == t).sum())
        counts['attached_stereo'] += len(ss)
        counts['truth_stereo_matched'] += int(((truth == t) & stereo).sum())
    counts['matched_tracks'] = len(found)
    counts['clone_candidates'] = sum(max(n - 1, 0) for n in found.values())
    return counts


def summarize(counts, loss_sum, edges, candidates):
    types = {name: {'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
                   'precision': int(tp) / max(int(tp + fp), 1),
                   'recall': int(tp) / max(int(tp + fn), 1),
                   'f1': 2 * int(tp) / max(int(2 * tp + fp + fn), 1)}
             for name, (tp, fp, fn) in zip(['AA', 'AS', 'SS'], counts)}
    c = dict(candidates)
    for name, num, den in [('axial_efficiency', 'matched_tracks', 'truth_tracks'),
                           ('fake_fraction', 'fake_candidates', 'output_candidates'),
                           ('clone_fraction', 'clone_candidates', 'output_candidates'),
                           ('axial_completeness_matched', 'axial_completeness_sum', 'matched_candidates'),
                           ('stereo_purity', 'correct_stereo', 'attached_stereo'),
                           ('stereo_completeness_matched', 'correct_stereo', 'truth_stereo_matched')]:
        c[name] = c.get(num, 0) / max(c.get(den, 0), 1)
    return {'loss': loss_sum / max(edges, 1), 'edges': edges, 'types': types,
            'macro_f1': sum(t['f1'] for t in types.values()) / 3, 'candidate_diagnostic': c}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-tag', required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--chunk-size', type=int, default=65536)
    args = p.parse_args()
    torch.set_num_threads(2)
    out = MODEL_ROOT / args.run_tag
    target = out / f'snapshot_evaluation_{args.split}.json'
    if target.exists():
        raise FileExistsError(target)
    config = json.loads((out / 'model_metadata_joint_no_time.json').read_text(encoding='utf-8'))
    if config['smoke'] or sha(DATA_ROOT / 'dataset_metadata_joint_no_time.json') != config['dataset_metadata_sha256']:
        raise ValueError('Invalid model or changed dataset')
    weight = out / 'best_interaction_network_joint_no_time.pth'
    weight_hash = sha(weight)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JointInteractionNetwork(chunk_size=args.chunk_size).to(device)
    model.load_state_dict(torch.load(weight, weights_only=True, map_location=device))
    model.eval()
    data = runtime_dataset(DATA_ROOT, args.split)
    exposed = set()
    for path, key in [(ROOT / 'pilot_v1/manifest.json', 'pilot'), (ROOT / 'memory_safe_gpu/fresh_test_ids.json', 'fresh')]:
        if path.exists():
            value = json.loads(path.read_text(encoding='utf-8'))
            exposed.update(value['ids']['test'] if key == 'pilot' else value['ids'])
    totals = np.zeros((2, 3, 3), dtype=np.int64)
    losses, edges, events = [0., 0.], [0, 0], [0, 0]
    candidates = [Counter(), Counter()]
    criterion = FocalLoss()
    started = time.time()
    per_event = out / f'snapshot_events_{args.split}.jsonl'
    with per_event.open('x', encoding='utf-8') as event_log:
        for index, g in enumerate(DevicePrefetch(runtime_loader(data, 1), device)):
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(g.x, g.edge_index, g.edge_attr)
            loss = criterion(logits, g.y).item()
            selected = (logits.sigmoid() >= .5).cpu().numpy()
            y, kind = g.y.bool().cpu().numpy(), g.edge_kind.cpu().numpy()
            counts = np.zeros((3, 3), dtype=np.int64)
            for k in range(3):
                mask = kind == k
                counts[k] = [(selected & y & mask).sum(), (selected & ~y & mask).sum(), (~selected & y & mask).sum()]
            cc = candidate_counts(g, selected)
            eid = int(g.event_id.item())
            for subset in ([0, 1] if eid not in exposed else [0]):
                totals[subset] += counts
                losses[subset] += loss * len(y)
                edges[subset] += len(y)
                events[subset] += 1
                candidates[subset].update(cc)
            event_log.write(json.dumps({'event_id': eid, 'edge_counts': counts.tolist(), 'candidate_counts': dict(cc)}) + '\n')
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(data):
                progress = {'split': args.split, 'events': index + 1, 'total_events': len(data),
                            'seconds': time.time() - started, 'updated_unix': time.time()}
                save_json(out / f'snapshot_progress_{args.split}.json', progress)
                event_log.flush()
                print(json.dumps(progress), flush=True)
    if weight_hash != sha(weight):
        raise RuntimeError('Weights changed during evaluation')
    report = {'split': args.split, 'events': events[0], 'result': summarize(totals[0], losses[0], edges[0], candidates[0]),
              'edge_threshold': .5, 'weight_sha256': weight_hash, 'device': str(device),
              'chunk_size': args.chunk_size, 'torch_version': str(torch.__version__),
              'dataset_metadata_sha256': sha(DATA_ROOT / 'dataset_metadata_joint_no_time.json'),
              'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
              'seconds': time.time() - started,
              'selected_epoch': json.loads((out / 'best_model_selection.json').read_text(encoding='utf-8'))['epoch'],
              'scope': 'edge association and connected-component diagnostic; no geometry fit or 3D momentum',
              'candidate_rule': {'threshold': .5, 'min_axial_hits': 5, 'min_axial_purity': .7,
                                 'min_axial_completeness': .7, 'threshold_tuned_on_test': False}}
    if args.split == 'test':
        report['previously_inspected_pilot_test_ids'] = sorted(exposed & set(data.ids))
        report['historical_pilot_excluded_subset'] = {'events': events[1], 'result': summarize(totals[1], losses[1], edges[1], candidates[1])}
        with (MODEL_ROOT / 'evaluation_exposure_history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'run_tag': args.run_tag, 'unix': time.time(), 'weight_sha256': weight_hash, 'event_ids': data.ids}) + '\n')
    save_json(target, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
