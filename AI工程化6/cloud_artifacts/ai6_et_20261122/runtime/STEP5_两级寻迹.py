"""Paired validation experiment: joint edges, axial two-pass, stereo attachment.

Reconstruction receives measurements and predictions only. Truth is read by
score_tracks after reconstruction. No time fields or stereo XY circle fit.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from common import ROOT, DATA_ROOT, MODEL_ROOT, JointEventDataset, JointInteractionNetwork, sha, save_json
from server_runtime import loader, DevicePrefetch
from two_stage_core import run_two_pass

DEFAULT_CONFIG = dict(strict_threshold=.70, relaxed_threshold=.575, strict_rms=1., relaxed_rms=1.5,
                      as_threshold=.70, assignment_margin=.10, ss_threshold=.70,
                      weak_as_threshold=.45, assisted_threshold=.65, geometry_tolerance_mm=2.,
                      min_axial_hits=5, scoring_purity=.80, scoring_completeness=.50,
                      min_stereo_truth_hits=3, stereo_scoring_purity=.80, stereo_scoring_completeness=.50)


def undirected_predictions(edge_index, probabilities):
    edges = np.asarray(edge_index)
    n = edges.shape[1] // 2
    if edges.shape[1] != 2 * n or not np.array_equal(edges[:, :n], edges[::-1, n:]):
        raise ValueError('Unexpected bidirectional graph ordering')
    return edges[:, :n].copy(), (probabilities[:n] + probabilities[n:]) / 2


def simple_components(size, pairs, p):
    keep = p >= .5
    adj = coo_matrix((np.ones(keep.sum()), (pairs[0, keep], pairs[1, keep])), shape=(size, size))
    _, labels = connected_components(adj, directed=False)
    return [dict(indices=np.flatnonzero(labels == label).tolist(), stage='simple') for label in np.unique(labels)]


def axial_two_pass(geometry, drift, pairs, p):
    axial = np.flatnonzero(np.linalg.norm(geometry[:, 2:4], axis=1) <= 1e-9)
    mapping = np.full(len(geometry), -1, dtype=np.int64)
    mapping[axial] = np.arange(len(axial))
    keep = (mapping[pairs[0]] >= 0) & (mapping[pairs[1]] >= 0)
    data = Data(wire_x=torch.tensor(geometry[axial, 0], dtype=torch.float64),
                wire_y=torch.tensor(geometry[axial, 1], dtype=torch.float64),
                drift_d=torch.tensor(drift[axial], dtype=torch.float64),
                edge_index=torch.tensor(mapping[pairs[:, keep]], dtype=torch.long), num_nodes=len(axial))
    tracks, diagnostics = run_two_pass(data, None, None, None, .70, .575, probabilities=p[keep])
    for track in tracks:
        track['indices'] = axial[track['indices']].tolist()
    return tracks, diagnostics


def circle_wire_lower_bound(circle, geometry):
    """Necessary XY compatibility over the complete finite wire, not a 3D fit."""
    center = np.asarray(circle[:2]); radius = circle[2]
    half = geometry[:, 2:4] * geometry[:, 5:6] / 2
    start = geometry[:, :2] - half
    delta = 2 * half
    fraction = np.clip(((center - start) * delta).sum(1) / np.maximum((delta * delta).sum(1), 1e-20), 0, 1)
    rmin = np.linalg.norm(start + fraction[:, None] * delta - center, axis=1)
    rmax = np.maximum(np.linalg.norm(start - center, axis=1), np.linalg.norm(start + delta - center, axis=1))
    return np.maximum(np.maximum(rmin - radius, radius - rmax), 0)


def unique_support(values, cell_ids, threshold):
    return np.array([len(np.unique(cell_ids[row >= threshold])) for row in values])


def assign_unique(scores, threshold, margin):
    if not scores.shape[1]:
        return np.full(len(scores), -1, dtype=int)
    best = np.argmax(scores, axis=1)
    top = scores[np.arange(len(scores)), best]
    second = np.partition(scores, -2, axis=1)[:, -2] if scores.shape[1] > 1 else np.zeros(len(scores))
    return np.where((top >= threshold) & (top - second >= margin), best, -1)


def attach_stereo(tracks, geometry, drift, cells, pairs, p, config):
    result = [dict(t, indices=list(t['indices'])) for t in tracks]
    stereo = np.flatnonzero(np.linalg.norm(geometry[:, 2:4], axis=1) > 1e-9)
    if not len(stereo) or not result:
        return result, dict(direct_assignments=0, ss_assisted_assignments=0, unassigned_stereo=len(stereo))
    matrix = np.zeros((len(geometry), len(geometry)), dtype=np.float32)
    matrix[pairs[0], pairs[1]] = p
    matrix[pairs[1], pairs[0]] = p
    direct = np.zeros((len(stereo), len(result)), dtype=np.float32)
    weak = np.zeros_like(direct, dtype=bool)
    compatible = np.zeros_like(direct, dtype=bool)
    as_mean = np.zeros_like(direct)
    for t, track in enumerate(result):
        aa = np.asarray(track['indices'])
        values = matrix[np.ix_(stereo, aa)]
        mean = np.sort(values, axis=1)[:, -2:].mean(1)
        compatible[:, t] = circle_wire_lower_bound(track['fit'], geometry[stereo]) <= drift[stereo] + config['geometry_tolerance_mm']
        strong = unique_support(values, cells[aa], config['as_threshold']) >= 2
        weak[:, t] = unique_support(values, cells[aa], config['weak_as_threshold']) >= 1
        direct[:, t] = np.where(strong & compatible[:, t], mean, 0)
        as_mean[:, t] = mean
    owner = assign_unique(direct, config['as_threshold'], config['assignment_margin'])
    seeds = owner.copy()  # One bounded pass; no iterative SS flooding.
    assisted = np.zeros_like(direct)
    for t in range(len(result)):
        seeds_t = stereo[seeds == t]
        if len(seeds_t) < 2:
            continue
        values = matrix[np.ix_(stereo, seeds_t)]
        ss_mean = np.sort(values, axis=1)[:, -2:].mean(1)
        supported = unique_support(values, cells[seeds_t], config['ss_threshold']) >= 2
        assisted[:, t] = np.where(supported & weak[:, t] & compatible[:, t], .75 * as_mean[:, t] + .25 * ss_mean, 0)
    secondary = assign_unique(assisted, config['assisted_threshold'], config['assignment_margin'])
    fill = (owner < 0) & (secondary >= 0)
    owner[fill] = secondary[fill]
    for t, track in enumerate(result):
        track['stereo_indices'] = stereo[owner == t].tolist()
        track['indices'].extend(track['stereo_indices'])
    return result, dict(direct_assignments=int((seeds >= 0).sum()), ss_assisted_assignments=int(fill.sum()),
                        unassigned_stereo=int((owner < 0).sum()))


def score_tracks(tracks, truth, stereo, config):
    """Common denominator and 80%/50% axial matching across all three methods."""
    aa = ~stereo
    ids, sizes = np.unique(truth[aa & (truth >= 0)], return_counts=True)
    eligible = dict(zip(ids[sizes >= 5], sizes[sizes >= 5]))
    eligible_s = {t for t in eligible if ((truth == t) & stereo).sum() >= config['min_stereo_truth_hits']}
    counts = Counter(truth_tracks=len(eligible), stereo_eligible_tracks=len(eligible_s),
                     truth_stereo_all=sum(int(((truth == t) & stereo).sum()) for t in eligible))
    found = set(); joint_found = set()
    for track in tracks:
        indices = np.asarray(track['indices'], dtype=int)
        axial = indices[~stereo[indices]]
        if len(axial) < 5:
            continue
        counts['output_candidates'] += 1
        tids, ns = np.unique(truth[axial], return_counts=True)
        t = tids[ns.argmax()]; n = int(ns.max())
        if t not in eligible or n / len(axial) < config['scoring_purity'] or n / eligible[t] < config['scoring_completeness']:
            counts['fake_candidates'] += 1
            continue
        if t in found:
            counts['clone_candidates'] += 1
            continue
        found.add(t)
        counts['axial_completeness_sum'] += n / eligible[t]
        ss = indices[stereo[indices]]
        correct = int((truth[ss] == t).sum()); total_s = int(((truth == t) & stereo).sum())
        counts['correct_stereo_matched'] += correct; counts['attached_stereo_matched'] += len(ss)
        counts['truth_stereo_matched'] += total_s
        if t in eligible_s and correct / max(len(ss), 1) >= config['stereo_scoring_purity'] and correct / total_s >= config['stereo_scoring_completeness']:
            joint_found.add(t)
    counts['matched_tracks'] = len(found); counts['joint_matched_tracks'] = len(joint_found)
    return dict(counts), sorted(map(int, found))


def summarize(rows):
    total = Counter()
    for row in rows:
        total.update(row)
    result = dict(total)
    for name, num, den in [('track_efficiency', 'matched_tracks', 'truth_tracks'),
                           ('fake_fraction', 'fake_candidates', 'output_candidates'),
                           ('clone_fraction', 'clone_candidates', 'output_candidates'),
                           ('stereo_purity_matched', 'correct_stereo_matched', 'attached_stereo_matched'),
                           ('stereo_completeness_matched', 'correct_stereo_matched', 'truth_stereo_matched'),
                           ('stereo_recall_all_eligible_tracks', 'correct_stereo_matched', 'truth_stereo_all'),
                           ('joint_track_efficiency', 'joint_matched_tracks', 'stereo_eligible_tracks')]:
        result[name] = total[num] / max(total[den], 1)
    return result


def process_event(eid, cache, config):
    started = time.time()
    with np.load(Path(cache) / f'event_{eid:05d}.npz') as f:
        pairs, p = f['pairs'], f['probabilities']
    with np.load(DATA_ROOT / 'events' / f'event_{eid:05d}.npz') as f:
        cells, drift, truth = f['cell_id'], f['d'], f['truth']
    geometry = np.load(DATA_ROOT / 'wire_geometry.npy')[cells]
    stereo = np.linalg.norm(geometry[:, 2:4], axis=1) > 1e-9
    simple = simple_components(len(cells), pairs, p)
    baseline, diag = axial_two_pass(geometry, drift, pairs, p)
    aided, assignment = attach_stereo(baseline, geometry, drift, cells, pairs, p, config)
    methods = {}; matched = {}
    for name, tracks in [('simple_joint', simple), ('axial_two_pass', baseline), ('stereo_assisted', aided)]:
        methods[name], matched[name] = score_tracks(tracks, truth, stereo, config)
    # Attaching stereo cannot silently change the axial memberships or denominator.
    assert matched['axial_two_pass'] == matched['stereo_assisted']
    return dict(event_id=eid, seconds=time.time() - started, methods=methods, matched_truth_ids=matched,
                two_pass_diagnostics=diag, stereo_assignment=assignment,
                tracks={name: tracks for name, tracks in [('axial_two_pass', baseline), ('stereo_assisted', aided)]})


@torch.no_grad()
def prepare_predictions(snapshot, ids, cache):
    dataset = JointEventDataset(DATA_ROOT, 'validation')
    indices = {eid: i for i, eid in enumerate(dataset.ids)}
    remaining = [eid for eid in ids if not (cache / f'event_{eid:05d}.npz').exists()]
    if not remaining:
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JointInteractionNetwork(chunk_size=65536).to(device)
    model.load_state_dict(torch.load(snapshot / 'best_interaction_network_joint_no_time.pth', map_location=device, weights_only=True))
    model.eval(); torch.set_num_threads(2)
    data = torch.utils.data.Subset(dataset, [indices[eid] for eid in remaining])
    for i, g in enumerate(DevicePrefetch(loader(data, 1), device)):
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            probabilities = model(g.x, g.edge_index, g.edge_attr).sigmoid().float().cpu().numpy()
        pairs, probabilities = undirected_predictions(g.edge_index.cpu().numpy(), probabilities)
        eid = int(g.event_id.item())
        np.savez_compressed(cache / f'event_{eid:05d}.npz', pairs=pairs.astype(np.int32), probabilities=probabilities)
        if i == 0 or (i + 1) % 25 == 0:
            print(json.dumps(dict(phase='predicting', events=i + 1, total=len(remaining))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default='local_evaluation_epoch051_20261106')
    parser.add_argument('--experiment', default='two_stage_validation_v1')
    parser.add_argument('--events', type=int, default=200)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.events <= 1500:
        raise ValueError('Validation event count must be between 1 and 1500')
    snapshot = MODEL_ROOT / args.snapshot
    out = snapshot / args.experiment
    out.mkdir(exist_ok=True)
    config = dict(DEFAULT_CONFIG)
    config['source_hashes'] = {name: sha(ROOT / name) for name in ['STEP5_两级寻迹.py', 'two_stage_core.py', 'common.py']}
    config['weight_sha256'] = sha(snapshot / 'best_interaction_network_joint_no_time.pth')
    config['dataset_metadata_sha256'] = sha(DATA_ROOT / 'dataset_metadata_joint_no_time.json')
    metadata = json.loads((snapshot / 'model_metadata_joint_no_time.json').read_text(encoding='utf-8'))
    if metadata['smoke'] or metadata['dataset_metadata_sha256'] != config['dataset_metadata_sha256']:
        raise ValueError('Snapshot/dataset mismatch')
    ids = json.loads((DATA_ROOT / 'dataset_metadata_joint_no_time.json').read_text(encoding='utf-8'))['event_ids']['validation']
    ids = list(map(int, np.random.default_rng(20261106).permutation(ids)[:args.events]))
    config.update(event_ids=ids, split='validation', selection_seed=20261106,
                  thresholds_tuned=False, edge_symmetrization='mean of both directions',
                  geometry_dtype='float64', selected_epoch=json.loads((snapshot / 'best_model_selection.json').read_text())['epoch'])
    existing = out / 'experiment_contract.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8')) != config:
        raise ValueError('Experiment contract changed; choose a new experiment name')
    save_json(existing, config)
    cache = snapshot / ('validation_prediction_cache_' + config['weight_sha256'][:12])
    cache.mkdir(exist_ok=True)
    cache_contract = {key: config[key] for key in ['weight_sha256', 'dataset_metadata_sha256', 'edge_symmetrization']}
    cache_contract['graph_builder_sha256'] = sha(ROOT / 'common.py')
    if (cache / 'contract.json').exists() and json.loads((cache / 'contract.json').read_text()) != cache_contract:
        raise ValueError('Prediction cache contract mismatch')
    save_json(cache / 'contract.json', cache_contract)
    prepare_predictions(snapshot, ids, cache)
    rows = {}
    events_dir = out / 'events'; events_dir.mkdir(exist_ok=True)
    for eid in ids:
        path = events_dir / f'event_{eid:05d}.json'
        if path.exists():
            rows[eid] = json.loads(path.read_text(encoding='utf-8'))
    started = time.time()
    pending = [eid for eid in ids if eid not in rows]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_event, eid, str(cache), config): eid for eid in pending}
        for future in as_completed(futures):
            row = future.result(); eid = row['event_id']; rows[eid] = row
            save_json(events_dir / f'event_{eid:05d}.json', row)
            progress = dict(phase='tracking', events=len(rows), total=len(ids), seconds=time.time() - started,
                            last_event=eid, last_event_seconds=row['seconds'], updated_unix=time.time())
            save_json(out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    ordered = [rows[eid] for eid in ids]
    methods = {name: summarize([row['methods'][name] for row in ordered])
               for name in ['simple_joint', 'axial_two_pass', 'stereo_assisted']}
    if sha(snapshot / 'best_interaction_network_joint_no_time.pth') != config['weight_sha256']:
        raise RuntimeError('Snapshot changed')
    report = dict(config=config, events=len(ids), methods=methods,
                  scope='validation feasibility experiment; stereo assignment uses finite-wire XY lower bound, not full 3D helix fitting',
                  total_tracking_event_seconds=sum(row['seconds'] for row in ordered))
    save_json(out / 'summary.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
