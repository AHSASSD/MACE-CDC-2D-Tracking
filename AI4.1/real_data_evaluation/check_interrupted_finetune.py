"""Read-only comparison of frozen no-time weights on whole-event validation."""
from collections import Counter
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import InteractionNetwork, model_dimensions
from STEP2_训练模型 import FocalLoss
from STEP4_两级寻迹 import predict, run_two_pass, evaluate, truth_sets

DATA = ROOT / 'whole_event_retraining'
OUT = DATA / 'interrupted_no_time_assessment'
STRICT, RELAXED = .70, .575


def init_worker():
    torch.set_num_threads(1)


def tracking(task):
    evt, data, probabilities = task
    tracks, _ = run_two_pass(data, None, None, 'no-time', STRICT, RELAXED, probabilities)
    result = evaluate(data, tracks)
    return {'EvtID': evt, 'output_tracks': len(tracks), **result}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage2', action='store_true')
    args = parser.parse_args()
    if args.stage2:
        OUT = DATA / 'stage2_no_time_assessment'
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parts = sorted((DATA / 'dataset_validation.pt.parts').glob('part_*.pt'))
    graphs = {}
    for path in parts:
        for graph in torch.load(path, weights_only=False):
            evt = int(graph.event_id.item())
            if evt in graphs:
                raise ValueError('Duplicate validation event')
            graphs[evt] = graph
    split = json.loads((DATA / 'event_split.json').read_text(encoding='utf-8'))
    assert set(graphs) == set(split['val_evt_ids'])
    assert not set(graphs) & set(split['test_evt_ids'])
    ids = sorted(graphs)
    inventory = pd.read_csv(DATA / 'track_inventory.csv')
    inventory = inventory.loc[inventory.EvtID.isin(ids) & inventory.hit_count.ge(5)].copy()
    assert len(inventory) == sum(sum(v >= 5 for v in Counter(k for values in truth_sets(g) for k in values).values()) for g in graphs.values())
    models = {
        'baseline': ROOT / 'best_interaction_network_no_time_clean_physics_v2.pth',
        'finetuned': DATA / 'best_interaction_network_no_time_finetune_whole_events_sigma05.pth',
    }
    if args.stage2:
        state = torch.load(DATA / 'training_state_no_time_finetune_stage2_lr3e4.pth', map_location='cpu', weights_only=False)
        last = OUT / f'last_epoch_{state["epoch"]}.pth'
        torch.save(state['model'], last)
        best = DATA / 'best_interaction_network_no_time_finetune_stage2_lr3e4.pth'
        best_values = torch.load(best, map_location='cpu', weights_only=True)
        assert all(torch.equal(best_values[k], v) for k, v in state['best_model'].items())
        models = {'baseline': models['finetuned'], 'finetuned': best, 'last': last, 'oracle': None}
        old = DATA / 'interrupted_no_time_assessment'
        for source, target in [('probabilities_finetuned.pt', 'probabilities_baseline.pt'), ('tracking_finetuned.json', 'tracking_baseline.json')]:
            if not (OUT / target).exists():
                import shutil
                shutil.copyfile(old / source, OUT / target)
    input_signature = [{'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in parts]
    summaries, row_sets = {}, {}
    for name, weight in models.items():
        sha = hashlib.sha256(weight.read_bytes()).hexdigest() if weight else 'truth_labels_offline_oracle'
        signature = {'weight_sha256': sha, 'validation_parts': input_signature}
        snapshot = OUT / f'{name}_{sha[:12]}.pth'
        if weight and not snapshot.exists():
            snapshot.write_bytes(weight.read_bytes())
        cache = OUT / f'probabilities_{name}.pt'
        saved = torch.load(cache, weights_only=False) if cache.exists() else None
        if weight is None:
            values = {evt: graph.y.numpy() for evt, graph in graphs.items()}
            edge_stats = {'offline_truth': True}
        elif saved and saved['signature'] == signature:
            values, edge_stats = saved['values'], saved['edge_stats']
        else:
            model = InteractionNetwork(*model_dimensions('no-time')).to(device)
            model.load_state_dict(torch.load(snapshot, map_location=device, weights_only=True))
            model.eval()
            values = {}
            loss_sum = edges = 0
            grid = np.arange(.30, .951, .05)
            tp = np.zeros(len(grid), dtype=np.int64)
            fp = tp.copy()
            fn = tp.copy()
            criterion = FocalLoss()
            for index, evt in enumerate(ids, 1):
                data = graphs[evt]
                values[evt] = probability = predict(data, model, device, 'no-time')
                labels = data.y.numpy().astype(bool)
                clipped = np.clip(probability, 1e-7, 1 - 1e-7)
                bce = -(labels * np.log(clipped) + ~labels * np.log1p(-clipped))
                loss_sum += float(np.sum(.5 * (1 - np.exp(-bce)) ** 2 * bce))
                edges += len(labels)
                for i, threshold in enumerate(grid):
                    predicted = probability >= threshold
                    tp[i] += int((predicted & labels).sum())
                    fp[i] += int((predicted & ~labels).sum())
                    fn[i] += int((~predicted & labels).sum())
                if index % 100 == 0:
                    print(name, 'inference', index, '/', len(ids), flush=True)
            table = [{'threshold': round(float(t), 2), 'precision': int(a) / max(int(a + b), 1),
                      'recall': int(a) / max(int(a + c), 1), 'f1': int(2 * a) / max(int(2 * a + b + c), 1)}
                     for t, a, b, c in zip(grid, tp, fp, fn)]
            edge_stats = {'loss': loss_sum / edges, 'edges': edges, 'thresholds': table,
                          'at_0p5': next(r for r in table if r['threshold'] == .5),
                          'best_validation_edge_f1': max(table, key=lambda r: r['f1'])}
            torch.save({'signature': signature, 'values': values, 'edge_stats': edge_stats}, cache)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        output = OUT / f'tracking_{name}.json'
        tracking_signature = {**signature, 'thresholds': [STRICT, RELAXED],
                              'tracker_sha256': hashlib.sha256((ROOT / 'STEP4_两级寻迹.py').read_bytes()).hexdigest()}
        saved_report = json.loads(output.read_text(encoding='utf-8')) if output.exists() else None
        if saved_report and saved_report['signature'] == tracking_signature:
            rows = saved_report['events']
        else:
            rows = []
            with ProcessPoolExecutor(max_workers=2, initializer=init_worker) as pool:
                tasks = ((evt, graphs[evt], values[evt]) for evt in ids)
                for index, row in enumerate(pool.map(tracking, tasks, chunksize=1), 1):
                    rows.append(row)
                    if index % 50 == 0:
                        print(name, 'tracking', index, '/', len(ids), flush=True)
            output.write_text(json.dumps({'signature': tracking_signature, 'events': rows}, ensure_ascii=False), encoding='utf-8')
        total = {k: sum(r[k] for r in rows) for k in ['truth_tracks', 'matched_tracks', 'fake_tracks', 'clone_tracks', 'output_tracks']}
        missed = {tuple(k) for row in rows for k in row['missed_tracks']}
        table = inventory.copy()
        table['matched'] = [(int(e), int(t)) not in missed for e, t in zip(table.EvtID, table.TrkID)]
        table['radius_bin'] = pd.cut(table.birth_radius_mm, [0, 1, 5, 15, 60, np.inf], right=False,
                                    labels=['0_1', '1_5', '5_15', '15_60', '60_plus'])
        table['hit_bin'] = pd.cut(table.hit_count, [5, 7, 11, 21, np.inf], right=False,
                                 labels=['5_6', '7_10', '11_20', '21_plus'])
        groups = {}
        for column in ['radius_bin', 'hit_bin']:
            groups[column] = {str(bin_name): {'truth': len(group), 'matched': int(group.matched.sum()),
                              'efficiency': float(group.matched.mean())}
                              for bin_name, group in table.groupby(column, observed=True)}
        summaries[name] = {**total, 'events': len(rows),
                           'efficiency': total['matched_tracks'] / total['truth_tracks'],
                           'fake_rate': total['fake_tracks'] / max(total['output_tracks'], 1),
                           'clone_rate': total['clone_tracks'] / max(total['output_tracks'], 1),
                           'fake_per_event': total['fake_tracks'] / len(rows),
                           'edge': edge_stats, 'by_truth': groups, 'weight_sha256': sha}
        row_sets[name] = rows
        print(name, 'RESULT', json.dumps({k: v for k, v in summaries[name].items() if k not in ['edge', 'by_truth']}, ensure_ascii=False), flush=True)
        (OUT / 'comparison_in_progress.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    base_rows, fine_rows = row_sets['baseline'], row_sets['finetuned']
    assert [r['EvtID'] for r in base_rows] == [r['EvtID'] for r in fine_rows]
    rng = np.random.default_rng(20261014)
    indices = rng.integers(0, len(ids), (1000, len(ids)))
    truth = np.array([r['truth_tracks'] for r in base_rows])
    changes = np.array([f['matched_tracks'] - b['matched_tracks'] for b, f in zip(base_rows, fine_rows)])
    deltas = changes[indices].sum(axis=1) / truth[indices].sum(axis=1)
    report = {'partition': 'validation only; independent test untouched', 'thresholds': [STRICT, RELAXED],
              'truth_matching': '>=5 hits; purity>=0.8; completeness>=0.5; fake/clone denominator=all output tracks',
              'summaries': summaries, 'efficiency_gain_95pct_event_bootstrap': np.quantile(deltas, [.025, .975]).tolist()}
    history_run = 'no_time_finetune_stage2_lr3e4' if args.stage2 else 'no_time_finetune_whole_events_sigma05'
    accumulator = EventAccumulator(str(DATA / 'runs' / history_run), size_guidance={'scalars': 0})
    accumulator.Reload()
    report['training_history'] = {t: [{'epoch': s.step, 'value': s.value, 'wall_time': s.wall_time}
                                     for s in accumulator.Scalars(t)] for t in accumulator.Tags()['scalars']}
    if args.stage2:
        report['checkpoint'] = {'epoch': state['epoch'], 'best_epoch': state['best_epoch'], 'best_loss': state['best_loss']}
        assert abs(summaries['finetuned']['edge']['loss'] - state['best_loss']) < 1e-6
    for summary in summaries.values():
        assert summary['output_tracks'] == summary['matched_tracks'] + summary['fake_tracks'] + summary['clone_tracks']
    (OUT / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('COMPLETE', OUT, flush=True)


if __name__ == '__main__':
    main()
