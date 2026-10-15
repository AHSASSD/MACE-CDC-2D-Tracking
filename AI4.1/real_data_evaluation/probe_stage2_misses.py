"""Offline diagnostics on a seeded validation subset; truth is never model input."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import fit_2d_circle
from STEP4_两级寻迹 import run_two_pass, truth_sets, evaluate
from STEP4_困难事件分析 import graph_components, best_track_match, classify_miss

DATA = ROOT / 'whole_event_retraining'
OUT = DATA / 'stage2_no_time_assessment'


def main():
    torch.set_num_threads(1)
    split = json.loads((DATA / 'event_split.json').read_text(encoding='utf-8'))
    ids = set(map(int, np.random.default_rng(20261014).choice(sorted(split['val_evt_ids']), 200, replace=False)))
    assert not ids & set(split['test_evt_ids'])
    graphs = {}
    for path in sorted((DATA / 'dataset_validation.pt.parts').glob('part_*.pt')):
        for graph in torch.load(path, weights_only=False):
            evt = int(graph.event_id.item())
            if evt in ids:
                graphs[evt] = graph
    saved = torch.load(OUT / 'probabilities_finetuned.pt', weights_only=False)
    assert len(graphs) == len(ids)
    frames = []
    columns = ['EvtID', 'TrkID', 'CellID', 'Good', 'WireType', 'CreatProc', 'd', 'd_true']
    for path in sorted((ROOT / 'whole data').glob('CDCSimHit_*.csv')):
        for chunk in pd.read_csv(path, usecols=columns, chunksize=100000):
            frames.append(chunk.loc[chunk.EvtID.isin(ids) & chunk.Good.eq(True) & chunk.WireType.eq('Axial') & chunk.CreatProc.eq('ExtendedDecayWithSpin')])
    frames = {int(evt): frame.reset_index(drop=True) for evt, frame in pd.concat(frames).groupby('EvtID', sort=True)}
    causes = Counter()
    cause_by_hits = defaultdict(Counter)
    fits = defaultdict(list)
    alternate = []
    event_results = []
    for index, evt in enumerate(sorted(ids), 1):
        graph, frame = graphs[evt], frames[evt]
        associations = truth_sets(graph)
        assert np.array_equal(graph.cell_id.numpy(), frame.CellID.to_numpy())
        assert np.allclose(graph.drift_d.numpy(), frame.d.to_numpy())
        assert [next(iter(k)) for k in associations] == list(zip(frame.EvtID, frame.TrkID))
        probability = saved['values'][evt]
        tracks, _ = run_two_pass(graph, None, None, 'no-time', .70, .575, probability)
        result = evaluate(graph, tracks)
        event_results.append({'EvtID': evt, **result})
        strict, smap = graph_components(graph, probability, .70)
        relaxed, rmap = graph_components(graph, probability, .575)
        nodes = defaultdict(set)
        for i, keys in enumerate(associations):
            for key in keys:
                nodes[key].add(i)
        missed = set(map(tuple, result['missed_tracks']))
        for key, members in nodes.items():
            if len(members) < 5:
                continue
            hit_bin = '21_plus' if len(members) >= 21 else '5_20'
            if key in missed:
                best = best_track_match(key, members, tracks, associations)
                cause, _ = classify_miss(members, associations, strict, smap, relaxed, rmap, best)
                causes[cause] += 1
                cause_by_hits[hit_bin][cause] += 1
            indices = sorted(members)
            x, y = graph.wire_x[indices].numpy().astype(np.float64), graph.wire_y[indices].numpy().astype(np.float64)
            for label, distances in [('measured_sigma05', frame.d.to_numpy()[indices]), ('true_distance', frame.d_true.to_numpy()[indices])]:
                fit = fit_2d_circle(x, y, distances)
                rms = np.inf if fit is None else float(np.sqrt(np.mean((np.abs(np.hypot(x - fit[0], y - fit[1]) - fit[2]) - distances) ** 2)))
                fits[(hit_bin, label)].append(rms)
                if label == 'true_distance' and rms > 1.5 and (hit_bin == '21_plus' or sum(r['hit_bin'] == '5_20' for r in alternate) < 200):
                    # Diagnostic only: an independent wire-circle seed tests initialization failures.
                    design = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
                    a, b, c = np.linalg.lstsq(design, x * x + y * y, rcond=None)[0]
                    radius = np.sqrt(max(c + a * a + b * b, 1e-6))
                    residual = lambda p: np.abs(np.hypot(x - p[0], y - p[1]) - p[2]) - distances
                    best_rms = rms
                    for factor in [.8, 1., 1.2]:
                        candidate = least_squares(residual, [a, b, radius * factor], bounds=([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, np.inf]), max_nfev=1000, ftol=1e-8, xtol=1e-8, gtol=1e-8)
                        best_rms = min(best_rms, float(np.sqrt(np.mean(candidate.fun ** 2))))
                    alternate.append({'truth': list(key), 'hit_bin': hit_bin, 'hits': len(indices), 'original_rms_mm': rms, 'alternate_rms_mm': best_rms})
        if index % 25 == 0:
            print('probe', index, '/', len(ids), flush=True)
    fit_summary = {f'{bin_name}:{label}': {'tracks': len(values), 'rms_gt_1mm': sum(x > 1 for x in values), 'rms_gt_1p5mm': sum(x > 1.5 for x in values), 'median_rms_mm': float(np.median(values))} for (bin_name, label), values in fits.items()}
    summary = {'seed': 20261014, 'event_ids': sorted(ids), 'partition': 'validation', 'probability_signature': saved['signature'], 'causes': dict(causes), 'cause_by_hits': {k: dict(v) for k, v in cause_by_hits.items()}, 'truth_group_circle_fits': fit_summary, 'alternate_initialization': alternate, 'events': event_results, 'note': 'Cause labels are rule-based symptoms, not independent causal attribution. Truth-distance fits use known true groups, unchanged fitter, and no trimming; this is not deployable tracking. Alternate initialization probes the first 200 ordinary failed fits and every long failed fit in the seeded sample, not a representative success rate.'}
    summary['truth_group_circle_fit_precision'] = 'float64 throughout; corrected mixed float32 coordinates / float64 distances'
    assert sum(causes.values()) == sum(r['truth_tracks'] - r['matched_tracks'] for r in event_results)
    (OUT / 'miss_probes.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['causes', 'cause_by_hits', 'truth_group_circle_fits']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
