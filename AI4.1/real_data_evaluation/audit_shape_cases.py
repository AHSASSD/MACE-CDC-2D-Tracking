"""Audit the three supplied tracks and separate pure fragments from mixed fake candidates."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import fit_2d_circle
from STEP4_两级寻迹 import run_two_pass, evaluate, truth_sets, components
from STEP6_事件切片可视化 import load_graphs

DATA = ROOT / 'whole_event_retraining'
OUT = DATA / 'shape_audit'
ASSESS = DATA / 'stage2_no_time_assessment'
CASES = [(174, 1637), (177, 4532), (188, 3769)]


def vector(value):
    result = np.array([float(v) for v in value.strip('{}').split(',')])
    assert result.shape == (3,) and np.isfinite(result).all()
    return result


def independent_fit(x, y, d):
    x, y, d = (np.asarray(value, dtype=np.float64) for value in [x, y, d])
    a, b, c = np.linalg.lstsq(np.column_stack([2*x, 2*y, np.ones(len(x))]), x*x+y*y, rcond=None)[0]
    radius = np.sqrt(max(c+a*a+b*b, 1e-6))
    residual = lambda p: np.abs(np.hypot(x-p[0], y-p[1])-p[2])-d
    attempts = [least_squares(residual, [a, b, radius*factor], bounds=([-np.inf, -np.inf, .001], [np.inf, np.inf, np.inf]), max_nfev=2000) for factor in [.8, 1, 1.2]]
    best = min(attempts, key=lambda value: np.mean(value.fun**2))
    return best.x.tolist(), float(np.sqrt(np.mean(best.fun**2)))


def cases():
    graphs = load_graphs({evt: 'validation' for evt, _ in CASES})
    results = []
    for evt, trk in CASES:
        frame = pd.read_csv(OUT / f'event_{evt}_track_{trk}_all_hits.csv')
        selected = frame.loc[frame.Good.eq(True) & frame.WireType.eq('Axial')].sort_values('tHit', kind='stable').reset_index(drop=True)
        xyz = np.array([vector(v) for v in selected.x])
        momentum = np.array([vector(v) for v in selected.p])
        graph = graphs[evt]
        associations = truth_sets(graph)
        indices = [i for i, keys in enumerate(associations) if (evt, trk) in keys]
        assert len(indices) == len(selected)
        # CSV order and graph order are both the original hit order, not time sorted.
        original = frame.loc[frame.Good.eq(True) & frame.WireType.eq('Axial')]
        assert np.array_equal(graph.cell_id[indices].numpy(), original.CellID.to_numpy())
        wire_x, wire_y = graph.wire_x[indices].numpy(), graph.wire_y[indices].numpy()
        fits = {}
        for label, distances in [('observed', original.d.to_numpy()), ('true_distance', original.d_true.to_numpy())]:
            old = fit_2d_circle(wire_x, wire_y, distances.astype(np.float32))
            old_rms = float(np.sqrt(np.mean((np.abs(np.hypot(wire_x-old[0], wire_y-old[1])-old[2])-distances)**2)))
            new, new_rms = independent_fit(wire_x, wire_y, distances)
            precise = fit_2d_circle(wire_x.astype(np.float64), wire_y.astype(np.float64), distances.astype(np.float64))
            precise_rms = float(np.sqrt(np.mean((np.abs(np.hypot(wire_x.astype(np.float64)-precise[0], wire_y.astype(np.float64)-precise[1])-precise[2])-distances)**2)))
            fits[label] = {'original_rms_mm': old_rms, 'original_fitter_float64_rms_mm': precise_rms, 'new_seed_rms_mm': new_rms, 'new_seed_circle': new}
        point_fit, point_rms = independent_fit(xyz[:, 0], xyz[:, 1], np.zeros(len(xyz)))
        hop = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        dt = np.diff(selected.tHit.to_numpy())
        jumps = []
        for index in np.flatnonzero(hop > 50):
            jumps.append({'from_HitID': int(selected.HitID.iloc[index]), 'to_HitID': int(selected.HitID.iloc[index+1]), 'separation_3d_mm': float(hop[index]), 'delta_tHit_ns': float(dt[index]), 'Ek_from_MeV': float(selected.Ek.iloc[index]), 'Ek_to_MeV': float(selected.Ek.iloc[index+1])})
        identities = frame[['SourceShard', 'NativeEvtID', 'PoolEvtID', 'PDGID', 'x0', 'Ek0']].drop_duplicates()
        assert len(identities) == 1
        result = {'EvtID': evt, 'TrkID': trk, 'all_hits': len(frame), 'selected_axial_hits': len(selected), 'unique_source_particle_identity': identities.iloc[0].to_dict(),
                  't0_range_ns': float(frame.t0.max()-frame.t0.min()), 'Ek_min_MeV': float(selected.Ek.min()), 'Ek_max_MeV': float(selected.Ek.max()),
                  'tHit_span_ns': float(selected.tHit.max()-selected.tHit.min()), 'energy_increases_gt_1MeV_in_tHit_order': int(np.count_nonzero(np.diff(selected.Ek.to_numpy())>1)),
                  'repeated_selected_CellID_hits': int(selected.CellID.duplicated().sum()), 'point_circle_rms_mm': point_rms, 'point_circle': point_fit, 'drift_circle_fits': fits,
                  'plot_chords_longer_than_50mm': jumps}
        oracle_components = components(graph, graph.y.numpy(), .70)
        result['true_edge_connected_component_sizes'] = sorted([len(set(indices)&set(part)) for part in oracle_components if set(indices)&set(part)], reverse=True)
        results.append(result)
        selected.to_csv(OUT / f'event_{evt}_track_{trk}_time_sorted.csv', index=False, encoding='utf-8-sig')
    (OUT / 'three_cases.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('CASES', json.dumps(results, ensure_ascii=False), flush=True)


def worker_init():
    torch.set_num_threads(1)


def fake_audit(task):
    evt, graph, probability = task
    tracks, _ = run_two_pass(graph, None, None, 'no-time', .70, .575, probability)
    result = evaluate(graph, tracks)
    sizes = Counter(key for keys in truth_sets(graph) for key in keys)
    counts = Counter()
    details = []
    for track in tracks:
        key = tuple(track['truth'])
        if sizes[key] >= 5 and track['purity'] >= .8 and track['completeness'] >= .5:
            continue
        cause = 'mixed' if track['purity'] < .8 else 'short_truth' if sizes[key] < 5 else 'pure_incomplete'
        length = 'long_21_plus' if sizes[key] >= 21 else 'ordinary_5_20'
        counts[f'{cause}:{length}'] += 1
        details.append({'truth': list(key), 'truth_hits': sizes[key], 'candidate_hits': track['hits'], 'purity': track['purity'], 'completeness': track['completeness'], 'rms_mm': track['rms'], 'cause': cause})
    assert sum(counts.values()) == result['fake_tracks']
    missed_long = sum(sizes[tuple(key)] >= 21 for key in result['missed_tracks'])
    return {'EvtID': evt, **result, 'missed_long_21_plus': missed_long, 'missed_ordinary_5_20': len(result['missed_tracks'])-missed_long, 'fake_causes': dict(counts), 'fake_candidates': details}


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    cases()
    split = json.loads((DATA / 'event_split.json').read_text(encoding='utf-8'))
    graphs = load_graphs({evt: 'validation' for evt in split['val_evt_ids']})
    cached = torch.load(ASSESS / 'probabilities_last.pt', weights_only=False)
    reference = json.loads((ASSESS / 'tracking_last.json').read_text(encoding='utf-8'))
    assert reference['signature']['weight_sha256'] == cached['signature']['weight_sha256']
    reference = {r['EvtID']: r for r in reference['events']}
    rows = []
    with ProcessPoolExecutor(max_workers=2, initializer=worker_init) as pool:
        for i, row in enumerate(pool.map(fake_audit, ((evt, graphs[evt], cached['values'][evt]) for evt in sorted(graphs)), chunksize=1), 1):
            for key in ['truth_tracks', 'matched_tracks', 'fake_tracks', 'clone_tracks', 'missed_tracks']:
                assert row[key] == reference[row['EvtID']][key]
            rows.append(row)
            if i % 100 == 0:
                print('fake audit', i, '/', len(graphs), flush=True)
    counts = Counter()
    for row in rows:
        counts.update(row['fake_causes'])
    report = {'partition': 'validation', 'events': len(rows), 'fake_causes': dict(counts), 'missed_long_21_plus': sum(r['missed_long_21_plus'] for r in rows), 'missed_ordinary_5_20': sum(r['missed_ordinary_5_20'] for r in rows), 'event_results': rows, 'probability_signature': cached['signature']}
    assert sum(counts.values()) == sum(r['fake_tracks'] for r in rows)
    (OUT / 'fake_fragment_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SUMMARY', json.dumps({k: v for k, v in report.items() if k not in ['event_results', 'probability_signature']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
