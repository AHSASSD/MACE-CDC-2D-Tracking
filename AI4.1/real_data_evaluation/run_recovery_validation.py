"""Ordered validation-only ablations for extension, merging and residual Hough."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from STEP6_事件切片可视化 import load_graphs
from STEP4_两级寻迹 import run_two_pass, evaluate
from fragment_recovery import recover_fragments, hough_remaining
import fragment_recovery

DATA = ROOT / 'whole_event_retraining'
ASSESS = DATA / 'stage2_no_time_assessment'
OUT = DATA / 'recovery_validation'
PROFILES = ['extend_geometry', 'merge_geometry', 'merge_time']


def first_phase_signature():
    code = ''.join(inspect.getsource(getattr(fragment_recovery, name)) for name in ['measurements', 'residual', 'wire_seed', 'fit_members', 'candidate', 'recover_fragments'])
    return hashlib.sha256(code.encode()).hexdigest()


def init_worker():
    torch.set_num_threads(1)


def metrics(data, tracks):
    mixed = fragments = 0
    tagged = copy.deepcopy(tracks)
    result = evaluate(data, tagged)
    for track in tagged:
        if track['purity'] < .8:
            mixed += 1
        elif track['completeness'] < .5:
            fragments += 1
    return {**result, 'output_tracks': len(tracks), 'mixed_candidates': mixed, 'incomplete_candidates': fragments}


def worker(task):
    evt, data, probability, base, profiles, parent = task
    started = time.perf_counter()
    if base is None:
        base, _ = run_two_pass(data, None, None, 'no-time', .70, .575, probability)
    rows = {'baseline': metrics(data, base)}
    outputs = {'baseline': base}
    for profile in profiles:
        if profile == 'baseline':
            continue
        if profile in ['hough_geometry', 'hough_time']:
            tracks = hough_remaining(data, outputs[parent], probability, use_time=profile == 'hough_time', seed=20261014+evt)
        else:
            tracks = recover_fragments(data, base, probability, use_time=profile == 'merge_time', merge=profile != 'extend_geometry')
        rows[profile] = metrics(data, tracks)
        outputs[profile] = tracks
    return {'EvtID': evt, 'metrics': rows, 'tracks': outputs, 'elapsed_seconds': time.perf_counter()-started}


def summarize(rows, profile):
    results = [r['metrics'][profile] for r in rows]
    keys = ['truth_tracks', 'matched_tracks', 'fake_tracks', 'clone_tracks', 'output_tracks', 'mixed_candidates', 'incomplete_candidates']
    total = {key: sum(r[key] for r in results) for key in keys}
    assert total['output_tracks'] == total['matched_tracks']+total['fake_tracks']+total['clone_tracks']
    return {**total, 'events': len(rows), 'efficiency': total['matched_tracks']/total['truth_tracks'],
            'fake_rate': total['fake_tracks']/max(total['output_tracks'], 1),
            'clone_rate': total['clone_tracks']/max(total['output_tracks'], 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['pilot', 'confirm', 'hough-pilot', 'hough'], required=True)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    split = json.loads((DATA / 'event_split.json').read_text(encoding='utf-8'))
    validation = sorted(split['val_evt_ids'])
    pilot = set(map(int, np.random.default_rng(20261014).choice(validation, 200, replace=False)))
    pilot.update([174, 177, 188])
    ids = sorted(pilot if args.phase in ['pilot', 'hough-pilot'] else set(validation)-pilot)
    assert not set(ids) & set(split['test_evt_ids'])
    assert set(ids) <= set(validation)
    probability = torch.load(ASSESS / 'probabilities_last.pt', map_location='cpu', weights_only=False)
    weight_hash = hashlib.sha256((ASSESS / 'last_epoch_62.pth').read_bytes()).hexdigest()
    assert probability['signature']['weight_sha256'] == weight_hash
    parts = sorted((DATA / 'dataset_validation.pt.parts').glob('part_*.pt'))
    assert probability['signature']['validation_parts'] == [{'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in parts]
    signature = {**probability['signature'], 'tracker_sha256': hashlib.sha256((ROOT / 'STEP4_两级寻迹.py').read_bytes()).hexdigest(),
                 'common_sha256': hashlib.sha256((ROOT / 'common.py').read_bytes()).hexdigest()}
    cache_path = OUT / 'baseline_tracks.pt'
    cached = torch.load(cache_path, weights_only=False) if cache_path.exists() else {'signature': signature, 'values': {}}
    if cached['signature'] != signature:
        raise ValueError('基线缓存不匹配，需使用新的实验输出目录')
    profiles = PROFILES
    parent = 'baseline'
    if args.phase == 'confirm':
        selection = json.loads((OUT / 'pilot.json').read_text(encoding='utf-8'))
        assert selection['signature'] == signature
        assert selection['first_phase_function_sha256'] == first_phase_signature()
        profiles = [selection['selected_profile']]
    elif args.phase in ['hough-pilot', 'hough']:
        selection = json.loads((OUT / 'pilot.json').read_text(encoding='utf-8'))
        assert selection['signature'] == signature
        assert selection['first_phase_function_sha256'] == first_phase_signature()
        parent = selection['selected_profile']
        if args.phase == 'hough-pilot':
            profiles = [parent, 'hough_geometry', 'hough_time']
        else:
            hough_selection = json.loads((OUT / 'hough-pilot.json').read_text(encoding='utf-8'))
            assert hough_selection['signature'] == signature
            assert hough_selection['recovery_sha256'] == hashlib.sha256((ROOT / 'fragment_recovery.py').read_bytes()).hexdigest()
            profiles = list(dict.fromkeys([parent, hough_selection['selected_profile']]))
    graphs = load_graphs({evt: 'validation' for evt in ids})
    reference = {r['EvtID']: r for r in json.loads((ASSESS / 'tracking_last.json').read_text(encoding='utf-8'))['events']}
    rows = []
    with ProcessPoolExecutor(max_workers=2, initializer=init_worker) as pool:
        tasks = ((evt, graphs[evt], probability['values'][evt], cached['values'].get(evt), profiles, parent) for evt in ids)
        for i, row in enumerate(pool.map(worker, tasks, chunksize=1), 1):
            evt = row['EvtID']
            for key in ['truth_tracks', 'matched_tracks', 'fake_tracks', 'clone_tracks', 'missed_tracks']:
                assert row['metrics']['baseline'][key] == reference[evt][key]
            cached['values'][evt] = row['tracks']['baseline']
            rows.append(row)
            if i % 25 == 0:
                torch.save(cached, cache_path.with_suffix('.tmp'))
                cache_path.with_suffix('.tmp').replace(cache_path)
                print(args.phase, i, '/', len(ids), flush=True)
    torch.save(cached, cache_path.with_suffix('.tmp'))
    cache_path.with_suffix('.tmp').replace(cache_path)
    summaries = {profile: summarize(rows, profile) for profile in ['baseline']+[p for p in profiles if p != 'baseline']}
    report = {'phase': args.phase, 'partition': 'validation only; internal pilot/confirmation split, not an independent test',
              'pilot_event_ids': sorted(pilot), 'event_ids': ids, 'signature': signature,
              'recovery_sha256': hashlib.sha256((ROOT / 'fragment_recovery.py').read_bytes()).hexdigest(),
              'first_phase_function_sha256': first_phase_signature(),
              'summaries': summaries, 'events': rows}
    if args.phase == 'pilot':
        baseline = summaries['baseline']
        eligible = [p for p in PROFILES if summaries[p]['fake_tracks'] <= baseline['fake_tracks'] and summaries[p]['clone_tracks'] <= baseline['clone_tracks']]
        report['selected_profile'] = max(['baseline']+eligible, key=lambda p: summaries[p]['matched_tracks'])
        report['selection_rule'] = 'maximize matched tracks, with no increase in fake or clone count on pilot; ties keep baseline'
    elif args.phase == 'hough-pilot':
        reference_summary = summaries[parent]
        eligible = [p for p in ['hough_geometry', 'hough_time'] if summaries[p]['fake_tracks'] <= reference_summary['fake_tracks'] and summaries[p]['clone_tracks'] <= reference_summary['clone_tracks']]
        report['selected_profile'] = max([parent]+eligible, key=lambda p: summaries[p]['matched_tracks'])
        report['selection_rule'] = 'maximize matched tracks, with no increase in fake or clone count relative to frozen fragment-recovery parent'
    report['hough_parent'] = parent
    for profile in summaries:
        if profile == 'baseline':
            continue
        recovered = lost = 0
        for row in rows:
            before = set(map(tuple, row['metrics']['baseline']['missed_tracks']))
            after = set(map(tuple, row['metrics'][profile]['missed_tracks']))
            recovered += len(before-after)
            lost += len(after-before)
        report['summaries'][profile].update({'newly_recovered': recovered, 'newly_lost': lost})
        assert recovered-lost == summaries[profile]['matched_tracks']-summaries['baseline']['matched_tracks']
    (OUT / f'{args.phase}.json').write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('RESULT', json.dumps({'summaries': summaries, 'selected': report.get('selected_profile')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
