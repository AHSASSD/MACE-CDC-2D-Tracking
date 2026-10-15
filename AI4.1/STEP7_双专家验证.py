"""固定no-time空间专家与新训练time专家的概率融合基线；不是已训练门控MoE。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from common import ROOT, InteractionNetwork, model_dimensions
from STEP4_两级寻迹 import predict, run_two_pass, evaluate
from STEP6_事件切片可视化 import load_graphs

DATA = ROOT / 'whole_event_retraining'
ASSESS = DATA / 'stage2_no_time_assessment'
OUT = DATA / 'dual_expert_validation'


def fuse_probabilities(spatial, timing, timing_weight):
    spatial, timing = np.asarray(spatial), np.asarray(timing)
    if spatial.shape != timing.shape or not 0 <= timing_weight <= 1:
        raise ValueError('专家概率形状或融合权重无效')
    if not np.isfinite(spatial).all() or not np.isfinite(timing).all() or np.any((spatial < 0) | (spatial > 1)) or np.any((timing < 0) | (timing > 1)):
        raise ValueError('专家输出必须是有限的0～1概率')
    return (1-timing_weight)*spatial+timing_weight*timing


def init_worker():
    torch.set_num_threads(1)


def evaluate_event(task):
    evt, graph, spatial, timing, weights = task
    results = {}
    for weight in weights:
        probability = fuse_probabilities(spatial, timing, weight)
        tracks, _ = run_two_pass(graph, None, None, 'no-time', .70, .575, probability)
        results[str(weight)] = {**evaluate(graph, tracks), 'output_tracks': len(tracks)}
    return {'EvtID': evt, 'results': results}


def summary(rows, weight):
    keys = ['truth_tracks', 'matched_tracks', 'fake_tracks', 'clone_tracks', 'output_tracks']
    total = {key: sum(r['results'][str(weight)][key] for r in rows) for key in keys}
    assert total['output_tracks'] == total['matched_tracks']+total['fake_tracks']+total['clone_tracks']
    return {**total, 'efficiency': total['matched_tracks']/total['truth_tracks'], 'fake_rate': total['fake_tracks']/max(total['output_tracks'], 1), 'clone_rate': total['clone_tracks']/max(total['output_tracks'], 1)}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time-model', type=Path, default=DATA / 'best_interaction_network_time_expert_whole_events_sigma05.pth', help='time权重或完整训练断点；完整断点使用最后一轮model')
    args = parser.parse_args()
    if not args.time_model.exists():
        parser.error('缺少新数据time专家权重。先按回收实验说明启动time微调；原始time权重不能当成已适配新数据的专家。')
    time_hash = hashlib.sha256(args.time_model.read_bytes()).hexdigest()
    OUT = OUT / time_hash[:12]
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((DATA / 'event_split.json').read_text(encoding='utf-8'))
    ids = sorted(split['val_evt_ids'])
    assert not set(ids) & set(split['test_evt_ids'])
    graph_map = load_graphs({evt: 'validation' for evt in ids})
    spatial = torch.load(ASSESS / 'probabilities_last.pt', weights_only=False)
    assert spatial['signature']['weight_sha256'] == hashlib.sha256((ASSESS / 'last_epoch_62.pth').read_bytes()).hexdigest()
    parts = sorted((DATA / 'dataset_validation.pt.parts').glob('part_*.pt'))
    assert spatial['signature']['validation_parts'] == [{'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in parts]
    signature = {**spatial['signature'], 'time_weight_sha256': time_hash}
    path = OUT / 'time_probabilities.pt'
    timing = torch.load(path, weights_only=False) if path.exists() else None
    if timing is None or timing['signature'] != signature:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = InteractionNetwork(*model_dimensions('time')).to(device)
        saved = torch.load(args.time_model, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'] if 'model' in saved and 'optimizer' in saved else saved)
        model.eval()
        values = {}
        for i, evt in enumerate(ids, 1):
            values[evt] = predict(graph_map[evt], model, device, 'time')
            if i % 100 == 0:
                print('time专家推理', i, '/', len(ids), flush=True)
        timing = {'signature': signature, 'values': values}
        torch.save(timing, path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    pilot = set(json.loads((DATA / 'recovery_validation/pilot.json').read_text(encoding='utf-8'))['pilot_event_ids'])
    pilot_rows = []
    weights = [0., .25, .5, .75, 1.]
    with ProcessPoolExecutor(max_workers=2, initializer=init_worker) as pool:
        tasks = ((evt, graph_map[evt], spatial['values'][evt], timing['values'][evt], weights) for evt in sorted(pilot))
        for row in pool.map(evaluate_event, tasks, chunksize=1):
            pilot_rows.append(row)
    summaries = {str(w): summary(pilot_rows, w) for w in weights}
    baseline = summaries['0.0']
    eligible = [w for w in weights if summaries[str(w)]['fake_tracks'] <= baseline['fake_tracks'] and summaries[str(w)]['clone_tracks'] <= baseline['clone_tracks']]
    chosen = max(eligible, key=lambda w: summaries[str(w)]['matched_tracks'])
    confirmation = []
    with ProcessPoolExecutor(max_workers=2, initializer=init_worker) as pool:
        tasks = ((evt, graph_map[evt], spatial['values'][evt], timing['values'][evt], sorted(set([0., 1., chosen]))) for evt in ids if evt not in pilot)
        for row in pool.map(evaluate_event, tasks, chunksize=1):
            confirmation.append(row)
            if len(confirmation) % 100 == 0:
                print('双专家确认', len(confirmation), flush=True)
    report = {'type': 'two-view fixed-weight ensemble baseline; not a learned gate', 'partition': 'validation only', 'signature': signature,
              'time_model_path': str(args.time_model.resolve()),
              'selected_time_weight': chosen, 'pilot': summaries,
              'confirmation': {str(w): summary(confirmation, w) for w in sorted(set([0., 1., chosen]))},
              'pilot_events': pilot_rows, 'confirmation_events': confirmation}
    (OUT / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['confirmation'], ensure_ascii=False), flush=True)
    print('双专家报告：', OUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
