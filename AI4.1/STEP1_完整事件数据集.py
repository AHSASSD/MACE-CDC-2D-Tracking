"""Reuse AI4 graphs while preserving whole events and measured d."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import ROOT, build_graph, load_cell_map, save_json

OUT = ROOT / 'whole_event_retraining'
TIME_SCALE_NS = 10000.0


def event_graph(frame):
    frame = frame.copy().reset_index(drop=True)
    frame['t_obs'] = frame.t - frame.t.min()
    frame['truth_tracks'] = [[(int(e), int(t))] for e, t in zip(frame.EvtID, frame.TrkID)]
    frame['is_noise'] = False
    frame['shared_hit'] = False
    data = build_graph(frame)
    if data is None:
        raise ValueError('Complete event has too few nodes or no candidate edges')
    data.x[:, 6] *= 300.0 / TIME_SCALE_NS
    data.edge_attr[:, 4:6] *= 300.0 / TIME_SCALE_NS
    data.event_id = torch.tensor([int(frame.EvtID.iloc[0])])
    return data


def prepare_file(path):
    torch.set_num_threads(1)
    inventory = pd.read_csv(OUT / 'event_inventory.csv')
    partition = inventory.set_index('EvtID').partition.to_dict()
    geometry = load_cell_map()[['CellID', 'wire_x', 'wire_y', 'wire_r']]
    columns = ['EvtID', 'TrkID', 'CellID', 'Good', 'CreatProc', 'WireType', 'd', 't']
    frame = pd.read_csv(path, usecols=columns)
    frame = frame.loc[frame.Good.eq(True) & frame.WireType.eq('Axial') &
                      frame.CreatProc.eq('ExtendedDecayWithSpin')].merge(
        geometry, on='CellID', how='left', validate='many_to_one')
    if not np.isfinite(frame[['d', 't', 'wire_x', 'wire_y', 'wire_r']].to_numpy()).all():
        raise ValueError(f'Invalid measured values/geometry: {path}')
    buffers = {name: [] for name in ['train', 'validation', 'test']}
    part_numbers = dict.fromkeys(buffers, 0)
    file_index = Path(path).name.split('_')[1]
    counts = dict.fromkeys(buffers, 0)
    nodes = edges = 0
    for evt, group in frame.groupby('EvtID', sort=True):
        name = partition[int(evt)]
        data = event_graph(group)
        nodes += data.num_nodes
        edges += data.num_edges
        buffers[name].append(data)
        counts[name] += 1
        if len(buffers[name]) == 100:
            target = OUT / f'dataset_{name}.pt.parts' / f'part_{file_index}_{part_numbers[name]:03d}.pt'
            torch.save(buffers[name], target)
            buffers[name] = []
            part_numbers[name] += 1
    for name, buffer in buffers.items():
        if buffer:
            torch.save(buffer, OUT / f'dataset_{name}.pt.parts' / f'part_{file_index}_{part_numbers[name]:03d}.pt')
    print(Path(path).name, counts, 'nodes', nodes, 'edges', edges, flush=True)
    return {'file': Path(path).name, 'counts': counts, 'nodes': nodes, 'edges': edges}


def self_check():
    frame = pd.DataFrame({'EvtID': [7] * 6, 'TrkID': [1] * 3 + [2] * 3,
                          'wire_x': np.arange(6) * 10., 'wire_y': [20.] * 6,
                          'd': [1.] * 6, 't': np.arange(6) * 5. + 1000})
    frame['wire_r'] = np.hypot(frame.wire_x, frame.wire_y)
    frame['CellID'] = np.arange(6)
    first = event_graph(frame)
    shifted = frame.copy()
    shifted['t'] += 100000
    shifted['t0'] = np.arange(6) * 99999
    shifted['d_true'] = 9999
    second = event_graph(shifted)
    assert torch.equal(first.x, second.x) and torch.equal(first.edge_attr, second.edge_attr)
    assert torch.equal(first.y, second.y)
    assert first.x.shape[1] == 7 and first.edge_attr.shape[1] == 6
    print('Self-check passed: event time origin invariant; t0/d_true unused', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.workers < 1:
        raise ValueError('workers must be positive')
    audit = json.loads((OUT / 'data_audit.json').read_text(encoding='utf-8'))
    inventory = pd.read_csv(OUT / 'event_inventory.csv')
    for name in ['train', 'validation', 'test']:
        directory = OUT / f'dataset_{name}.pt.parts'
        directory.mkdir(exist_ok=True)
        if any(directory.iterdir()):
            raise FileExistsError(f'Refusing to overwrite graph shards: {directory}')
    paths = sorted((ROOT / 'whole data').glob('CDCSimHit_*.csv'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(prepare_file, paths))
    split = {f'{name}_evt_ids': inventory.loc[inventory.partition.eq(part), 'EvtID'].astype(int).tolist()
             for name, part in [('train', 'train'), ('val', 'validation'), ('test', 'test')]}
    assert not set(split['train_evt_ids']) & set(split['val_evt_ids'])
    assert not set(split['train_evt_ids']) & set(split['test_evt_ids'])
    assert not set(split['val_evt_ids']) & set(split['test_evt_ids'])
    for name in ['train', 'validation', 'test']:
        count = sum(r['counts'][name] for r in reports)
        assert count == audit['partitions'][name]
    save_json(OUT / 'event_split.json', split)
    save_json(OUT / 'dataset_metadata.json', {
        'profile': 'whole-events-v1', 'source_group_disjoint': True,
        'event_mixing': False, 'source_audit': str(OUT / 'data_audit.json'),
        'distance': {'already_measured': True, 'declared_sigma_mm': .5, 'added_sigma_mm': 0},
        'timing': {'basis': 'observed_event_min', 'scale_ns': TIME_SCALE_NS, 'added_sigma_ns': 0},
        'noise': {'added': False}, 'detector': {'added_dead_time_ns': 0},
        'features': {'time': {'node': 7, 'edge': 6}, 'no-time': {'node': 6, 'edge': 4},
                     'excluded_truth_inputs': ['EvtID', 'TrkID', 'Good', 'CreatProc', 'x0',
                        'd_true', 'd_error_mm', 'is_noise', 'shared_hit', 't0', 'tHit',
                        't_window', 't_rel', 'dt_rel']},
        'summary': {'events': len(inventory), 'partitions': audit['partitions'], 'files': reports}})
    print('Complete whole-event dataset:', OUT, flush=True)


if __name__ == '__main__':
    main()
