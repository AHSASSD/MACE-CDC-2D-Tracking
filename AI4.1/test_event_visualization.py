"""Boundary/edge-direction checks and agreement with the frozen real-event assessment."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile

import numpy as np
import pandas as pd
import torch

from STEP6_事件切片可视化 import ROOT, select_ids, undirected_edges, circle_arc, nearby_truth_segments, load_external_source, event_graph
from STEP6_绘图界面 import build_command


class VisualizationTests(unittest.TestCase):
    def test_external_source_preserves_event_identity_and_rejects_cross_file_overlap(self):
        frame = pd.DataFrame({'evtid': [1503]*6 + [1504], 'TrkID': [1]*3 + [2]*3 + [9],
                              'CellID': range(7), 'Good': [' TRUE ']*7, 'WireType': ['Axial']*7,
                              'CreatProc': ['ExtendedDecayWithSpin']*7, 'd': [0.5]*7,
                              't': np.arange(7)*5. + 1000, 'tHit': np.arange(7)*5.,
                              'x': ['{10,20,0}']*7})
        geometry = pd.DataFrame({'CellID': range(7), 'x': np.arange(7)*10., 'y': [20.]*7, 'u': [0]*7, 'v': [0]*7})
        with tempfile.TemporaryDirectory() as directory, patch('STEP6_事件切片可视化.pd.read_excel', return_value=geometry):
            path = Path(directory)/'hits.csv'
            frame.to_csv(path, index=False)
            source = load_external_source(path, 1503, 1503)
            self.assertEqual(set(source), {1503})
            selected = source[1503]
            self.assertEqual(selected.HitID.tolist(), list(range(2, 8)))
            self.assertEqual(selected.TrkID.tolist(), [1]*3 + [2]*3)
            graph = event_graph(selected)
            self.assertEqual(graph.num_nodes, 6)
            self.assertEqual(graph.event_id.item(), 1503)
            self.assertEqual(set(map(tuple, graph.truth_pairs.tolist())), {(1503, 1), (1503, 2)})
            pd.DataFrame({'not_hits': [1]}).to_csv(Path(directory)/'other.csv', index=False)
            self.assertEqual(len(load_external_source(Path(directory), 1503, 1503)[1503]), 6)
            frame.to_csv(Path(directory)/'duplicate.csv', index=False)
            with self.assertRaisesRegex(ValueError, '同时出现在'):
                load_external_source(Path(directory), 1503, 1503)

    def test_gui_command_uses_current_python_and_external_split(self):
        import sys
        values = dict(start='1503', end='1510', strict='.7', relaxed='.575', recovery='hough', model='last',
                      source='D:/new source/hits.xlsx', cell_map='', sheet='Hits', split='validation', track='', output='')
        command, output = build_command(values)
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[command.index('--source')+1], values['source'])
        self.assertEqual(command[command.index('--split')+1], 'all')
        self.assertEqual(command[command.index('--sheet')+1], 'Hits')
        self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            build_command({**values, 'end': '1502'})

    def test_interval_is_inclusive_and_partitioned(self):
        split = {'train_evt_ids': [150], 'val_evt_ids': [99, 100, 200, 201], 'test_evt_ids': [151]}
        self.assertEqual(select_ids(split, 100, 200, 'validation'), {100: 'validation', 200: 'validation'})
        self.assertEqual(set(select_ids(split, 100, 200, 'all')), {100, 150, 151, 200})
        with self.assertRaises(ValueError):
            select_ids(split, 200, 100, 'all')

    def test_either_direction_keeps_an_undirected_connection(self):
        graph = SimpleNamespace(edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), y=torch.tensor([1., 1., 0., 0.]))
        pairs, probability, labels = undirected_edges(graph, np.array([.2, .75, .85, .1]))
        np.testing.assert_array_equal(pairs, [[0, 1], [1, 2]])
        np.testing.assert_allclose(probability, [.75, .85])
        np.testing.assert_array_equal(labels, [True, False])

    def test_circle_arc_crosses_angle_zero_without_drawing_almost_a_full_circle(self):
        theta = np.deg2rad([350, 0, 10])
        positions = np.column_stack([100*np.cos(theta), 100*np.sin(theta)])
        arc = circle_arc([0, 0, 100], positions)
        np.testing.assert_allclose(np.linalg.norm(arc, axis=1), 100)
        self.assertLess(np.linalg.norm(np.diff(arc, axis=0), axis=1).sum(), 60)

    def test_truth_drawing_does_not_bridge_unobserved_gaps(self):
        positions = np.array([[0., 0.], [10., 0.], [300., 0.], [310., 0.]])
        segments = nearby_truth_segments(positions, 50)
        np.testing.assert_array_equal(segments, [[[0., 0.], [10., 0.]], [[300., 0.], [310., 0.]]])
        self.assertEqual(len(positions), 4)
        self.assertEqual(nearby_truth_segments(positions[:1], 50).shape, (0, 2, 2))

    def test_real_slice_matches_existing_tracking(self):
        directory = ROOT / 'whole_event_retraining/event_visualizations/last_validation_100_200'
        if not (directory / 'summary.json').exists():
            self.skipTest('Optional event-gallery fixture is not distributed')
        summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        reference = json.loads((ROOT / 'whole_event_retraining/stage2_no_time_assessment/tracking_last.json').read_text(encoding='utf-8'))
        reference = {r['EvtID']: r for r in reference['events']}
        expected = {evt for evt in reference if 100 <= evt <= 200}
        self.assertEqual({r['EvtID'] for r in summary['events']}, expected)
        for row in summary['events']:
            evt = row['EvtID']
            for key in ['truth_tracks', 'matched_tracks', 'fake_tracks', 'clone_tracks', 'missed_tracks']:
                self.assertEqual(row[key], reference[evt][key])
            record = json.loads((directory / f'event_{evt}.json').read_text(encoding='utf-8'))
            self.assertEqual(len(record['miss_diagnosis']), row['truth_tracks']-row['matched_tracks'])
            for circle in record['circles']:
                self.assertEqual(circle['indices'], [h['node_index'] for h in circle['source_hits']])
                self.assertTrue(all(h['EvtID'] == evt for h in circle['source_hits']))
            self.assertGreater((directory / row['png']).stat().st_size, 10000)


if __name__ == '__main__':
    unittest.main()
