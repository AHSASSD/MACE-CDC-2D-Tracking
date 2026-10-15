import copy
import unittest

import numpy as np
import torch

from common import build_graph
from STEP1_完整事件数据集 import event_graph
from fragment_recovery import tangent_circles, candidate, measurements, recover_fragments, hough_remaining
from STEP7_双专家验证 import fuse_probabilities
import pandas as pd


class RecoveryTests(unittest.TestCase):
    def test_expert_fusion_endpoints_and_validation(self):
        a, b = np.array([.1, .9]), np.array([.7, .3])
        np.testing.assert_array_equal(fuse_probabilities(a, b, 0), a)
        np.testing.assert_array_equal(fuse_probabilities(a, b, 1), b)
        np.testing.assert_allclose(fuse_probabilities(a, b, .5), [.4, .6])
        with self.assertRaises(ValueError):
            fuse_probabilities(a, np.array([np.nan, .3]), .5)

    def test_drift_circle_hypotheses_include_known_circle(self):
        angles = np.array([.1, 1.5, 3.])
        d = np.array([1., 2., 3.])
        xy = np.column_stack([100+(200+d)*np.cos(angles), 50+(200+d)*np.sin(angles)])
        fits = tangent_circles(xy, d)
        self.assertTrue(any(np.linalg.norm(fit-np.array([100., 50., 200.])) < 1e-5 for fit in fits))

    def test_time_gate_and_truth_blindness(self):
        angles = np.linspace(.4, 1.5, 8)
        distance = np.linspace(.3, 2, 8)
        x = 100+(200+distance)*np.cos(angles)
        y = 50+(200+distance)*np.sin(angles)
        frame = pd.DataFrame({'EvtID': [3]*16, 'TrkID': [1]*8+[2]*8, 'CellID': range(16),
                              'wire_x': np.tile(x, 2), 'wire_y': np.tile(y, 2), 'd': np.tile(distance, 2),
                              't': np.r_[distance/.035, distance/.035+500]})
        frame['wire_r'] = np.hypot(frame.wire_x, frame.wire_y)
        graph = event_graph(frame)
        probability = np.full(graph.num_edges, .95, dtype=np.float32)
        xy, d, _ = measurements(graph)
        tracks = [candidate(graph, probability, xy, d, list(range(i, i+8)), [[100., 50., 200.]], 'seed') for i in [0, 8]]
        result = recover_fragments(graph, tracks, probability, use_time=True)
        self.assertEqual(len(result), 2)
        changed = copy.deepcopy(graph)
        changed.y.fill_(0)
        changed.truth_pairs.fill_(999)
        self.assertEqual(result, recover_fragments(changed, tracks, probability, use_time=True))
        self.assertEqual(sum(t['hits'] for t in result), 16)
        ambiguous = candidate(graph, probability, xy, d, list(range(16)), [[100., 50., 200.]], 'ambiguous')
        recovered = hough_remaining(graph, [ambiguous], probability, use_time=True, seed=12)
        self.assertEqual(sorted(t['hits'] for t in recovered), [8, 8])
        self.assertEqual(recovered, hough_remaining(changed, [ambiguous], probability, use_time=True, seed=12))


if __name__ == '__main__':
    unittest.main()
