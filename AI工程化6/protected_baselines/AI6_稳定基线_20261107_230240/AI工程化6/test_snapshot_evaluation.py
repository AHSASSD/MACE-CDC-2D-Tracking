"""Regression: merged components must not count as recovered true tracks."""
import numpy as np
import torch
from torch_geometric.data import Data
from STEP4_本机快照评估 import candidate_counts, summarize


def main():
    # Two tracks: five axial hits and one stereo hit each.
    truth = torch.tensor([1] * 6 + [2] * 6)
    stereo = torch.tensor([False] * 5 + [True] + [False] * 5 + [True])
    chain = [(i, i + 1) for i in range(5)] + [(i, i + 1) for i in range(6, 11)]
    g = Data(truth=truth, stereo=stereo, num_nodes=12,
             edge_index=torch.tensor(chain + [(4, 6)]).T)
    separate = np.array([True] * 10 + [False])
    counts = candidate_counts(g, separate)
    assert counts['matched_tracks'] == 2 and counts['output_candidates'] == 2
    assert counts['correct_stereo'] == counts['attached_stereo'] == 2
    merged = candidate_counts(g, np.ones(11, dtype=bool))
    assert merged['matched_tracks'] == 0 and merged['fake_candidates'] == 1
    empty = candidate_counts(g, np.zeros(11, dtype=bool))
    assert empty['output_candidates'] == 0 and empty['truth_tracks'] == 2
    result = summarize(np.array([[5, 1, 2], [2, 0, 1], [0, 0, 0]]), 4., 8, counts)
    assert result['loss'] == .5 and result['types']['AA']['f1'] == 10 / 13
    assert result['candidate_diagnostic']['axial_efficiency'] == 1.
    # The scoring truth cannot affect reconstructed candidate counts.
    g.truth[:] = 7
    assert candidate_counts(g, separate)['output_candidates'] == 2
    print('snapshot evaluation regression checks passed')


if __name__ == '__main__':
    main()
