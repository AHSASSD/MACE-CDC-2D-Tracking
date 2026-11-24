"""Check geometry, stereo competition, bidirectional scores and shared scoring."""
import numpy as np
from STEP5_两级寻迹 import (DEFAULT_CONFIG, attach_stereo, axial_two_pass,
                            circle_wire_lower_bound, score_tracks, undirected_predictions)


def main():
    pairs, p = undirected_predictions(np.array([[0, 1, 1, 2], [1, 2, 0, 1]]), np.array([.8, .6, .4, .2]))
    assert np.allclose(p, [.6, .4]) and pairs.shape == (2, 2)
    # The reference point is 20 mm outside the circle, but the finite wire crosses it.
    wire = np.array([[120., 0, 1., 0, 0., 80.]])
    assert circle_wire_lower_bound([0, 0, 100], wire)[0] == 0
    assert circle_wire_lower_bound([0, 0, 10], wire)[0] == 70
    theta = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    g = np.zeros((14, 6)); g[:, 4] = 1; g[:, 5] = 200
    g[:5, :2] = np.c_[50 * np.cos(theta), 50 * np.sin(theta)]
    g[5:10, :2] = np.c_[300 + 50 * np.cos(theta), 50 * np.sin(theta)]
    g[10:, 0] = 50; g[10:, 2] = .1; g[10:, 4] = np.sqrt(.99)
    truth = np.array([1] * 5 + [2] * 5 + [1, 1, 1, 2])
    tracks = [dict(indices=list(range(5)), fit=[0, 0, 50]), dict(indices=list(range(5, 10)), fit=[300, 0, 50])]
    edges = [(10, 0), (10, 1), (11, 0), (11, 1), (12, 0), (12, 1),
             (12, 10), (12, 11), (13, 0), (13, 1), (13, 5), (13, 6)]
    probabilities = np.array([.9] * 4 + [.6] * 2 + [.9] * 6)
    config = dict(DEFAULT_CONFIG, geometry_tolerance_mm=1000.)
    result, diag = attach_stereo(tracks, g, np.zeros(14), np.arange(14), np.array(edges).T, probabilities, config)
    assert result[0]['stereo_indices'] == [10, 11, 12]
    assert result[1]['stereo_indices'] == []  # Equal competing owners leave hit 13 unassigned.
    assert diag['direct_assignments'] == 2 and diag['ss_assisted_assignments'] == 1
    all_members = [hit for track in result for hit in track['indices']]
    assert len(set(all_members)) == len(all_members)
    assert tracks[0]['indices'] == list(range(5))  # Caller membership unchanged.
    counts, found = score_tracks(result, truth, np.arange(14) >= 10, config)
    assert found == [1, 2] and counts['joint_matched_tracks'] == 1
    assert counts['truth_tracks'] == 2 and counts['correct_stereo_matched'] == 3
    # Repeated independent hits in one cell must not count as two AS witnesses.
    cells = np.arange(14); cells[1] = cells[0]
    rejected, _ = attach_stereo(tracks, g, np.zeros(14), cells, np.array(edges).T, probabilities, config)
    assert not rejected[0]['stereo_indices']
    axial_edges = np.array([(a, b) for a in range(5) for b in range(a + 1, 5)]).T
    fitted, _ = axial_two_pass(g[:5], np.zeros(5), axial_edges, np.full(axial_edges.shape[1], .95))
    assert len(fitted) == 1 and fitted[0]['rms'] < .01
    print('joint two-stage regression checks passed')


if __name__ == '__main__':
    main()
