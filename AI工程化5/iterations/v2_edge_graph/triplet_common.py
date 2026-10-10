from __future__ import annotations

import numpy as np


def truth_sets(data):
    ptr = data.truth_ptr.cpu().numpy()
    pairs = data.truth_pairs.cpu().numpy()
    return [set(map(tuple, pairs[ptr[i]:ptr[i + 1]])) for i in range(data.num_nodes)]


def pair_features(data, seed, left, right, left_score, right_score, shared_probability=0.0):
    sx, sy = float(data.wire_x[seed]), float(data.wire_y[seed])
    lv = np.array([float(data.wire_x[left]) - sx, float(data.wire_y[left]) - sy])
    rv = np.array([float(data.wire_x[right]) - sx, float(data.wire_y[right]) - sy])
    denom = max(float(np.linalg.norm(lv) * np.linalg.norm(rv)), 1e-9)
    cosine = float(np.dot(lv, rv) / denom)
    opposite = max(0.0, min(1.0, (1.0 - cosine) / 2.0))
    return np.asarray([
        left_score, right_score, min(left_score, right_score), left_score * right_score,
        opposite, cosine, float(np.linalg.norm(lv)), float(np.linalg.norm(rv)),
        abs(float(data.wire_x[left]) - float(data.wire_x[right])),
        abs(float(data.wire_y[left]) - float(data.wire_y[right])),
        float(shared_probability),
    ], dtype=np.float32)


FEATURE_NAMES = [
    "left_edge", "right_edge", "min_edge", "edge_product", "opposite",
    "cosine", "left_radius", "right_radius", "delta_x", "delta_y", "shared_probability",
]
