import numpy as np

from collections import Counter

from STEP4_评估节点分类器 import evaluate_binary, finalize_tracks


def main():
    result = evaluate_binary(np.array([0.99, 0.01]), np.array([True, False]), 0.95)
    assert result["precision"] == 1.0 and result["recall"] == 1.0
    tracks = finalize_tracks(Counter({
        "source_eligible_tracks": 10, "electronics_visible_tracks": 8,
        "algorithm_matched_tracks": 6, "visible_5_6_tracks": 4,
        "matched_5_6_tracks": 3, "fake_tracks": 2, "clone_tracks": 1,
    }), 2)
    assert tracks["algorithm_efficiency_given_visible"] == 0.75
    assert tracks["algorithm_efficiency_5_6_visible"] == 0.75
    assert tracks["mean_fake_tracks"] == 1.0
    print("node classifier evaluation self-check passed")


if __name__ == "__main__":
    main()
