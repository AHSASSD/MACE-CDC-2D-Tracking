from STEP4_经典残差寻迹 import arbitrate, hybrid_arbitrate


def main():
    candidates = [
        {"indices": [0, 1, 2, 3, 4], "hits": 5, "rms": 0.2, "accepted_geometry": True},
        {"indices": [0, 5, 6, 7, 8], "hits": 5, "rms": 0.1, "accepted_geometry": True},
    ]
    accepted = arbitrate(candidates)
    assert len(accepted) == 1
    assert accepted[0]["rms"] == 0.1
    hybrid = hybrid_arbitrate(
        [{"indices": [0, 1, 2, 3, 4], "hits": 5, "rms": 0.2, "accepted_geometry": True}],
        [{"indices": [5, 6, 7, 8, 9], "hits": 5, "rms": 0.7, "accepted_geometry": True}],
    )
    assert len(hybrid) == 2
    print("classic residual tracking self-check passed")


if __name__ == "__main__":
    main()
