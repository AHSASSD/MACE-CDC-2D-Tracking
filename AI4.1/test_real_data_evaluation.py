"""真实数据契约和评估分母检查：python test_real_data_evaluation.py。"""
import numpy as np
import pandas as pd
from common import build_graph, select_features
from STEP4_两级寻迹 import evaluate
from STEP4_真实数据评估 import prepare_frame, split_events, summarize


def main():
    hits = pd.DataFrame({"EvtID": [0] * 14, "TrkID": [10] * 5 + [20] * 9,
                         "HitID": range(14), "CellID": list(range(13)) + [0], "d": [1.] * 14,
                         "Good": [True] * 14, "WireType": ["Axial"] * 14,
                         "CreatProc": ["ExtendedDecayWithSpin"] * 14,
                         "t": [1001.] * 7 + [2011.] * 7, "t0": [1000.] * 7 + [2000.] * 7})
    geometry = pd.DataFrame({"CellID": range(13), "x": np.arange(13) * 10., "y": [100.] * 13,
                             "u": [0.] * 13, "v": [0.] * 13})
    frame = prepare_frame(hits, geometry)
    assert frame.t_obs.tolist() == [1.] * 7 + [11.] * 7
    assert len(frame) == 14 and frame.CellID.duplicated().sum() == 1
    graph = build_graph(frame)
    assert np.isclose(graph.x[0, 6], 1 / 300)
    assert np.isclose(graph.x[7, 6], 11 / 300)
    before = select_features(graph, "no-time")[0].clone()
    graph.x[:, 6] = 123
    assert np.array_equal(before, select_features(graph, "no-time")[0])
    tracks = [{"indices": indices, "hits": len(indices)} for indices in
              [list(range(5)), list(range(5)), [0, 1, 5, 6, 7]]]
    report = evaluate(graph, tracks)
    assert (report["truth_tracks"], report["matched_tracks"], report["fake_tracks"], report["clone_tracks"]) == (2, 1, 1, 1)
    summary = summarize([{**report, "output_tracks": 3, "elapsed_seconds": 0,
                          "hit_bins": {"5_6": 1, "7_10": 1}, "matched_bins": {"5_6": 1}, "miss_causes": {}}])
    assert summary["track_efficiency"] == .5
    assert summary["fake_rate"] == summary["clone_rate"] == 1 / 3
    split = split_events(range(1000))
    assert split == split_events(reversed(range(1000)))
    assert list(split.values()).count("holdout") == 200
    assert all(split[i] == "development" for i in [0, 1, 2])
    print("Real-data evaluation checks passed")


if __name__ == "__main__":
    main()
