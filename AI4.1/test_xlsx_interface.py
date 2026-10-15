"""运行：python test_xlsx_interface.py。无需模型权重的接口回归检查。"""
import numpy as np
import pandas as pd
from common import build_graph, select_features
from STEP4_test_xlsx_绘图 import prepare_hits, event_result, track_event


def main():
    hits = pd.DataFrame({"EvtID": [0] * 6 + [1], "HitID": list(range(6)) + [0],
                         "CellID": [0, 1, 2, 3, 4, 5, 0], "d": [1.] * 7,
                         "WireType": ["Axial"] * 5 + ["Stereo", "Axial"], "Good": [True] * 7,
                         "CreatProc": ["ExtendedDecayWithSpin"] * 7})
    geometry = pd.DataFrame({"CellID": range(6), "x": np.arange(6) * 10., "y": [100.] * 6,
                             "u": [0.] * 6, "v": [0.] * 5 + [0.1]})
    original, axial = prepare_hits(hits, geometry)
    assert len(original) == 7 and len(axial) == 6
    frame = axial.loc[axial.EvtID.eq(0)].reset_index(drop=True)
    assert frame.HitID.tolist() == list(range(5))
    contaminated = hits.copy()
    contaminated.loc[0, "Good"] = False
    contaminated.loc[1, "CreatProc"] = "eIoni"
    assert prepare_hits(contaminated, geometry)[1].HitID.tolist() == [2, 3, 4, 0]
    graph = build_graph(frame)
    x, edges = select_features(graph, "no-time")
    assert x.shape == (5, 6) and edges.shape[1] == 4
    graph.x[:, 6] = 999  # Time must never enter no-time inputs.
    assert np.array_equal(x.numpy(), select_features(graph, "no-time")[0].numpy())
    tracks = [{"indices": [0, 2, 4], "fit": [1., 2., 3.], "rms": 0.1, "stage": "strict"}]
    result = event_result(0, frame, tracks, "ok", 6)
    assert [h["HitID"] for h in result["circles"][0]["hits"]] == [0, 2, 4]
    assert [h["excel_row"] for h in result["circles"][0]["hits"]] == [2, 4, 6]
    assert [h["HitID"] for h in result["unassigned_hits"]] == [1, 3]
    assert track_event(axial.loc[axial.EvtID.eq(1)], None, None, .5, .35)[0] == []
    bad = hits.copy()
    bad.loc[0, "CellID"] = 999
    try:
        prepare_hits(bad, geometry)
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown CellID must fail")
    print("Excel interface checks passed")


if __name__ == "__main__":
    main()
