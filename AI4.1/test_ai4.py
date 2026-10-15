from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import DATASET_META_FILE, apply_dead_time, build_graph, prepare_trigger, select_features
from STEP2_训练模型 import validate_dataset_metadata


def main():
    validate_dataset_metadata(DATASET_META_FILE, "time")
    validate_dataset_metadata(DATASET_META_FILE, "no-time")
    cell_map = pd.DataFrame({
        "CellID": [1, 2, 3, 4, 5],
        "wire_x": [0.0, 20.0, 40.0, 60.0, 80.0],
        "wire_y": [0.0, 0.0, 0.0, 0.0, 0.0],
        "wire_r": [0.0, 20.0, 40.0, 60.0, 80.0],
    })
    hits = pd.DataFrame([
        {"EvtID": 1, "TrkID": 4, "CellID": 1, "d": 1.0, "t": 12.0, "t0": 0.0, "tHit": 1.0, "wire_x": 0.0, "wire_y": 0.0, "wire_r": 0.0},
        {"EvtID": 1, "TrkID": 5, "CellID": 1, "d": 2.0, "t": 10.0, "t0": 0.0, "tHit": 1.2, "wire_x": 0.0, "wire_y": 0.0, "wire_r": 0.0},
        {"EvtID": 1, "TrkID": 5, "CellID": 2, "d": 2.0, "t": 20.0, "t0": 0.0, "tHit": 2.0, "wire_x": 20.0, "wire_y": 0.0, "wire_r": 20.0},
        {"EvtID": 1, "TrkID": 5, "CellID": 3, "d": 2.0, "t": 30.0, "t0": 0.0, "tHit": 3.0, "wire_x": 40.0, "wire_y": 0.0, "wire_r": 40.0},
        {"EvtID": 1, "TrkID": 5, "CellID": 4, "d": 2.0, "t": 40.0, "t0": 0.0, "tHit": 4.0, "wire_x": 60.0, "wire_y": 0.0, "wire_r": 60.0},
        {"EvtID": 1, "TrkID": 5, "CellID": 5, "d": 2.0, "t": 50.0, "t0": 0.0, "tHit": 5.0, "wire_x": 80.0, "wire_y": 0.0, "wire_r": 80.0},
    ])
    hits["t_window"] = hits["t"]
    hits["t_obs"] = hits["t"]
    hits["truth_tracks"] = [[(int(e), int(t))] for e, t in zip(hits.EvtID, hits.TrkID)]
    hits["is_noise"] = False
    hits["shared_hit"] = False
    hits["noise_multiplicity"] = 0
    candidates = pd.DataFrame([
        {**hits.iloc[0].to_dict(), "t_obs": 10.0, "is_noise": True, "truth_tracks": []},
        {**hits.iloc[0].to_dict(), "t_obs": 50.0},
        {**hits.iloc[0].to_dict(), "t_obs": 115.0},
        {**hits.iloc[0].to_dict(), "t_obs": 220.0},
    ])
    reduced = apply_dead_time(candidates)
    assert reduced.t_obs.tolist() == [10.0, 115.0, 220.0]
    assert bool(reduced.iloc[0].is_noise)
    trigger = prepare_trigger(hits, cell_map, 0.0, 0.0, np.random.default_rng(1))
    graph = build_graph(trigger)
    assert graph is not None
    time_x, time_edge = select_features(graph, "time")
    no_time_x, no_time_edge = select_features(graph, "no-time")
    assert time_x.shape[1] == 7 and no_time_x.shape[1] == 6
    assert time_edge.shape[1] == 6 and no_time_edge.shape[1] == 4
    assert not bool(graph.shared_hit.any())
    first_start, first_end = int(graph.truth_ptr[0]), int(graph.truth_ptr[1])
    assert len(graph.truth_pairs[first_start:first_end]) == 1
    shifted = hits.copy()
    shifted["t"] += 5000.0
    shifted["t0"] += 5000.0
    shifted_graph = build_graph(prepare_trigger(shifted, cell_map, 0.0, 0.0, np.random.default_rng(1)))
    assert shifted_graph is not None
    assert torch.equal(time_x, select_features(shifted_graph, "time")[0])
    # Offline truth flags may exist for evaluation, but must not affect inputs.
    baseline = time_x.clone()
    graph.is_noise[:] = ~graph.is_noise
    graph.shared_hit[:] = ~graph.shared_hit
    changed_time_x, changed_time_edge = select_features(graph, "time")
    changed_no_time_x, changed_no_time_edge = select_features(graph, "no-time")
    assert torch.equal(baseline, changed_time_x)
    assert torch.equal(no_time_x, changed_no_time_x)
    print("AI4 self-check passed")


if __name__ == "__main__":
    main()
