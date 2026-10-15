"""成对输入消融；固定权重、0.70/0.575 阈值和原两级后处理。"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import (MAP_FILE, apply_dead_time, build_graph, inject_noise, load_cell_map,
                    prepare_trigger, InteractionNetwork, model_dimensions)
from STEP4_真实数据评估 import evaluate_event, init_worker, summarize, file_stamp
from STEP4_两级寻迹 import predict

OUT = Path(__file__).resolve().parent / "paired_diagnosis"
SCENARIOS = {
    "raw": (False, False, False, False),
    "deadtime_only": (True, False, False, False),
    "noise_only": (False, True, False, False),
    "distance_only": (False, False, True, False),
    "time_only": (False, False, False, True),
    "measurement_only": (False, False, True, True),
    "full_old_contract": (True, True, True, True),
    "full_without_deadtime": (False, True, True, True),
}
RECIPE = {"name": "paired_fixed", "strict": .70, "relaxed": .575}


def prepare_variant(source, cell_map, scenario, seed):
    deadtime, noise, distance_error, time_error = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    real = source.copy().reset_index(drop=True)
    real["t_window"] = real.t.to_numpy(dtype=float) - real.t0.to_numpy(dtype=float)
    # 始终消费相同的两组随机数，使所有条件复用相同误差与噪声样本。
    time_jitter = rng.normal(0, 3.0, len(real))
    distance_jitter = rng.normal(0, .075, len(real))
    real["t_obs"] = real.t_window + (time_jitter if time_error else 0)
    real["d"] = np.abs(real.d.to_numpy(dtype=float) + (distance_jitter if distance_error else 0))
    real["truth_tracks"] = [[(int(e), int(t))] for e, t in zip(real.EvtID, real.TrkID)]
    real["is_noise"] = False
    real["shared_hit"] = False
    real["noise_multiplicity"] = 0
    frame = inject_noise(real, cell_map, 100_000.0 if noise else 0.0,
                         3.0 if time_error else 0.0, rng)
    if deadtime:
        frame = apply_dead_time(frame)
    return frame.reset_index(drop=True)


def reference_metrics(event, row):
    source = event["source_frame"]
    source_sizes = Counter((int(e), int(t)) for e, t in zip(source.EvtID, source.TrkID))
    eligible = {k for k, n in source_sizes.items() if n >= 5}
    frame = event["frame"]
    observed_sizes = Counter(k for values in frame.truth_tracks for k in values)
    observable = {k for k, n in observed_sizes.items() if n >= 5 and k in eligible}
    observed_missed = set(map(tuple, row["missed_tracks"]))
    matched_observed = observable - observed_missed
    # 第二套固定源 hit 完整度口径，防止删点改变分母后虚增成功数。
    matched_source_hits = set()
    for track in row["tracks"]:
        counts = Counter(k for index in track["indices"] for k in frame.iloc[index].truth_tracks)
        if not counts:
            continue
        key, overlap = counts.most_common(1)[0]
        if key in eligible and overlap / track["hits"] >= .8 and overlap / source_sizes[key] >= .5:
            matched_source_hits.add(key)
    anchor = event["evtid"]
    anchor_eligible = {k for k in eligible if k[0] == anchor}
    anchors_observable = observable & anchor_eligible
    anchors_matched = matched_observed & anchor_eligible
    return {"source_truth_tracks": len(eligible), "observable_truth_tracks": len(observable),
            "source_eligible_not_observable": len(eligible - observable),
            "matched_observable_source_tracks": len(matched_observed),
            "matched_fixed_source_hit_tracks": len(matched_source_hits),
            "anchor_source_truth_tracks": len(anchor_eligible),
            "anchor_observable_truth_tracks": len(anchors_observable),
            "anchor_matched_tracks": len(anchors_matched),
            "anchor_matched_fixed_source_hit_tracks": len(matched_source_hits & anchor_eligible),
            "anchor_missed_source_keys": [list(k) for k in sorted(anchor_eligible - anchors_matched)],
            "source_hits": len(source), "observed_real_hits": int((~frame.is_noise).sum()),
            "noise_hits": int(frame.is_noise.sum()),
            "removed_real_hits": len(source) - int((~frame.is_noise).sum()),
            "mean_incoming_degree": event["data"].edge_index.shape[1] / len(frame) if event["data"] is not None else 0}


def paired_worker(task):
    event, variant, probabilities = task
    row = evaluate_event((event, variant, probabilities, RECIPE))
    row.update(reference_metrics(event, row))
    if event["data"] is not None:
        labels = event["data"].y.numpy().astype(bool)
        prediction = probabilities >= .5
        row["edge_tp"] = int((prediction & labels).sum())
        row["edge_fp"] = int((prediction & ~labels).sum())
        row["edge_fn"] = int((~prediction & labels).sum())
    else:
        row.update(edge_tp=0, edge_fp=0, edge_fn=0)
    return row


def summary(rows):
    result = summarize(rows)
    fields = ["source_truth_tracks", "observable_truth_tracks", "source_eligible_not_observable",
              "matched_observable_source_tracks", "matched_fixed_source_hit_tracks", "source_hits",
              "observed_real_hits", "noise_hits", "removed_real_hits", "anchor_source_truth_tracks",
              "anchor_observable_truth_tracks", "anchor_matched_tracks", "anchor_matched_fixed_source_hit_tracks"]
    result.update({f: sum(r[f] for r in rows) for f in fields})
    s, o = result["source_truth_tracks"], result["observable_truth_tracks"]
    result["physics_visibility"] = o / s if s else None
    result["conditional_algorithm_efficiency"] = result["matched_observable_source_tracks"] / o if o else None
    result["source_end_to_end_efficiency"] = result["matched_observable_source_tracks"] / s if s else None
    result["fixed_source_hit_efficiency"] = result["matched_fixed_source_hit_tracks"] / s if s else None
    a, ao = result["anchor_source_truth_tracks"], result["anchor_observable_truth_tracks"]
    result["anchor_end_to_end_efficiency"] = result["anchor_matched_tracks"] / a if a else None
    result["anchor_conditional_efficiency"] = result["anchor_matched_tracks"] / ao if ao else None
    result["anchor_fixed_source_hit_efficiency"] = result["anchor_matched_fixed_source_hit_tracks"] / a if a else None
    tp, fp, fn = (sum(r[f] for r in rows) for f in ["edge_tp", "edge_fp", "edge_fn"])
    result["edge_metrics_at_050"] = {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                                    "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
    result["mean_nodes"] = sum(r["nodes"] for r in rows) / len(rows)
    result["mean_incoming_degree"] = sum(r["mean_incoming_degree"] for r in rows) / len(rows)
    if s and o:
        assert np.isclose(result["source_end_to_end_efficiency"],
                          result["physics_visibility"] * result["conditional_algorithm_efficiency"])
    return result


def make_events(source_events, scenarios, seed, context):
    cell_map = load_cell_map()
    lookup = {e["evtid"]: e for e in source_events}
    chosen_ids = sorted(lookup)
    outputs = {}
    for scenario in scenarios:
        events = []
        for original in source_events:
            evt = original["evtid"]
            source = original["frame"]
            if context == "triple":
                rng = np.random.default_rng(seed + evt * 1013 + 700000)
                partners = rng.choice([i for i in chosen_ids if i != evt], 2, replace=False)
                source = pd.concat([source, *[lookup[int(i)]["frame"] for i in partners]], ignore_index=True)
            frame = prepare_variant(source, cell_map, scenario, seed + evt * 1009)
            events.append({"evtid": evt, "partition": "diagnosis", "source_frame": source,
                           "frame": frame, "data": build_graph(frame)})
        outputs[scenario] = events
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--context", choices=["single", "triple"], default="single")
    parser.add_argument("--features", choices=["both", "time", "no-time"], default="both")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1 or (args.context == "triple" and args.count < 3):
        raise ValueError("count/workers 无效")
    torch.set_num_threads(1)
    originals = torch.load(ROOT / "real_data_evaluation" / "events.pt", weights_only=False)
    if args.count < len(originals):
        pool = [e for e in originals if e["partition"] == "development"]
        if args.count > len(pool):
            raise ValueError("子集实验不能超出开发池")
        rng = np.random.default_rng(73013)
        ids = set(map(int, rng.choice([e["evtid"] for e in pool], args.count, replace=False)))
        originals = [e for e in originals if e["evtid"] in ids]
    directory = OUT / f"n{len(originals)}_{args.context}_seed{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    generated = make_events(originals, args.scenarios, args.seed, args.context)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for variant in (["no-time", "time"] if args.features == "both" else [args.features]):
        model = InteractionNetwork(*model_dimensions(variant)).to(device)
        stem = f"{variant.replace('-', '_')}_clean_physics_v2"
        model.load_state_dict(torch.load(ROOT / f"best_interaction_network_{stem}.pth", map_location=device, weights_only=True))
        model.eval()
        for scenario, events in generated.items():
            path = directory / f"{variant}_{scenario}.json"
            signature = {"source": file_stamp(ROOT / "real_data_evaluation" / "events.pt"),
                         "weight": file_stamp(ROOT / f"best_interaction_network_{stem}.pth"),
                         "cell_map": file_stamp(MAP_FILE), "feature_code": file_stamp(ROOT / "common.py"),
                         "version": "paired_inputs_v1", "scenario_flags": SCENARIOS[scenario],
                         "event_ids": [e["evtid"] for e in originals]}
            if path.exists():
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("signature") == json.loads(json.dumps(signature)):
                    print(f"Cached {variant}/{scenario}", flush=True)
                    continue
            probabilities = {e["evtid"]: predict(e["data"], model, device, variant)
                             for e in events if e["data"] is not None}
            tasks = ((e, variant, probabilities.get(e["evtid"])) for e in events)
            rows = []
            with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
                for index, row in enumerate(pool.map(paired_worker, tasks, chunksize=1), 1):
                    rows.append(row)
                    if index % 50 == 0:
                        print(f"[paired] {variant}/{scenario}/{args.context}: {index}/{len(events)}", flush=True)
            report = {"variant": variant, "scenario": scenario, "context": args.context, "seed": args.seed,
                      "signature": signature,
                      "thresholds": RECIPE, "source_event_ids": [e["evtid"] for e in originals],
                      "summary": summary(rows), "events": rows,
                      "input_note": "Triple is diagnostic synthetic context only; anchors compared on same source particles"}
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Paired result", variant, scenario, report["summary"], flush=True)


def self_check():
    cell_map = pd.DataFrame({"CellID": [0, 1], "wire_x": [100., 110.], "wire_y": [0., 0.], "wire_r": [100., 110.]})
    source = pd.DataFrame({"CellID": [0, 0, 0, 1], "EvtID": [0] * 4, "TrkID": [1, 2, 3, 4],
                           "wire_x": [100., 100., 100., 110.], "wire_y": [0.] * 4, "wire_r": [100., 100., 100., 110.],
                           "d": [1., 2., 3., 4.], "t": [1001., 1010., 1150., 2005.], "t0": [1000., 1000., 1000., 2000.]})
    raw = prepare_variant(source, cell_map, "raw", 123)
    assert raw.d.tolist() == source.d.tolist() and raw.t_obs.tolist() == [1., 10., 150., 5.]
    dead = prepare_variant(source, cell_map, "deadtime_only", 123)
    assert dead.TrkID.tolist() == [1, 3, 4]
    reference = prepare_trigger(source, cell_map, rng=np.random.default_rng(123))
    actual = prepare_variant(source, cell_map, "full_old_contract", 123)
    for column in ["CellID", "wire_x", "wire_y", "d", "t_obs", "is_noise"]:
        assert np.array_equal(actual[column].to_numpy(), reference[column].to_numpy())
    assert actual.truth_tracks.tolist() == reference.truth_tracks.tolist()
    distance = prepare_variant(source, cell_map, "distance_only", 123)
    both = prepare_variant(source, cell_map, "measurement_only", 123)
    assert np.array_equal(distance.set_index("TrkID").loc[both.TrkID, "d"].to_numpy(), both.d.to_numpy())
    print("Paired input contract checks passed")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        main()
