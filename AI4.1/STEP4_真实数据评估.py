"""无合成噪声/误差，按原始 EvtID 比较 time/no-time 并保留命中真值审计。"""
# Historical Excel timing contract t-t0; do not use this branch to evaluate the whole-event time33 model.
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from common import ROOT, MAP_FILE, InteractionNetwork, build_graph, model_dimensions
from STEP4_test_xlsx_绘图 import prepare_hits
from STEP4_两级寻迹 import evaluate, predict, run_two_pass, truth_sets
from STEP4_困难事件分析 import best_track_match, classify_miss, graph_components
from step4_short_core_experiment.run_experiment import run_with_short_core
from STEP4_剩余混轨恢复 import recover_remaining
from step4_short_rerun_experiment.run_experiment import run_with_short_rerun

OUT = ROOT / "real_data_evaluation"
CORE_PROFILE = {"skip_parent_rms": .20, "max_core_rms": .80, "min_rms_gain": .10,
                "min_probability_gain": .05, "min_coverage": .40}
_RERUN_MODELS = {}


def file_stamp(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prepare_frame(source, geometry):
    _, frame = prepare_hits(source, geometry)
    if "TrkID" not in frame or not np.isfinite(frame.TrkID.to_numpy(dtype=float)).all():
        raise ValueError("评估必须有完整 TrkID 真值")
    if (frame.TrkID % 1 != 0).any():
        raise ValueError("TrkID 必须是整数")
    frame["t_obs"] = pd.to_numeric(frame.t, errors="raise") - pd.to_numeric(frame.t0, errors="raise")
    if not np.isfinite(frame.t_obs).all():
        raise ValueError("t-t0 含无效值")
    frame["truth_tracks"] = [[(int(e), int(t))] for e, t in zip(frame.EvtID, frame.TrkID)]
    return frame


def split_events(ids):
    # 0/1/2 的切片已经用于诊断，必须留在开发集。
    ids = sorted(map(int, ids))
    pool = np.array([evt for evt in ids if evt not in {0, 1, 2}], dtype=int)
    np.random.default_rng(20261013).shuffle(pool)
    holdout = set(map(int, pool[:min(len(pool), len(ids) // 5)]))
    return {evt: "holdout" if evt in holdout else "development" for evt in ids}


def build_cache(input_path, output):
    stamp = {"input": file_stamp(input_path), "geometry": file_stamp(MAP_FILE),
             "contract": "Good=True;Axial;ExtendedDecayWithSpin;t_obs=t-t0;new_generation_80_20_v2"}
    cache = output / "events.pt"
    stamp_file = output / "input_manifest.json"
    if cache.exists() and stamp_file.exists() and json.loads(stamp_file.read_text(encoding="utf-8"))["stamp"] == stamp:
        return torch.load(cache, weights_only=False)
    started = time.perf_counter()
    columns = ["EvtID", "HitID", "TrkID", "CellID", "Good", "CreatProc", "WireType", "d", "t", "t0"]
    source = pd.read_excel(input_path, usecols=columns)
    frame = prepare_frame(source, pd.read_excel(MAP_FILE))
    partitions = split_events(frame.EvtID.unique())
    events = []
    for evt, group in frame.groupby("EvtID", sort=True):
        group = group.reset_index(drop=True)
        events.append({"evtid": int(evt), "partition": partitions[int(evt)],
                       "frame": group, "data": build_graph(group)})
    torch.save(events, cache)
    manifest = {"stamp": stamp, "source_rows": len(source), "selected_rows": len(frame),
                "events": len(events), "partitions": dict(Counter(partitions.values())),
                "identity_duplicates": int(frame.duplicated(["EvtID", "HitID"]).sum()),
                "repeated_cells": int(frame.duplicated(["EvtID", "CellID"]).sum()),
                "time_ns_min": float(frame.t_obs.min()), "time_ns_max": float(frame.t_obs.max()),
                "outside_training_time_window": int((~frame.t_obs.between(0, 300, inclusive="left")).sum()),
                "read_and_build_seconds": time.perf_counter() - started,
                "source_independence": "User confirmed a newly generated simulation batch, independent of original training data"}
    stamp_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Prepared", manifest, flush=True)
    return events


def infer_cache(events, variant, output):
    stem = f"{variant.replace('-', '_')}_clean_physics_v2"
    weight = ROOT / f"best_interaction_network_{stem}.pth"
    signature = {"model": file_stamp(weight), "input_cache": file_stamp(output / "events.pt"),
                 "event_ids": [e["evtid"] for e in events]}
    path = output / f"probabilities_{variant}.pt"
    if path.exists():
        cached = torch.load(path, weights_only=False)
        if cached["signature"] == signature:
            return cached["values"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionNetwork(*model_dimensions(variant)).to(device)
    model.load_state_dict(torch.load(weight, map_location=device, weights_only=True))
    model.eval()
    values = {}
    for index, event in enumerate(events, 1):
        if event["data"] is not None:
            values[event["evtid"]] = predict(event["data"], model, device, variant)
        if index % 100 == 0:
            print(f"Inference {variant}: {index}/{len(events)}", flush=True)
    torch.save({"signature": signature, "values": values}, path)
    return values


def init_worker():
    torch.set_num_threads(1)


def evaluate_event(task):
    event, variant, probabilities, recipe = task
    data, frame = event["data"], event["frame"]
    started = time.perf_counter()
    strict, relaxed = recipe["strict"], recipe["relaxed"]
    if data is None:
        tracks, diagnostics = [], {"no_graph": True}
        sizes = Counter((int(e), int(t)) for e, t in zip(frame.EvtID, frame.TrkID))
        eligible = {key for key, size in sizes.items() if size >= 5}
        result = {"truth_tracks": len(eligible), "excluded_short_truth_tracks": len(sizes) - len(eligible),
                  "matched_tracks": 0, "fake_tracks": 0, "clone_tracks": 0,
                  "missed_tracks": [list(k) for k in sorted(eligible)]}
        causes = Counter({"no_candidate_graph": len(eligible)})
    else:
        if recipe.get("short_rerun"):
            if variant not in _RERUN_MODELS:
                model = InteractionNetwork(*model_dimensions(variant))
                stem = f"{variant.replace('-', '_')}_clean_physics_v2"
                model.load_state_dict(torch.load(ROOT / f"best_interaction_network_{stem}.pth",
                                                map_location="cpu", weights_only=True))
                _RERUN_MODELS[variant] = model.eval()
            tracks, diagnostics = run_with_short_rerun(
                data, _RERUN_MODELS[variant], torch.device("cpu"), variant, probabilities,
                strict, relaxed, .55, .20, CORE_PROFILE)
        elif recipe.get("short_core"):
            tracks, diagnostics = run_with_short_core(data, probabilities, strict, relaxed, CORE_PROFILE)
        else:
            tracks, diagnostics = run_two_pass(data, None, None, variant, strict, relaxed, probabilities)
        if recipe.get("recover_remaining"):
            before = len(tracks)
            tracks = recover_remaining(data, tracks, probabilities,
                                       min_probability=recipe.get("recover_min_probability", .80))
            diagnostics["recovered_candidates"] = len(tracks) - before
        result = evaluate(data, tracks)
        associations = truth_sets(data)
        sizes = Counter(k for values in associations for k in values)
        causes = Counter()
        if result["missed_tracks"]:
            sc, sm = graph_components(data, probabilities, strict)
            rc, rm = graph_components(data, probabilities, relaxed)
            for key in map(tuple, result["missed_tracks"]):
                nodes = {i for i, values in enumerate(associations) if key in values}
                best = best_track_match(key, nodes, tracks, associations)
                causes[classify_miss(nodes, associations, sc, sm, rc, rm, best)[0]] += 1
    missed = set(map(tuple, result["missed_tracks"]))
    hit_bins = Counter()
    matched_bins = Counter()
    for key, size in sizes.items():
        if size >= 5:
            name = "5_6" if size <= 6 else "7_10" if size <= 10 else "11_20" if size <= 20 else "21_plus"
            hit_bins[name] += 1
            matched_bins[name] += key not in missed
    return {"EvtID": event["evtid"], "partition": event["partition"], "nodes": len(frame),
            **result, "output_tracks": len(tracks), "elapsed_seconds": time.perf_counter() - started,
            "hit_bins": dict(hit_bins), "matched_bins": dict(matched_bins), "miss_causes": dict(causes),
            "diagnostics": diagnostics, "tracks": tracks}


def summarize(rows):
    n = len(rows)
    total = {key: sum(r[key] for r in rows) for key in
             ["truth_tracks", "matched_tracks", "fake_tracks", "clone_tracks", "output_tracks", "excluded_short_truth_tracks"]}
    causes, bins, matches = Counter(), Counter(), Counter()
    for row in rows:
        causes.update(row["miss_causes"])
        bins.update(row["hit_bins"])
        matches.update(row["matched_bins"])
    return {"events": n, **total,
            "track_efficiency": total["matched_tracks"] / total["truth_tracks"] if total["truth_tracks"] else None,
            "fake_rate": total["fake_tracks"] / total["output_tracks"] if total["output_tracks"] else None,
            "clone_rate": total["clone_tracks"] / total["output_tracks"] if total["output_tracks"] else None,
            "fake_per_event": total["fake_tracks"] / n if n else None,
            "clone_per_event": total["clone_tracks"] / n if n else None,
            "mean_elapsed_seconds": sum(r["elapsed_seconds"] for r in rows) / n if n else None,
            "miss_causes": dict(causes),
            "by_truth_hits": {b: {"truth": bins[b], "matched": matches[b], "efficiency": matches[b] / bins[b]}
                              for b in bins}}


def run_recipe(events, probabilities, variant, recipe, output, workers):
    parts = {e["partition"] for e in events}
    suffix = f"_{next(iter(parts))}" if len(parts) == 1 else ""
    path = output / f"{variant}_{recipe['name']}{suffix}.json"
    tasks = ((e, variant, probabilities.get(e["evtid"]), recipe) for e in events)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        for index, row in enumerate(pool.map(evaluate_event, tasks, chunksize=1), 1):
            rows.append(row)
            if index % 50 == 0 or index == len(events):
                print(f"Tracking {variant}/{recipe['name']}: {index}/{len(events)}", flush=True)
    report = {"variant": variant, "recipe": recipe, "input": "raw, no noise or Gaussian error; t_obs=t-t0",
              "metrics": "truth>=5;purity>=0.8;completeness>=0.5;fake/clone denominator=all outputs",
              "summary": summarize(rows),
              "partitions": {p: summarize([r for r in rows if r["partition"] == p])
                             for p in sorted({r["partition"] for r in rows})}, "events": rows}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Development result", variant, recipe["name"], report["partitions"].get("development"), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "test.xlsx")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--features", choices=["both", "time", "no-time"], default="both")
    parser.add_argument("--development-candidates", action="store_true")
    parser.add_argument("--recipe", choices=["baseline", "higher_threshold", "short_core", "residual_recursive", "conservative_residual", "historical_strongest"], default="baseline")
    parser.add_argument("--partition", choices=["all", "development", "holdout"], default="all")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("workers/limit 必须大于零")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = build_cache(args.input, args.output_dir)
    if args.limit:
        events = events[:args.limit]
    for variant in (["no-time", "time"] if args.features == "both" else [args.features]):
        probabilities = infer_cache(events, variant, args.output_dir)
        metadata = json.loads((ROOT / f"model_metadata_{variant.replace('-', '_')}_clean_physics_v2.json").read_text(encoding="utf-8"))
        strict = float(metadata["edge_threshold"])
        baseline = {"name": "baseline", "strict": strict, "relaxed": max(.35, strict - .15)}
        if args.development_candidates:
            dev = [e for e in events if e["partition"] == "development"]
            for recipe in [baseline, {"name": "higher_threshold", "strict": .70, "relaxed": .575},
                           {"name": "short_core", "strict": .70, "relaxed": .575, "short_core": True}]:
                run_recipe(dev, probabilities, variant, recipe, args.output_dir, args.workers)
        else:
            recipe = {"baseline": baseline,
                      "higher_threshold": {"name": "higher_threshold", "strict": .70, "relaxed": .575},
                      "short_core": {"name": "short_core", "strict": .70, "relaxed": .575, "short_core": True},
                      "residual_recursive": {"name": "residual_recursive", "strict": strict,
                                             "relaxed": max(.35, strict - .15), "recover_remaining": True},
                      "conservative_residual": {"name": "conservative_residual", "strict": strict,
                                                "relaxed": max(.35, strict - .15), "recover_remaining": True,
                                                "recover_min_probability": .92},
                      "historical_strongest": {"name": "historical_strongest", "strict": .70,
                                               "relaxed": .575, "short_rerun": True}}[args.recipe]
            selected = events if args.partition == "all" else [e for e in events if e["partition"] == args.partition]
            run_recipe(selected, probabilities, variant, recipe, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
