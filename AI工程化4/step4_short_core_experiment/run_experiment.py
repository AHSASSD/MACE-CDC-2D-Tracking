from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import InteractionNetwork, model_dimensions, save_json  # noqa: E402
from STEP4_两级寻迹 import (  # noqa: E402
    RELAXED_RMS,
    STRICT_RMS,
    components,
    evaluate,
    fit_candidate,
    hierarchy_partition,
    predict,
    run_two_pass,
    score,
    split_component,
    trim_outliers,
    truth_sets,
)
from STEP4_困难事件分析 import best_track_match, classify_miss, graph_components  # noqa: E402

MIN_COMPONENT_SIZE = 7
MAX_COMPONENT_SIZE = 14


def short_core_extract(data, component, probabilities, strict_threshold, profile):
    if not MIN_COMPONENT_SIZE <= len(component) <= MAX_COMPONENT_SIZE:
        return None

    parent = fit_candidate(data, component, probabilities, strict_threshold, STRICT_RMS, "strict")
    if parent is not None and parent["accepted_geometry"] and parent["rms"] <= profile["skip_parent_rms"]:
        return None

    best = None
    for threshold in np.arange(strict_threshold + 0.025, 0.901, 0.025):
        threshold = round(float(threshold), 3)
        for core in components(data, probabilities, threshold, component):
            if len(core) == len(component) or len(core) / len(component) < profile["min_coverage"]:
                continue
            candidate = trim_outliers(data, core, probabilities, threshold, profile["max_core_rms"], "short_core")
            if candidate is None:
                continue
            rms_gain = parent["rms"] - candidate["rms"] if parent is not None else float("inf")
            probability_gain = candidate["mean_probability"] - parent["mean_probability"] if parent is not None else float("inf")
            if rms_gain < profile["min_rms_gain"] and probability_gain < profile["min_probability_gain"]:
                continue
            utility = (
                1.5 * len(candidate["indices"]) / len(component)
                + candidate["mean_probability"]
                - candidate["rms"]
            )
            if best is None or utility > best[0]:
                best = (utility, candidate, threshold, rms_gain, probability_gain)

    if best is None:
        return None
    _, candidate, threshold, rms_gain, probability_gain = best
    candidate.update({
        "core_threshold": threshold,
        "core_parent_hits": len(component),
        "core_rms_gain": rms_gain,
        "core_probability_gain": probability_gain,
        "removed_indices": sorted(set(component) - set(candidate["indices"])),
    })
    return candidate


def run_with_short_core(data, probabilities, strict_threshold, relaxed_threshold, profile):
    strict_components = components(data, probabilities, strict_threshold)
    strict_candidates = []
    for component in strict_components:
        core = short_core_extract(data, component, probabilities, strict_threshold, profile)
        if core is not None:
            strict_candidates.append(core)
            continue
        hierarchy = hierarchy_partition(data, component, probabilities, strict_threshold)
        strict_candidates.extend(hierarchy or split_component(data, component, probabilities, strict_threshold, STRICT_RMS))

    used = {index for candidate in strict_candidates for index in candidate["indices"]}
    leftover = set(range(data.num_nodes)) - used
    relaxed_candidates = [
        trim_outliers(data, component, probabilities, relaxed_threshold, RELAXED_RMS, "relaxed")
        for component in components(data, probabilities, relaxed_threshold, leftover)
    ]
    candidates = strict_candidates + [candidate for candidate in relaxed_candidates if candidate is not None]
    candidates.sort(key=score, reverse=True)
    final = []
    final_used = set()
    for candidate in candidates:
        if candidate["accepted_geometry"] and not final_used.intersection(candidate["indices"]):
            candidate["score"] = score(candidate)
            final.append(candidate)
            final_used.update(candidate["indices"])
    return final, {
        "short_core_candidates": sum(candidate["stage"] == "short_core" for candidate in candidates),
        "accepted_short_cores": sum(candidate["stage"] == "short_core" for candidate in final),
    }


def truth_result(data, tracks, probabilities, strict_threshold, relaxed_threshold):
    associations = truth_sets(data)
    truth_sizes = Counter(key for values in associations for key in values)
    eligible = {key for key, size in truth_sizes.items() if size >= 5}
    report = evaluate(data, tracks)
    missed = {tuple(key) for key in report["missed_tracks"]}
    matched = eligible - missed
    strict_components, strict_map = graph_components(data, probabilities, strict_threshold)
    relaxed_components, relaxed_map = graph_components(data, probabilities, relaxed_threshold)
    causes = Counter()
    for key in missed:
        nodes = {index for index, values in enumerate(associations) if key in values}
        best = best_track_match(key, nodes, tracks, associations)
        cause, _ = classify_miss(
            nodes, associations, strict_components, strict_map, relaxed_components, relaxed_map, best
        )
        causes[cause] += 1
    bins = Counter()
    matched_bins = Counter()
    for key in eligible:
        size = truth_sizes[key]
        hit_bin = "5_6" if size <= 6 else "7_10" if size <= 10 else "11_20" if size <= 20 else "21_plus"
        bins[hit_bin] += 1
        matched_bins[hit_bin] += key in matched
    short_cores = [track for track in tracks if track["stage"] == "short_core"]
    return report, matched, bins, matched_bins, short_cores, causes, truth_sizes


def part_paths(data_path):
    directory = data_path if data_path.is_dir() else data_path.with_suffix(data_path.suffix + ".parts")
    paths = sorted(directory.glob("part_*.pt"))
    if not paths:
        raise FileNotFoundError(f"找不到数据分片：{directory}")
    return paths


def new_accumulator():
    return {
        "events": 0, "truth": 0, "matched": 0, "fake": 0, "clone": 0,
        "bins": Counter(), "matched_bins": Counter(), "elapsed": 0.0,
        "short_cores": 0, "matched_short_cores": 0, "miss_causes": Counter(),
    }


def add_result(acc, report, bins, matched_bins, elapsed, short_cores=(), causes=()):
    acc["events"] += 1
    acc["truth"] += report["truth_tracks"]
    acc["matched"] += report["matched_tracks"]
    acc["fake"] += report["fake_tracks"]
    acc["clone"] += report["clone_tracks"]
    acc["bins"].update(bins)
    acc["matched_bins"].update(matched_bins)
    acc["elapsed"] += elapsed
    acc["short_cores"] += len(short_cores)
    acc["matched_short_cores"] += sum(track["purity"] >= 0.80 and track["completeness"] >= 0.50 for track in short_cores)
    acc["miss_causes"].update(causes)


def summarize(acc):
    events = max(acc["events"], 1)
    return {
        "events": acc["events"],
        "micro_track_efficiency": acc["matched"] / max(acc["truth"], 1),
        "mean_fake_tracks": acc["fake"] / events,
        "mean_clone_tracks": acc["clone"] / events,
        "mean_elapsed_seconds": acc["elapsed"] / events,
        "total_truth_tracks": acc["truth"],
        "total_matched_tracks": acc["matched"],
        "efficiency_by_truth_hits": {
            key: {
                "truth_tracks": acc["bins"][key],
                "matched_tracks": acc["matched_bins"][key],
                "efficiency": acc["matched_bins"][key] / max(acc["bins"][key], 1),
            }
            for key in ("5_6", "7_10", "11_20", "21_plus")
        },
        "short_core_tracks": acc["short_cores"],
        "matched_short_core_tracks": acc["matched_short_cores"],
        "miss_causes": dict(acc["miss_causes"]),
    }


def main():
    parser = argparse.ArgumentParser(description="STEP4 短候选核心提取可行性实验")
    parser.add_argument("--features", choices=["time", "no-time"], default="time")
    parser.add_argument("--run-tag", default="clean_physics_v2")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset_calibration_threshold_calibration_v1.pt")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--strict-threshold", type=float, default=0.70)
    parser.add_argument("--relaxed-threshold", type=float, default=0.575)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-parent-rms", type=float, default=0.20)
    parser.add_argument("--max-core-rms", type=float, default=0.80)
    parser.add_argument("--min-rms-gain", type=float, default=0.10)
    parser.add_argument("--min-probability-gain", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.40)
    args = parser.parse_args()
    if args.rounds < 1 or not 0 < args.relaxed_threshold <= args.strict_threshold < 1:
        raise ValueError("rounds 和阈值参数无效")

    profile = {
        "skip_parent_rms": args.skip_parent_rms,
        "max_core_rms": args.max_core_rms,
        "min_rms_gain": args.min_rms_gain,
        "min_probability_gain": args.min_probability_gain,
        "min_coverage": args.min_coverage,
    }
    stem = f"{args.features.replace('-', '_')}_{args.run_tag}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionNetwork(*model_dimensions(args.features)).to(device)
    model.load_state_dict(torch.load(ROOT / f"best_interaction_network_{stem}.pth", map_location=device))
    model.eval()

    baseline_acc = new_accumulator()
    experiment_acc = new_accumulator()
    recovered = newly_lost = seen = 0
    recovered_bins = Counter()
    newly_lost_bins = Counter()
    for part_path in part_paths(args.data.resolve()):
        for data in torch.load(part_path, weights_only=False):
            if seen >= args.rounds:
                break
            probabilities = predict(data, model, device, args.features)

            started = time.perf_counter()
            baseline_tracks, _ = run_two_pass(
                data, None, None, args.features, args.strict_threshold, args.relaxed_threshold, probabilities
            )
            baseline_result = truth_result(
                data, baseline_tracks, probabilities, args.strict_threshold, args.relaxed_threshold
            )
            baseline_report, baseline_matched, bins, baseline_matched_bins, _, causes, truth_sizes = baseline_result
            add_result(
                baseline_acc, baseline_report, bins, baseline_matched_bins,
                time.perf_counter() - started, causes=causes,
            )

            started = time.perf_counter()
            experiment_tracks, _ = run_with_short_core(
                data, probabilities, args.strict_threshold, args.relaxed_threshold, profile
            )
            experiment_result = truth_result(
                data, experiment_tracks, probabilities, args.strict_threshold, args.relaxed_threshold
            )
            experiment_report, experiment_matched, bins, experiment_matched_bins, short_cores, causes, _ = experiment_result
            add_result(
                experiment_acc, experiment_report, bins, experiment_matched_bins,
                time.perf_counter() - started, short_cores, causes,
            )
            event_recovered = experiment_matched - baseline_matched
            event_lost = baseline_matched - experiment_matched
            recovered += len(event_recovered)
            newly_lost += len(event_lost)
            for target, keys in ((recovered_bins, event_recovered), (newly_lost_bins, event_lost)):
                for key in keys:
                    size = truth_sizes[key]
                    target["5_6" if size <= 6 else "7_10" if size <= 10 else "11_20" if size <= 20 else "21_plus"] += 1
            seen += 1
            if seen % 50 == 0:
                print(f"已完成 {seen}/{args.rounds}", flush=True)
        if seen >= args.rounds:
            break

    baseline = summarize(baseline_acc)
    experiment = summarize(experiment_acc)
    comparison = {
        "efficiency_delta": experiment["micro_track_efficiency"] - baseline["micro_track_efficiency"],
        "fake_tracks_per_event_delta": experiment["mean_fake_tracks"] - baseline["mean_fake_tracks"],
        "clone_tracks_per_event_delta": experiment["mean_clone_tracks"] - baseline["mean_clone_tracks"],
        "recovered_truth_tracks": recovered,
        "newly_lost_truth_tracks": newly_lost,
        "net_recovered_truth_tracks": recovered - newly_lost,
        "recovered_by_truth_hits": dict(recovered_bins),
        "newly_lost_by_truth_hits": dict(newly_lost_bins),
    }
    output = args.output or Path(__file__).with_name(f"result_{stem}_{seen}.json")
    save_json(output, {
        "features": args.features,
        "data": str(args.data.resolve()),
        "strict_threshold": args.strict_threshold,
        "relaxed_threshold": args.relaxed_threshold,
        "profile": profile,
        "baseline": baseline,
        "short_core": experiment,
        "comparison": comparison,
    })
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print("报告：", output.resolve())


if __name__ == "__main__":
    main()
