from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import InteractionNetwork, model_dimensions, save_json  # noqa: E402
from STEP4_两级寻迹 import (  # noqa: E402
    RELAXED_RMS,
    STRICT_RMS,
    components,
    hierarchy_partition,
    predict,
    score,
    split_component,
    trim_outliers,
)
from step4_short_core_experiment.run_experiment import (  # noqa: E402
    add_result,
    new_accumulator,
    part_paths,
    run_with_short_core,
    short_core_extract,
    summarize,
    truth_result,
)


def short_components(data, probabilities, threshold, allowed):
    allowed = set(allowed)
    graph = nx.Graph()
    graph.add_nodes_from(allowed)
    edges = data.edge_index.t().cpu().numpy()
    mask = np.zeros(data.num_nodes, dtype=bool)
    mask[list(allowed)] = True
    keep = (probabilities >= threshold) & mask[edges[:, 0]] & mask[edges[:, 1]]
    graph.add_edges_from((int(u), int(v)) for u, v in edges[keep])
    return [sorted(value) for value in nx.connected_components(graph) if len(value) <= 4]


def rerun_small_graph(data, model, device, variant, fragments):
    nodes = sorted({node for fragment in fragments for node in fragment})
    if len(nodes) < 5:
        return None, None

    node_mask = np.zeros(data.num_nodes, dtype=bool)
    node_mask[nodes] = True
    edges = data.edge_index.t().cpu().numpy()
    edge_mask = node_mask[edges[:, 0]] & node_mask[edges[:, 1]]
    if not edge_mask.any():
        return None, None

    mapping = np.full(data.num_nodes, -1, dtype=np.int64)
    mapping[nodes] = np.arange(len(nodes))
    sub_edges = torch.from_numpy(mapping[edges[edge_mask]].T).long().to(device)
    x = data.x[nodes, :7 if variant == "time" else 6].to(device)
    edge_attr = data.edge_attr[edge_mask, :6 if variant == "time" else 4].to(device)
    with torch.no_grad():
        values = torch.sigmoid(model(x, sub_edges, edge_attr)).cpu().numpy()

    full_probabilities = np.full(len(edges), -np.inf, dtype=np.float32)
    full_probabilities[edge_mask] = values
    fragment_id = {node: index for index, fragment in enumerate(fragments) for node in fragment}
    fragment_sizes = {index: len(fragment) for index, fragment in enumerate(fragments)}
    return full_probabilities, (fragment_id, fragment_sizes)


def run_with_short_rerun(
    data, rerun_model, device, variant, probabilities, strict_threshold, relaxed_threshold,
    rerun_threshold, rerun_rms, core_profile,
):
    strict_components = components(data, probabilities, strict_threshold)
    strict_candidates = []
    for component in strict_components:
        core = short_core_extract(data, component, probabilities, strict_threshold, core_profile)
        if core is not None:
            strict_candidates.append(core)
            continue
        hierarchy = hierarchy_partition(data, component, probabilities, strict_threshold)
        strict_candidates.extend(hierarchy or split_component(data, component, probabilities, strict_threshold, STRICT_RMS))

    used = {index for candidate in strict_candidates for index in candidate["indices"]}
    leftover = set(range(data.num_nodes)) - used
    relaxed_components = components(data, probabilities, relaxed_threshold, leftover)
    relaxed_candidates = [
        trim_outliers(data, component, probabilities, relaxed_threshold, RELAXED_RMS, "relaxed")
        for component in relaxed_components
    ]
    relaxed_candidates = [candidate for candidate in relaxed_candidates if candidate is not None]
    relaxed_used = {index for candidate in relaxed_candidates for index in candidate["indices"]}
    fragments = short_components(data, probabilities, relaxed_threshold, leftover - relaxed_used)

    rerun_probabilities, fragment_info = rerun_small_graph(data, rerun_model, device, variant, fragments)
    rerun_candidates = []
    if rerun_probabilities is not None:
        fragment_id, fragment_sizes = fragment_info
        seed_nodes = set(fragment_id)
        for component in components(data, rerun_probabilities, rerun_threshold, seed_nodes):
            source_ids = {fragment_id[node] for node in component}
            if len(source_ids) < 2 or max(fragment_sizes[index] for index in source_ids) < 3:
                continue
            candidate = trim_outliers(
                data, component, rerun_probabilities, rerun_threshold, rerun_rms, "short_graph_rerun"
            )
            if candidate is not None:
                candidate["source_fragments"] = len({fragment_id[node] for node in candidate["indices"]})
                rerun_candidates.append(candidate)

    candidates = strict_candidates + relaxed_candidates + rerun_candidates
    candidates.sort(key=score, reverse=True)
    final = []
    final_used = set()
    for candidate in candidates:
        if candidate["accepted_geometry"] and not final_used.intersection(candidate["indices"]):
            candidate["score"] = score(candidate)
            final.append(candidate)
            final_used.update(candidate["indices"])
    return final, {
        "small_fragments": sum(len(fragment) >= 3 for fragment in fragments),
        "rerun_nodes": sum(map(len, fragments)),
        "rerun_candidates": len(rerun_candidates),
        "accepted_rerun_tracks": sum(track["stage"] == "short_graph_rerun" for track in final),
    }


def hit_bin(size):
    return "5_6" if size <= 6 else "7_10" if size <= 10 else "11_20" if size <= 20 else "21_plus"


def main():
    parser = argparse.ArgumentParser(description="STEP4 小碎片子图二次推理实验")
    parser.add_argument("--features", choices=["time", "no-time"], default="time")
    parser.add_argument("--run-tag", default="clean_physics_v2")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset_calibration_threshold_calibration_v1.pt")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--strict-threshold", type=float, default=0.70)
    parser.add_argument("--relaxed-threshold", type=float, default=0.575)
    parser.add_argument("--rerun-threshold", type=float, default=0.55)
    parser.add_argument("--rerun-rms", type=float, default=0.20)
    parser.add_argument("--rerun-model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1 or not 0 < args.rerun_threshold < 1:
        raise ValueError("rounds 或 rerun-threshold 无效")

    core_profile = {
        "skip_parent_rms": 0.20,
        "max_core_rms": 0.80,
        "min_rms_gain": 0.10,
        "min_probability_gain": 0.05,
        "min_coverage": 0.40,
    }
    stem = f"{args.features.replace('-', '_')}_{args.run_tag}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionNetwork(*model_dimensions(args.features)).to(device)
    model.load_state_dict(torch.load(ROOT / f"best_interaction_network_{stem}.pth", map_location=device))
    model.eval()
    rerun_model = model
    if args.rerun_model is not None:
        rerun_model = InteractionNetwork(*model_dimensions(args.features)).to(device)
        rerun_model.load_state_dict(torch.load(args.rerun_model.resolve(), map_location=device))
        rerun_model.eval()

    baseline_acc = new_accumulator()
    experiment_acc = new_accumulator()
    recovered = newly_lost = seen = rerun_tracks = matched_rerun_tracks = 0
    recovered_bins = Counter()
    newly_lost_bins = Counter()
    diagnostic_sums = Counter()
    for part_path in part_paths(args.data.resolve()):
        for data in torch.load(part_path, weights_only=False):
            if seen >= args.rounds:
                break
            probabilities = predict(data, model, device, args.features)

            started = time.perf_counter()
            baseline_tracks, _ = run_with_short_core(
                data, probabilities, args.strict_threshold, args.relaxed_threshold, core_profile
            )
            baseline_result = truth_result(
                data, baseline_tracks, probabilities, args.strict_threshold, args.relaxed_threshold
            )
            baseline_report, baseline_matched, bins, baseline_matched_bins, short_cores, causes, truth_sizes = baseline_result
            add_result(
                baseline_acc, baseline_report, bins, baseline_matched_bins,
                time.perf_counter() - started, short_cores, causes,
            )

            started = time.perf_counter()
            experiment_tracks, diagnostics = run_with_short_rerun(
                data, rerun_model, device, args.features, probabilities,
                args.strict_threshold, args.relaxed_threshold,
                args.rerun_threshold, args.rerun_rms, core_profile,
            )
            experiment_result = truth_result(
                data, experiment_tracks, probabilities, args.strict_threshold, args.relaxed_threshold
            )
            experiment_report, experiment_matched, bins, experiment_matched_bins, short_cores, causes, _ = experiment_result
            add_result(
                experiment_acc, experiment_report, bins, experiment_matched_bins,
                time.perf_counter() - started, short_cores, causes,
            )
            tracks = [track for track in experiment_tracks if track["stage"] == "short_graph_rerun"]
            rerun_tracks += len(tracks)
            matched_rerun_tracks += sum(track["purity"] >= 0.80 and track["completeness"] >= 0.50 for track in tracks)
            diagnostic_sums.update(diagnostics)

            event_recovered = experiment_matched - baseline_matched
            event_lost = baseline_matched - experiment_matched
            recovered += len(event_recovered)
            newly_lost += len(event_lost)
            for target, keys in ((recovered_bins, event_recovered), (newly_lost_bins, event_lost)):
                for key in keys:
                    target[hit_bin(truth_sizes[key])] += 1
            seen += 1
            if seen % 50 == 0:
                print(f"已完成 {seen}/{args.rounds}", flush=True)
        if seen >= args.rounds:
            break

    baseline = summarize(baseline_acc)
    experiment = summarize(experiment_acc)
    experiment.update({
        "mean_small_fragments": diagnostic_sums["small_fragments"] / max(seen, 1),
        "mean_rerun_nodes": diagnostic_sums["rerun_nodes"] / max(seen, 1),
        "rerun_tracks": rerun_tracks,
        "matched_rerun_tracks": matched_rerun_tracks,
    })
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
        "rerun_threshold": args.rerun_threshold,
        "rerun_rms": args.rerun_rms,
        "rerun_model": str(args.rerun_model.resolve()) if args.rerun_model else str(ROOT / f"best_interaction_network_{stem}.pth"),
        "baseline_short_core": baseline,
        "short_graph_rerun": experiment,
        "comparison": comparison,
    })
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print("报告：", output.resolve())


if __name__ == "__main__":
    main()
