from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from common import ROOT, TEST_DATA, InteractionNetwork, load_graph_dataset, model_dimensions, save_json
from STEP4_两级寻迹 import MIN_HITS, predict, run_two_pass, truth_sets


def graph_components(data, probabilities, threshold):
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    edges = data.edge_index.t().cpu().numpy()
    graph.add_edges_from((int(u), int(v)) for u, v in edges[probabilities >= threshold])
    components = [set(value) for value in nx.connected_components(graph)]
    node_component = {}
    for index, component in enumerate(components):
        for node in component:
            node_component[node] = index
    return components, node_component


def best_track_match(key, nodes, tracks, associations):
    best = {"overlap": 0, "purity": 0.0, "completeness": 0.0, "stage": None, "rms": None,
            "contaminant_nodes": 0, "noise_contaminants": 0, "other_truth_contaminants": 0}
    for track in tracks:
        overlap = len(nodes.intersection(track["indices"]))
        purity = overlap / max(track["hits"], 1)
        completeness = overlap / max(len(nodes), 1)
        if (completeness, purity) > (best["completeness"], best["purity"]):
            contaminants = set(track["indices"]) - nodes
            best = {
                "overlap": overlap,
                "purity": purity,
                "completeness": completeness,
                "stage": track.get("stage"),
                "rms": track.get("rms"),
                "contaminant_nodes": len(contaminants),
                "noise_contaminants": sum(not associations[index] for index in contaminants),
                "other_truth_contaminants": sum(bool(associations[index] - {key}) for index in contaminants),
            }
    return best


def classify_miss(nodes, associations, strict_components, strict_map, relaxed_components, relaxed_map, best):
    strict_ids = {strict_map[index] for index in nodes}
    relaxed_ids = {relaxed_map[index] for index in nodes}
    strict_largest = max((len(nodes & strict_components[index]) for index in strict_ids), default=0)
    relaxed_largest = max((len(nodes & relaxed_components[index]) for index in relaxed_ids), default=0)
    strict_others = set()
    strict_noise = 0
    for component_id in strict_ids:
        component = strict_components[component_id]
        strict_noise += sum(not associations[index] for index in component)
        strict_others.update(key for index in component for key in associations[index])
    own_key = next(iter(associations[next(iter(nodes))]))
    strict_others.discard(own_key)

    if best["completeness"] >= 0.50 and best["purity"] < 0.80:
        cause = "complete_but_contaminated"
    elif relaxed_largest < MIN_HITS:
        cause = "edge_fragmentation_below_min_hits"
    elif strict_largest < 0.50 * len(nodes) and relaxed_largest >= 0.50 * len(nodes):
        cause = "strict_threshold_fragmentation"
    elif strict_others:
        cause = "merged_with_other_tracks"
    elif best["overlap"] == 0:
        cause = "geometry_or_candidate_rejection"
    else:
        cause = "partial_candidate_loss"
    return cause, {
        "strict_component_count": len(strict_ids),
        "strict_largest_truth_hits": strict_largest,
        "relaxed_component_count": len(relaxed_ids),
        "relaxed_largest_truth_hits": relaxed_largest,
        "other_truth_tracks_in_strict_components": len(strict_others),
        "noise_nodes_in_strict_components": strict_noise,
    }


def analyze_variant(variant, dataset, hard_indices, strict, relaxed, run_tag, device):
    stem = f"{variant.replace('-', '_')}_{run_tag}"
    model = InteractionNetwork(*model_dimensions(variant)).to(device)
    model.load_state_dict(torch.load(ROOT / f"best_interaction_network_{stem}.pth", map_location=device))
    model.eval()
    cause_counts = Counter()
    hit_bins = Counter()
    misses = []
    events = []

    for event_index in hard_indices:
        data = dataset[event_index]
        probabilities = predict(data, model, device, variant)
        tracks, diagnostics = run_two_pass(data, model, device, variant, strict, relaxed, probabilities)
        associations = truth_sets(data)
        truth_sizes = Counter(key for values in associations for key in values)
        eligible = {key for key, size in truth_sizes.items() if size >= MIN_HITS}
        strict_components, strict_map = graph_components(data, probabilities, strict)
        relaxed_components, relaxed_map = graph_components(data, probabilities, relaxed)
        event_misses = []
        matched = 0
        for key in eligible:
            nodes = {index for index, values in enumerate(associations) if key in values}
            best = best_track_match(key, nodes, tracks, associations)
            if best["purity"] >= 0.80 and best["completeness"] >= 0.50:
                matched += 1
                continue
            cause, details = classify_miss(nodes, associations, strict_components, strict_map, relaxed_components, relaxed_map, best)
            cause_counts[cause] += 1
            size = len(nodes)
            hit_bin = "5_6" if size <= 6 else "7_10" if size <= 10 else "11_20" if size <= 20 else "21_plus"
            hit_bins[(hit_bin, cause)] += 1
            item = {
                "truth": list(key),
                "truth_hits": size,
                "cause": cause,
                "best_final": best,
                **details,
            }
            misses.append({"event": event_index + 1, **item})
            event_misses.append(item)
        events.append({
            "event": event_index + 1,
            "truth_tracks": len(eligible),
            "matched_tracks": matched,
            "efficiency": matched / max(len(eligible), 1),
            "noise_nodes": int(data.is_noise.sum()),
            "nodes": data.num_nodes,
            "misses": event_misses,
            "diagnostics": diagnostics,
        })
    return {
        "variant": variant,
        "hard_events": len(hard_indices),
        "missed_tracks": len(misses),
        "cause_counts": dict(cause_counts),
        "cause_by_hit_bin": {f"{hit_bin}:{cause}": count for (hit_bin, cause), count in sorted(hit_bins.items())},
        "mean_missed_truth_hits": float(np.mean([item["truth_hits"] for item in misses])) if misses else 0.0,
        "events": events,
        "misses": misses,
    }


def main():
    parser = argparse.ArgumentParser(description="分析 STEP4 冻结测试中的困难事件")
    parser.add_argument("--run-tag", default="clean_physics_v2")
    parser.add_argument("--strict-threshold", type=float, default=0.70)
    parser.add_argument("--relaxed-threshold", type=float, default=0.575)
    parser.add_argument("--difficulty", type=float, default=0.90)
    parser.add_argument("--output", type=Path, default=ROOT / "step4_difficult_event_analysis.json")
    args = parser.parse_args()

    report_paths = {
        "no-time": ROOT / f"step4_ai4_no_time_{args.run_tag}_test_frozen.json",
        "time": ROOT / f"step4_ai4_time_{args.run_tag}_test_frozen.json",
    }
    reports = {variant: json.loads(path.read_text(encoding="utf-8")) for variant, path in report_paths.items()}
    hard_by_variant = {
        variant: {int(item["round"]) - 1 for item in report["rounds"] if item["track_efficiency"] < args.difficulty}
        for variant, report in reports.items()
    }
    hard_indices = sorted(set().union(*hard_by_variant.values()))
    dataset = load_graph_dataset(TEST_DATA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyses = {
        variant: analyze_variant(variant, dataset, hard_indices, args.strict_threshold, args.relaxed_threshold, args.run_tag, device)
        for variant in ("no-time", "time")
    }
    save_json(args.output, {
        "strict_threshold": args.strict_threshold,
        "relaxed_threshold": args.relaxed_threshold,
        "difficulty_efficiency_below": args.difficulty,
        "hard_events_by_variant": {variant: sorted(index + 1 for index in values) for variant, values in hard_by_variant.items()},
        "union_hard_events": [index + 1 for index in hard_indices],
        "hard_event_overlap": len(hard_by_variant["no-time"] & hard_by_variant["time"]),
        "analyses": analyses,
    })
    for variant, analysis in analyses.items():
        print(variant, analysis["cause_counts"], "missed_tracks=", analysis["missed_tracks"])
    print("报告：", args.output)


if __name__ == "__main__":
    main()
