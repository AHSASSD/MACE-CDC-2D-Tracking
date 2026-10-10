from __future__ import annotations

import argparse
import json
import itertools
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from common import ROOT, TEST_DATA, VAL_DATA, InteractionNetwork, fit_2d_circle, load_graph_dataset, model_dimensions, save_json, select_features

MIN_HITS = 5
STRICT_RMS = 1.0
RELAXED_RMS = 1.5
MAX_SPLIT_DEPTH = 8
MAX_TRIMMED_HITS = 2
MIN_HIERARCHY_COVERAGE = 0.70
MIN_HIERARCHY_RMS_GAIN = 0.15
MIN_HIERARCHY_SIZE = 15


def predict(data, model, device, variant):
    x, edge_attr = select_features(data, variant)
    with torch.no_grad():
        logits = model(x.to(device), data.edge_index.to(device), edge_attr.to(device))
    return torch.sigmoid(logits).cpu().numpy()


def components(data, probabilities, threshold, allowed=None):
    allowed = set(range(data.num_nodes)) if allowed is None else set(allowed)
    graph = nx.Graph()
    graph.add_nodes_from(allowed)
    edges = data.edge_index.t().cpu().numpy()
    keep = probabilities >= threshold
    if len(allowed) != data.num_nodes:
        allowed_mask = np.zeros(data.num_nodes, dtype=bool)
        allowed_mask[list(allowed)] = True
        keep &= allowed_mask[edges[:, 0]] & allowed_mask[edges[:, 1]]
    graph.add_edges_from((int(u), int(v), {"weight": float(p)}) for (u, v), p in zip(edges[keep], probabilities[keep]))
    return [sorted(value) for value in nx.connected_components(graph) if len(value) >= MIN_HITS]


def fit_candidate(data, indices, probabilities, threshold, rms_limit, stage):
    indices = sorted(set(indices))
    if len(indices) < MIN_HITS:
        return None
    x = data.wire_x[indices].numpy()
    y = data.wire_y[indices].numpy()
    d = data.drift_d[indices].numpy()
    fit = fit_2d_circle(x, y, d)
    if fit is None or not np.all(np.isfinite(fit)):
        return None
    a, b, radius = fit
    residuals = np.abs(np.hypot(x - a, y - b) - radius) - d
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    edges = data.edge_index.cpu().numpy()
    member_mask = np.zeros(data.num_nodes, dtype=bool)
    member_mask[indices] = True
    keep = member_mask[edges[0]] & member_mask[edges[1]] & (probabilities >= threshold)
    edge_values = probabilities[keep]
    return {
        "indices": indices,
        "fit": [float(v) for v in fit],
        "hits": len(indices),
        "rms": rms,
        "mean_probability": float(np.mean(edge_values)) if edge_values.size else 0.0,
        "stage": stage,
        "accepted_geometry": rms <= rms_limit,
    }


def trim_outliers(data, component, probabilities, threshold, rms_limit, stage):
    indices = sorted(set(component))
    removed = []
    for _ in range(MAX_TRIMMED_HITS + 1):
        candidate = fit_candidate(data, indices, probabilities, threshold, rms_limit, stage)
        if candidate is None:
            return None
        if candidate["accepted_geometry"]:
            candidate["removed_indices"] = removed
            return candidate
        if len(indices) <= MIN_HITS or len(removed) >= MAX_TRIMMED_HITS:
            return None
        a, b, radius = candidate["fit"]
        x = data.wire_x[indices].numpy()
        y = data.wire_y[indices].numpy()
        d = data.drift_d[indices].numpy()
        residuals = np.abs(np.abs(np.hypot(x - a, y - b) - radius) - d)
        worst = int(np.argmax(residuals))
        removed.append(indices.pop(worst))
    return None


def split_component(data, component, probabilities, threshold, rms_limit, depth=0):
    candidate = fit_candidate(data, component, probabilities, threshold, rms_limit, "strict")
    if candidate is not None and candidate["accepted_geometry"]:
        return [candidate]
    if candidate is None:
        if depth >= MAX_SPLIT_DEPTH or len(component) < 2 * MIN_HITS:
            return []
    if depth >= MAX_SPLIT_DEPTH or len(component) < 2 * MIN_HITS:
        return [candidate] if candidate is not None and candidate["accepted_geometry"] else []
    members = set(component)
    graph = nx.Graph()
    graph.add_nodes_from(component)
    for (u, v), probability in zip(data.edge_index.t().tolist(), probabilities):
        if u in members and v in members and probability >= threshold:
            graph.add_edge(u, v, weight=float(probability))
    if not graph.number_of_edges():
        trimmed = trim_outliers(data, component, probabilities, threshold, rms_limit, "trimmed")
        return [trimmed] if trimmed is not None else []
    tree = nx.maximum_spanning_tree(graph)
    weak_edges = sorted(tree.edges(data=True), key=lambda edge: edge[2]["weight"])[:5]
    original_score = -np.inf
    best_parts = None
    best_score = original_score
    # Try one- and two-edge cuts. The small cap keeps STEP4 bounded on dense graphs.
    for cut_count in (1, 2):
        for cuts in itertools.combinations(weak_edges, cut_count):
            trial = tree.copy()
            trial.remove_edges_from((u, v) for u, v, _ in cuts)
            parts = [sorted(value) for value in nx.connected_components(trial) if len(value) >= MIN_HITS]
            if len(parts) < 2:
                continue
            fitted = [fit_candidate(data, part, probabilities, threshold, rms_limit, "split") for part in parts]
            if any(value is None or not value["accepted_geometry"] for value in fitted):
                continue
            total = sum(score(value) for value in fitted)
            if total > best_score + 0.5:
                best_score = total
                best_parts = parts
    if best_parts is None:
        trimmed = trim_outliers(data, component, probabilities, threshold, rms_limit, "trimmed")
        return [trimmed] if trimmed is not None else []
    result = []
    for part in best_parts:
        result.extend(split_component(data, part, probabilities, threshold, rms_limit, depth + 1))
    return result


def score(candidate):
    return 2.0 * min(candidate["hits"] / 10.0, 2.0) + min(candidate["mean_probability"], 1.0) - min(candidate["rms"] / 2.0, 2.0)


def hierarchy_partition(data, component, probabilities, strict_threshold):
    parent = fit_candidate(data, component, probabilities, strict_threshold, STRICT_RMS, "strict")
    if len(component) < MIN_HIERARCHY_SIZE or (parent is not None and parent["accepted_geometry"] and parent["rms"] < 0.20):
        return None
    best = None
    start = int(np.floor(strict_threshold * 20.0)) + 1
    for value in range(start, 17):
        threshold = value / 20.0
        parts = components(data, probabilities, threshold, component)
        candidates = [trim_outliers(data, part, probabilities, threshold, STRICT_RMS, "hierarchy") for part in parts]
        candidates = [candidate for candidate in candidates if candidate is not None]
        if len(candidates) < 2:
            continue
        covered = len({index for candidate in candidates for index in candidate["indices"]}) / len(component)
        if covered < MIN_HIERARCHY_COVERAGE:
            continue
        weighted_rms = sum(candidate["rms"] * candidate["hits"] for candidate in candidates) / sum(candidate["hits"] for candidate in candidates)
        rms_gain = (parent["rms"] - weighted_rms) if parent is not None else float("inf")
        if parent is not None and parent["accepted_geometry"] and rms_gain < MIN_HIERARCHY_RMS_GAIN:
            continue
        mean_probability = float(np.mean([candidate["mean_probability"] for candidate in candidates]))
        utility = covered + 0.25 * mean_probability - 0.5 * weighted_rms
        if best is None or utility > best[0]:
            best = (utility, candidates, threshold, covered, rms_gain)
    if best is None:
        return None
    _, candidates, threshold, covered, rms_gain = best
    for candidate in candidates:
        candidate["hierarchy_threshold"] = threshold
        candidate["hierarchy_coverage"] = covered
        candidate["hierarchy_rms_gain"] = rms_gain
    return candidates


def run_two_pass(data, model, device, variant, strict_threshold, relaxed_threshold, probabilities=None):
    probabilities = predict(data, model, device, variant) if probabilities is None else probabilities
    strict_components = components(data, probabilities, strict_threshold)
    strict_candidates = []
    for component in strict_components:
        hierarchy = hierarchy_partition(data, component, probabilities, strict_threshold)
        strict_candidates.extend(hierarchy or split_component(data, component, probabilities, strict_threshold, STRICT_RMS))
    used = {index for candidate in strict_candidates for index in candidate["indices"]}
    leftover = set(range(data.num_nodes)) - used
    relaxed_candidates = [
        trim_outliers(data, component, probabilities, relaxed_threshold, RELAXED_RMS, "relaxed")
        for component in components(data, probabilities, relaxed_threshold, leftover)
    ]
    candidates = strict_candidates + [value for value in relaxed_candidates if value is not None]
    candidates.sort(key=score, reverse=True)
    final = []
    final_used = set()
    rejected = Counter()
    for candidate in candidates:
        if not candidate["accepted_geometry"]:
            rejected["rms_or_geometry"] += 1
            continue
        if final_used.intersection(candidate["indices"]):
            rejected["hit_conflict"] += 1
            continue
        candidate["score"] = score(candidate)
        final.append(candidate)
        final_used.update(candidate["indices"])
    return final, {
        "strict_components": len(strict_components),
        "candidate_count": len(candidates),
        "accepted_count": len(final),
        "split_candidates": sum(candidate["stage"] == "split" for candidate in candidates),
        "hierarchy_candidates": sum(candidate["stage"] == "hierarchy" for candidate in candidates),
        "trimmed_candidates": sum(bool(candidate.get("removed_indices")) for candidate in candidates),
        "rejected": dict(rejected),
    }


def truth_sets(data):
    result = []
    for index in range(data.num_nodes):
        start, end = int(data.truth_ptr[index]), int(data.truth_ptr[index + 1])
        result.append(set(map(tuple, data.truth_pairs[start:end].tolist())))
    return result


def evaluate(data, tracks):
    associations = truth_sets(data)
    truth_sizes = Counter(key for values in associations for key in values)
    eligible = {key for key, size in truth_sizes.items() if size >= MIN_HITS}
    matched = set()
    fake = clones = 0
    for track in tracks:
        counts = Counter(key for index in track["indices"] for key in associations[index])
        dominant, count = counts.most_common(1)[0] if counts else ((-1, -1), 0)
        purity = count / max(track["hits"], 1)
        completeness = count / max(truth_sizes[dominant], 1)
        track.update({"truth": list(dominant), "purity": purity, "completeness": completeness})
        if dominant not in eligible or purity < 0.80 or completeness < 0.50:
            fake += 1
        elif dominant in matched:
            clones += 1
        else:
            matched.add(dominant)
    return {
        "truth_tracks": len(eligible),
        "excluded_short_truth_tracks": len(truth_sizes) - len(eligible),
        "matched_tracks": len(matched),
        "track_efficiency": len(matched) / max(len(eligible), 1),
        "fake_tracks": fake,
        "clone_tracks": clones,
        "missed_tracks": [list(value) for value in sorted(eligible - matched)],
        "noise_nodes": int(data.is_noise.sum()),
        "shared_nodes": int(data.shared_hit.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="AI工程化4 两级寻迹")
    parser.add_argument("--features", choices=["time", "no-time"], required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--run-tag", default="clean_physics_v2")
    parser.add_argument("--strict-threshold", type=float)
    parser.add_argument("--relaxed-threshold", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset = load_graph_dataset(VAL_DATA if args.split == "validation" else TEST_DATA)
    stem = f"{args.features.replace('-', '_')}_{args.run_tag}"
    metadata = json.loads((ROOT / f"model_metadata_{stem}.json").read_text(encoding="utf-8"))
    strict_threshold = float(args.strict_threshold if args.strict_threshold is not None else metadata["edge_threshold"])
    relaxed_threshold = float(args.relaxed_threshold if args.relaxed_threshold is not None else max(0.35, strict_threshold - 0.15))
    if not 0.0 < relaxed_threshold <= strict_threshold < 1.0:
        raise ValueError("必须满足 0 < relaxed-threshold <= strict-threshold < 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionNetwork(*model_dimensions(args.features)).to(device)
    model.load_state_dict(torch.load(ROOT / f"best_interaction_network_{stem}.pth", map_location=device))
    model.eval()
    reports = []
    for index, data in enumerate(dataset[:args.rounds], 1):
        started = time.perf_counter()
        tracks, diagnostics = run_two_pass(data, model, device, args.features, strict_threshold, relaxed_threshold)
        result = evaluate(data, tracks)
        result.update({"round": index, "elapsed_seconds": time.perf_counter() - started, "diagnostics": diagnostics})
        reports.append(result)
    total_truth = sum(r["truth_tracks"] for r in reports)
    total_matched = sum(r["matched_tracks"] for r in reports)
    total_fake = sum(r["fake_tracks"] for r in reports)
    total_clone = sum(r["clone_tracks"] for r in reports)
    def efficiency_bin(predicate):
        selected = [r for r in reports if predicate(r)]
        truth = sum(r["truth_tracks"] for r in selected)
        return {
            "events": len(selected),
            "truth_tracks": truth,
            "matched_tracks": sum(r["matched_tracks"] for r in selected),
            "micro_efficiency": sum(r["matched_tracks"] for r in selected) / max(truth, 1),
            "mean_fake_tracks": float(np.mean([r["fake_tracks"] for r in selected])) if selected else 0.0,
        }
    summary = {
        "mean_track_efficiency": float(np.mean([r["track_efficiency"] for r in reports])),
        "micro_track_efficiency": total_matched / max(total_truth, 1),
        "mean_fake_tracks": float(np.mean([r["fake_tracks"] for r in reports])),
        "mean_clone_tracks": float(np.mean([r["clone_tracks"] for r in reports])),
        "mean_noise_nodes": float(np.mean([r["noise_nodes"] for r in reports])),
        "mean_elapsed_seconds": float(np.mean([r["elapsed_seconds"] for r in reports])),
        "total_truth_tracks": total_truth,
        "total_matched_tracks": total_matched,
        "total_fake_tracks": total_fake,
        "total_clone_tracks": total_clone,
        "events_efficiency_below_90pct": sum(r["track_efficiency"] < 0.90 for r in reports),
        "events_efficiency_below_80pct": sum(r["track_efficiency"] < 0.80 for r in reports),
        "efficiency_by_noise_nodes": {
            "0_30": efficiency_bin(lambda r: r["noise_nodes"] < 30),
            "30_45": efficiency_bin(lambda r: 30 <= r["noise_nodes"] < 45),
            "45_plus": efficiency_bin(lambda r: r["noise_nodes"] >= 45),
        },
    }
    output = args.output or ROOT / f"step4_ai4_{stem}_{args.split}.json"
    save_json(output, {"features": args.features, "split": args.split, "strict_threshold": strict_threshold, "relaxed_threshold": relaxed_threshold, "summary": summary, "rounds": reports})
    print(summary)
    print("报告：", output)


if __name__ == "__main__":
    main()
