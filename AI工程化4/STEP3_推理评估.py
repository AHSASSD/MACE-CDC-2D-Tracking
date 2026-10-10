from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from common import (
    ROOT, TEST_DATA, VAL_DATA, InteractionNetwork, fit_2d_circle,
    load_graph_dataset, model_dimensions, save_json, select_features,
)

MIN_HITS = 5
MIN_PURITY = 0.80
MIN_COMPLETENESS = 0.50


def infer(data, model, device, variant, threshold):
    x, edge_attr = select_features(data, variant)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x.to(device), data.edge_index.to(device), edge_attr.to(device))).cpu().numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    keep = probabilities >= threshold
    graph.add_edges_from(data.edge_index[:, keep].t().cpu().tolist())
    tracks = []
    for component in nx.connected_components(graph):
        indices = sorted(component)
        if len(indices) < MIN_HITS:
            continue
        x_values = data.wire_x[indices].numpy()
        y_values = data.wire_y[indices].numpy()
        d_values = data.drift_d[indices].numpy()
        fit = fit_2d_circle(x_values, y_values, d_values)
        if fit is None:
            continue
        a, b, radius = fit
        residuals = np.abs(np.hypot(x_values - a, y_values - b) - radius) - d_values
        keys = []
        for index in indices:
            start, end = int(data.truth_ptr[index]), int(data.truth_ptr[index + 1])
            keys.extend(map(tuple, data.truth_pairs[start:end].tolist()))
        dominant, count = Counter(keys).most_common(1)[0] if keys else ((-1, -1), 0)
        tracks.append({
            "indices": indices,
            "fit": [float(v) for v in fit],
            "hits": len(indices),
            "rms": float(np.sqrt(np.mean(residuals ** 2))),
            "truth": [int(dominant[0]), int(dominant[1])],
            "truth_hits": count,
        })
    return tracks


def evaluate(data, tracks):
    truth_sizes = Counter()
    for index in range(data.num_nodes):
        start, end = int(data.truth_ptr[index]), int(data.truth_ptr[index + 1])
        for key in map(tuple, data.truth_pairs[start:end].tolist()):
            truth_sizes[key] += 1
    eligible_truth = {key for key, size in truth_sizes.items() if size >= MIN_HITS}
    matched = set()
    fake = clones = 0
    for track in tracks:
        truth = tuple(track["truth"])
        purity = track["truth_hits"] / max(track["hits"], 1)
        completeness = track["truth_hits"] / max(truth_sizes[truth], 1)
        track.update({"purity": purity, "completeness": completeness})
        if purity < MIN_PURITY or completeness < MIN_COMPLETENESS or truth == (-1, -1):
            fake += 1
        elif truth not in eligible_truth:
            fake += 1
        elif truth in matched:
            clones += 1
        else:
            matched.add(truth)
    return {
        "truth_tracks": len(eligible_truth),
        "excluded_short_truth_tracks": len(truth_sizes) - len(eligible_truth),
        "matched_tracks": len(matched),
        "track_efficiency": len(matched) / max(len(eligible_truth), 1),
        "fake_tracks": fake,
        "clone_tracks": clones,
        "missed_tracks": [list(x) for x in eligible_truth - matched],
        "tracks": tracks,
        "noise_nodes": int(data.is_noise.sum()),
        "shared_nodes": int(data.shared_hit.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="AI工程化4 独立推理评估")
    parser.add_argument("--features", choices=["time", "no-time"], required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--run-tag", default="clean_physics_v2")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stem = f"{args.features.replace('-', '_')}_{args.run_tag}"
    model_file = args.model or ROOT / f"best_interaction_network_{stem}.pth"
    meta_file = ROOT / f"model_metadata_{stem}.json"
    data_file = args.data or (VAL_DATA if args.split == "validation" else TEST_DATA)
    dataset = load_graph_dataset(data_file)
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_in, edge_in = model_dimensions(args.features)
    model = InteractionNetwork(node_in, edge_in).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    threshold = float(metadata["edge_threshold"])
    rng = np.random.default_rng(20260823)
    reports = []
    for index in range(min(args.rounds, len(dataset))):
        data = dataset[int(rng.integers(len(dataset)))]
        started = time.perf_counter()
        tracks = infer(data, model, device, args.features, threshold)
        result = evaluate(data, tracks)
        result.update({"round": index + 1, "elapsed_seconds": time.perf_counter() - started})
        reports.append(result)
    summary = {
        "mean_track_efficiency": float(np.mean([r["track_efficiency"] for r in reports])) if reports else 0.0,
        "mean_fake_tracks": float(np.mean([r["fake_tracks"] for r in reports])) if reports else 0.0,
        "mean_clone_tracks": float(np.mean([r["clone_tracks"] for r in reports])) if reports else 0.0,
        "mean_noise_nodes": float(np.mean([r["noise_nodes"] for r in reports])) if reports else 0.0,
        "mean_elapsed_seconds": float(np.mean([r["elapsed_seconds"] for r in reports])) if reports else 0.0,
    }
    output = args.output or ROOT / f"ai4_{args.features.replace('-', '_')}_{args.split}_report.json"
    save_json(output, {"features": args.features, "split": args.split, "threshold": threshold, "summary": summary, "rounds": reports})
    print(summary)
    print("报告：", output)


if __name__ == "__main__":
    main()
