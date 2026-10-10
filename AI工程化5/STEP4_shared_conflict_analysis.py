from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
AI4 = ROOT.parent / "AI工程化4"
sys.path.insert(0, str(AI4))

from common import InteractionNetwork, fit_2d_circle, load_axial_hits, model_dimensions  # noqa: E402
from STEP4_两级寻迹 import predict  # noqa: E402
from step4_hit_audit.analyze_physics_algorithm_loss import CORE_PROFILE, truth_maps  # noqa: E402
from step4_short_rerun_experiment.run_experiment import run_with_short_rerun  # noqa: E402


def fit(data, indices):
    if len(indices) < 5:
        return None
    indices = sorted(set(indices))
    values = fit_2d_circle(
        data.wire_x[indices].numpy(), data.wire_y[indices].numpy(), data.drift_d[indices].numpy()
    )
    if values is None:
        return None
    a, b, radius = values
    residuals = np.abs(np.hypot(data.wire_x[indices].numpy() - a, data.wire_y[indices].numpy() - b) - radius) - data.drift_d[indices].numpy()
    return {"fit": [float(v) for v in values], "rms": float(np.sqrt(np.mean(residuals ** 2))), "hits": len(indices)}


def track_truth(track, associations):
    return Counter(key for node in track["indices"] for key in associations[node])


def main():
    parser = argparse.ArgumentParser(description="AI5 shared 点候选冲突与拟合后验证")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--features", choices=["time", "no-time"], default="time")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--strict-threshold", type=float, default=0.70)
    parser.add_argument("--relaxed-threshold", type=float, default=0.575)
    parser.add_argument("--rerun-threshold", type=float, default=0.55)
    parser.add_argument("--rerun-rms", type=float, default=0.20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graphs = torch.load(args.audit_dir / "graphs.pt", weights_only=False)
    records = json.loads((args.audit_dir / "audit_records.json").read_text(encoding="utf-8"))
    limit = min(args.rounds, len(graphs), len(records))
    source = load_axial_hits()
    source_by_event = {int(evt): group for evt, group in source.groupby("EvtID")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionNetwork(*model_dimensions(args.features)).to(device)
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.eval()

    stats = Counter()
    rms_deltas = []
    parameter_deltas = []
    conflict_graph = nx.Graph()
    conflict_edges = 0

    for graph_index, (data, record) in enumerate(zip(graphs[:limit], records[:limit])):
        source_sizes = Counter()
        for evt in record["selected_event_ids"]:
            group = source_by_event[int(evt)]
            source_sizes.update((int(e), int(t)) for e, t in zip(group.EvtID, group.TrkID))
        probabilities = predict(data, model, device, args.features)
        tracks, _ = run_with_short_rerun(
            data, model, device, args.features, probabilities,
            args.strict_threshold, args.relaxed_threshold,
            args.rerun_threshold, args.rerun_rms, CORE_PROFILE,
        )
        associations = truth_maps(data)["exclusive_winner"]
        event_nodes = []
        for track_index, track in enumerate(tracks):
            conflict_graph.add_node((graph_index, track_index))
            indices = track["indices"]
            shared = [node for node in indices if bool(data.shared_hit[node])]
            exclusive = [node for node in indices if not bool(data.shared_hit[node])]
            full_fit = fit(data, indices)
            exclusive_fit = fit(data, exclusive)
            if shared:
                stats["tracks_with_shared_nodes"] += 1
                stats["shared_nodes_in_tracks"] += len(shared)
            if full_fit and exclusive_fit:
                rms_deltas.append(full_fit["rms"] - exclusive_fit["rms"])
                parameter_deltas.append(float(np.linalg.norm(np.asarray(full_fit["fit"]) - np.asarray(exclusive_fit["fit"]))))
            truth = track_truth(track, associations)
            event_nodes.append((track_index, set(truth), set(shared)))
        for left in range(len(event_nodes)):
            for right in range(left + 1, len(event_nodes)):
                li, lt, ls = event_nodes[left]
                ri, rt, rs = event_nodes[right]
                overlap = lt & rt
                shared_overlap = ls & rs
                if overlap or shared_overlap:
                    conflict_graph.add_edge((graph_index, li), (graph_index, ri), overlap=len(overlap), shared_overlap=len(shared_overlap))
                    conflict_edges += 1
        stats["tracks"] += len(tracks)
        stats["graphs"] += 1

    report = {
        "audit_dir": str(args.audit_dir.resolve()),
        "model_file": str(args.model_file.resolve()),
        "graphs": stats["graphs"],
        "tracks": stats["tracks"],
        "tracks_with_shared_nodes": stats["tracks_with_shared_nodes"],
        "shared_nodes_in_tracks": stats["shared_nodes_in_tracks"],
        "conflict_edges": conflict_edges,
        "mean_rms_delta_full_minus_exclusive": float(np.mean(rms_deltas)) if rms_deltas else 0.0,
        "mean_abs_parameter_delta": float(np.mean(parameter_deltas)) if parameter_deltas else 0.0,
        "p95_abs_parameter_delta": float(np.percentile(np.abs(parameter_deltas), 95)) if parameter_deltas else 0.0,
        "interpretation": "Shared points are diagnostic evidence only; exclusive points define the first fit.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
