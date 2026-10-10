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
from STEP1_生成节点分类派生数据 import node_context  # noqa: E402
from STEP2_训练节点分类器 import NodeStateClassifier, binary_uncertainty  # noqa: E402
from STEP4_评估节点分类器 import add_metrics, evaluate_tracks, finalize_tracks  # noqa: E402


def anchor_fragments(data, probabilities, allowed, threshold):
    graph = nx.Graph()
    graph.add_nodes_from(allowed)
    edges = data.edge_index.cpu().numpy().T
    allowed_mask = np.zeros(data.num_nodes, dtype=bool)
    allowed_mask[list(allowed)] = True
    keep = (probabilities >= threshold) & allowed_mask[edges[:, 0]] & allowed_mask[edges[:, 1]]
    graph.add_edges_from((int(u), int(v)) for u, v in edges[keep])
    return [sorted(nodes) for nodes in nx.connected_components(graph) if 2 <= len(nodes) <= 6]


def anchored_candidate(
    data, anchors, shared_nodes, probabilities, shared_probability, uncertainty,
    owner_tracks, anchor_rms_limit, shared_residual_limit,
):
    if len(anchors) < 3:
        return None
    x = data.wire_x[anchors].cpu().numpy()
    y = data.wire_y[anchors].cpu().numpy()
    d = data.drift_d[anchors].cpu().numpy()
    fit = fit_2d_circle(x, y, d)
    if fit is None or not np.all(np.isfinite(fit)):
        return None
    a, b, radius = fit
    anchor_residuals = np.abs(np.hypot(x - a, y - b) - radius) - d
    anchor_rms = float(np.sqrt(np.mean(anchor_residuals ** 2)))
    if anchor_rms > anchor_rms_limit:
        return None
    compatible = []
    residuals = []
    for node in shared_nodes:
        value = abs(abs(np.hypot(float(data.wire_x[node]) - a, float(data.wire_y[node]) - b) - radius) - float(data.drift_d[node]))
        if value <= shared_residual_limit:
            compatible.append(node)
            residuals.append(value)
    indices = sorted(set(anchors + compatible))
    if len(indices) < 5:
        return None
    edges = data.edge_index.cpu().numpy().T
    member = np.zeros(data.num_nodes, dtype=bool)
    member[indices] = True
    internal = member[edges[:, 0]] & member[edges[:, 1]]
    edge_values = probabilities[internal]
    owner_anchor_residuals = []
    for node in compatible:
        owner = owner_tracks.get(node)
        if owner is None:
            continue
        oa, ob, owner_radius = owner["fit"]
        owner_anchor_residuals.extend(np.abs(np.hypot(x - oa, y - ob) - owner_radius) - d)
    return {
        "indices": indices,
        "hits": len(indices),
        "fit": [float(a), float(b), float(radius)],
        "rms": anchor_rms,
        "anchor_hits": len(anchors),
        "borrowed_shared_hits": len(compatible),
        "mean_shared_residual": float(np.mean(residuals)) if residuals else 0.0,
        "mean_edge_probability": float(np.mean(edge_values)) if edge_values.size else 0.0,
        "min_edge_probability": float(np.min(edge_values)) if edge_values.size else 0.0,
        "mean_shared_probability": float(np.mean(shared_probability[compatible])) if compatible else 0.0,
        "min_shared_probability": float(np.min(shared_probability[compatible])) if compatible else 0.0,
        "mean_uncertainty": float(np.mean(uncertainty[indices])),
        "mean_owner_residual_on_anchors": float(np.mean(owner_anchor_residuals)) if owner_anchor_residuals else 0.0,
        "stage": "shared_local_reconnect",
        "accepted_geometry": True,
    }


def local_reconnect(
    data, probabilities, baseline_tracks, noise_probability, shared_probability, uncertainty,
    noise_threshold, shared_threshold, fragment_threshold, borrow_edge_threshold,
    anchor_rms_limit, shared_residual_limit,
):
    used = {node for track in baseline_tracks for node in track["indices"]}
    owner_tracks = {node: track for track in baseline_tracks for node in track["indices"]}
    rejected_noise = set(np.flatnonzero(noise_probability >= noise_threshold).tolist())
    unused = set(range(data.num_nodes)) - used - rejected_noise
    fragments = anchor_fragments(data, probabilities, unused, fragment_threshold)
    edges = data.edge_index.cpu().numpy().T
    adjacency = {}
    for (u, v), probability in zip(edges, probabilities):
        if probability >= borrow_edge_threshold:
            adjacency.setdefault(int(u), set()).add(int(v))

    candidates = []
    for anchors in fragments:
        if len(anchors) < 3:
            continue
        borrowed = {
            node
            for anchor in anchors
            for node in adjacency.get(anchor, ())
            if node in used
            and node not in rejected_noise
            and shared_probability[node] >= shared_threshold
        }
        candidate = anchored_candidate(
            data, anchors, sorted(borrowed), probabilities, shared_probability, uncertainty,
            owner_tracks, anchor_rms_limit, shared_residual_limit
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(
        key=lambda value: (value["anchor_hits"], -value["rms"], -value["mean_shared_residual"]),
        reverse=True,
    )
    accepted = []
    used_anchors = set()
    for candidate in candidates:
        anchors = set(candidate["indices"]) - used
        if anchors & used_anchors:
            continue
        accepted.append(candidate)
        used_anchors.update(anchors)
    return baseline_tracks + accepted, {
        "fragments": len(fragments),
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "borrowed_shared_hits": sum(value["borrowed_shared_hits"] for value in accepted),
    }, accepted


def main():
    parser = argparse.ArgumentParser(description="AI5 shared 节点局部非破坏性重连")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--node-model-file", type=Path, required=True)
    parser.add_argument("--features", choices=["time"], default="time")
    parser.add_argument("--noise-threshold", type=float, default=0.95)
    parser.add_argument("--shared-threshold", type=float, default=0.45)
    parser.add_argument("--fragment-threshold", type=float, default=0.55)
    parser.add_argument("--borrow-edge-threshold", type=float, default=0.55)
    parser.add_argument("--anchor-rms-limit", type=float, default=1.0)
    parser.add_argument("--shared-residual-limit", type=float, default=1.0)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graphs = torch.load(args.audit_dir / "graphs.pt", weights_only=False)
    records = json.loads((args.audit_dir / "audit_records.json").read_text(encoding="utf-8"))
    limit = min(len(graphs), len(records), args.rounds or len(graphs))
    source = load_axial_hits()
    source_by_event = {int(evt): group for evt, group in source.groupby("EvtID")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_model = InteractionNetwork(*model_dimensions(args.features)).to(device)
    edge_model.load_state_dict(torch.load(args.model_file, map_location=device))
    edge_model.eval()
    node_model = NodeStateClassifier(node_in=13, edge_in=8).to(device)
    node_model.load_state_dict(torch.load(args.node_model_file, map_location=device, weights_only=False))
    node_model.eval()

    baseline_total, experiment_total, diagnostics = Counter(), Counter(), Counter()
    for index, (data, record) in enumerate(zip(graphs[:limit], records[:limit]), 1):
        probabilities = predict(data, edge_model, device, args.features)
        baseline_tracks, _ = run_with_short_rerun(
            data, edge_model, device, args.features, probabilities,
            0.70, 0.575, 0.55, 0.20, CORE_PROFILE,
        )
        context, edge_context = node_context(data, probabilities, baseline_tracks)
        with torch.no_grad():
            noise_logits, shared_logits = node_model(
                data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device),
                context.to(device), edge_context.to(device),
            )
        uncertainty = np.maximum(
            binary_uncertainty(noise_logits).cpu().numpy(),
            binary_uncertainty(shared_logits).cpu().numpy(),
        )
        tracks, event_diagnostics, _ = local_reconnect(
            data, probabilities, baseline_tracks,
            torch.sigmoid(noise_logits).cpu().numpy(), torch.sigmoid(shared_logits).cpu().numpy(), uncertainty,
            args.noise_threshold, args.shared_threshold, args.fragment_threshold,
            args.borrow_edge_threshold, args.anchor_rms_limit, args.shared_residual_limit,
        )
        diagnostics.update(event_diagnostics)
        source_sizes = Counter()
        for evt in record["selected_event_ids"]:
            group = source_by_event[int(evt)]
            source_sizes.update((int(e), int(t)) for e, t in zip(group.EvtID, group.TrkID))
        associations = truth_maps(data)["exclusive_winner"]
        add_metrics(baseline_total, evaluate_tracks(baseline_tracks, associations, source_sizes))
        add_metrics(experiment_total, evaluate_tracks(tracks, associations, source_sizes))
        if index % 100 == 0:
            print(f"已完成 {index}/{limit}", flush=True)

    baseline = finalize_tracks(baseline_total, limit)
    experiment = finalize_tracks(experiment_total, limit)
    report = {
        "events": limit,
        "audit_dir": str(args.audit_dir.resolve()),
        "edge_model_file": str(args.model_file.resolve()),
        "node_model_file": str(args.node_model_file.resolve()),
        "parameters": {
            "noise_threshold": args.noise_threshold,
            "shared_threshold": args.shared_threshold,
            "fragment_threshold": args.fragment_threshold,
            "borrow_edge_threshold": args.borrow_edge_threshold,
            "anchor_rms_limit": args.anchor_rms_limit,
            "shared_residual_limit": args.shared_residual_limit,
        },
        "baseline": baseline,
        "shared_local_reconnect": experiment,
        "delta": {
            key: experiment[key] - baseline[key]
            for key in (
                "algorithm_matched_tracks", "algorithm_efficiency_given_visible",
                "end_to_end_efficiency_from_source", "algorithm_efficiency_5_6_visible",
                "mean_fake_tracks", "mean_clone_tracks",
            )
        },
        "diagnostics": dict(diagnostics),
        "contract": "long tracks are frozen; shared nodes may be borrowed but anchor-only fit defines parameters",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["delta"], ensure_ascii=False, indent=2))
    print("报告：", args.output.resolve())


if __name__ == "__main__":
    main()
