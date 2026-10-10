from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
AI4 = ROOT.parent / "AI工程化4"
CLASSIC = ROOT.parent / "工程化2"
sys.path.insert(0, str(AI4))
sys.path.insert(0, str(CLASSIC))

from ca_tracker_core import CATracker, DriftWire  # noqa: E402
from hough_tracker_core import HoughPipeline  # noqa: E402
from common import InteractionNetwork, fit_2d_circle, load_axial_hits, model_dimensions  # noqa: E402
from STEP4_两级寻迹 import predict  # noqa: E402
from step4_hit_audit.analyze_physics_algorithm_loss import CORE_PROFILE, truth_maps  # noqa: E402
from step4_short_rerun_experiment.run_experiment import run_with_short_rerun  # noqa: E402
from STEP1_生成节点分类派生数据 import node_context  # noqa: E402
from STEP2_训练节点分类器 import NodeStateClassifier, binary_uncertainty  # noqa: E402
from STEP4_评估节点分类器 import add_metrics, evaluate_tracks, finalize_tracks  # noqa: E402
from STEP4_训练经典候选裁决器 import features as adjudicator_features, probability_column  # noqa: E402


def residual_items(data, nodes):
    return [
        {
            "cell_id": int(node),
            "detector_cell_id": int(data.cell_id[node]),
            "a": float(data.wire_x[node]),
            "b": float(data.wire_y[node]),
            "d": float(data.drift_d[node]),
            "true_xy": (0.0, 0.0),
        }
        for node in nodes
    ]


def fit_candidate(data, indices, source):
    indices = sorted(set(map(int, indices)))
    if len(indices) < 5:
        return None
    fit = fit_2d_circle(
        data.wire_x[indices].cpu().numpy(),
        data.wire_y[indices].cpu().numpy(),
        data.drift_d[indices].cpu().numpy(),
    )
    if fit is None or not np.all(np.isfinite(fit)):
        return None
    a, b, radius = fit
    residuals = np.abs(
        np.hypot(data.wire_x[indices].cpu().numpy() - a, data.wire_y[indices].cpu().numpy() - b)
        - radius
    ) - data.drift_d[indices].cpu().numpy()
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return {
        "indices": indices,
        "hits": len(indices),
        "fit": [float(a), float(b), float(radius)],
        "rms": rms,
        "dca": abs(float(np.hypot(a, b) - radius)),
        "stage": source,
        "accepted_geometry": rms <= 1.0,
    }


def ca_candidates(data, items):
    tracker = CATracker(search_dist=180.0, gap_tol=2.0, angle_cut=0.75)
    tracker.set_wires([
        DriftWire(item["cell_id"], item["a"], item["b"], item["d"], original_item=item)
        for item in items
    ])
    result = []
    for cells in tracker.run_pipeline(min_len=4):
        indices = {cell.w1.id for cell in cells} | {cell.w2.id for cell in cells}
        candidate = fit_candidate(data, indices, "classic_ca")
        if candidate is not None:
            result.append(candidate)
    return result


def hough_candidates(data, items):
    engine = HoughPipeline(
        theta_bins=500, rho_bins=500, top_k=15, nms_window=12,
        score_min=4, residual_cut_mm=2.5, rms_cut=1.0,
    )
    result = []
    for candidate in engine.get_candidates(items):
        fitted = fit_candidate(data, candidate["wire_ids"], "classic_hough")
        if fitted is not None:
            result.append(fitted)
    return result


def consensus_fusion(data, ca_pool, hough_pool, iou_threshold):
    proposals = []
    used_ca, used_hough = set(), set()
    for ca_index, ca in enumerate(ca_pool):
        ca_nodes = set(ca["indices"])
        best = None
        for hough_index, hough in enumerate(hough_pool):
            if hough_index in used_hough:
                continue
            hough_nodes = set(hough["indices"])
            union = ca_nodes | hough_nodes
            iou = len(ca_nodes & hough_nodes) / max(len(union), 1)
            if best is None or iou > best[0]:
                best = (iou, hough_index, union)
        if best is None or best[0] < iou_threshold:
            continue
        iou, hough_index, union = best
        candidate = fit_candidate(data, union, "classic_consensus")
        if candidate is not None and candidate["accepted_geometry"]:
            candidate["iou"] = iou
            proposals.append(candidate)
            used_ca.add(ca_index)
            used_hough.add(hough_index)
    leftover = [candidate for index, candidate in enumerate(ca_pool) if index not in used_ca]
    leftover.extend(candidate for index, candidate in enumerate(hough_pool) if index not in used_hough)
    return proposals, leftover


def arbitrate(candidates):
    candidates = [candidate for candidate in candidates if candidate["accepted_geometry"]]
    candidates.sort(key=lambda value: (-value["hits"], value["rms"]))
    accepted, used = [], set()
    for candidate in candidates:
        nodes = set(candidate["indices"])
        if nodes & used:
            continue
        accepted.append(candidate)
        used.update(nodes)
    return accepted


def hybrid_arbitrate(consensus_pool, solo_pool):
    consensus = arbitrate(consensus_pool)
    used = {node for candidate in consensus for node in candidate["indices"]}
    solo = [candidate for candidate in solo_pool if candidate["accepted_geometry"] and candidate["rms"] < 0.8]
    solo.sort(key=lambda value: (-value["hits"], value["rms"]))
    accepted = list(consensus)
    for candidate in solo:
        nodes = set(candidate["indices"])
        if nodes & used:
            continue
        accepted.append(candidate)
        used.update(nodes)
    return accepted


def augment_candidates_with_local_shared(
    data, candidates, used, shared_probability, edge_probability,
    shared_threshold=0.45, edge_threshold=0.55, max_shared=3,
):
    edges = data.edge_index.cpu().numpy().T
    augmented = []
    for candidate in candidates:
        members = set(candidate["indices"])
        proposals = {}
        for (u, v), probability in zip(edges, edge_probability):
            u, v = int(u), int(v)
            if probability < edge_threshold:
                continue
            if u in members and v in used and v not in members:
                node = v
            elif v in members and u in used and u not in members:
                node = u
            else:
                continue
            if shared_probability[node] < shared_threshold:
                continue
            score = float(probability * shared_probability[node])
            proposals[node] = max(proposals.get(node, 0.0), score)
        selected = [node for node, _ in sorted(proposals.items(), key=lambda item: -item[1])[:max_shared]]
        if not selected:
            continue
        fitted = fit_candidate(data, list(members) + selected, "classic_shared_candidate")
        if fitted is not None and fitted["accepted_geometry"]:
            fitted["shared_added_hits"] = len(selected)
            augmented.append(fitted)
    return augmented


def merge_candidate_fragments(data, candidates, edge_probability, cross_edge_min=30, rms_limit=1.0):
    if len(candidates) < 2:
        return list(candidates)
    edges = data.edge_index.cpu().numpy().T
    adjacency = [set() for _ in candidates]
    for (left_index, left), (right_index, right) in itertools.combinations(enumerate(candidates), 2):
        left_nodes, right_nodes = set(left["indices"]), set(right["indices"])
        cross_support = sum(
            probability >= 0.5 for (u, v), probability in zip(edges, edge_probability)
            if (int(u) in left_nodes and int(v) in right_nodes)
            or (int(u) in right_nodes and int(v) in left_nodes)
        )
        pair = fit_candidate(data, left["indices"] + right["indices"], "classic_pair")
        geometry_support = pair is not None and pair["rms"] <= rms_limit
        if cross_support >= cross_edge_min or geometry_support:
            adjacency[left_index].add(right_index)
            adjacency[right_index].add(left_index)

    merged, visited = [], set()
    for start in range(len(candidates)):
        if start in visited:
            continue
        stack, component = [start], []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        if len(component) == 1:
            merged.append(candidates[start])
            continue
        indices = [node for index in component for node in candidates[index]["indices"]]
        candidate = fit_candidate(data, indices, "classic_merged")
        if candidate is not None:
            candidate["merged_fragments"] = len(component)
            merged.append(candidate)
        else:
            merged.extend(candidates[index] for index in component)
    return merged


def leave_one_out_features(data, indices):
    held_out = []
    for node in indices:
        kept = [value for value in indices if value != node]
        fit = fit_2d_circle(
            data.wire_x[kept].cpu().numpy(), data.wire_y[kept].cpu().numpy(),
            data.drift_d[kept].cpu().numpy(),
        )
        if fit is None or not np.all(np.isfinite(fit)):
            continue
        a, b, radius = fit
        held_out.append(abs(
            abs(np.hypot(float(data.wire_x[node]) - a, float(data.wire_y[node]) - b) - radius)
            - float(data.drift_d[node])
        ))
    if not held_out:
        return float("inf"), float("inf")
    return float(np.mean(held_out)), float(np.max(held_out))


def nearest_baseline_features(data, indices, baseline_tracks):
    x = data.wire_x[indices].cpu().numpy()
    y = data.wire_y[indices].cpu().numpy()
    d = data.drift_d[indices].cpu().numpy()
    best = None
    for track in baseline_tracks:
        fit = track.get("fit")
        if fit is None:
            fit = fit_2d_circle(
                data.wire_x[track["indices"]].cpu().numpy(),
                data.wire_y[track["indices"]].cpu().numpy(),
                data.drift_d[track["indices"]].cpu().numpy(),
            )
        if fit is None or not np.all(np.isfinite(fit)):
            continue
        a, b, radius = fit
        residuals = np.abs(np.abs(np.hypot(x - a, y - b) - radius) - d)
        value = (
            float(np.mean(residuals)), float(np.max(residuals)),
            float(np.mean(residuals <= 1.0)), len(track["indices"]),
        )
        if best is None or value[0] < best[0]:
            best = value
    return best or (float("inf"), float("inf"), 0.0, 0)


def candidate_feature_row(data, candidate, baseline_tracks, edge_probability, noise, uncertainty):
    indices = candidate["indices"]
    member = np.zeros(data.num_nodes, dtype=bool)
    member[indices] = True
    edges = data.edge_index.cpu().numpy().T
    internal = member[edges[:, 0]] & member[edges[:, 1]]
    values = edge_probability[internal]
    support = {node: 0 for node in indices}
    for (u, v), probability in zip(edges[internal], values):
        if probability >= 0.5:
            support[int(u)] += 1
            support[int(v)] += 1
    loo_mean, loo_max = leave_one_out_features(data, indices)
    baseline_mean, baseline_max, baseline_fraction, baseline_hits = nearest_baseline_features(
        data, indices, baseline_tracks
    )
    return {
        "source": candidate["stage"], "hits": candidate["hits"], "rms": candidate["rms"],
        "dca": candidate["dca"], "consensus_iou": float(candidate.get("iou", 0.0)),
        "mean_edge_probability": float(np.mean(values)) if values.size else 0.0,
        "min_edge_probability": float(np.min(values)) if values.size else 0.0,
        "min_internal_support": min(support.values(), default=0),
        "mean_noise_probability": float(np.mean(noise[indices])),
        "max_noise_probability": float(np.max(noise[indices])),
        "mean_uncertainty": float(np.mean(uncertainty[indices])),
        "loo_mean_heldout_residual": loo_mean, "loo_max_heldout_residual": loo_max,
        "nearest_baseline_mean_residual": baseline_mean,
        "nearest_baseline_max_residual": baseline_max,
        "nearest_baseline_compatible_fraction": baseline_fraction,
        "nearest_baseline_track_hits": baseline_hits,
    }


def classify_candidates(data, candidates, baseline_tracks, edge_probability, noise, uncertainty, bundle):
    if not candidates:
        return [], []
    frame = pd.DataFrame([
        candidate_feature_row(
            data, candidate, baseline_tracks, edge_probability, noise, uncertainty
        ) for candidate in candidates
    ])
    probabilities = bundle["model"].predict_proba(adjudicator_features(frame))
    accept_probability = probability_column(bundle["model"], probabilities, "accept")
    fragment_probability = probability_column(bundle["model"], probabilities, "fragment")
    accepted, fragments = [], []
    for candidate, accept_value, fragment_value in zip(
        candidates, accept_probability, fragment_probability
    ):
        if accept_value >= bundle["accept_threshold"]:
            candidate["adjudication"] = "accept"
            accepted.append(candidate)
        elif fragment_value >= bundle["fragment_threshold"]:
            candidate["adjudication"] = "fragment"
            fragments.append(candidate)
    return accepted, fragments


def generate_hybrid(data, nodes, probabilities, iou_threshold):
    items = residual_items(data, nodes)
    if len(items) < 5:
        return [], [], [], [], []
    ca_pool = ca_candidates(data, items)
    hough_pool = hough_candidates(data, items)
    consensus_pool, solo_pool = consensus_fusion(data, ca_pool, hough_pool, iou_threshold)
    hybrid = hybrid_arbitrate(consensus_pool, solo_pool)
    return ca_pool, hough_pool, consensus_pool, solo_pool, hybrid


def expand_residual_with_local_shared(
    data, residual, used, noise, shared_probability, edge_probability,
    noise_threshold, shared_threshold, neighbor_edge_threshold,
):
    residual_set = set(residual)
    occupied_shared = {
        node for node in used
        if noise[node] < noise_threshold and shared_probability[node] >= shared_threshold
    }
    if not occupied_shared:
        return list(residual), 0
    edges = data.edge_index.cpu().numpy().T
    eligible = set()
    for (u, v), probability in zip(edges, edge_probability):
        u, v = int(u), int(v)
        if probability < neighbor_edge_threshold:
            continue
        if u in residual_set and v in occupied_shared:
            eligible.add(v)
        elif v in residual_set and u in occupied_shared:
            eligible.add(u)
    expanded = sorted(residual_set | eligible)
    return expanded, len(eligible)


def augment_hybrid_with_shared(base_candidates, shared_candidates):
    accepted = list(base_candidates)
    used = {node for candidate in accepted for node in candidate["indices"]}
    additions = sorted(
        shared_candidates,
        key=lambda value: (-value["hits"], value["rms"]),
    )
    for candidate in additions:
        nodes = set(candidate["indices"])
        if nodes & used:
            continue
        accepted.append(candidate)
        used.update(nodes)
    return accepted


def main():
    parser = argparse.ArgumentParser(description="AI5 残差图经典 CA/Hough/IoU 寻迹实验")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--node-model-file", type=Path, required=True)
    parser.add_argument("--noise-threshold", type=float, default=0.95)
    parser.add_argument("--shared-threshold", type=float, default=0.45)
    parser.add_argument("--shared-neighbor-edge-threshold", type=float, default=0.55)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument(
        "--adjudicator-model", type=Path,
        default=ROOT / "models" / "classic_candidate_adjudicator_v1.joblib",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graphs = torch.load(args.audit_dir / "graphs.pt", weights_only=False)
    records = json.loads((args.audit_dir / "audit_records.json").read_text(encoding="utf-8"))
    limit = min(len(graphs), len(records), args.rounds)
    source = load_axial_hits()
    source_by_event = {int(evt): group for evt, group in source.groupby("EvtID")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_model = InteractionNetwork(*model_dimensions("time")).to(device)
    edge_model.load_state_dict(torch.load(args.model_file, map_location=device))
    edge_model.eval()
    node_model = NodeStateClassifier(node_in=13, edge_in=8).to(device)
    node_model.load_state_dict(torch.load(args.node_model_file, map_location=device, weights_only=False))
    node_model.eval()
    adjudicator = joblib.load(args.adjudicator_model)

    names = (
        "baseline", "ca_only", "hough_only", "consensus", "hybrid", "hybrid_merged",
        "hybrid_accept_only", "hybrid_fragment_merge", "hybrid_adjudicated",
        "shared_residual_hybrid", "shared_residual_hybrid_merged",
        "hybrid_shared_augment", "hybrid_shared_augment_merged",
        "shared_candidate_augmented", "shared_candidate_augmented_merged",
    )
    totals = {name: Counter() for name in names}
    diagnostics = Counter()
    for index, (data, record) in enumerate(zip(graphs[:limit], records[:limit]), 1):
        probabilities = predict(data, edge_model, device, "time")
        baseline_tracks, _ = run_with_short_rerun(
            data, edge_model, device, "time", probabilities, 0.70, 0.575, 0.55, 0.20, CORE_PROFILE
        )
        context, edge_context = node_context(data, probabilities, baseline_tracks)
        with torch.no_grad():
            noise_logits, shared_logits = node_model(
                data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device),
                context.to(device), edge_context.to(device),
            )
        noise = torch.sigmoid(noise_logits).cpu().numpy()
        shared_probability = torch.sigmoid(shared_logits).cpu().numpy()
        uncertainty = np.maximum(
            binary_uncertainty(noise_logits).cpu().numpy(),
            binary_uncertainty(shared_logits).cpu().numpy(),
        )
        used = {node for track in baseline_tracks for node in track["indices"]}
        residual = [node for node in range(data.num_nodes) if node not in used and noise[node] < args.noise_threshold]
        ca_pool, hough_pool, consensus_pool, solo_pool, hybrid = generate_hybrid(
            data, residual, probabilities, args.iou_threshold
        )
        hybrid_merged = merge_candidate_fragments(data, hybrid, probabilities)
        shared_candidate_pool = augment_candidates_with_local_shared(
            data, hybrid, used, shared_probability, probabilities,
            args.shared_threshold, args.shared_neighbor_edge_threshold,
        )
        shared_candidate_augmented = arbitrate(shared_candidate_pool)
        shared_candidate_augmented_merged = merge_candidate_fragments(
            data, shared_candidate_augmented, probabilities
        )
        accepted, fragments = classify_candidates(
            data, hybrid, baseline_tracks, probabilities, noise, uncertainty, adjudicator
        )
        fragment_merge = [
            candidate for candidate in merge_candidate_fragments(
                data, fragments, probabilities
            ) if candidate["stage"] == "classic_merged"
        ]
        combined = merge_candidate_fragments(data, accepted + fragments, probabilities)
        hybrid_adjudicated = [
            candidate for candidate in combined
            if candidate["stage"] == "classic_merged" or candidate.get("adjudication") == "accept"
        ]
        shared_residual, shared_nodes_added = expand_residual_with_local_shared(
            data, residual, used, noise, shared_probability, probabilities,
            args.noise_threshold, args.shared_threshold, args.shared_neighbor_edge_threshold,
        )
        shared_ca, shared_hough, shared_consensus, shared_solo, shared_hybrid = generate_hybrid(
            data, shared_residual, probabilities, args.iou_threshold
        )
        shared_hybrid_merged = merge_candidate_fragments(data, shared_hybrid, probabilities)
        shared_augment = augment_hybrid_with_shared(hybrid, shared_hybrid)
        shared_augment_merged = merge_candidate_fragments(data, shared_augment, probabilities)
        variants = {
            "baseline": baseline_tracks,
            "ca_only": baseline_tracks + arbitrate(ca_pool),
            "hough_only": baseline_tracks + arbitrate(hough_pool),
            "consensus": baseline_tracks + arbitrate(consensus_pool),
            "hybrid": baseline_tracks + hybrid,
            "hybrid_merged": baseline_tracks + hybrid_merged,
            "shared_candidate_augmented": baseline_tracks + shared_candidate_augmented,
            "shared_candidate_augmented_merged": baseline_tracks + shared_candidate_augmented_merged,
            "hybrid_accept_only": baseline_tracks + accepted,
            "hybrid_fragment_merge": baseline_tracks + fragment_merge,
            "hybrid_adjudicated": baseline_tracks + hybrid_adjudicated,
            "shared_residual_hybrid": baseline_tracks + shared_hybrid,
            "shared_residual_hybrid_merged": baseline_tracks + shared_hybrid_merged,
            "hybrid_shared_augment": baseline_tracks + shared_augment,
            "hybrid_shared_augment_merged": baseline_tracks + shared_augment_merged,
        }
        diagnostics.update({
            "residual_nodes": len(residual), "ca_candidates": len(ca_pool),
            "hough_candidates": len(hough_pool), "consensus_candidates": len(consensus_pool),
            "solo_candidates": len(solo_pool),
            "merged_candidate_reduction": len(hybrid) - len(hybrid_merged),
            "shared_residual_nodes": len(shared_residual),
            "shared_nodes_added": shared_nodes_added,
            "shared_hybrid_candidates": len(shared_hybrid),
            "shared_merged_candidate_reduction": len(shared_hybrid) - len(shared_hybrid_merged),
        })
        source_sizes = Counter()
        for evt in record["selected_event_ids"]:
            group = source_by_event[int(evt)]
            source_sizes.update((int(e), int(t)) for e, t in zip(group.EvtID, group.TrkID))
        associations = truth_maps(data)["exclusive_winner"]
        for name, tracks in variants.items():
            add_metrics(totals[name], evaluate_tracks(tracks, associations, source_sizes))
        if index % 50 == 0:
            print(f"已完成 {index}/{limit}", flush=True)

    metrics = {name: finalize_tracks(total, limit) for name, total in totals.items()}
    baseline = metrics["baseline"]
    report = {
        "events": limit,
        "parameters": {
            "noise_threshold": args.noise_threshold,
            "shared_threshold": args.shared_threshold,
            "iou_threshold": args.iou_threshold,
            "shared_neighbor_edge_threshold": args.shared_neighbor_edge_threshold,
        },
        "metrics": metrics,
        "delta": {
            name: {
                key: metrics[name][key] - baseline[key]
                for key in (
                    "algorithm_matched_tracks", "algorithm_efficiency_given_visible",
                    "end_to_end_efficiency_from_source", "algorithm_efficiency_5_6_visible",
                    "mean_fake_tracks", "mean_clone_tracks",
                )
            }
            for name in names if name != "baseline"
        },
        "diagnostics": dict(diagnostics),
        "contract": "classic trackers see only unused non-noise nodes; truth is used only for offline evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["delta"], ensure_ascii=False, indent=2))
    print("报告：", args.output.resolve())


if __name__ == "__main__":
    main()
