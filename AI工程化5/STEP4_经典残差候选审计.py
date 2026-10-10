from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

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
from STEP4_shared候选审计 import label_candidate, matched_truths  # noqa: E402
from STEP4_经典残差寻迹 import (  # noqa: E402
    ca_candidates, consensus_fusion, fit_candidate, hough_candidates, hybrid_arbitrate, residual_items,
)


def leave_one_out(data, indices):
    held_out = []
    parameter_shift = []
    full = fit_2d_circle(
        data.wire_x[indices].cpu().numpy(), data.wire_y[indices].cpu().numpy(),
        data.drift_d[indices].cpu().numpy(),
    )
    if full is None:
        return float("inf"), float("inf"), float("inf")
    for node in indices:
        kept = [value for value in indices if value != node]
        if len(kept) < 3:
            continue
        fit = fit_2d_circle(
            data.wire_x[kept].cpu().numpy(), data.wire_y[kept].cpu().numpy(),
            data.drift_d[kept].cpu().numpy(),
        )
        if fit is None or not np.all(np.isfinite(fit)):
            continue
        a, b, radius = fit
        residual = abs(
            abs(np.hypot(float(data.wire_x[node]) - a, float(data.wire_y[node]) - b) - radius)
            - float(data.drift_d[node])
        )
        held_out.append(residual)
        parameter_shift.append(float(np.linalg.norm(np.asarray(fit) - np.asarray(full))))
    if not held_out:
        return float("inf"), float("inf"), float("inf")
    return float(np.mean(held_out)), float(np.max(held_out)), float(np.max(parameter_shift))


def baseline_compatibility(data, indices, baseline_tracks):
    x = data.wire_x[indices].cpu().numpy()
    y = data.wire_y[indices].cpu().numpy()
    d = data.drift_d[indices].cpu().numpy()
    best = None
    for track_index, track in enumerate(baseline_tracks):
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
            float(np.mean(residuals <= 1.0)), track_index, len(track["indices"]),
        )
        if best is None or value[0] < best[0]:
            best = value
    if best is None:
        return {
            "nearest_baseline_mean_residual": float("inf"),
            "nearest_baseline_max_residual": float("inf"),
            "nearest_baseline_compatible_fraction": 0.0,
            "nearest_baseline_track_index": -1,
            "nearest_baseline_track_hits": 0,
        }
    mean_residual, max_residual, compatible_fraction, track_index, track_hits = best
    return {
        "nearest_baseline_mean_residual": mean_residual,
        "nearest_baseline_max_residual": max_residual,
        "nearest_baseline_compatible_fraction": compatible_fraction,
        "nearest_baseline_track_index": track_index,
        "nearest_baseline_track_hits": track_hits,
    }


def observables(data, candidate, baseline_tracks, edge_probability, noise_probability, uncertainty):
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
    loo_mean, loo_max, loo_parameter_shift = leave_one_out(data, indices)
    return {
        "source": candidate["stage"],
        "hits": candidate["hits"],
        "rms": candidate["rms"],
        "dca": candidate["dca"],
        "consensus_iou": float(candidate.get("iou", 0.0)),
        "mean_edge_probability": float(np.mean(values)) if values.size else 0.0,
        "min_edge_probability": float(np.min(values)) if values.size else 0.0,
        "min_internal_support": min(support.values(), default=0),
        "mean_noise_probability": float(np.mean(noise_probability[indices])),
        "max_noise_probability": float(np.max(noise_probability[indices])),
        "mean_uncertainty": float(np.mean(uncertainty[indices])),
        "loo_mean_heldout_residual": loo_mean,
        "loo_max_heldout_residual": loo_max,
        "loo_max_parameter_shift": loo_parameter_shift,
        **baseline_compatibility(data, indices, baseline_tracks),
    }


def audit_category(dominant, purity, completeness, visible, baseline_matched):
    if dominant not in visible:
        return "not_visible"
    if purity < 0.80:
        return "contaminated"
    if dominant in baseline_matched:
        return "baseline_clone" if completeness >= 0.50 else "baseline_fragment"
    if completeness < 0.50:
        return "unmatched_fragment"
    return "new_match"


def pair_observables(data, left, right, edge_probability):
    merged = fit_candidate(data, left["indices"] + right["indices"], "classic_pair")
    if merged is None:
        return None, {}
    left_nodes, right_nodes = set(left["indices"]), set(right["indices"])
    edges = data.edge_index.cpu().numpy().T
    cross_values = [
        float(probability) for (u, v), probability in zip(edges, edge_probability)
        if (int(u) in left_nodes and int(v) in right_nodes)
        or (int(u) in right_nodes and int(v) in left_nodes)
    ]
    return merged, {
        "left_source": left["stage"], "right_source": right["stage"],
        "left_hits": left["hits"], "right_hits": right["hits"],
        "left_rms": left["rms"], "right_rms": right["rms"],
        "merged_hits": merged["hits"], "merged_rms": merged["rms"],
        "cross_edge_count": len(cross_values),
        "cross_edge_mean_probability": float(np.mean(cross_values)) if cross_values else 0.0,
        "cross_edge_max_probability": float(np.max(cross_values)) if cross_values else 0.0,
        "cross_edges_over_0_2": sum(value >= 0.2 for value in cross_values),
        "cross_edges_over_0_5": sum(value >= 0.5 for value in cross_values),
    }


def load_sharded_dataset(path, limit=None):
    parts_dir = path.with_suffix(path.suffix + ".parts")
    if parts_dir.is_dir():
        graphs = []
        for part in sorted(parts_dir.glob("part_*.pt")):
            graphs.extend(torch.load(part, weights_only=False))
            if limit is not None and len(graphs) >= limit:
                return graphs[:limit]
        return graphs
    return torch.load(path, weights_only=False)


def main():
    parser = argparse.ArgumentParser(description="AI5 经典残差 hybrid 候选级审计")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--audit-dir", type=Path)
    source_group.add_argument("--dataset-file", type=Path)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--node-model-file", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.audit_dir is not None:
        graphs = torch.load(args.audit_dir / "graphs.pt", weights_only=False)
        records = json.loads((args.audit_dir / "audit_records.json").read_text(encoding="utf-8"))
        source = load_axial_hits()
        source_by_event = {int(evt): group for evt, group in source.groupby("EvtID")}
    else:
        graphs = load_sharded_dataset(args.dataset_file, args.rounds)
        records = [None] * len(graphs)
        source_by_event = None
    limit = min(len(graphs), len(records), args.rounds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_model = InteractionNetwork(*model_dimensions("time")).to(device)
    edge_model.load_state_dict(torch.load(args.model_file, map_location=device))
    edge_model.eval()
    node_model = NodeStateClassifier(node_in=13, edge_in=8).to(device)
    node_model.load_state_dict(torch.load(args.node_model_file, map_location=device, weights_only=False))
    node_model.eval()

    rows = []
    pair_rows = []
    for graph_index, (data, record) in enumerate(zip(graphs[:limit], records[:limit])):
        edge_probability = predict(data, edge_model, device, "time")
        baseline_tracks, _ = run_with_short_rerun(
            data, edge_model, device, "time", edge_probability, 0.70, 0.575, 0.55, 0.20, CORE_PROFILE
        )
        context, edge_context = node_context(data, edge_probability, baseline_tracks)
        with torch.no_grad():
            noise_logits, shared_logits = node_model(
                data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device),
                context.to(device), edge_context.to(device),
            )
        noise_probability = torch.sigmoid(noise_logits).cpu().numpy()
        uncertainty = np.maximum(
            binary_uncertainty(noise_logits).cpu().numpy(),
            binary_uncertainty(shared_logits).cpu().numpy(),
        )
        used = {node for track in baseline_tracks for node in track["indices"]}
        residual = [node for node in range(data.num_nodes) if node not in used and noise_probability[node] < 0.95]
        items = residual_items(data, residual)
        ca_pool = ca_candidates(data, items) if len(items) >= 5 else []
        hough_pool = hough_candidates(data, items) if len(items) >= 5 else []
        consensus_pool, solo_pool = consensus_fusion(data, ca_pool, hough_pool, 0.5)
        candidates = hybrid_arbitrate(consensus_pool, solo_pool)

        associations = truth_maps(data)["exclusive_winner"]
        visible_sizes = Counter(key for values in associations for key in values)
        if record is not None:
            source_sizes = Counter()
            for evt in record["selected_event_ids"]:
                group = source_by_event[int(evt)]
                source_sizes.update((int(e), int(t)) for e, t in zip(group.EvtID, group.TrkID))
            eligible = {key for key, size in source_sizes.items() if size >= 5}
            visible = {key for key in eligible if visible_sizes[key] >= 5}
        else:
            visible = {key for key, size in visible_sizes.items() if size >= 5}
        baseline_matched = matched_truths(baseline_tracks, associations, visible_sizes, visible)
        labeled_candidates = []
        for candidate_index, candidate in enumerate(candidates):
            outcome, dominant, purity, completeness = label_candidate(
                candidate, associations, visible_sizes, visible, baseline_matched
            )
            category = audit_category(dominant, purity, completeness, visible, baseline_matched)
            rows.append({
                "graph_index": graph_index, "candidate_index": candidate_index,
                **observables(
                    data, candidate, baseline_tracks, edge_probability, noise_probability, uncertainty
                ),
                "outcome": outcome, "truth_evt": dominant[0], "truth_track": dominant[1],
                "visible_hits": visible_sizes[dominant], "purity": purity, "completeness": completeness,
                "dominant_already_matched": dominant in baseline_matched,
                "audit_category": category,
            })
            labeled_candidates.append((candidate_index, candidate, outcome, dominant, category))
        for (left_index, left, _, left_truth, left_category), (
            right_index, right, _, right_truth, right_category
        ) in itertools.combinations(labeled_candidates, 2):
            merged, pair_values = pair_observables(data, left, right, edge_probability)
            if merged is None:
                continue
            merged_outcome, merged_truth, merged_purity, merged_completeness = label_candidate(
                merged, associations, visible_sizes, visible, baseline_matched
            )
            pair_rows.append({
                "graph_index": graph_index, "left_candidate_index": left_index,
                "right_candidate_index": right_index, **pair_values,
                "same_dominant_truth": left_truth == right_truth,
                "left_category": left_category, "right_category": right_category,
                "merged_outcome": merged_outcome,
                "merged_category": audit_category(
                    merged_truth, merged_purity, merged_completeness, visible, baseline_matched
                ),
                "merged_truth_evt": merged_truth[0], "merged_truth_track": merged_truth[1],
                "merged_visible_hits": visible_sizes[merged_truth],
                "merged_purity": merged_purity, "merged_completeness": merged_completeness,
            })
        if (graph_index + 1) % 50 == 0:
            print(f"已完成 {graph_index + 1}/{limit}", flush=True)

    csv_path = args.output.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    pair_csv_path = args.output.with_name(args.output.stem + "_pairs.csv")
    if pair_rows:
        with pair_csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
            writer.writeheader()
            writer.writerows(pair_rows)
    by_source = defaultdict(Counter)
    for row in rows:
        by_source[row["source"]][row["outcome"]] += 1
    report = {
        "events": limit,
        "candidates": len(rows),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "audit_categories": dict(Counter(row["audit_category"] for row in rows)),
        "by_source": {key: dict(value) for key, value in by_source.items()},
        "candidate_csv": str(csv_path.resolve()),
        "candidate_pairs": len(pair_rows),
        "same_truth_pairs": sum(row["same_dominant_truth"] for row in pair_rows),
        "merged_pair_outcomes": dict(Counter(row["merged_outcome"] for row in pair_rows)),
        "pair_csv": str(pair_csv_path.resolve()),
        "contract": (
            "classic algorithms generate residual candidates; truth is used only for offline labels; "
            "dataset-file mode defines eligibility from visible truth only"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
