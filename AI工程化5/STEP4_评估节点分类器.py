from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
AI4 = ROOT.parent / "AI工程化4"
sys.path.insert(0, str(AI4))

from common import InteractionNetwork, load_axial_hits, model_dimensions  # noqa: E402
from STEP4_两级寻迹 import predict  # noqa: E402
from step4_hit_audit.analyze_physics_algorithm_loss import CORE_PROFILE, truth_maps  # noqa: E402
from step4_short_rerun_experiment.run_experiment import run_with_short_rerun  # noqa: E402
from STEP1_生成节点分类派生数据 import node_context  # noqa: E402
from STEP2_训练节点分类器 import NodeStateClassifier, binary_uncertainty  # noqa: E402


def evaluate_binary(probability, target, threshold):
    prediction = probability >= threshold
    target = target.astype(bool)
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    tn = int((~prediction & ~target).sum())
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def evaluate_tracks(tracks, associations, source_sizes):
    visible_sizes = Counter(key for values in associations for key in values)
    eligible = {key for key, size in source_sizes.items() if size >= 5}
    visible = {key for key in eligible if visible_sizes[key] >= 5}
    matched, fake, clones = set(), 0, 0
    for track in tracks:
        counts = Counter(key for node in track["indices"] for key in associations[node])
        dominant, count = counts.most_common(1)[0] if counts else ((-1, -1), 0)
        purity = count / max(len(track["indices"]), 1)
        completeness = count / max(visible_sizes[dominant], 1)
        if dominant not in visible or purity < 0.80 or completeness < 0.50:
            fake += 1
        elif dominant in matched:
            clones += 1
        else:
            matched.add(dominant)
    visible_bins = Counter("5_6" if visible_sizes[k] <= 6 else "other" for k in visible)
    matched_bins = Counter("5_6" if visible_sizes[k] <= 6 else "other" for k in matched)
    return {
        "source_eligible_tracks": len(eligible),
        "electronics_visible_tracks": len(visible),
        "algorithm_matched_tracks": len(matched),
        "algorithm_lost_tracks": len(visible - matched),
        "fake_tracks": fake,
        "clone_tracks": clones,
        "visible_5_6_tracks": visible_bins["5_6"],
        "matched_5_6_tracks": matched_bins["5_6"],
    }


def add_metrics(total, value):
    for key, item in value.items():
        total[key] += item


def finalize_tracks(total, events):
    result = dict(total)
    result.update({
        "algorithm_efficiency_given_visible": total["algorithm_matched_tracks"] / max(total["electronics_visible_tracks"], 1),
        "end_to_end_efficiency_from_source": total["algorithm_matched_tracks"] / max(total["source_eligible_tracks"], 1),
        "algorithm_efficiency_5_6_visible": total["matched_5_6_tracks"] / max(total["visible_5_6_tracks"], 1),
        "mean_fake_tracks": total["fake_tracks"] / max(events, 1),
        "mean_clone_tracks": total["clone_tracks"] / max(events, 1),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="固定评估 AI5 节点分类器及噪声硬清理")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--node-model-file", type=Path, required=True)
    parser.add_argument("--features", choices=["time", "no-time"], default="time")
    parser.add_argument("--noise-threshold", type=float, default=0.95)
    parser.add_argument("--shared-threshold", type=float, default=0.45)
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

    first = torch.load(args.node_model_file, map_location=device, weights_only=False)
    node_model = NodeStateClassifier(node_in=13, edge_in=8).to(device)
    node_model.load_state_dict(first)
    node_model.eval()

    node_probs = {"noise": [], "shared": [], "uncertainty": []}
    node_targets = {"noise": [], "shared": []}
    baseline_total = Counter()
    hard_noise_total = Counter()
    noise_removed = 0
    shared_candidates = 0

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
        noise = torch.sigmoid(noise_logits).cpu().numpy()
        shared = torch.sigmoid(shared_logits).cpu().numpy()
        uncertainty = torch.maximum(binary_uncertainty(noise_logits), binary_uncertainty(shared_logits)).cpu().numpy()
        node_probs["noise"].extend(noise.tolist())
        node_probs["shared"].extend(shared.tolist())
        node_probs["uncertainty"].extend(uncertainty.tolist())
        node_targets["noise"].extend(data.is_noise.cpu().numpy().astype(bool).tolist())
        truth_counts = (data.truth_ptr[1:] - data.truth_ptr[:-1]).cpu().numpy()
        node_targets["shared"].extend(((truth_counts > 1) & ~data.is_noise.cpu().numpy()).tolist())

        source_sizes = Counter()
        for evt in record["selected_event_ids"]:
            group = source_by_event[int(evt)]
            source_sizes.update((int(e), int(t)) for e, t in zip(group.EvtID, group.TrkID))
        associations = truth_maps(data)["exclusive_winner"]
        add_metrics(baseline_total, evaluate_tracks(baseline_tracks, associations, source_sizes))

        filtered = probabilities.copy()
        edges = data.edge_index.cpu().numpy()
        rejected = noise >= args.noise_threshold
        filtered[rejected[edges[0]] | rejected[edges[1]]] = -np.inf
        hard_tracks, _ = run_with_short_rerun(
            data, edge_model, device, args.features, filtered,
            0.70, 0.575, 0.55, 0.20, CORE_PROFILE,
        )
        add_metrics(hard_noise_total, evaluate_tracks(hard_tracks, associations, source_sizes))
        noise_removed += int(rejected.sum())
        shared_candidates += int((shared >= args.shared_threshold).sum())
        if index % 100 == 0:
            print(f"已完成 {index}/{limit}", flush=True)

    report = {
        "audit_dir": str(args.audit_dir.resolve()),
        "edge_model_file": str(args.model_file.resolve()),
        "node_model_file": str(args.node_model_file.resolve()),
        "events": limit,
        "thresholds": {"noise": args.noise_threshold, "shared": args.shared_threshold},
        "node_metrics": {
            "noise": evaluate_binary(np.asarray(node_probs["noise"]), np.asarray(node_targets["noise"]), args.noise_threshold),
            "shared": evaluate_binary(np.asarray(node_probs["shared"]), np.asarray(node_targets["shared"]), args.shared_threshold),
            "mean_uncertainty": float(np.mean(node_probs["uncertainty"])) if node_probs["uncertainty"] else 0.0,
        },
        "candidate_metrics": {
            "baseline": finalize_tracks(baseline_total, limit),
            "hard_noise_filter": finalize_tracks(hard_noise_total, limit),
        },
        "diagnostics": {"hard_noise_nodes": noise_removed, "shared_candidate_nodes": shared_candidates},
        "contract": "node truth is used only for offline targets; hard filter is an ablation, no shared rerun yet",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["node_metrics"], ensure_ascii=False, indent=2))
    print("报告：", args.output.resolve())


if __name__ == "__main__":
    main()
