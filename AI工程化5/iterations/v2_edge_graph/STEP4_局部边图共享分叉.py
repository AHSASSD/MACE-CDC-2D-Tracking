from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

HERE = Path(__file__).resolve().parent
AI5 = HERE.parents[1]
AI4 = AI5.parent / "AI工程化4"
sys.path.insert(0, str(AI5))
sys.path.insert(0, str(AI4))

from STEP4_经典残差寻迹 import (  # noqa: E402
    arbitrate,
    ca_candidates,
    consensus_fusion,
    fit_candidate,
    hough_candidates,
    hybrid_arbitrate,
    merge_candidate_fragments,
    residual_items,
)
from STEP4_两级寻迹 import predict  # noqa: E402
from STEP1_生成节点分类派生数据 import node_context  # noqa: E402
from STEP2_训练节点分类器 import NodeStateClassifier  # noqa: E402
from STEP4_评估节点分类器 import add_metrics, evaluate_tracks, finalize_tracks  # noqa: E402
from common import InteractionNetwork, load_axial_hits, model_dimensions  # noqa: E402
from step4_hit_audit.analyze_physics_algorithm_loss import CORE_PROFILE, truth_maps  # noqa: E402
from step4_short_rerun_experiment.run_experiment import run_with_short_rerun  # noqa: E402
from triplet_common import pair_features  # noqa: E402


def local_edge_graph_candidates(data, ca_pool, hough_pool, used, edge_probability,
                                shared_probability, shared_threshold, edge_threshold,
                                max_pair_hits=3):
    """Join two residual fragments through one shared hit.

    The shared hit is a reusable edge-graph articulation; all other hits must
    remain outside first-pass tracks and are fitted as exclusive anchors.
    """
    edges = data.edge_index.cpu().numpy().T
    incident = {}
    for (u, v), score in zip(edges, edge_probability):
        if score < edge_threshold:
            continue
        u, v = int(u), int(v)
        incident.setdefault(u, []).append(v)
        incident.setdefault(v, []).append(u)
    pools = list(ca_pool) + list(hough_pool)
    proposals = []
    for left_index, left in enumerate(pools):
        left_nodes = set(left["indices"])
        for right in pools[left_index + 1:]:
            right_nodes = set(right["indices"])
            if left_nodes & right_nodes:
                continue
            shared = set()
            for node in incident:
                if shared_probability[node] < shared_threshold or node not in used:
                    continue
                left_support = sum(neighbor in left_nodes for neighbor in incident[node])
                right_support = sum(neighbor in right_nodes for neighbor in incident[node])
                if left_support and right_support:
                    shared.add(node)
            if not shared:
                continue
            shared = sorted(shared, key=lambda node: -shared_probability[node])[:max_pair_hits]
            indices = sorted(left_nodes | right_nodes | set(shared))
            fitted = fit_candidate(data, indices, "edge_graph_shared_split")
            if fitted is None or not fitted["accepted_geometry"]:
                continue
            fitted["shared_articulations"] = shared
            fitted["triplet_support"] = len(shared)
            fitted["source_pair_hits"] = [left["hits"], right["hits"]]
            proposals.append(fitted)
    return proposals


def local_in_shared_seed_candidates(data, used, residual, edge_model, device,
                                    shared_probability, shared_threshold=0.45,
                                    seed_edge_threshold=0.35, local_edge_threshold=0.50,
                                    max_hops=2, triplet_threshold=0.30,
                                    min_branch_exclusive=2, min_local_edge=0.55,
                                    local_rms_limit=0.90, triplet_model=None,
                                    classifier_threshold=0.80, min_shared_split=0.35,
                                    max_fake_probability=0.20, split_margin=0.05):
    """Re-run the frozen Interaction Network on subgraphs around shared seeds."""
    edges = data.edge_index.cpu().numpy().T
    scores = np.asarray(predict(data, edge_model, device, "time"), dtype=float)
    adjacency = {}
    edge_scores = {}
    for (u, v), score in zip(edges, scores):
        if score < seed_edge_threshold:
            continue
        u, v = int(u), int(v)
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
        edge_scores[(u, v)] = max(edge_scores.get((u, v), 0.0), float(score))
        edge_scores[(v, u)] = max(edge_scores.get((v, u), 0.0), float(score))
    seeds = [node for node in used if shared_probability[node] >= shared_threshold]
    proposals = []
    residual_set = set(residual)
    for seed in seeds:
        local = {int(seed)}
        frontier = {int(seed)}
        for _ in range(max_hops):
            frontier = {neighbor for node in frontier for neighbor in adjacency.get(node, set())}
            local.update(frontier)
        local &= (residual_set | {int(seed)})
        if len(local) < 5 or len(local & residual_set) < 4:
            continue
        ordered = sorted(local)
        remap = {node: index for index, node in enumerate(ordered)}
        mask = np.array([(int(u) in remap and int(v) in remap) for u, v in edges])
        if not mask.any():
            continue
        local_data = Data(
            x=data.x[ordered], edge_index=torch.tensor(
                [[remap[int(u)] for u, v in edges[mask]],
                 [remap[int(v)] for u, v in edges[mask]]], dtype=torch.long),
            edge_attr=data.edge_attr[torch.from_numpy(mask)],
        )
        local_scores = np.asarray(predict(local_data, edge_model, device, "time"), dtype=float)
        local_edges = local_data.edge_index.cpu().numpy().T
        graph = {node: set() for node in ordered}
        for (u, v), score in zip(local_edges, local_scores):
            if score >= local_edge_threshold:
                graph[ordered[int(u)]].add(ordered[int(v)])
        # Triplet-like articulation test: retain only a geometrically
        # compatible pair of branches through the shared seed.  Opposite
        # vectors are the expected local signature of a through-going track.
        seed_neighbors = [node for node in graph[seed] if node in residual_set]
        best_pair = None
        sx, sy = float(data.wire_x[seed]), float(data.wire_y[seed])
        for left_index, left in enumerate(seed_neighbors):
            lv = np.array([float(data.wire_x[left]) - sx, float(data.wire_y[left]) - sy])
            for right in seed_neighbors[left_index + 1:]:
                rv = np.array([float(data.wire_x[right]) - sx, float(data.wire_y[right]) - sy])
                denom = float(np.linalg.norm(lv) * np.linalg.norm(rv))
                if denom <= 1e-9:
                    continue
                opposite = max(0.0, min(1.0, (1.0 - float(np.dot(lv, rv) / denom)) / 2.0))
                pair_score = opposite * min(
                    edge_scores.get((seed, left), 0.0), edge_scores.get((seed, right), 0.0)
                ) * float(shared_probability[seed])
                if triplet_model is not None:
                    features = pair_features(
                        data, seed, left, right,
                        edge_scores.get((seed, left), 0.0), edge_scores.get((seed, right), 0.0),
                        shared_probability[seed],
                    ).reshape(1, -1)
                    class_prob = triplet_model["model"].predict_proba(features)[0]
                    accepted_prob = float(class_prob[0] + class_prob[1])
                    if (
                        accepted_prob < classifier_threshold
                        or float(class_prob[1]) < min_shared_split
                        or float(class_prob[2]) > max_fake_probability
                        or float(class_prob[1] - class_prob[0]) < split_margin
                    ):
                        continue
                    pair_score *= accepted_prob
                if best_pair is None or pair_score > best_pair[0]:
                    best_pair = (pair_score, left, right)
        if best_pair is None or best_pair[0] < triplet_threshold:
            continue
        _, left_root, right_root = best_pair
        component = {seed}
        for root in (left_root, right_root):
            branch = {root}
            frontier = [root]
            for _ in range(max_hops - 1):
                next_frontier = []
                for node in frontier:
                    for neighbor in graph[node]:
                        if neighbor == seed or neighbor not in residual_set or neighbor in branch:
                            continue
                        branch.add(neighbor)
                        next_frontier.append(neighbor)
                frontier = next_frontier
            component.update(branch)
            if len(branch & residual_set) < min_branch_exclusive:
                component = set()
                break
        if not component:
            continue
        exclusive = component & residual_set
        if len(exclusive) < 4:
            continue
        fitted = fit_candidate(data, sorted(component), "edge_graph_local_in")
        if fitted is None or not fitted["accepted_geometry"]:
            continue
        local_internal = [
            float(score) for (u, v), score in zip(local_edges, local_scores)
            if int(u) in {remap[node] for node in component}
            and int(v) in {remap[node] for node in component}
        ]
        if not local_internal or float(np.mean(local_internal)) < min_local_edge or fitted["rms"] > local_rms_limit:
            continue
        fitted["shared_articulations"] = [seed]
        fitted["exclusive_hits"] = len(exclusive)
        fitted["local_nodes"] = len(local)
        fitted["triplet_score"] = float(best_pair[0])
        fitted["local_mean_edge_probability"] = float(np.mean(local_internal))
        proposals.append(fitted)
    # One proposal per shared articulation prevents clone bursts.
    best_by_seed = {}
    for candidate in proposals:
        seed = candidate["shared_articulations"][0]
        if seed not in best_by_seed or (
            candidate["triplet_score"], candidate["local_mean_edge_probability"], -candidate["rms"]
        ) > (
            best_by_seed[seed]["triplet_score"], best_by_seed[seed]["local_mean_edge_probability"],
            -best_by_seed[seed]["rms"],
        ):
            best_by_seed[seed] = candidate
    return list(best_by_seed.values())


def arbitrate_shared(candidates):
    """Arbitrate candidates while allowing reuse of declared shared nodes."""
    candidates = [candidate for candidate in candidates if candidate["accepted_geometry"]]
    candidates.sort(key=lambda value: (-value["hits"], value["rms"]))
    accepted, used_exclusive = [], set()
    for candidate in candidates:
        shared = set(candidate.get("shared_articulations", []))
        exclusive = set(candidate["indices"]) - shared
        if exclusive & used_exclusive:
            continue
        accepted.append(candidate)
        used_exclusive.update(exclusive)
    return accepted


def main():
    parser = argparse.ArgumentParser(description="AI5 v2 local edge-graph shared split experiment")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--node-model-file", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--shared-threshold", type=float, default=0.45)
    parser.add_argument("--edge-threshold", type=float, default=0.55)
    parser.add_argument("--triplet-threshold", type=float, default=0.30)
    parser.add_argument("--min-branch-exclusive", type=int, default=2)
    parser.add_argument("--min-local-edge", type=float, default=0.55)
    parser.add_argument("--local-rms-limit", type=float, default=0.90)
    parser.add_argument("--triplet-model-file", type=Path)
    parser.add_argument("--min-shared-split", type=float, default=0.35)
    parser.add_argument("--max-fake-probability", type=float, default=0.20)
    parser.add_argument("--split-margin", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
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
    triplet_model = None
    classifier_threshold = 0.80
    if args.triplet_model_file:
        import joblib
        # The first focal-loss training run was launched as a script, so
        # joblib recorded these classes under __main__. Register compatible
        # symbols before unpickling that artifact.
        from STEP5_训练Triplet分类器 import TripletFocalClassifier, _TripletNet
        import __main__
        __main__.TripletFocalClassifier = TripletFocalClassifier
        __main__._TripletNet = _TripletNet
        triplet_model = joblib.load(args.triplet_model_file)
        classifier_threshold = float(triplet_model.get("accept_threshold", classifier_threshold))
    totals = {name: Counter() for name in ("baseline", "hybrid_merged", "edge_graph_shared_split", "edge_graph_local_in")}
    diagnostics = Counter()
    for index, (data, record) in enumerate(zip(graphs[:limit], records[:limit]), 1):
        probabilities = predict(data, edge_model, device, "time")
        baseline, _ = run_with_short_rerun(
            data, edge_model, device, "time", probabilities, 0.70, 0.575, 0.55, 0.20, CORE_PROFILE
        )
        context, edge_context = node_context(data, probabilities, baseline)
        with torch.no_grad():
            _, shared_logits = node_model(
                data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device),
                context.to(device), edge_context.to(device),
            )
        shared_probability = torch.sigmoid(shared_logits).cpu().numpy()
        used = {node for track in baseline for node in track["indices"]}
        residual = [node for node in range(data.num_nodes) if node not in used]
        items = residual_items(data, residual)
        if len(items) < 5:
            hybrid = []
            ca_pool = hough_pool = []
        else:
            ca_pool = ca_candidates(data, items)
            hough_pool = hough_candidates(data, items)
            consensus, solo = consensus_fusion(data, ca_pool, hough_pool, args.iou_threshold)
            hybrid = hybrid_arbitrate(consensus, solo)
        hybrid_merged = merge_candidate_fragments(data, hybrid, probabilities)
        split_pool = local_edge_graph_candidates(
            data, ca_pool, hough_pool, used, probabilities, shared_probability,
            args.shared_threshold, args.edge_threshold,
        )
        split = arbitrate(split_pool)
        local_in_pool = local_in_shared_seed_candidates(
            data, used, residual, edge_model, device, shared_probability,
            triplet_threshold=args.triplet_threshold,
            min_branch_exclusive=args.min_branch_exclusive,
            min_local_edge=args.min_local_edge,
            local_rms_limit=args.local_rms_limit,
            triplet_model=triplet_model, classifier_threshold=classifier_threshold,
            min_shared_split=args.min_shared_split,
            max_fake_probability=args.max_fake_probability,
            split_margin=args.split_margin,
        )
        local_in = arbitrate_shared(local_in_pool)
        variants = {
            "baseline": baseline,
            "hybrid_merged": baseline + hybrid_merged,
            "edge_graph_shared_split": baseline + split,
            "edge_graph_local_in": baseline + local_in,
        }
        diagnostics.update({
            "residual_nodes": len(residual), "shared_split_candidates": len(split_pool),
            "shared_split_accepted": len(split),
            "local_in_candidates": len(local_in_pool), "local_in_accepted": len(local_in),
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
    keys = (
        "algorithm_matched_tracks", "algorithm_efficiency_given_visible",
        "end_to_end_efficiency_from_source", "algorithm_efficiency_5_6_visible",
        "mean_fake_tracks", "mean_clone_tracks",
    )
    report = {
        "events": limit,
        "parameters": {
            "shared_threshold": args.shared_threshold, "edge_threshold": args.edge_threshold,
            "triplet_threshold": args.triplet_threshold,
            "min_branch_exclusive": args.min_branch_exclusive,
            "min_local_edge": args.min_local_edge,
            "local_rms_limit": args.local_rms_limit,
            "triplet_model_file": str(args.triplet_model_file) if args.triplet_model_file else None,
            "classifier_threshold": classifier_threshold,
            "min_shared_split": args.min_shared_split,
            "max_fake_probability": args.max_fake_probability,
            "split_margin": args.split_margin,
        },
        "metrics": metrics,
        "delta": {name: {key: metrics[name][key] - baseline[key] for key in keys}
                  for name in metrics if name != "baseline"},
        "diagnostics": dict(diagnostics),
        "contract": "local edge graph uses observable edge/node scores; truth is offline only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["delta"], ensure_ascii=False, indent=2))
    print("报告：", args.output.resolve())


if __name__ == "__main__":
    main()
