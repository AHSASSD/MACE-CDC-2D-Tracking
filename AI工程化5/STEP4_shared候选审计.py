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

from common import InteractionNetwork, load_axial_hits, model_dimensions  # noqa: E402
from STEP4_两级寻迹 import predict  # noqa: E402
from step4_hit_audit.analyze_physics_algorithm_loss import CORE_PROFILE, truth_maps  # noqa: E402
from step4_short_rerun_experiment.run_experiment import run_with_short_rerun  # noqa: E402
from STEP1_生成节点分类派生数据 import node_context  # noqa: E402
from STEP2_训练节点分类器 import NodeStateClassifier, binary_uncertainty  # noqa: E402
from STEP4_shared局部重连 import local_reconnect  # noqa: E402


FEATURES = [
    "anchor_hits", "borrowed_shared_hits", "rms", "mean_shared_residual",
    "mean_edge_probability", "min_edge_probability", "mean_shared_probability",
    "min_shared_probability", "mean_uncertainty", "mean_owner_residual_on_anchors",
]

RULE_ATOMS = {
    "no_borrow": [
        ("anchor_hits", ">=", 5.0), ("anchor_hits", ">=", 6.0),
        ("rms", "<=", 0.2), ("rms", "<=", 0.4), ("rms", "<=", 0.6), ("rms", "<=", 0.8),
        ("min_edge_probability", ">=", 0.3), ("min_edge_probability", ">=", 0.4), ("min_edge_probability", ">=", 0.5),
        ("mean_edge_probability", ">=", 0.55), ("mean_edge_probability", ">=", 0.60),
        ("mean_edge_probability", ">=", 0.65), ("mean_edge_probability", ">=", 0.70),
        ("mean_uncertainty", "<=", 0.1), ("mean_uncertainty", "<=", 0.2), ("mean_uncertainty", "<=", 0.3),
    ],
    "borrowed": [
        ("anchor_hits", ">=", 4.0), ("anchor_hits", ">=", 5.0),
        ("rms", "<=", 0.2), ("rms", "<=", 0.4), ("rms", "<=", 0.6),
        ("mean_shared_residual", "<=", 0.2), ("mean_shared_residual", "<=", 0.5),
        ("mean_shared_residual", "<=", 0.8),
        ("min_shared_probability", ">=", 0.8), ("min_shared_probability", ">=", 0.9),
        ("min_shared_probability", ">=", 0.95),
        ("mean_uncertainty", "<=", 0.1), ("mean_uncertainty", "<=", 0.2), ("mean_uncertainty", "<=", 0.3),
        ("mean_owner_residual_on_anchors", ">=", 0.5),
        ("mean_owner_residual_on_anchors", ">=", 1.0),
        ("mean_owner_residual_on_anchors", ">=", 2.0),
    ],
}


def matched_truths(tracks, associations, visible_sizes, visible):
    matched = set()
    for track in tracks:
        counts = Counter(key for node in track["indices"] for key in associations[node])
        dominant, count = counts.most_common(1)[0] if counts else ((-1, -1), 0)
        purity = count / max(len(track["indices"]), 1)
        completeness = count / max(visible_sizes[dominant], 1)
        if dominant in visible and purity >= 0.80 and completeness >= 0.50:
            matched.add(dominant)
    return matched


def label_candidate(candidate, associations, visible_sizes, visible, baseline_matched):
    counts = Counter(key for node in candidate["indices"] for key in associations[node])
    dominant, count = counts.most_common(1)[0] if counts else ((-1, -1), 0)
    purity = count / max(len(candidate["indices"]), 1)
    completeness = count / max(visible_sizes[dominant], 1)
    if dominant not in visible or purity < 0.80 or completeness < 0.50:
        outcome = "fake"
    elif dominant in baseline_matched:
        outcome = "clone"
    else:
        outcome = "new_match"
    return outcome, dominant, purity, completeness


def branch(row):
    return "borrowed" if row["borrowed_shared_hits"] > 0 else "no_borrow"


def passes_condition(row, condition):
    feature, operator, threshold = condition
    value = float(row[feature])
    return value >= threshold if operator == ">=" else value <= threshold


def passes_rule(row, rule):
    return all(passes_condition(row, condition) for condition in rule)


def summarize(rows):
    outcomes = Counter(row["outcome"] for row in rows)
    return {
        "selected": len(rows),
        "new_match": outcomes["new_match"],
        "new_match_5_6": sum(row["outcome"] == "new_match" and row["visible_hits"] <= 6 for row in rows),
        "fake": outcomes["fake"],
        "clone": outcomes["clone"],
        "utility": outcomes["new_match"] - outcomes["fake"] - outcomes["clone"],
    }


def candidate_rules(name):
    atoms = RULE_ATOMS[name]
    rules = [(atom,) for atom in atoms]
    rules.extend(
        pair for pair in itertools.combinations(atoms, 2)
        if pair[0][0] != pair[1][0]
    )
    return rules


def select_rule(development_rows, name):
    rows = [row for row in development_rows if branch(row) == name]
    evaluated = []
    for rule in candidate_rules(name):
        stats = summarize([row for row in rows if passes_rule(row, rule)])
        evaluated.append({"conditions": [list(value) for value in rule], **stats})
    zero_bad = [value for value in evaluated if value["new_match"] >= 2 and value["fake"] == 0 and value["clone"] == 0]
    pool = zero_bad or evaluated
    selected = max(
        pool,
        key=lambda value: (
            value["new_match"], value["utility"], -value["fake"] - value["clone"],
            -len(value["conditions"]), value["selected"],
        ),
    )
    selected["selection_policy"] = "prefer >=2 new matches with zero fake/clone; otherwise maximize development recovery"
    return selected


def apply_selected_rule(rows, name, selected):
    rule = [tuple(value) for value in selected["conditions"]]
    return [row for row in rows if branch(row) == name and passes_rule(row, rule)]


def main():
    parser = argparse.ArgumentParser(description="AI5 shared 局部重连候选级离线审计")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--node-model-file", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--development-events", type=int, default=1500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graphs = torch.load(args.audit_dir / "graphs.pt", weights_only=False)
    records = json.loads((args.audit_dir / "audit_records.json").read_text(encoding="utf-8"))
    limit = min(len(graphs), len(records), args.rounds)
    if not 0 < args.development_events < limit:
        raise ValueError("development-events 必须位于 0 和 rounds 之间")
    source = load_axial_hits()
    source_by_event = {int(evt): group for evt, group in source.groupby("EvtID")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_model = InteractionNetwork(*model_dimensions("time")).to(device)
    edge_model.load_state_dict(torch.load(args.model_file, map_location=device))
    edge_model.eval()
    node_model = NodeStateClassifier(node_in=13, edge_in=8).to(device)
    node_model.load_state_dict(torch.load(args.node_model_file, map_location=device, weights_only=False))
    node_model.eval()

    rows = []
    for graph_index, (data, record) in enumerate(zip(graphs[:limit], records[:limit])):
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
        shared = torch.sigmoid(shared_logits).cpu().numpy()
        uncertainty = np.maximum(binary_uncertainty(noise_logits).cpu().numpy(), binary_uncertainty(shared_logits).cpu().numpy())
        _, _, candidates = local_reconnect(
            data, probabilities, baseline_tracks, noise, shared, uncertainty,
            0.95, 0.45, 0.55, 0.55, 1.0, 1.0,
        )
        associations = truth_maps(data)["exclusive_winner"]
        visible_sizes = Counter(key for values in associations for key in values)
        source_sizes = Counter()
        for evt in record["selected_event_ids"]:
            group = source_by_event[int(evt)]
            source_sizes.update((int(e), int(t)) for e, t in zip(group.EvtID, group.TrkID))
        eligible = {key for key, size in source_sizes.items() if size >= 5}
        visible = {key for key in eligible if visible_sizes[key] >= 5}
        baseline_matched = matched_truths(baseline_tracks, associations, visible_sizes, visible)
        for candidate_index, candidate in enumerate(candidates):
            outcome, dominant, purity, completeness = label_candidate(candidate, associations, visible_sizes, visible, baseline_matched)
            rows.append({
                "graph_index": graph_index, "candidate_index": candidate_index,
                **{key: candidate[key] for key in FEATURES},
                "outcome": outcome, "truth_evt": dominant[0], "truth_track": dominant[1],
                "visible_hits": visible_sizes[dominant],
                "purity": purity, "completeness": completeness,
            })
        if (graph_index + 1) % 100 == 0:
            print(f"已完成 {graph_index + 1}/{limit}", flush=True)

    development = [row for row in rows if row["graph_index"] < args.development_events]
    locked = [row for row in rows if row["graph_index"] >= args.development_events]
    selections = {name: select_rule(development, name) for name in ("no_borrow", "borrowed")}
    locked_selected = {
        name: apply_selected_rule(locked, name, selections[name])
        for name in ("no_borrow", "borrowed")
    }
    combined_locked = locked_selected["no_borrow"] + locked_selected["borrowed"]
    locked_summary = {name: summarize(value) for name, value in locked_selected.items()}
    locked_summary["combined"] = summarize(combined_locked)
    locked_summary["combined"]["gate_passed"] = (
        locked_summary["combined"]["new_match"] > 0
        and locked_summary["combined"]["new_match_5_6"] > 0
        and locked_summary["combined"]["fake"] == 0
        and locked_summary["combined"]["clone"] == 0
    )
    csv_path = args.output.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "events": limit,
        "candidates": len(rows),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "split": {"development_events": args.development_events, "locked_events": limit - args.development_events},
        "development": {
            "candidates": len(development),
            "by_branch": {
                name: summarize([row for row in development if branch(row) == name])
                for name in ("no_borrow", "borrowed")
            },
            "selected_rules": selections,
        },
        "locked": {
            "candidates": len(locked),
            "unfiltered_by_branch": {
                name: summarize([row for row in locked if branch(row) == name])
                for name in ("no_borrow", "borrowed")
            },
            "selected": locked_summary,
        },
        "candidate_csv": str(csv_path.resolve()),
        "contract": "rules are selected on development graphs only and evaluated once on locked graphs; truth labels are offline only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
