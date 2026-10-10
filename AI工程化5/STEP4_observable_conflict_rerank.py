from __future__ import annotations

import argparse
import itertools
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

from common import InteractionNetwork, load_axial_hits, model_dimensions  # noqa: E402
from STEP4_两级寻迹 import (  # noqa: E402
    RELAXED_RMS,
    STRICT_RMS,
    components,
    hierarchy_partition,
    predict,
    score,
    split_component,
    trim_outliers,
)
from step4_hit_audit.analyze_physics_algorithm_loss import CORE_PROFILE, truth_maps  # noqa: E402
from step4_short_rerun_experiment.run_experiment import short_components, rerun_small_graph  # noqa: E402


def collect_candidates(data, probabilities):
    """Build overlapping alternatives from observable probability thresholds."""
    candidates = []
    for threshold in (0.70, 0.65, 0.60, 0.575, 0.55):
        rms_limit = STRICT_RMS if threshold >= 0.60 else RELAXED_RMS
        for component in components(data, probabilities, threshold):
            candidate = trim_outliers(data, component, probabilities, threshold, rms_limit, f"threshold_{threshold:.3f}")
            if candidate is not None:
                candidate["proposal_threshold"] = threshold
                candidates.append(candidate)
    return candidates


def observable_rerank(candidates, overlap_penalty):
    """Greedy conflict resolution using only candidate-local observables."""
    remaining = [candidate for candidate in candidates if candidate["accepted_geometry"]]
    final = []
    used = set()
    conflict_edges = 0
    while remaining:
        scored = []
        for candidate in remaining:
            indices = set(candidate["indices"])
            overlap = sum(bool(indices & set(other["indices"])) for other in remaining if other is not candidate)
            observable = score(candidate) - overlap_penalty * overlap
            scored.append((observable, score(candidate), candidate))
        _, _, chosen = max(scored, key=lambda value: (value[0], value[1], value[2]["hits"]))
        chosen_indices = set(chosen["indices"])
        conflicts = [candidate for candidate in remaining if candidate is not chosen and chosen_indices & set(candidate["indices"])]
        conflict_edges += len(conflicts)
        final.append(chosen)
        used.update(chosen_indices)
        remaining = [candidate for candidate in remaining if not (chosen_indices & set(candidate["indices"]))]
    return final, conflict_edges


def evaluate(data, tracks, associations, source_sizes):
    visible_sizes = Counter(key for values in associations for key in values)
    eligible = {key for key, size in source_sizes.items() if size >= 5}
    visible = {key for key in eligible if visible_sizes[key] >= 5}
    matched = set()
    fake = clones = 0
    bins = Counter()
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
            bins["5_6" if visible_sizes[dominant] <= 6 else "7_10" if visible_sizes[dominant] <= 10 else "other"] += 1
    return {"source": len(eligible), "visible": len(visible), "matched": len(matched), "fake": fake, "clones": clones, "visible_5_6": sum(visible_sizes[k] <= 6 for k in visible), "matched_5_6": bins["5_6"]}


def main():
    parser = argparse.ArgumentParser(description="AI5 纯可观测候选冲突重排实验")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--features", choices=["time", "no-time"], default="time")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--overlap-penalty", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graphs = torch.load(args.audit_dir / "graphs.pt", weights_only=False)
    records = json.loads((args.audit_dir / "audit_records.json").read_text(encoding="utf-8"))
    source = load_axial_hits()
    source_by_event = {int(evt): group for evt, group in source.groupby("EvtID")}
    limit = min(args.rounds, len(graphs), len(records))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionNetwork(*model_dimensions(args.features)).to(device)
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.eval()
    totals = {"baseline": Counter(), "observable_rerank": Counter()}
    conflicts = 0
    for data, record in zip(graphs[:limit], records[:limit]):
        source_sizes = Counter()
        for evt in record["selected_event_ids"]:
            group = source_by_event[int(evt)]
            source_sizes.update((int(e), int(t)) for e, t in zip(group.EvtID, group.TrkID))
        probabilities = predict(data, model, device, args.features)
        candidates = collect_candidates(data, probabilities)
        baseline, _ = observable_rerank(candidates, 0.0)
        reranked, conflict_count = observable_rerank(candidates, args.overlap_penalty)
        conflicts += conflict_count
        associations = truth_maps(data)["exclusive_winner"]
        for name, tracks in (("baseline", baseline), ("observable_rerank", reranked)):
            result = evaluate(data, tracks, associations, source_sizes)
            for key, value in result.items():
                totals[name][key] += value
        totals["baseline"]["events"] += 1
        totals["observable_rerank"]["events"] += 1
    report = {"graphs": limit, "overlap_penalty": args.overlap_penalty, "observable_conflict_edges": conflicts}
    for name, total in totals.items():
        events = max(total["events"], 1)
        report[name] = {
            "physics_track_survival": total["visible"] / max(total["source"], 1),
            "algorithm_efficiency_given_visible": total["matched"] / max(total["visible"], 1),
            "end_to_end_efficiency_from_source": total["matched"] / max(total["source"], 1),
            "algorithm_efficiency_5_6_visible": total["matched_5_6"] / max(total["visible_5_6"], 1),
            "mean_fake_tracks": total["fake"] / events,
            "mean_clone_tracks": total["clones"] / events,
        }
    report["contract"] = "rerank uses candidate observables only; truth is read only for offline evaluation"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
