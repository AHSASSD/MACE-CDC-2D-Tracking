from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

AI4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI4))

from common import InteractionNetwork, load_axial_hits, model_dimensions  # noqa: E402
from STEP4_两级寻迹 import predict, run_two_pass  # noqa: E402
from step4_short_rerun_experiment.run_experiment import run_with_short_rerun  # noqa: E402


CORE_PROFILE = {
    "skip_parent_rms": 0.20,
    "max_core_rms": 0.80,
    "min_rms_gain": 0.10,
    "min_probability_gain": 0.05,
    "min_coverage": 0.40,
}


def truth_maps(data):
    inclusive = []
    exclusive = []
    for node in range(data.num_nodes):
        start, end = int(data.truth_ptr[node]), int(data.truth_ptr[node + 1])
        values = [tuple(map(int, pair)) for pair in data.truth_pairs[start:end].tolist()]
        inclusive.append(set(values))
        exclusive.append(set(values[:1]))
    return {"exclusive_winner": exclusive, "shared_credit": inclusive}


def evaluate_tracks(tracks, associations, source_sizes):
    visible_sizes = Counter(key for values in associations for key in values)
    source_eligible = {key for key, size in source_sizes.items() if size >= 5}
    visible_eligible = {key for key in source_eligible if visible_sizes[key] >= 5}
    matched = set()
    fake = clones = 0
    for track in tracks:
        counts = Counter(key for node in track["indices"] for key in associations[node])
        dominant, count = counts.most_common(1)[0] if counts else ((-1, -1), 0)
        purity = count / max(len(track["indices"]), 1)
        completeness = count / max(visible_sizes[dominant], 1)
        if dominant not in visible_eligible or purity < 0.80 or completeness < 0.50:
            fake += 1
        elif dominant in matched:
            clones += 1
        else:
            matched.add(dominant)
    physically_lost = source_eligible - visible_eligible
    algorithm_lost = visible_eligible - matched
    return {
        "source_eligible_tracks": len(source_eligible),
        "electronics_visible_tracks": len(visible_eligible),
        "physically_unreconstructable_tracks": len(physically_lost),
        "algorithm_matched_tracks": len(matched),
        "algorithm_lost_tracks": len(algorithm_lost),
        "fake_tracks": fake,
        "clone_tracks": clones,
        "source_hits": sum(source_sizes[key] for key in source_eligible),
        "visible_hits": sum(visible_sizes[key] for key in source_eligible),
        "source_hit_bins": Counter(hit_bin(source_sizes[key]) for key in source_eligible),
        "physical_loss_bins": Counter(hit_bin(source_sizes[key]) for key in physically_lost),
        "algorithm_loss_bins": Counter(hit_bin(visible_sizes[key]) for key in algorithm_lost),
    }


def hit_bin(size):
    return "5_6" if size <= 6 else "7_10" if size <= 10 else "11_20" if size <= 20 else "21_plus"


def add(total, result):
    for key, value in result.items():
        if isinstance(value, Counter):
            total.setdefault(key, Counter()).update(value)
        else:
            total[key] += value


def finalize(total, events):
    source = total["source_eligible_tracks"]
    visible = total["electronics_visible_tracks"]
    matched = total["algorithm_matched_tracks"]
    result = {key: (dict(value) if isinstance(value, Counter) else value) for key, value in total.items()}
    result.update({
        "physics_track_survival": visible / max(source, 1),
        "algorithm_efficiency_given_visible": matched / max(visible, 1),
        "end_to_end_efficiency_from_source": matched / max(source, 1),
        "physical_hit_retention": total["visible_hits"] / max(total["source_hits"], 1),
        "mean_fake_tracks": total["fake_tracks"] / max(events, 1),
        "mean_clone_tracks": total["clone_tracks"] / max(events, 1),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="严格拆分电子学物理损失与 STEP4 算法损失")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--features", choices=["time", "no-time"], default="time")
    parser.add_argument("--pipeline", choices=["formal", "strongest"], default="strongest")
    parser.add_argument("--run-tag", default="clean_physics_v2")
    parser.add_argument("--strict-threshold", type=float, default=0.70)
    parser.add_argument("--relaxed-threshold", type=float, default=0.575)
    parser.add_argument("--rerun-threshold", type=float, default=0.55)
    parser.add_argument("--rerun-rms", type=float, default=0.20)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    graphs = torch.load(args.audit_dir / "graphs.pt", weights_only=False)
    records = json.loads((args.audit_dir / "audit_records.json").read_text(encoding="utf-8"))
    limit = min(len(graphs), len(records), args.rounds or len(graphs))
    source = load_axial_hits()
    source_by_event = {int(evt): group for evt, group in source.groupby("EvtID")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stem = f"{args.features.replace('-', '_')}_{args.run_tag}"
    model = InteractionNetwork(*model_dimensions(args.features)).to(device)
    model.load_state_dict(torch.load(AI4 / f"best_interaction_network_{stem}.pth", map_location=device))
    model.eval()

    totals = {name: Counter() for name in ("exclusive_winner", "shared_credit")}
    reason_counts = Counter()
    started = time.perf_counter()
    for index, (data, record) in enumerate(zip(graphs[:limit], records[:limit]), 1):
        source_sizes = Counter()
        for evt in record["selected_event_ids"]:
            group = source_by_event[int(evt)]
            source_sizes.update((int(e), int(t)) for e, t in zip(group.EvtID, group.TrkID))
        reason_counts.update(record.get("real_candidate_counts_by_reason", {}))
        probabilities = predict(data, model, device, args.features)
        if args.pipeline == "strongest":
            tracks, _ = run_with_short_rerun(
                data, model, device, args.features, probabilities,
                args.strict_threshold, args.relaxed_threshold,
                args.rerun_threshold, args.rerun_rms, CORE_PROFILE,
            )
        else:
            tracks, _ = run_two_pass(
                data, model, device, args.features,
                args.strict_threshold, args.relaxed_threshold, probabilities,
            )
        for convention, associations in truth_maps(data).items():
            add(totals[convention], evaluate_tracks(tracks, associations, source_sizes))
        if index % 100 == 0:
            print(f"已完成 {index}/{limit}", flush=True)

    output = args.output or args.audit_dir / f"physics_algorithm_loss_{args.features}_{args.pipeline}.json"
    report = {
        "audit_dir": str(args.audit_dir.resolve()),
        "events": limit,
        "features": args.features,
        "pipeline": args.pipeline,
        "thresholds": {"strict": args.strict_threshold, "relaxed": args.relaxed_threshold,
                       "rerun": args.rerun_threshold, "rerun_rms": args.rerun_rms},
        "physical_hit_rejection_reasons": dict(reason_counts),
        "conventions": {name: finalize(total, limit) for name, total in totals.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "contract_note": {
            "exclusive_winner": "Only the earliest accepted rising edge owns a cell hit.",
            "shared_credit": "A surviving cell observation may support all truth tracks suppressed behind it.",
        },
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["conventions"], ensure_ascii=False, indent=2))
    print("报告：", output.resolve())


if __name__ == "__main__":
    main()
