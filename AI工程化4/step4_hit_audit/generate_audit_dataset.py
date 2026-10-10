from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    DEFAULT_NOISE_RATE_HZ,
    DEFAULT_SIGMA_T_NS,
    DEAD_TIME_NS,
    EVENT_WINDOW_NS,
    DRIFT_DISTANCE_SIGMA_MM,
    build_graph,
    load_axial_hits,
    load_cell_map,
)


def sample_target(rng, mean_tracks, minimum, maximum):
    while True:
        value = int(rng.poisson(mean_tracks))
        if minimum <= value <= maximum:
            return value


def sample_events(event_pool, tracks_per_event, target, rng):
    selected = []
    actual = 0
    for evt in rng.permutation(event_pool):
        selected.append(int(evt))
        actual += int(tracks_per_event[int(evt)])
        if actual >= target:
            break
    return selected, actual


def make_candidates(source, cell_map, noise_rate_hz, sigma_t_ns, rng):
    real = source.copy().reset_index(drop=True)
    real["candidate_id"] = np.arange(len(real), dtype=np.int64)
    real["source_kind"] = "real"
    real["truth_tracks"] = [[(int(e), int(t))] for e, t in zip(real.EvtID, real.TrkID)]
    real["t_window"] = real.t.to_numpy(float) - real.t0.to_numpy(float)
    real["t_obs"] = real.t_window.to_numpy(float) + rng.normal(0.0, sigma_t_ns, len(real))
    real["d"] = np.abs(real.d.to_numpy(float) + rng.normal(0.0, DRIFT_DISTANCE_SIGMA_MM, len(real)))
    real["is_noise"] = False
    real["shared_hit"] = False
    real["noise_multiplicity"] = 0

    mean_noise = noise_rate_hz * EVENT_WINDOW_NS * 1e-9
    multiplicity = rng.poisson(mean_noise, len(cell_map))
    repeated = np.repeat(np.arange(len(cell_map)), multiplicity)
    noise = cell_map.iloc[repeated][["CellID", "wire_x", "wire_y", "wire_r"]].copy().reset_index(drop=True)
    if len(noise):
        start = len(real)
        noise["candidate_id"] = np.arange(start, start + len(noise), dtype=np.int64)
        noise["source_kind"] = "noise"
        noise["EvtID"] = -1
        noise["TrkID"] = -1
        noise["d"] = rng.uniform(0.0, 6.0, len(noise))
        noise["t_window"] = rng.uniform(0.0, EVENT_WINDOW_NS, len(noise))
        noise["t_obs"] = noise.t_window.to_numpy(float) + rng.normal(0.0, sigma_t_ns, len(noise))
        noise["t0"] = np.nan
        noise["tHit"] = np.nan
        noise["truth_tracks"] = [[] for _ in range(len(noise))]
        noise["is_noise"] = True
        noise["shared_hit"] = False
        noise["noise_multiplicity"] = 1
        candidates = pd.concat([real, noise], ignore_index=True, sort=False)
    else:
        candidates = real
    candidates["in_window"] = candidates.t_obs.between(0.0, EVENT_WINDOW_NS, inclusive="left")
    candidates["rejection_reason"] = np.where(candidates.in_window, "pending", "outside_window")
    return candidates


def apply_audit_dead_time(candidates, dead_time_ns):
    frame = candidates.sort_values(["CellID", "t_obs", "candidate_id"], kind="stable").copy()
    accepted = []
    last_by_cell = {}
    accepted_by_cell = {}
    for idx, hit in frame.iterrows():
        if not bool(hit.in_window):
            continue
        cell = int(hit.CellID)
        previous = last_by_cell.get(cell)
        if previous is None or float(hit.t_obs) - previous["t_obs"] >= dead_time_ns:
            accepted.append(idx)
            last_by_cell[cell] = {"t_obs": float(hit.t_obs), "source_kind": hit.source_kind,
                                  "truth": tuple(hit.truth_tracks[0]) if hit.truth_tracks else None,
                                  "index": idx}
            accepted_by_cell.setdefault(cell, []).append(idx)
            frame.at[idx, "rejection_reason"] = "accepted"
        else:
            winner = previous
            if winner["source_kind"] == "noise":
                reason = "dead_time_by_noise"
            elif hit.source_kind == "noise":
                reason = "dead_time_by_real"
            elif winner["truth"] == (int(hit.EvtID), int(hit.TrkID)):
                reason = "dead_time_by_same_track"
            else:
                reason = "dead_time_by_other_track"
                # Preserve the ambiguity on the observed winner for offline audits.
                if hit.truth_tracks:
                    merged = list(frame.at[winner["index"], "truth_tracks"])
                    for truth in hit.truth_tracks:
                        if truth not in merged:
                            merged.append(truth)
                    frame.at[winner["index"], "truth_tracks"] = merged
            frame.at[idx, "rejection_reason"] = reason
    accepted_frame = frame.loc[accepted].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    # Mark true candidates that share a Cell with another truth track, including winners.
    for cell, rows in accepted_by_cell.items():
        truths = {truth for i in rows for truth in frame.loc[i].truth_tracks}
        if len(truths) > 1:
            accepted_frame.loc[accepted_frame.CellID.eq(cell), "shared_hit"] = True
    return accepted_frame, frame.reset_index(drop=True)


def audit_summary(all_candidates, accepted, source):
    real_truth = {(int(e), int(t)): 0 for e, t in zip(source.EvtID, source.TrkID)}
    for truth in source[["EvtID", "TrkID"]].itertuples(index=False):
        real_truth[(int(truth.EvtID), int(truth.TrkID))] += 1
    observed = {key: 0 for key in real_truth}
    for tracks in accepted.truth_tracks:
        for key in tracks:
            if key in observed:
                observed[key] += 1
    counts = all_candidates.rejection_reason.value_counts().to_dict()
    real_candidates = all_candidates[all_candidates.source_kind.eq("real")]
    real_counts = real_candidates.rejection_reason.value_counts().to_dict()
    loss_by_truth = {key: real_truth[key] - observed[key] for key in real_truth}
    return {
        "source_tracks": len(real_truth),
        "source_real_hits": int(len(source)),
        "accepted_nodes": int(len(accepted)),
        "noise_nodes": int(accepted.is_noise.sum()),
        "shared_nodes": int(accepted.shared_hit.sum()),
        "candidate_counts_by_reason": {str(k): int(v) for k, v in counts.items()},
        "real_candidate_counts_by_reason": {str(k): int(v) for k, v in real_counts.items()},
        "truth_hits_before_dead_time": int(len(source)),
        "truth_hits_after_dead_time": int(sum(observed.values())),
        "truth_hits_lost_total": int(sum(loss_by_truth.values())),
        "truth_hits_lost_outside_window": int((real_candidates.rejection_reason == "outside_window").sum()),
        "truth_hits_lost_to_dead_time": int(sum(loss_by_truth.values()) - (real_candidates.rejection_reason == "outside_window").sum()),
        "tracks_with_dead_time_loss": int(sum(v > 0 for v in loss_by_truth.values())),
        "max_truth_hits_lost_for_one_track": int(max(loss_by_truth.values(), default=0)),
    }


def main():
    parser = argparse.ArgumentParser(description="生成独立的 MACE 命中来源审计数据集")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--graphs", type=int, default=1000)
    parser.add_argument("--mean-tracks", type=float, default=30.0)
    parser.add_argument("--min-tracks", type=int, default=5)
    parser.add_argument("--max-tracks", type=int, default=60)
    parser.add_argument("--noise-rate-hz", type=float, default=DEFAULT_NOISE_RATE_HZ)
    parser.add_argument("--sigma-t-ns", type=float, default=DEFAULT_SIGMA_T_NS)
    parser.add_argument("--dead-time-ns", type=float, default=DEAD_TIME_NS)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "audit_test")
    args = parser.parse_args()

    split_path = Path(__file__).resolve().parents[1] / "event_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    event_pool = np.asarray(split[{"train": "train_evt_ids", "validation": "val_evt_ids", "test": "test_evt_ids"}[args.split]], dtype=int)
    source = load_axial_hits()
    cell_map = load_cell_map()
    tracks_per_event = source.groupby("EvtID").TrkID.nunique().astype(int).to_dict()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graphs = []
    audits = []
    for sample_index in range(args.graphs):
        rng = np.random.default_rng(args.seed + sample_index * 1009)
        target = sample_target(rng, args.mean_tracks, args.min_tracks, args.max_tracks)
        events, actual = sample_events(event_pool, tracks_per_event, target, rng)
        selected = source[source.EvtID.isin(events)].reset_index(drop=True)
        candidates = make_candidates(selected, cell_map, args.noise_rate_hz, args.sigma_t_ns, rng)
        accepted, audited = apply_audit_dead_time(candidates, args.dead_time_ns)
        graph = build_graph(accepted)
        if graph is None:
            continue
        graph.audit_candidate_id = torch.tensor(accepted.candidate_id.to_numpy(np.int64))
        graph.audit_rejection_reason = [str(v) for v in accepted.rejection_reason]
        graphs.append(graph)
        record = audit_summary(audited, accepted, selected)
        record.update({"sample_index": sample_index, "selected_event_ids": events, "target_tracks": target, "actual_tracks": actual})
        audits.append(record)

    torch.save(graphs, args.output_dir / "graphs.pt")
    (args.output_dir / "audit_records.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {"split": args.split, "graphs": len(graphs), "requested_graphs": args.graphs,
                "noise_rate_hz": args.noise_rate_hz, "sigma_t_ns": args.sigma_t_ns,
                "dead_time_ns": args.dead_time_ns, "event_window_ns": EVENT_WINDOW_NS,
                "distance_sigma_mm": DRIFT_DISTANCE_SIGMA_MM,
                "purpose": "offline provenance audit; not a training dataset"}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("输出：", args.output_dir.resolve())


if __name__ == "__main__":
    main()
