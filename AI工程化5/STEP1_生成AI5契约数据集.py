from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
AI4 = ROOT.parent / "AI工程化4"
sys.path.insert(0, str(AI4))

from common import build_graph, load_axial_hits, load_cell_map  # noqa: E402
from step4_hit_audit.generate_audit_dataset import (  # noqa: E402
    apply_audit_dead_time,
    make_candidates,
    sample_events,
    sample_target,
)


def apply_contract_labels(graph):
    inclusive = []
    exclusive = []
    for node in range(graph.num_nodes):
        start, end = int(graph.truth_ptr[node]), int(graph.truth_ptr[node + 1])
        values = [tuple(map(int, pair)) for pair in graph.truth_pairs[start:end].tolist()]
        inclusive.append(set(values))
        exclusive.append(set(values[:1]))

    edges = graph.edge_index.t().tolist()
    y_exclusive = torch.tensor(
        [bool(exclusive[u] & exclusive[v]) for u, v in edges], dtype=torch.float32
    )
    y_shared = torch.tensor(
        [bool(inclusive[u] & inclusive[v]) for u, v in edges], dtype=torch.bool
    )
    graph.y = y_exclusive
    graph.edge_shared_compatible = y_shared
    graph.edge_shared_ambiguous = y_shared & ~y_exclusive.bool()
    return graph


def generate_split(args, split_name, count, seed, source, cell_map, event_pool, tracks_per_event):
    output = ROOT / "data" / f"ai5_dataset_{split_name}_{args.tag}.pt"
    parts_dir = output.with_suffix(output.suffix + ".parts")
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    chunk = []
    part_index = graph_count = ambiguous_edges = shared_edges = 0
    for sample_index in tqdm(range(count), desc=split_name):
        rng = np.random.default_rng(seed + sample_index * 1009)
        target = sample_target(rng, args.mean_tracks, args.min_tracks, args.max_tracks)
        events, _ = sample_events(event_pool, tracks_per_event, target, rng)
        selected = source[source.EvtID.isin(events)].reset_index(drop=True)
        candidates = make_candidates(selected, cell_map, args.noise_rate_hz, args.sigma_t_ns, rng)
        accepted, _ = apply_audit_dead_time(candidates, args.dead_time_ns)
        graph = build_graph(accepted)
        if graph is None:
            continue
        graph = apply_contract_labels(graph)
        ambiguous_edges += int(graph.edge_shared_ambiguous.sum())
        shared_edges += int(graph.edge_shared_compatible.sum())
        chunk.append(graph)
        graph_count += 1
        if len(chunk) >= args.chunk_size:
            torch.save(chunk, parts_dir / f"part_{part_index:05d}.pt")
            chunk = []
            part_index += 1
    if chunk:
        torch.save(chunk, parts_dir / f"part_{part_index:05d}.pt")
        part_index += 1
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps({"parts_dir": str(parts_dir), "parts": part_index, "graphs": graph_count}, indent=2),
        encoding="utf-8",
    )
    return {"graphs": graph_count, "shared_edges": shared_edges, "ambiguous_edges": ambiguous_edges}


def main():
    parser = argparse.ArgumentParser(description="生成 AI5 exclusive + shared compatibility 契约数据")
    parser.add_argument("--train-graphs", type=int, default=4000)
    parser.add_argument("--val-graphs", type=int, default=1000)
    parser.add_argument("--mean-tracks", type=float, default=30.0)
    parser.add_argument("--min-tracks", type=int, default=5)
    parser.add_argument("--max-tracks", type=int, default=60)
    parser.add_argument("--noise-rate-hz", type=float, default=100000.0)
    parser.add_argument("--sigma-t-ns", type=float, default=3.0)
    parser.add_argument("--dead-time-ns", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--tag", default="contract_noise100k_sigma3")
    args = parser.parse_args()

    split = json.loads((AI4 / "event_split.json").read_text(encoding="utf-8"))
    source = load_axial_hits()
    cell_map = load_cell_map()
    tracks_per_event = source.groupby("EvtID").TrkID.nunique().astype(int).to_dict()
    summaries = {
        "train": generate_split(
            args, "train", args.train_graphs, args.seed, source, cell_map,
            np.asarray(split["train_evt_ids"], dtype=int), tracks_per_event,
        ),
        "val": generate_split(
            args, "val", args.val_graphs, args.seed + 1_000_000, source, cell_map,
            np.asarray(split["val_evt_ids"], dtype=int), tracks_per_event,
        ),
    }
    metadata = {
        "project": "AI5",
        "label_contract": "exclusive_winner",
        "shared_compatibility": {
            "field": "edge_shared_compatible",
            "ambiguous_field": "edge_shared_ambiguous",
            "definition": "shared-compatible but non-exclusive edges remain negative and may be downweighted",
            "leakage_guard": "offline training supervision only",
        },
        "seed": args.seed,
        "features": {
            "time": {"node": 7, "edge": 6},
            "no-time": {"node": 6, "edge": 4},
            "excluded_truth_inputs": ["is_noise", "shared_hit", "t0", "tHit", "t_window", "t_rel", "dt_rel"],
        },
        "detector": {"dead_time": "non-paralyzable per CellID, ordered by t_obs", "dead_time_ns": 100.0},
        "noise": {"all_cells_compete": True, "rate_hz": args.noise_rate_hz},
        "distance": {"sigma_mm": 0.075},
        "summary": summaries,
    }
    metadata_path = ROOT / "data" / f"ai5_dataset_metadata_{args.tag}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print("元数据：", metadata_path.resolve())


if __name__ == "__main__":
    main()
