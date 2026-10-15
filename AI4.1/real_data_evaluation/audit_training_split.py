"""逐分片核查实际 AI4 数据中的原始事件身份，不修改数据或模型。"""
from collections import Counter
import json
from pathlib import Path
import torch

SOURCE = Path(__file__).resolve().parents[2] / "AI工程化4"
OUTPUT = Path(__file__).resolve().parent / "training_split_audit.json"


def main():
    torch.set_num_threads(1)
    split = json.loads((SOURCE / "event_split.json").read_text(encoding="utf-8"))
    keys = {"train": "train_evt_ids", "validation": "val_evt_ids", "test": "test_evt_ids"}
    observed, results = {}, {}
    for name, key in keys.items():
        pool = set(split[key])
        filename = f"dataset_{'val' if name == 'validation' else name}_noise100k_sigma3.pt"
        paths = sorted((SOURCE / (filename + ".parts")).glob("part_*.pt"))
        if not paths:
            raise FileNotFoundError(filename)
        ids, track_ids, appearances = set(), set(), Counter()
        graphs = nodes = edges = noise = 0
        for part_index, path in enumerate(paths, 1):
            batch = torch.load(path, weights_only=False, map_location="cpu")
            for graph in batch:
                pairs = graph.truth_pairs.numpy()
                real = pairs[pairs[:, 0] >= 0]
                current_ids = set(map(int, real[:, 0]))
                assert current_ids <= pool, f"{name}/{path.name}: source identities outside declared pool"
                ids.update(current_ids)
                track_ids.update(map(tuple, real.tolist()))
                appearances.update(current_ids)
                graphs += 1
                nodes += graph.num_nodes
                edges += graph.edge_index.shape[1]
                noise += int(graph.is_noise.sum())
            del batch
            if part_index % 10 == 0 or part_index == len(paths):
                print(f"Audit {name}: {part_index}/{len(paths)} shards, {graphs} graphs", flush=True)
        observed[name] = ids
        results[name] = {"shards": len(paths), "graphs": graphs, "declared_source_events": len(pool),
                         "observed_source_events": len(ids), "observed_truth_tracks": len(track_ids),
                         "source_events_not_observed": sorted(pool - ids),
                         "mean_nodes": nodes / graphs, "mean_directed_edges": edges / graphs,
                         "mean_incoming_degree": edges / nodes, "mean_noise_nodes": noise / graphs,
                         "graphs_per_observed_source_event_mean": sum(appearances.values()) / len(ids),
                         "graphs_per_observed_source_event_min": min(appearances.values()),
                         "graphs_per_observed_source_event_max": max(appearances.values())}
    overlaps = {}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = observed[a] & observed[b]
        assert not overlap, f"Source events leak across {a}/{b}: {overlap}"
        overlaps[f"{a}_vs_{b}"] = len(overlap)
    report = {"source_directory": str(SOURCE), "identity": "(source EvtID,TrkID) recorded in saved graph.truth_pairs",
              "declared_pool_sizes": {n: len(split[k]) for n, k in keys.items()},
              "actual_source_event_intersections": overlaps, "datasets": results,
              "conclusion": "Saved graphs follow disjoint source-event pools; reuse occurs within each split, not across splits",
              "provenance_limit": "This audits currently stored datasets and code; metadata alone cannot prove historical weight file provenance"}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
