from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from common import InteractionNetwork, load_graph_dataset, model_dimensions


ROOT = Path(__file__).resolve().parent
AI4 = ROOT.parent / "AI工程化4"
sys.path.insert(0, str(AI4))

from STEP4_两级寻迹 import predict  # noqa: E402
from step4_hit_audit.analyze_physics_algorithm_loss import CORE_PROFILE  # noqa: E402
from step4_short_rerun_experiment.run_experiment import run_with_short_rerun  # noqa: E402


_WORKER_MODEL = None
_WORKER_FEATURES = None


def init_worker(model_file, features, torch_threads):
    global _WORKER_MODEL, _WORKER_FEATURES
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    _WORKER_FEATURES = features
    _WORKER_MODEL = InteractionNetwork(*model_dimensions(features))
    _WORKER_MODEL.load_state_dict(torch.load(model_file, map_location="cpu"))
    _WORKER_MODEL.eval()


def derive_graph_worker(data):
    return derive_graph(data, _WORKER_MODEL, torch.device("cpu"), _WORKER_FEATURES)


def node_context(data, probabilities, tracks):
    node_count = int(data.num_nodes)
    edges = data.edge_index.cpu().numpy()
    degree = np.zeros(node_count, dtype=np.float32)
    probability_sum = np.zeros(node_count, dtype=np.float32)
    probability_max = np.zeros(node_count, dtype=np.float32)
    for (u, v), value in zip(edges.T, probabilities):
        degree[v] += 1.0
        probability_sum[v] += float(value)
        probability_max[v] = max(probability_max[v], float(value))
    probability_mean = probability_sum / np.maximum(degree, 1.0)

    assignment = np.full(node_count, -1, dtype=np.int64)
    assigned_hits = np.zeros(node_count, dtype=np.float32)
    assigned_rms = np.zeros(node_count, dtype=np.float32)
    for track_id, track in enumerate(tracks):
        indices = np.asarray(track["indices"], dtype=np.int64)
        assignment[indices] = track_id
        assigned_hits[indices] = min(float(track["hits"]) / 20.0, 1.0)
        assigned_rms[indices] = min(float(track["rms"]) / 1.5, 1.0)

    degree_scale = max(float(np.log1p(node_count)), 1.0)
    context = np.column_stack([
        assignment >= 0,
        assigned_hits,
        assigned_rms,
        probability_mean,
        probability_max,
        np.log1p(degree) / degree_scale,
    ]).astype(np.float32)
    same_track = (
        (assignment[edges[0]] >= 0)
        & (assignment[edges[0]] == assignment[edges[1]])
    )
    edge_context = np.column_stack([probabilities, same_track]).astype(np.float32)
    return torch.from_numpy(context), torch.from_numpy(edge_context)


def derive_graph(data, model, device, features):
    probabilities = predict(data, model, device, features)
    tracks, _ = run_with_short_rerun(
        data, model, device, features, probabilities,
        0.70, 0.575, 0.55, 0.20, CORE_PROFILE,
    )
    context, edge_context = node_context(data, probabilities, tracks)
    truth_counts = data.truth_ptr[1:] - data.truth_ptr[:-1]
    y_noise = data.is_noise.bool().cpu()
    y_shared = (truth_counts > 1).bool().cpu() & ~y_noise
    return Data(
        x=data.x.cpu(),
        edge_index=data.edge_index.cpu(),
        edge_attr=data.edge_attr.cpu(),
        node_context=context,
        edge_context=edge_context,
        y_noise=y_noise.float(),
        y_shared=y_shared.float(),
        wire_x=data.wire_x.cpu(),
        wire_y=data.wire_y.cpu(),
        drift_d=data.drift_d.cpu(),
        cell_id=data.cell_id.cpu(),
    )


def generate_split(source_path, output_path, model, device, features, chunk_size, rounds, pool=None):
    graphs = load_graph_dataset(source_path)
    limit = min(len(graphs), rounds or len(graphs))
    parts_dir = output_path.with_suffix(output_path.suffix + ".parts")
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)
    chunk = []
    totals = {"graphs": 0, "nodes": 0, "noise_nodes": 0, "shared_nodes": 0}
    part_index = 0
    selected = graphs[:limit]
    derived_graphs = (
        pool.imap(derive_graph_worker, selected, chunksize=1)
        if pool is not None
        else (derive_graph(data, model, device, features) for data in selected)
    )
    for derived in tqdm(derived_graphs, total=limit, desc=output_path.stem):
        chunk.append(derived)
        totals["graphs"] += 1
        totals["nodes"] += int(derived.num_nodes)
        totals["noise_nodes"] += int(derived.y_noise.sum())
        totals["shared_nodes"] += int(derived.y_shared.sum())
        if len(chunk) >= chunk_size:
            torch.save(chunk, parts_dir / f"part_{part_index:05d}.pt")
            chunk = []
            part_index += 1
    if chunk:
        torch.save(chunk, parts_dir / f"part_{part_index:05d}.pt")
        part_index += 1
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps({"parts_dir": str(parts_dir), "parts": part_index, **totals}, indent=2),
        encoding="utf-8",
    )
    return totals


def main():
    parser = argparse.ArgumentParser(description="生成 AI5 节点分类派生图")
    parser.add_argument("--train-data", type=Path, default=ROOT / "data" / "ai5_dataset_train_contract_noise100k_sigma3.pt")
    parser.add_argument("--val-data", type=Path, default=ROOT / "data" / "ai5_dataset_val_contract_noise100k_sigma3.pt")
    parser.add_argument("--model-file", type=Path, default=AI4 / "best_interaction_network_time_clean_physics_v2.pth")
    parser.add_argument("--features", choices=["time", "no-time"], default="time")
    parser.add_argument("--tag", default="ai4_strongest_v1")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--workers", type=int, default=1,
                        help="CPU 派生进程数；1 保留原单进程 GPU/CPU 路径")
    parser.add_argument("--worker-torch-threads", type=int, default=1,
                        help="每个 CPU worker 内部的 PyTorch 线程数")
    args = parser.parse_args()
    if (args.chunk_size < 1 or args.workers < 1 or args.worker_torch_threads < 1
            or (args.rounds is not None and args.rounds < 1)):
        raise ValueError("chunk-size、workers、worker-torch-threads 和 rounds 必须为正数")

    model = device = None
    if args.workers == 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = InteractionNetwork(*model_dimensions(args.features)).to(device)
        model.load_state_dict(torch.load(args.model_file, map_location=device))
        model.eval()
    else:
        print(f"CPU 多进程模式：workers={args.workers}, 每进程 PyTorch 线程={args.worker_torch_threads}")
    summaries = {}
    pool_context = (
        mp.get_context("spawn").Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(args.model_file, args.features, args.worker_torch_threads),
        )
        if args.workers > 1 else None
    )
    try:
        for split, source in (("train", args.train_data), ("val", args.val_data)):
            output = ROOT / "data" / f"ai5_node_classifier_{split}_{args.tag}.pt"
            summaries[split] = generate_split(
                source, output, model, device, args.features, args.chunk_size, args.rounds,
                pool_context,
            )
    except BaseException:
        if pool_context is not None:
            pool_context.terminate()
            pool_context.join()
        raise
    else:
        if pool_context is not None:
            pool_context.close()
            pool_context.join()
    metadata = {
        "project": "AI5",
        "task": "observable_node_classification",
        "source_model": str(args.model_file.resolve()),
        "source_pipeline": "AI4 time strongest",
        "features": args.features,
        "execution": {
            "mode": "cpu_multiprocessing" if args.workers > 1 else str(device),
            "workers": args.workers,
            "worker_torch_threads": args.worker_torch_threads,
        },
        "node_context": [
            "first_pass_assigned", "assigned_track_hits_scaled", "assigned_track_rms_scaled",
            "mean_incident_edge_probability", "max_incident_edge_probability", "candidate_degree_scaled",
        ],
        "edge_context": ["ai4_edge_probability", "edge_same_first_pass_track"],
        "targets": {
            "noise": "is_noise",
            "shared": "non-noise and truth_tracks count > 1",
        },
        "leakage_guard": "truth fields are targets only and are not stored as model inputs",
        "summary": summaries,
    }
    metadata_path = ROOT / "data" / f"ai5_node_classifier_metadata_{args.tag}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print("元数据：", metadata_path.resolve())


if __name__ == "__main__":
    mp.freeze_support()
    main()
