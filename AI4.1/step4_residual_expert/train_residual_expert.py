from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import InteractionNetwork, TRAIN_DATA, VAL_DATA, model_dimensions, save_json, select_features  # noqa: E402
from STEP2_训练模型 import FocalLoss, evaluate, select_threshold  # noqa: E402
from STEP4_两级寻迹 import (  # noqa: E402
    RELAXED_RMS,
    STRICT_RMS,
    components,
    hierarchy_partition,
    predict,
    split_component,
    trim_outliers,
)
from step4_short_core_experiment.run_experiment import short_core_extract  # noqa: E402
from step4_short_rerun_experiment.run_experiment import short_components  # noqa: E402

SEED = 20260828
CORE_PROFILE = {
    "skip_parent_rms": 0.20,
    "max_core_rms": 0.80,
    "min_rms_gain": 0.10,
    "min_probability_gain": 0.05,
    "min_coverage": 0.40,
}


def part_paths(path):
    directory = path.with_suffix(path.suffix + ".parts")
    paths = sorted(directory.glob("part_*.pt"))
    if not paths:
        raise FileNotFoundError(f"找不到数据分片：{directory}")
    return paths


def residual_graph(data, probabilities, strict_threshold, relaxed_threshold):
    strict_candidates = []
    for component in components(data, probabilities, strict_threshold):
        core = short_core_extract(data, component, probabilities, strict_threshold, CORE_PROFILE)
        if core is not None:
            strict_candidates.append(core)
            continue
        hierarchy = hierarchy_partition(data, component, probabilities, strict_threshold)
        strict_candidates.extend(hierarchy or split_component(data, component, probabilities, strict_threshold, STRICT_RMS))

    used = {index for candidate in strict_candidates for index in candidate["indices"]}
    leftover = set(range(data.num_nodes)) - used
    relaxed_candidates = [
        trim_outliers(data, component, probabilities, relaxed_threshold, RELAXED_RMS, "relaxed")
        for component in components(data, probabilities, relaxed_threshold, leftover)
    ]
    relaxed_used = {
        index for candidate in relaxed_candidates if candidate is not None for index in candidate["indices"]
    }
    fragments = short_components(data, probabilities, relaxed_threshold, leftover - relaxed_used)
    if not any(len(fragment) >= 3 for fragment in fragments):
        return None
    nodes = sorted({node for fragment in fragments for node in fragment})
    if len(nodes) < 5:
        return None

    edges = data.edge_index.t().cpu().numpy()
    node_mask = np.zeros(data.num_nodes, dtype=bool)
    node_mask[nodes] = True
    edge_mask = node_mask[edges[:, 0]] & node_mask[edges[:, 1]]
    if not edge_mask.any():
        return None
    mapping = np.full(data.num_nodes, -1, dtype=np.int64)
    mapping[nodes] = np.arange(len(nodes))
    return Data(
        x=data.x[nodes].clone(),
        edge_index=torch.from_numpy(mapping[edges[edge_mask]].T).long(),
        edge_attr=data.edge_attr[edge_mask].clone(),
        y=data.y[edge_mask].clone(),
    )


def prepare_split(name, source, output_dir, base_model, device, variant, strict, relaxed):
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = {"source_graphs": 0, "residual_graphs": 0, "nodes": 0, "edges": 0, "positive_edges": 0}
    for part_index, source_path in enumerate(part_paths(source)):
        output_path = output_dir / f"part_{part_index:05d}.pt"
        if output_path.is_file():
            graphs = torch.load(output_path, weights_only=False)
            totals["source_graphs"] += 100
        else:
            graphs = []
            source_graphs = torch.load(source_path, weights_only=False)
            totals["source_graphs"] += len(source_graphs)
            for data in source_graphs:
                probabilities = predict(data, base_model, device, variant)
                graph = residual_graph(data, probabilities, strict, relaxed)
                if graph is not None:
                    graphs.append(graph)
            torch.save(graphs, output_path)
        totals["residual_graphs"] += len(graphs)
        totals["nodes"] += sum(graph.num_nodes for graph in graphs)
        totals["edges"] += sum(graph.num_edges for graph in graphs)
        totals["positive_edges"] += sum(int(graph.y.sum()) for graph in graphs)
        print(f"{name} 残差分片 {part_index + 1}: {totals['residual_graphs']} 张图", flush=True)
    totals["positive_edge_fraction"] = totals["positive_edges"] / max(totals["edges"], 1)
    return totals


def load_parts(directory):
    graphs = []
    for path in sorted(directory.glob("part_*.pt")):
        graphs.extend(torch.load(path, weights_only=False))
    if not graphs:
        raise RuntimeError(f"残差数据集为空：{directory}")
    return graphs


def main():
    parser = argparse.ArgumentParser(description="生成残差子图并微调短径迹专家")
    parser.add_argument("--features", choices=["time", "no-time"], default="time")
    parser.add_argument("--run-tag", default="clean_physics_v2")
    parser.add_argument("--expert-tag", default="residual_v1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("MACE_BATCH_SIZE", "2")))
    parser.add_argument("--strict-threshold", type=float, default=0.70)
    parser.add_argument("--relaxed-threshold", type=float, default=0.575)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs、batch-size 和 learning-rate 必须为正数")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stem = f"{args.features.replace('-', '_')}_{args.run_tag}"
    base_path = ROOT / f"best_interaction_network_{stem}.pth"
    base_model = InteractionNetwork(*model_dimensions(args.features)).to(device)
    base_model.load_state_dict(torch.load(base_path, map_location=device))
    base_model.eval()

    routing_tag = f"s{args.strict_threshold:.3f}_r{args.relaxed_threshold:.3f}".replace(".", "")
    cache_root = Path(__file__).resolve().parent / (
        f"dataset_{args.features.replace('-', '_')}_{args.run_tag}_{args.expert_tag}_{routing_tag}"
    )
    train_dir = cache_root / "train.parts"
    val_dir = cache_root / "validation.parts"
    if args.rebuild:
        for directory in (train_dir, val_dir):
            if directory.is_dir():
                for path in directory.glob("part_*.pt"):
                    path.unlink()

    started = time.time()
    train_summary = prepare_split(
        "train", TRAIN_DATA, train_dir, base_model, device, args.features,
        args.strict_threshold, args.relaxed_threshold,
    )
    val_summary = prepare_split(
        "validation", VAL_DATA, val_dir, base_model, device, args.features,
        args.strict_threshold, args.relaxed_threshold,
    )
    dataset_meta = cache_root / "metadata.json"
    save_json(dataset_meta, {
        "purpose": "residual-subgraph expert fine-tuning",
        "routing_uses_truth": False,
        "base_model": str(base_path),
        "features": args.features,
        "strict_threshold": args.strict_threshold,
        "relaxed_threshold": args.relaxed_threshold,
        "core_profile": CORE_PROFILE,
        "train": train_summary,
        "validation": val_summary,
    })
    if args.prepare_only:
        print("残差数据准备完成：", dataset_meta)
        return

    train = load_parts(train_dir)
    val = load_parts(val_dir)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    expert = InteractionNetwork(*model_dimensions(args.features)).to(device)
    expert.load_state_dict(torch.load(base_path, map_location=device))
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(expert.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    expert_stem = f"residual_{args.features.replace('-', '_')}_{args.run_tag}_{args.expert_tag}"
    model_path = ROOT / f"best_interaction_network_{expert_stem}.pth"
    meta_path = ROOT / f"model_metadata_{expert_stem}.json"
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        expert.train()
        loss_sum = edge_count = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            x, edge_attr = select_features(data, args.features)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = criterion(expert(x, data.edge_index, edge_attr), data.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(expert.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * data.y.numel()
            edge_count += data.y.numel()
        metrics = evaluate(expert, val_loader, criterion, device, args.features)
        print(
            f"{epoch:03d} train={loss_sum / max(edge_count, 1):.5f} "
            f"val={metrics['loss']:.5f} P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f}", flush=True,
        )
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(expert.state_dict(), model_path)

    expert.load_state_dict(torch.load(model_path, map_location=device))
    threshold = select_threshold(expert, val_loader, device, args.features)
    save_json(meta_path, {
        "feature_variant": args.features,
        "expert_tag": args.expert_tag,
        "base_model": str(base_path),
        "residual_dataset": str(dataset_meta),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "training_seed": SEED,
        "best_validation_loss": best_loss,
        "validation_edge_threshold": threshold["threshold"],
        "validation_edge_precision": threshold["precision"],
        "validation_edge_recall": threshold["recall"],
        "validation_edge_f1": threshold["f1"],
        "elapsed_seconds": time.time() - started,
    })
    print("完成：", model_path)
    print("元信息：", meta_path)
    print(f"残差边最佳阈值={threshold['threshold']:.2f} F1={threshold['f1']:.4f}")


if __name__ == "__main__":
    main()
