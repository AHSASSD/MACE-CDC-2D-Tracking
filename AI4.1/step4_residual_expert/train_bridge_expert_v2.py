from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import InteractionNetwork, TRAIN_DATA, VAL_DATA, model_dimensions, save_json, select_features  # noqa: E402
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

SEED = 20260829
CORE_PROFILE = {
    "skip_parent_rms": 0.20,
    "max_core_rms": 0.80,
    "min_rms_gain": 0.10,
    "min_probability_gain": 0.05,
    "min_coverage": 0.40,
}
THRESHOLDS = [value / 100 for value in range(5, 81, 5)]


def source_parts(path, limit):
    paths = sorted(path.with_suffix(path.suffix + ".parts").glob("part_*.pt"))
    if not paths:
        raise FileNotFoundError(f"找不到数据分片：{path}")
    return paths[:limit] if limit else paths


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

    fragment_id = np.full(data.num_nodes, -1, dtype=np.int64)
    for index, fragment in enumerate(fragments):
        fragment_id[fragment] = index
    edges = data.edge_index.t().cpu().numpy()
    node_mask = np.zeros(data.num_nodes, dtype=bool)
    node_mask[nodes] = True
    edge_mask = node_mask[edges[:, 0]] & node_mask[edges[:, 1]]
    if not edge_mask.any():
        return None
    selected_edges = edges[edge_mask]
    mapping = np.full(data.num_nodes, -1, dtype=np.int64)
    mapping[nodes] = np.arange(len(nodes))
    bridge_mask = fragment_id[selected_edges[:, 0]] != fragment_id[selected_edges[:, 1]]
    return Data(
        x=data.x[nodes].clone(),
        edge_index=torch.from_numpy(mapping[selected_edges].T).long(),
        edge_attr=data.edge_attr[edge_mask].clone(),
        y=data.y[edge_mask].clone(),
        bridge_mask=torch.from_numpy(bridge_mask),
    )


@torch.no_grad()
def add_teacher_logits(graph, base_model, device, variant):
    batch = graph.clone().to(device)
    x, edge_attr = select_features(batch, variant)
    graph.teacher_logits = base_model(x, batch.edge_index, edge_attr).float().cpu()
    return graph


def summarize_graphs(graphs):
    edges = sum(graph.num_edges for graph in graphs)
    positive_edges = sum(int(graph.y.sum()) for graph in graphs)
    bridge_edges = sum(int(graph.bridge_mask.sum()) for graph in graphs)
    positive_bridge_edges = sum(int((graph.bridge_mask & graph.y.bool()).sum()) for graph in graphs)
    return {
        "residual_graphs": len(graphs),
        "nodes": sum(graph.num_nodes for graph in graphs),
        "edges": edges,
        "positive_edges": positive_edges,
        "positive_edge_fraction": positive_edges / max(edges, 1),
        "bridge_edges": bridge_edges,
        "positive_bridge_edges": positive_bridge_edges,
        "positive_bridge_fraction": positive_bridge_edges / max(bridge_edges, 1),
    }


def prepare_split(name, source, output_dir, base_model, device, variant, strict, relaxed, max_parts):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_graphs = []
    paths = source_parts(source, max_parts)
    for part_index, source_path in enumerate(paths):
        output_path = output_dir / f"part_{part_index:05d}.pt"
        if output_path.is_file():
            graphs = torch.load(output_path, weights_only=False)
            required = all(hasattr(graph, "bridge_mask") and hasattr(graph, "teacher_logits") for graph in graphs)
            if not required:
                raise RuntimeError(f"缓存格式不是 V2，请使用 --rebuild：{output_path}")
        else:
            graphs = []
            for data in torch.load(source_path, weights_only=False):
                probabilities = predict(data, base_model, device, variant)
                graph = residual_graph(data, probabilities, strict, relaxed)
                if graph is not None:
                    graphs.append(add_teacher_logits(graph, base_model, device, variant))
            torch.save(graphs, output_path)
        all_graphs.extend(graphs)
        print(f"{name} V2 分片 {part_index + 1}/{len(paths)}: {len(all_graphs)} 张残差图", flush=True)
    return all_graphs, summarize_graphs(all_graphs)


def weighted_focal_loss(logits, target, bridge_mask, bridge_positive_weight, bridge_negative_weight):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = 0.5 * target + 0.5 * (1 - target)
    losses = alpha_t * (1 - pt).pow(2.0) * bce
    weights = torch.ones_like(losses)
    weights[bridge_mask & target.bool()] = bridge_positive_weight
    weights[bridge_mask & ~target.bool()] = bridge_negative_weight
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def metric_result(probabilities, labels, bridge_mask, threshold):
    prediction = probabilities >= threshold

    def metrics(mask, beta):
        pred = prediction[mask]
        truth = labels[mask]
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        beta2 = beta * beta
        fbeta = (1 + beta2) * tp / max((1 + beta2) * tp + beta2 * fn + fp, 1)
        return {"precision": precision, "recall": recall, f"f{beta:g}": fbeta, "tp": tp, "fp": fp, "fn": fn}

    return {
        "threshold": threshold,
        "all_edges": metrics(torch.ones_like(labels, dtype=torch.bool), 1.0),
        "bridge_edges": metrics(bridge_mask, 2.0),
    }


@torch.no_grad()
def evaluate_thresholds(model, loader, device, variant):
    model.eval()
    probabilities = []
    labels = []
    bridges = []
    for data in loader:
        data = data.to(device)
        x, edge_attr = select_features(data, variant)
        probabilities.append(torch.sigmoid(model(x, data.edge_index, edge_attr)).cpu())
        labels.append(data.y.bool().cpu())
        bridges.append(data.bridge_mask.bool().cpu())
    probabilities = torch.cat(probabilities)
    labels = torch.cat(labels)
    bridges = torch.cat(bridges)
    results = [metric_result(probabilities, labels, bridges, threshold) for threshold in THRESHOLDS]
    return max(results, key=lambda item: (item["bridge_edges"]["f2"], item["all_edges"]["f1"]))


def train_loss(model, data, variant, bridge_positive_weight, bridge_negative_weight, distill_weight):
    x, edge_attr = select_features(data, variant)
    logits = model(x, data.edge_index, edge_attr)
    supervised = weighted_focal_loss(
        logits, data.y, data.bridge_mask.bool(), bridge_positive_weight, bridge_negative_weight
    )
    teacher = torch.sigmoid(data.teacher_logits)
    distillation = F.binary_cross_entropy_with_logits(logits, teacher)
    return supervised + distill_weight * distillation, supervised, distillation


def main():
    parser = argparse.ArgumentParser(description="V2 跨碎片桥接残差专家训练")
    parser.add_argument("--features", choices=["time", "no-time"], default="time")
    parser.add_argument("--run-tag", default="clean_physics_v2")
    parser.add_argument("--expert-tag", default="bridge_v2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("MACE_BATCH_SIZE", "2")))
    parser.add_argument("--bridge-positive-weight", type=float, default=5.0)
    parser.add_argument("--bridge-negative-weight", type=float, default=1.5)
    parser.add_argument("--distill-weight", type=float, default=0.10)
    parser.add_argument("--strict-threshold", type=float, default=0.70)
    parser.add_argument("--relaxed-threshold", type=float, default=0.575)
    parser.add_argument("--max-train-parts", type=int, default=0)
    parser.add_argument("--max-val-parts", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    positive = [args.epochs, args.patience, args.batch_size, args.learning_rate,
                args.bridge_positive_weight, args.bridge_negative_weight]
    if any(value <= 0 for value in positive) or args.distill_weight < 0:
        raise ValueError("训练轮数、耐心、批量、学习率和边权重必须为正，蒸馏权重不得为负")

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
    limit_tag = f"tp{args.max_train_parts or 'all'}_vp{args.max_val_parts or 'all'}"
    cache_root = Path(__file__).resolve().parent / (
        f"dataset_v2_{args.features.replace('-', '_')}_{args.run_tag}_{routing_tag}_{limit_tag}"
    )
    train_dir = cache_root / "train.parts"
    val_dir = cache_root / "validation.parts"
    if args.rebuild:
        for directory in (train_dir, val_dir):
            for path in directory.glob("part_*.pt") if directory.is_dir() else ():
                path.unlink()

    started = time.time()
    train, train_summary = prepare_split(
        "train", TRAIN_DATA, train_dir, base_model, device, args.features,
        args.strict_threshold, args.relaxed_threshold, args.max_train_parts,
    )
    val, val_summary = prepare_split(
        "validation", VAL_DATA, val_dir, base_model, device, args.features,
        args.strict_threshold, args.relaxed_threshold, args.max_val_parts,
    )
    if not train or not val:
        raise RuntimeError("训练或验证残差子图为空")
    if val_summary["positive_bridge_edges"] == 0:
        raise RuntimeError("验证集没有跨碎片正边，无法使用桥接 F2 选模")
    dataset_meta = cache_root / "metadata.json"
    save_json(dataset_meta, {
        "purpose": "bridge-aware residual expert fine-tuning",
        "routing_uses_truth": False,
        "truth_usage": "edge supervision only after routing",
        "teacher_context": "base model rerun on the same residual subgraph",
        "base_model": str(base_path),
        "features": args.features,
        "strict_threshold": args.strict_threshold,
        "relaxed_threshold": args.relaxed_threshold,
        "core_profile": CORE_PROFILE,
        "train": train_summary,
        "validation": val_summary,
    })
    if args.prepare_only:
        print("V2 残差数据准备完成：", dataset_meta)
        return

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    expert = InteractionNetwork(*model_dimensions(args.features)).to(device)
    expert.load_state_dict(torch.load(base_path, map_location=device))
    for name, parameter in expert.named_parameters():
        parameter.requires_grad = name.startswith("edge_classifier.")
    trainable = [parameter for parameter in expert.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in expert.named_parameters() if parameter.requires_grad]
    if not trainable_names or any(not name.startswith("edge_classifier.") for name in trainable_names):
        raise RuntimeError(f"可训练参数范围异常：{trainable_names}")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    expert_stem = f"residual_{args.features.replace('-', '_')}_{args.run_tag}_{args.expert_tag}"
    model_path = ROOT / f"best_interaction_network_{expert_stem}.pth"
    meta_path = ROOT / f"model_metadata_{expert_stem}.json"

    baseline = evaluate_thresholds(expert, val_loader, device, args.features)
    print("base residual metrics:", json.dumps(baseline, ensure_ascii=False), flush=True)
    best = (
        (baseline["bridge_edges"]["f2"], baseline["all_edges"]["f1"]),
        baseline,
        0,
    )
    torch.save(expert.state_dict(), model_path)
    epochs_without_improvement = 0
    completed_epochs = 0
    for epoch in range(1, args.epochs + 1):
        expert.train()
        total_sum = supervised_sum = distill_sum = edge_count = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                total, supervised, distillation = train_loss(
                    expert, data, args.features, args.bridge_positive_weight,
                    args.bridge_negative_weight, args.distill_weight,
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(optimizer)
            scaler.update()
            count = data.y.numel()
            total_sum += total.item() * count
            supervised_sum += supervised.item() * count
            distill_sum += distillation.item() * count
            edge_count += count

        metrics = evaluate_thresholds(expert, val_loader, device, args.features)
        bridge_f2 = metrics["bridge_edges"]["f2"]
        all_f1 = metrics["all_edges"]["f1"]
        print(
            f"{epoch:03d} loss={total_sum / edge_count:.5f} "
            f"supervised={supervised_sum / edge_count:.5f} distill={distill_sum / edge_count:.5f} "
            f"threshold={metrics['threshold']:.2f} bridge_F2={bridge_f2:.4f} "
            f"bridge_P={metrics['bridge_edges']['precision']:.4f} "
            f"bridge_R={metrics['bridge_edges']['recall']:.4f} all_F1={all_f1:.4f}", flush=True,
        )
        completed_epochs = epoch
        rank = (bridge_f2, all_f1)
        if rank > best[0]:
            best = (rank, metrics, epoch)
            torch.save(expert.state_dict(), model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"早停：连续 {args.patience} 轮桥接 F2 未提升", flush=True)
                break

    _, best_metrics, best_epoch = best
    save_json(meta_path, {
        "feature_variant": args.features,
        "expert_tag": args.expert_tag,
        "version": 2,
        "base_model": str(base_path),
        "residual_dataset": str(dataset_meta),
        "trainable_module": "edge_classifier",
        "trainable_parameters": trainable_names,
        "loss": {
            "bridge_positive_weight": args.bridge_positive_weight,
            "bridge_negative_weight": args.bridge_negative_weight,
            "distill_weight": args.distill_weight,
        },
        "selection_metric": "validation bridge-edge F2, tie-break all-edge F1",
        "baseline_validation": baseline,
        "best_validation": best_metrics,
        "best_epoch": best_epoch,
        "improved_over_base": best_epoch > 0,
        "completed_epochs": completed_epochs,
        "learning_rate": args.learning_rate,
        "training_seed": SEED,
        "elapsed_seconds": time.time() - started,
    })
    print("完成：", model_path)
    print("元信息：", meta_path)
    print(f"推荐二次阈值={best_metrics['threshold']:.2f} bridge_F2={best_metrics['bridge_edges']['f2']:.4f}")


if __name__ == "__main__":
    main()
