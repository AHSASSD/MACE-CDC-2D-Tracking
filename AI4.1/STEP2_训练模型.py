from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common import (
    DATASET_META_FILE, ROOT, SPLIT_FILE, TRAIN_DATA, VAL_DATA,
    DEAD_TIME_NS, DRIFT_DISTANCE_SIGMA_MM, InteractionNetwork,
    load_graph_dataset, model_dimensions, save_json, select_features,
)


# 低显存默认配置：不改变网络结构，只降低单次显存峰值。
BATCH_SIZE = int(os.getenv("MACE_BATCH_SIZE", "2"))
VAL_BATCH_SIZE = int(os.getenv("MACE_VAL_BATCH_SIZE", "2"))
ACCUMULATION_STEPS = int(os.getenv("MACE_ACCUMULATION_STEPS", "16"))
USE_AMP = os.getenv("MACE_USE_AMP", "1") != "0"
TRAIN_SEED = int(os.getenv("MACE_TRAIN_SEED", "20260823"))

if BATCH_SIZE < 1 or VAL_BATCH_SIZE < 1 or ACCUMULATION_STEPS < 1:
    raise ValueError("MACE_BATCH_SIZE、MACE_VAL_BATCH_SIZE、MACE_ACCUMULATION_STEPS 必须 >= 1")


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


@torch.no_grad()
def evaluate(model, loader, criterion, device, variant, threshold=0.5):
    model.eval()
    loss_sum = edges = tp = fp = fn = 0
    for data in loader:
        data = data.to(device)
        x, edge_attr = select_features(data, variant)
        logits = model(x, data.edge_index, edge_attr)
        loss_sum += criterion(logits, data.y).item() * data.y.numel()
        edges += data.y.numel()
        pred = torch.sigmoid(logits) >= threshold
        target = data.y.bool()
        tp += int((pred & target).sum())
        fp += int((pred & ~target).sum())
        fn += int((~pred & target).sum())
    return {"loss": loss_sum / max(edges, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


@torch.no_grad()
def select_threshold(model, loader, device, variant):
    model.eval()
    probabilities = []
    labels = []
    for data in loader:
        data = data.to(device)
        x, edge_attr = select_features(data, variant)
        probabilities.append(torch.sigmoid(model(x, data.edge_index, edge_attr)).cpu())
        labels.append(data.y.bool().cpu())
    probabilities = torch.cat(probabilities)
    labels = torch.cat(labels)
    best = None
    for threshold in [x / 100 for x in range(30, 96, 5)]:
        pred = probabilities >= threshold
        tp = int((pred & labels).sum())
        fp = int((pred & ~labels).sum())
        fn = int((~pred & labels).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        result = {"threshold": threshold, "precision": precision, "recall": recall, "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
        if best is None or result["f1"] > best["f1"]:
            best = result
    return best


def validate_dataset_metadata(path, variant):
    if not path.is_file():
        raise FileNotFoundError(f"缺少数据集元信息：{path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    expected_node, expected_edge = model_dimensions(variant)
    actual = metadata.get("features", {}).get(variant, {})
    if (actual.get("node"), actual.get("edge")) != (expected_node, expected_edge):
        raise ValueError(f"数据集维度不匹配：{actual} != node={expected_node}, edge={expected_edge}")
    detector = metadata.get("detector", {})
    if metadata.get("profile") == "whole-events-v1":
        if metadata.get("event_mixing") is not False or metadata.get("source_group_disjoint") is not True:
            raise ValueError("完整事件来源划分契约不匹配")
        if metadata.get("distance", {}).get("already_measured") is not True or metadata["distance"].get("added_sigma_mm") != 0:
            raise ValueError("完整事件必须直接使用已测量d")
        timing = metadata.get("timing", {})
        if timing.get("basis") != "observed_event_min" or timing.get("scale_ns", 0) <= 0:
            raise ValueError("完整事件时间必须来自可观测事件基准")
    else:
        if detector.get("dead_time_ns") != DEAD_TIME_NS or "non-paralyzable" not in detector.get("dead_time", ""):
            raise ValueError("数据集死时间契约不匹配")
        if metadata.get("noise", {}).get("all_cells_compete") is not True:
            raise ValueError("数据集未启用全 Cell 噪声竞争")
        if metadata.get("distance", {}).get("sigma_mm") != DRIFT_DISTANCE_SIGMA_MM:
            raise ValueError("数据集漂移距离分辨率契约不匹配")
    excluded = set(metadata.get("features", {}).get("excluded_truth_inputs", []))
    required = {"is_noise", "shared_hit", "t0", "tHit", "t_window"}
    if not required.issubset(excluded):
        raise ValueError(f"数据集仍暴露真值输入：缺少排除项 {sorted(required - excluded)}")
    for forbidden in ("t_rel", "dt_rel"):
        if forbidden not in excluded:
            raise ValueError(f"数据集元信息未明确排除作弊特征：{forbidden}")
    return metadata


def load_initial_weights(model, path):
    path = path.resolve()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save_atomic(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_weights(model, path):
    save_atomic(model.state_dict(), path)


def restore_training_state(state, model, optimizer, scheduler, scaler):
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def dataset_signature(metadata_path, split_path, train_path, val_path):
    files = []
    for path in (train_path, val_path):
        parts = path.with_suffix(path.suffix + ".parts")
        files.extend(sorted(parts.glob("part_*.pt")) if parts.is_dir() else [path])
    return {"metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "files": [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files]}


def main():
    parser = argparse.ArgumentParser(description="训练 AI工程化4 时间消融模型")
    parser.add_argument("--features", choices=["time", "no-time"], required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--run-tag", default="clean_physics_v2")
    parser.add_argument("--dataset-root", type=Path, help="完整事件数据集目录；不指定时保留原AI4数据集")
    parser.add_argument("--init-model", type=Path, help="加载已有权重做全网络微调；优化器重新初始化")
    parser.add_argument("--lr", type=float, help="学习率；微调默认1e-4，从头训练默认1e-3")
    parser.add_argument("--resume", type=Path, help="恢复完整训练断点；epochs表示本阶段目标总轮数")
    args = parser.parse_args()
    if args.resume and (args.init_model or args.lr is not None):
        parser.error("--resume不能与--init-model或--lr同时使用；断点会恢复学习率")
    learning_rate = args.lr if args.lr is not None else (1e-4 if args.init_model else 1e-3)
    if args.epochs < 1 or not 0 < learning_rate < float("inf"):
        parser.error("epochs和学习率必须为正数，学习率必须有限")
    batch_size = BATCH_SIZE
    val_batch_size = VAL_BATCH_SIZE
    accumulation = ACCUMULATION_STEPS
    use_amp = USE_AMP
    stem = f"{args.features.replace('-', '_')}_{args.run_tag}"
    output_root = args.dataset_root.resolve() if args.dataset_root else ROOT
    train_path = output_root / "dataset_train.pt" if args.dataset_root else TRAIN_DATA
    val_path = output_root / "dataset_validation.pt" if args.dataset_root else VAL_DATA
    metadata_path = output_root / "dataset_metadata.json" if args.dataset_root else DATASET_META_FILE
    split_path = output_root / "event_split.json" if args.dataset_root else SPLIT_FILE
    model_file = output_root / f"best_interaction_network_{stem}.pth"
    meta_file = output_root / f"model_metadata_{stem}.json"
    checkpoint_file = output_root / f"training_state_{stem}.pth"
    if not args.resume and (args.init_model or args.dataset_root) and any(p.exists() for p in (model_file, meta_file, checkpoint_file)):
        raise FileExistsError(f"微调输出已存在，请换一个--run-tag以保留已有结果：{model_file}")

    dataset_metadata = validate_dataset_metadata(metadata_path, args.features)
    resume_state = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    config = {"features": args.features, "run_tag": args.run_tag,
              "model_output": str(model_file.resolve()),
              "dataset": dataset_signature(metadata_path, split_path, train_path, val_path),
              "batch_size": batch_size, "val_batch_size": val_batch_size,
              "accumulation": accumulation, "seed": TRAIN_SEED}
    if resume_state is not None and (resume_state.get("version") != 1 or resume_state.get("config") != config):
        raise ValueError("断点的模型、数据、输出标签或训练配置不匹配")
    torch.manual_seed(TRAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAIN_SEED)

    node_in, edge_in = model_dimensions(args.features)
    model = InteractionNetwork(node_in, edge_in)
    initialization = resume_state["initialization"] if resume_state else (load_initial_weights(model, args.init_model) if args.init_model else None)
    if not resume_state:
        print(f"初始化={'微调: ' + initialization['path'] if initialization else '随机权重'}；学习率={learning_rate:g}；输出={model_file}", flush=True)

    train = load_graph_dataset(train_path)
    val = load_graph_dataset(val_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assert not set(split["train_evt_ids"]) & set(split["val_evt_ids"])
    assert not set(split["train_evt_ids"]) & set(split["test_evt_ids"])
    assert not set(split["val_evt_ids"]) & set(split["test_evt_ids"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = use_amp and device.type == "cuda"
    model = model.to(device)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, exclude_keys=["truth_ptr", "truth_pairs", "all_truth_pairs"])
    val_loader = DataLoader(val, batch_size=val_batch_size, exclude_keys=["truth_ptr", "truth_pairs", "all_truth_pairs"])
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_loss = float("inf")
    best_epoch = 0
    best_model = None
    start_epoch = 1
    elapsed_before = 0
    started = time.time()

    def save_training_checkpoint(epoch):
        save_atomic({"version": 1, "config": config, "epoch": epoch,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                     "amp_enabled": amp_enabled, "device_type": device.type, "best_model": best_model,
                     "best_loss": best_loss, "best_epoch": best_epoch,
                     "initialization": initialization, "initial_learning_rate": learning_rate,
                     "elapsed_seconds": elapsed_before + time.time() - started}, checkpoint_file)
    if resume_state:
        if resume_state["amp_enabled"] != amp_enabled or resume_state["device_type"] != device.type:
            raise ValueError("恢复时混合精度/计算设备配置必须与断点一致")
        restore_training_state(resume_state, model, optimizer, scheduler, scaler)
        best_loss, best_epoch = resume_state["best_loss"], resume_state["best_epoch"]
        best_model = resume_state["best_model"]
        start_epoch = resume_state["epoch"] + 1
        elapsed_before = resume_state["elapsed_seconds"]
        learning_rate = resume_state["initial_learning_rate"]
        if best_model is not None:
            save_atomic(best_model, model_file)
        print(f"恢复第{start_epoch - 1}轮；当前学习率={optimizer.param_groups[0]['lr']:g}；最佳轮={best_epoch}", flush=True)
    elif initialization:
        initial_metrics = evaluate(model, val_loader, criterion, device, args.features)
        best_loss = initial_metrics["loss"]
        best_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        save_atomic(best_model, model_file)
        print(f"微调前 val={best_loss:.5f} P={initial_metrics['precision']:.4f} R={initial_metrics['recall']:.4f} F1={initial_metrics['f1']:.4f}", flush=True)
    if not resume_state:
        save_training_checkpoint(0)
    writer = SummaryWriter(output_root / "runs" / stem, purge_step=start_epoch if resume_state else None)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_sum = edge_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, data in enumerate(train_loader):
            data = data.to(device)
            x, edge_attr = select_features(data, args.features)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                raw_loss = criterion(model(x, data.edge_index, edge_attr), data.y)
                loss = raw_loss / accumulation
            if not torch.isfinite(raw_loss).item():
                raise FloatingPointError("训练loss非有限值，停止本轮；上一个完整断点仍保留")
            scaler.scale(loss).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_sum += raw_loss.item() * data.y.numel()
            edge_count += data.y.numel()
        metrics = evaluate(model, val_loader, criterion, device, args.features)
        if not all(float('-inf') < value < float('inf') for value in metrics.values()):
            raise FloatingPointError("验证指标非有限值，停止本轮；上一个完整断点仍保留")
        train_loss = train_sum / max(edge_count, 1)
        used_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(metrics["loss"])
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", metrics["loss"], epoch)
        writer.add_scalar("F1/validation", metrics["f1"], epoch)
        writer.add_scalar("LR/used", used_lr, epoch)
        writer.add_scalar("LR/next", optimizer.param_groups[0]["lr"], epoch)
        print(f"{epoch:03d} train={train_loss:.5f} val={metrics['loss']:.5f} P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} lr={used_lr:g} next_lr={optimizer.param_groups[0]['lr']:g}", flush=True)
        improved = metrics["loss"] < best_loss
        if improved:
            best_loss = metrics["loss"]
            best_epoch = epoch
            best_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        save_training_checkpoint(epoch)
        if improved:
            save_atomic(best_model, model_file)
        writer.flush()
    writer.close()
    model.load_state_dict(torch.load(model_file, map_location=device))
    threshold = select_threshold(model, val_loader, device, args.features)
    save_json(meta_file, {
        "feature_variant": args.features,
        "run_tag": args.run_tag,
        "training_seed": TRAIN_SEED,
        "initialization": initialization,
        "initial_learning_rate": learning_rate,
        "optimizer_state_reused": bool(resume_state),
        "resume_from": str(args.resume.resolve()) if args.resume else None,
        "best_epoch": best_epoch,
        "training_checkpoint": str(checkpoint_file),
        "dataset_metadata": str(metadata_path),
        "dataset_summary": dataset_metadata.get("summary", {}),
        "selection_dataset": "validation",
        "edge_threshold": threshold["threshold"],
        "validation_edge_precision": threshold["precision"],
        "validation_edge_recall": threshold["recall"],
        "validation_edge_f1": threshold["f1"],
        "model": {"node_in": node_in, "edge_in": edge_in, "hidden": 256, "n_passes": 8},
        "elapsed_seconds": elapsed_before + time.time() - started,
    })
    print(f"完成：{model_file}")
    print(f"验证集最佳边阈值={threshold['threshold']:.2f} F1={threshold['f1']:.4f}")


if __name__ == "__main__":
    main()
