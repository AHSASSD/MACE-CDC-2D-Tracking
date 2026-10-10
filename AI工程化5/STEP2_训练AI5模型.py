from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common import (
    DATA_DIR, DATASET_META_FILE, MODEL_DIR, ROOT, SPLIT_FILE, TRAIN_DATA, VAL_DATA,
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

    def forward(self, logits, target, weight=None):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        if weight is not None:
            return (loss * weight).sum() / weight.sum().clamp_min(1.0)
        return loss.mean()


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


def main():
    parser = argparse.ArgumentParser(description="训练 AI5 独立模型")
    parser.add_argument("--features", choices=["time", "no-time"], required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--run-tag", default="clean_physics_v2")
    parser.add_argument("--data-tag", default="noise100k_sigma3", help="AI5 数据集标签")
    parser.add_argument("--shared-compat-weight", type=float, default=1.0,
                        help="shared-compatible 但非 exclusive 的负边权重；基线为 1.0")
    args = parser.parse_args()
    if not 0.0 < args.shared_compat_weight <= 1.0:
        raise ValueError("--shared-compat-weight 必须在 (0, 1] 内")
    batch_size = BATCH_SIZE
    val_batch_size = VAL_BATCH_SIZE
    accumulation = ACCUMULATION_STEPS
    use_amp = USE_AMP
    stem = f"{args.features.replace('-', '_')}_{args.run_tag}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_file = MODEL_DIR / f"best_interaction_network_{stem}.pth"
    meta_file = MODEL_DIR / f"model_metadata_{stem}.json"

    if args.data_tag == "noise100k_sigma3":
        train_data, val_data, metadata_file = TRAIN_DATA, VAL_DATA, DATASET_META_FILE
    else:
        train_data = DATA_DIR / f"ai5_dataset_train_{args.data_tag}.pt"
        val_data = DATA_DIR / f"ai5_dataset_val_{args.data_tag}.pt"
        metadata_file = DATA_DIR / f"ai5_dataset_metadata_{args.data_tag}.json"
    dataset_metadata = validate_dataset_metadata(metadata_file, args.features)
    torch.manual_seed(TRAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAIN_SEED)

    train = load_graph_dataset(train_data)
    val = load_graph_dataset(val_data)
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    assert not set(split["train_evt_ids"]) & set(split["val_evt_ids"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = use_amp and device.type == "cuda"
    node_in, edge_in = model_dimensions(args.features)
    model = InteractionNetwork(node_in, edge_in).to(device)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, exclude_keys=["truth_ptr", "truth_pairs", "all_truth_pairs"])
    val_loader = DataLoader(val, batch_size=val_batch_size, exclude_keys=["truth_ptr", "truth_pairs", "all_truth_pairs"])
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    writer = SummaryWriter(ROOT / "runs" / stem)
    best_loss = float("inf")
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = edge_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, data in enumerate(train_loader):
            data = data.to(device)
            x, edge_attr = select_features(data, args.features)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                weight = None
                if args.shared_compat_weight != 1.0:
                    if not hasattr(data, "edge_shared_ambiguous"):
                        raise ValueError("数据集缺少 edge_shared_ambiguous，不能启用 shared compatibility 权重")
                    weight = torch.ones_like(data.y)
                    weight[data.edge_shared_ambiguous.bool()] = args.shared_compat_weight
                raw_loss = criterion(model(x, data.edge_index, edge_attr), data.y, weight)
                loss = raw_loss / accumulation
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
        train_loss = train_sum / max(edge_count, 1)
        scheduler.step(metrics["loss"])
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", metrics["loss"], epoch)
        writer.add_scalar("F1/validation", metrics["f1"], epoch)
        print(f"{epoch:03d} train={train_loss:.5f} val={metrics['loss']:.5f} P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(model.state_dict(), model_file)
    writer.close()
    model.load_state_dict(torch.load(model_file, map_location=device))
    threshold = select_threshold(model, val_loader, device, args.features)
    save_json(meta_file, {
        "project": "AI5",
        "label_contract": "exclusive_winner",
        "shared_compatibility": {
            "status": "not_used_in_baseline" if args.shared_compat_weight == 1.0 else "ambiguous_negative_downweight",
        },
        "shared_compat_weight": args.shared_compat_weight,
        "feature_variant": args.features,
        "run_tag": args.run_tag,
        "training_seed": TRAIN_SEED,
        "dataset_metadata": str(metadata_file),
        "dataset_summary": dataset_metadata.get("summary", {}),
        "selection_dataset": "validation",
        "edge_threshold": threshold["threshold"],
        "validation_edge_precision": threshold["precision"],
        "validation_edge_recall": threshold["recall"],
        "validation_edge_f1": threshold["f1"],
        "model": {"node_in": node_in, "edge_in": edge_in, "hidden": 256, "n_passes": 8},
        "elapsed_seconds": time.time() - started,
    })
    print(f"完成：{model_file}")
    print(f"验证集最佳边阈值={threshold['threshold']:.2f} F1={threshold['f1']:.4f}")


if __name__ == "__main__":
    main()
