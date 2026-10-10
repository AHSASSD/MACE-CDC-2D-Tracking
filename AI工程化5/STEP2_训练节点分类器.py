from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from common import ROOT, load_graph_dataset, save_json


class NodeStateClassifier(nn.Module):
    def __init__(self, node_in, edge_in, hidden=96, passes=3):
        super().__init__()
        self.passes = passes
        self.node_encoder = nn.Sequential(nn.Linear(node_in, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.edge_encoder = nn.Sequential(nn.Linear(edge_in, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.message = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.noise_head = nn.Linear(hidden, 1)
        self.shared_head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_attr, node_context, edge_context):
        nodes = self.node_encoder(torch.cat([x, node_context], dim=-1))
        edges = self.edge_encoder(torch.cat([edge_attr, edge_context], dim=-1))
        row, col = edge_index
        for _ in range(self.passes):
            messages = self.message(torch.cat([nodes[row], nodes[col], edges], dim=-1))
            aggregate = torch.zeros_like(nodes)
            aggregate.index_add_(0, col, messages)
            nodes = nodes + self.update(torch.cat([nodes, aggregate], dim=-1))
        return self.noise_head(nodes).squeeze(-1), self.shared_head(nodes).squeeze(-1)


def binary_uncertainty(logits):
    probability = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    return -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log()) / math.log(2.0)


def metrics(logits, target, threshold):
    prediction = torch.sigmoid(logits) >= threshold
    truth = target.bool()
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    values = {"noise_logits": [], "noise_target": [], "shared_logits": [], "shared_target": []}
    for data in loader:
        data = data.to(device)
        noise, shared = model(data.x, data.edge_index, data.edge_attr, data.node_context, data.edge_context)
        values["noise_logits"].append(noise.cpu())
        values["noise_target"].append(data.y_noise.cpu())
        values["shared_logits"].append(shared.cpu())
        values["shared_target"].append(data.y_shared.cpu())
    return {key: torch.cat(value) for key, value in values.items()}


def select_threshold(logits, target):
    candidates = []
    for value in range(5, 100, 5):
        threshold = value / 100.0
        candidates.append({"threshold": threshold, **metrics(logits, target, threshold)})
    return max(candidates, key=lambda item: item["f1"])


def main():
    parser = argparse.ArgumentParser(description="训练 AI5 可观测节点分类器")
    parser.add_argument("--data-tag", default="ai4_strongest_v1")
    parser.add_argument("--run-tag", default="node_state_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs 和 batch-size 必须为正数")

    train_path = ROOT / "data" / f"ai5_node_classifier_train_{args.data_tag}.pt"
    val_path = ROOT / "data" / f"ai5_node_classifier_val_{args.data_tag}.pt"
    train = load_graph_dataset(train_path)
    val = load_graph_dataset(val_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_in = int(train[0].x.shape[1] + train[0].node_context.shape[1])
    edge_in = int(train[0].edge_attr.shape[1] + train[0].edge_context.shape[1])
    model = NodeStateClassifier(node_in, edge_in).to(device)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    noise_loss = nn.BCEWithLogitsLoss()
    shared_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"best_node_state_classifier_{args.run_tag}.pth"
    best_loss = float("inf")
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = nodes = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            noise, shared = model(data.x, data.edge_index, data.edge_attr, data.node_context, data.edge_context)
            loss = noise_loss(noise, data.y_noise) + shared_loss(shared, data.y_shared)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_sum += float(loss.detach()) * data.num_nodes
            nodes += data.num_nodes
        values = collect(model, val_loader, device)
        validation_loss = float(
            noise_loss(values["noise_logits"].to(device), values["noise_target"].to(device))
            + shared_loss(values["shared_logits"].to(device), values["shared_target"].to(device))
        )
        print(f"{epoch:03d} train={train_sum / max(nodes, 1):.5f} val={validation_loss:.5f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path, map_location=device))
    values = collect(model, val_loader, device)
    noise = select_threshold(values["noise_logits"], values["noise_target"])
    shared = select_threshold(values["shared_logits"], values["shared_target"])
    metadata = {
        "project": "AI5",
        "task": "noise_and_shared_node_classification",
        "data_tag": args.data_tag,
        "run_tag": args.run_tag,
        "model": {"node_in": node_in, "edge_in": edge_in, "hidden": 96, "passes": 3},
        "uncertainty": "maximum normalized binary entropy of noise/shared outputs; not a supervised class",
        "validation": {"noise": noise, "shared": shared, "loss": best_loss},
        "elapsed_seconds": time.time() - started,
    }
    metadata_path = model_dir / f"node_state_classifier_metadata_{args.run_tag}.json"
    save_json(metadata_path, metadata)
    print(json.dumps(metadata["validation"], ensure_ascii=False, indent=2))
    print("模型：", model_path.resolve())


if __name__ == "__main__":
    main()
