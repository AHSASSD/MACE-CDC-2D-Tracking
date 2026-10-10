from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
AI5 = HERE.parents[1]
import sys
sys.path.insert(0, str(AI5))
sys.path.insert(0, str(AI5.parent / "AI工程化4"))

from common import InteractionNetwork, load_graph_dataset, model_dimensions  # noqa: E402
from STEP4_两级寻迹 import predict  # noqa: E402
from STEP1_生成节点分类派生数据 import node_context  # noqa: E402
from STEP2_训练节点分类器 import NodeStateClassifier  # noqa: E402
from triplet_common import pair_features, truth_sets  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-file", type=Path, required=True)
    p.add_argument("--model-file", type=Path, required=True)
    p.add_argument("--node-model-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-graphs", type=int, default=4000)
    p.add_argument("--development-fraction", type=float, default=0.25)
    args = p.parse_args()
    graphs = load_graph_dataset(args.dataset_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_model = InteractionNetwork(*model_dimensions("time")).to(device)
    edge_model.load_state_dict(torch.load(args.model_file, map_location=device))
    edge_model.eval()
    node_model = NodeStateClassifier(node_in=13, edge_in=8).to(device)
    node_model.load_state_dict(torch.load(args.node_model_file, map_location=device, weights_only=False))
    node_model.eval()
    rows, labels, groups = [], [], []
    limit = min(len(graphs), args.max_graphs)
    for graph_id, data in enumerate(graphs[:limit]):
        edge_scores = np.asarray(predict(data, edge_model, device, "time"), dtype=float)
        context, edge_context = node_context(data, edge_scores, [])
        with torch.no_grad():
            _, shared_logits = node_model(
                data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device),
                context.to(device), edge_context.to(device),
            )
        shared = torch.sigmoid(shared_logits).cpu().numpy()
        truth = truth_sets(data)
        incident = {}
        for (u, v), score in zip(data.edge_index.cpu().numpy().T, edge_scores):
            if score < 0.30:
                continue
            incident.setdefault(int(u), []).append((int(v), float(score)))
        for seed, neighbors in incident.items():
            neighbors = sorted(neighbors, key=lambda item: -item[1])[:8]
            for i, (left, left_score) in enumerate(neighbors):
                for right, right_score in neighbors[i + 1:]:
                    left_truth = truth[seed] & truth[left]
                    right_truth = truth[seed] & truth[right]
                    if left_truth and right_truth and left_truth == right_truth:
                        label = 0  # continuation
                    elif left_truth and right_truth and left_truth != right_truth:
                        label = 1  # shared split
                    else:
                        label = 2  # fake/contaminated
                    rows.append(pair_features(data, seed, left, right, left_score, right_score, shared[seed]))
                    labels.append(label)
                    groups.append(graph_id)
        if graph_id and graph_id % 100 == 0:
            print(f"已完成 {graph_id}/{limit}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": torch.tensor(np.asarray(rows)), "y": torch.tensor(labels),
                "group": torch.tensor(groups), "feature_names": pair_features.__name__}, args.output)
    print(f"triplets={len(labels)} graphs={limit} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
