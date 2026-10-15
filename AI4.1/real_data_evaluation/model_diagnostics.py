"""固定旧权重的边分类质量、占用率与真值连边诊断。"""
from collections import Counter
import json
from pathlib import Path
import networkx as nx
import numpy as np
import torch

OUT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    events = torch.load(OUT / "events.pt", weights_only=False)
    node_count = sum(e["data"].num_nodes for e in events)
    edge_count = sum(e["data"].edge_index.shape[1] for e in events)
    labels = np.concatenate([e["data"].y.numpy().astype(bool) for e in events])
    output = {"new_batch_nodes_mean": node_count / len(events),
              "new_batch_directed_edges_mean": edge_count / len(events),
              "new_batch_mean_incoming_degree": edge_count / node_count,
              "new_batch_edge_metrics": {}}
    for variant in ["no-time", "time"]:
        values = torch.load(OUT / f"probabilities_{variant}.pt", weights_only=False)["values"]
        probability = np.concatenate([values[e["evtid"]] for e in events])
        assert probability.shape == labels.shape
        prediction = probability >= .5
        tp = int((prediction & labels).sum())
        fp = int((prediction & ~labels).sum())
        fn = int((~prediction & labels).sum())
        output["new_batch_edge_metrics"][variant] = {
            "true_positive_edges": tp, "false_positive_edges": fp, "false_negative_edges": fn,
            "precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn)}
    sizes = Counter()
    disconnected = disconnected_below_half = 0
    t0_nunique, t0_range = [], []
    for event in events:
        frame, data = event["frame"], event["data"]
        truth_graph = nx.Graph()
        truth_graph.add_nodes_from(range(len(frame)))
        edges = data.edge_index.t().numpy()
        truth_graph.add_edges_from(map(tuple, edges[data.y.numpy().astype(bool)]))
        component_of = {node: part for part in nx.connected_components(truth_graph) for node in part}
        for _, track in frame.groupby("TrkID"):
            size = len(track)
            sizes[size] += 1
            t0_nunique.append(int(track.t0.nunique()))
            t0_range.append(float(track.t0.max() - track.t0.min()))
            largest = max(len(component_of[i]) for i in track.index)
            disconnected += largest < size
            disconnected_below_half += largest < size * .5
    output["truth_graph"] = {"tracks": sum(sizes.values()), "tracks_not_fully_connected": disconnected,
                             "tracks_no_component_reaching_half": disconnected_below_half}
    output["t0_within_truth_track"] = {"mean_unique_values": float(np.mean(t0_nunique)),
                                       "tracks_with_multiple_t0": sum(n > 1 for n in t0_nunique),
                                       "median_range_ns": float(np.median(t0_range)), "max_range_ns": max(t0_range),
                                       "warning": "t0 field semantics and online observability need simulator/hardware definition"}
    oracle = json.loads((OUT / "oracle_baseline.json").read_text(encoding="utf-8"))
    output["perfect_truth_edges_fixed_postprocessing"] = oracle["summary"]
    (OUT / "model_diagnostics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
