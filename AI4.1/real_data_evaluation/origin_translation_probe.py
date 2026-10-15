"""离线真值单轨平移诊断：固定拓扑、距离、时间，只改变坐标相对原点的位置。

使用出生位置和真轨分组，仅用于归因，不可用于实际寻迹或汇报部署效率。
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import build_graph, InteractionNetwork, model_dimensions, select_features

OUT = Path(__file__).resolve().parent / "paired_diagnosis"


def main():
    torch.set_num_threads(1)
    ids = set(json.loads((OUT / "n200_single_seed20261013" / "time_raw.json").read_text(encoding="utf-8"))["source_event_ids"])
    events = torch.load(ROOT / "real_data_evaluation" / "events.pt", weights_only=False)
    physics = pd.read_csv(OUT / "new_particle_physics.csv").set_index(["EvtID", "TrkID"])
    originals, translated, identities = [], [], []
    changed_topology = skipped = 0
    for event in events:
        if event["evtid"] not in ids:
            continue
        for trk, group in event["frame"].groupby("TrkID"):
            if not 5 <= len(group) <= 20:
                continue
            group = group.reset_index(drop=True)
            key = (event["evtid"], int(trk))
            entry = physics.loc[key]
            shifted = group.copy()
            shifted["wire_x"] -= entry.birth_radius_mm * np.cos(entry.birth_phi)
            shifted["wire_y"] -= entry.birth_radius_mm * np.sin(entry.birth_phi)
            shifted["wire_r"] = np.hypot(shifted.wire_x, shifted.wire_y)
            a, b = build_graph(group), build_graph(shifted)
            if a is None or b is None:
                skipped += 1
                continue
            pairs = list(map(tuple, a.edge_index.t().tolist()))
            other_pairs = list(map(tuple, b.edge_index.t().tolist()))
            if set(pairs) != set(other_pairs):
                changed_topology += 1
                continue
            lookup = {pair: attr for pair, attr in zip(other_pairs, b.edge_attr)}
            b.edge_index = a.edge_index.clone()
            b.edge_attr = torch.stack([lookup[pair] for pair in pairs])
            assert torch.equal(a.drift_d, b.drift_d) and torch.equal(a.x[:, 6], b.x[:, 6])
            assert torch.all(a.y == 1) and torch.all(b.y == 1)
            originals.append(a)
            translated.append(b)
            identities.append({"EvtID": key[0], "TrkID": key[1], "birth_radius_mm": float(entry.birth_radius_mm),
                               "edges": len(pairs)})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"diagnostic_only": True, "not_a_reconstruction_efficiency": True,
              "tracks": len(identities), "topology_changed_excluded": changed_topology, "no_graph_excluded": skipped,
              "description": "Truth-isolated positive-edge graphs before/after translating all nodes by that particle's x0,y0; same topology,d,time,labels", "models": {}}
    for variant in ["no-time", "time"]:
        model = InteractionNetwork(*model_dimensions(variant)).to(device)
        stem = f"{variant.replace('-', '_')}_clean_physics_v2"
        model.load_state_dict(torch.load(ROOT / f"best_interaction_network_{stem}.pth", map_location=device, weights_only=True))
        model.eval()
        condition_metrics = {}
        for condition, graphs in [("original_coordinates", originals), ("birth_at_origin", translated)]:
            values = []
            for batch in DataLoader(graphs, batch_size=32, shuffle=False, exclude_keys=["truth_ptr", "truth_pairs"]):
                x, attr = select_features(batch, variant)
                with torch.no_grad():
                    values.append(torch.sigmoid(model(x.to(device), batch.edge_index.to(device), attr.to(device))).cpu().numpy())
            probabilities = np.concatenate(values)
            rows = []
            offset = 0
            for info in identities:
                n = info["edges"]
                p = probabilities[offset:offset+n]
                offset += n
                rows.append({**info, "mean_probability": float(p.mean()),
                             "positive_edges_retained_at_050": int((p >= .5).sum()),
                             "positive_edges_retained_at_070": int((p >= .7).sum())})
            assert offset == len(probabilities)
            frame = pd.DataFrame(rows)
            frame["radius_bin"] = pd.cut(frame.birth_radius_mm, [-1, 1, 5, 15, 60, 100],
                                           labels=["0_1", "1_5", "5_15", "15_60", "60_plus"])
            totals = frame.groupby("radius_bin", observed=True).agg(tracks=("TrkID", "size"), edges=("edges", "sum"),
                mean_probability=("mean_probability", "mean"), retained_050=("positive_edges_retained_at_050", "sum"),
                retained_070=("positive_edges_retained_at_070", "sum"))
            totals["positive_edge_recall_050"] = totals.retained_050 / totals.edges
            totals["positive_edge_recall_070"] = totals.retained_070 / totals.edges
            condition_metrics[condition] = {"all_mean_probability": float(probabilities.mean()),
                                           "all_positive_edge_recall_050": float((probabilities >= .5).mean()),
                                           "all_positive_edge_recall_070": float((probabilities >= .7).mean()),
                                           "by_birth_radius": totals.to_dict("index")}
            frame.to_csv(OUT / f"origin_probe_{variant}_{condition}.csv", index=False, encoding="utf-8-sig")
        report["models"][variant] = condition_metrics
    (OUT / "origin_translation_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
