from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import ROOT, SPLIT_FILE, InteractionNetwork, build_graph, load_axial_hits, load_cell_map, model_dimensions, prepare_trigger, save_json
from STEP1_生成AI4数据集 import sample_events, sample_target
from STEP4_两级寻迹 import evaluate, run_two_pass


NOISE_RATES_HZ = [10_000.0, 100_000.0, 300_000.0, 1_000_000.0]
SIGMA_T_NS = [1.0, 3.0, 5.0, 10.0]


def fixed_samples(frame, event_pool, count, seed):
    rng = np.random.default_rng(seed)
    tracks_per_event = frame.groupby("EvtID")["TrkID"].nunique().astype(int).to_dict()
    samples = []
    for _ in range(count):
        target = sample_target(rng, 30.0, 5, 60)
        events, _ = sample_events(event_pool, tracks_per_event, target, rng)
        samples.append(events)
    return samples


def load_model(variant, device, run_tag="clean_physics_v2"):
    stem = f"{variant.replace('-', '_')}_{run_tag}"
    metadata = json.loads((ROOT / f"model_metadata_{stem}.json").read_text(encoding="utf-8"))
    model = InteractionNetwork(*model_dimensions(variant)).to(device)
    model.load_state_dict(torch.load(ROOT / f"best_interaction_network_{stem}.pth", map_location=device))
    model.eval()
    strict = float(metadata["edge_threshold"])
    return model, strict, max(0.35, strict - 0.15)


def main():
    parser = argparse.ArgumentParser(description="AI工程化4 固定验证事件噪声/时间误差矩阵")
    parser.add_argument("--graphs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", type=Path, default=ROOT / "ai4_noise_time_validation_matrix.json")
    parser.add_argument("--run-tag", default="clean_physics_v2")
    args = parser.parse_args()
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    frame = load_axial_hits()
    cell_map = load_cell_map()
    validation_events = np.asarray(split["val_evt_ids"], dtype=int)
    samples = fixed_samples(frame, validation_events, args.graphs, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {variant: load_model(variant, device, args.run_tag) for variant in ("no-time", "time")}
    results = []
    for rate in NOISE_RATES_HZ:
        for sigma in SIGMA_T_NS:
            config_graphs = []
            for graph_index, events in enumerate(samples):
                rng = np.random.default_rng(args.seed + graph_index + int(rate) + int(sigma * 1000))
                source = frame[frame["EvtID"].isin(events)].reset_index(drop=True)
                graph = build_graph(prepare_trigger(source, cell_map, rate, sigma, rng))
                if graph is not None:
                    config_graphs.append(graph)
            for variant, (model, strict, relaxed) in models.items():
                reports = []
                for graph in config_graphs:
                    tracks, _ = run_two_pass(graph, model, device, variant, strict, relaxed)
                    reports.append(evaluate(graph, tracks))
                result = {
                    "feature_variant": variant,
                    "noise_rate_hz": rate,
                    "sigma_t_ns": sigma,
                    "graphs": len(reports),
                    "mean_track_efficiency": float(np.mean([x["track_efficiency"] for x in reports])),
                    "mean_fake_tracks": float(np.mean([x["fake_tracks"] for x in reports])),
                    "mean_clone_tracks": float(np.mean([x["clone_tracks"] for x in reports])),
                    "mean_noise_nodes": float(np.mean([x["noise_nodes"] for x in reports])),
                }
                results.append(result)
                print(result)
    save_json(args.output, {
        "selection_dataset": "validation",
        "fixed_event_samples": True,
        "noise_rates_hz": NOISE_RATES_HZ,
        "sigma_t_ns": SIGMA_T_NS,
        "results": results,
    })
    print("报告：", args.output)


if __name__ == "__main__":
    main()
