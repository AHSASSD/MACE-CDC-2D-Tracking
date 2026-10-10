from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from common import ROOT, TEST_DATA, VAL_DATA, InteractionNetwork, model_dimensions, save_json
from STEP4_两级寻迹 import evaluate, predict, run_two_pass


def parse_grid(value: str) -> list[float]:
    values = sorted({round(float(item), 3) for item in value.split(",")})
    if not values or any(item <= 0.0 or item >= 1.0 for item in values):
        raise ValueError("阈值必须位于 (0, 1)")
    return values


def part_paths(data_path: Path) -> list[Path]:
    directory = data_path if data_path.is_dir() else data_path.with_suffix(data_path.suffix + ".parts")
    paths = sorted(directory.glob("part_*.pt"))
    if not paths:
        raise FileNotFoundError(f"找不到数据分片目录：{directory}")
    return paths


def _scan_shard(task):
    data_path, probability_path, strict, relaxed, variant = task
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    dataset = torch.load(data_path, weights_only=False)
    probabilities = torch.load(probability_path, weights_only=False)
    sums = {"eff": 0.0, "fake": 0.0, "clone": 0.0, "trimmed": 0.0, "count": 0}
    for data, values in zip(dataset, probabilities):
        tracks, diagnostics = run_two_pass(data, None, None, variant, strict, relaxed, values)
        report = evaluate(data, tracks)
        sums["eff"] += report["track_efficiency"]
        sums["fake"] += report["fake_tracks"]
        sums["clone"] += report["clone_tracks"]
        sums["trimmed"] += diagnostics.get("trimmed_candidates", 0)
        sums["count"] += 1
    return strict, relaxed, sums


def aggregate(pair, sums):
    strict, relaxed = pair
    count = max(sums["count"], 1)
    efficiency = sums["eff"] / count
    fake = sums["fake"] / count
    clones = sums["clone"] / count
    return {
        "strict_threshold": strict,
        "relaxed_threshold": relaxed,
        "mean_track_efficiency": efficiency,
        "mean_fake_tracks": fake,
        "mean_clone_tracks": clones,
        "utility": efficiency - 0.01 * fake - 0.02 * clones,
        "mean_trimmed_candidates": sums["trimmed"] / count,
    }


def main():
    parser = argparse.ArgumentParser(description="AI4 STEP4 分片并行阈值扫描")
    parser.add_argument("--features", choices=["time", "no-time"], required=True)
    parser.add_argument("--run-tag", default="clean_physics_v2")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--thresholds")
    parser.add_argument("--strict-thresholds", default="0.625,0.65,0.675,0.70,0.725")
    parser.add_argument("--relaxed-thresholds", default="0.45,0.475,0.50,0.525,0.55")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 1) - 1)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers 必须 >= 1")

    strict_values = parse_grid(args.thresholds or args.strict_thresholds)
    relaxed_values = None if args.thresholds else parse_grid(args.relaxed_thresholds)
    pairs = [(s, max(0.30, s - 0.15)) for s in strict_values] if relaxed_values is None else [
        (s, r) for s in strict_values for r in relaxed_values if r <= s
    ]
    stem = f"{args.features.replace('-', '_')}_{args.run_tag}"
    metadata = json.loads((ROOT / f"model_metadata_{stem}.json").read_text(encoding="utf-8"))
    data_path = (args.data or (VAL_DATA if args.split == "validation" else TEST_DATA)).resolve()
    data_parts = part_paths(data_path)
    output = (args.output or ROOT / f"step4_threshold_scan_{stem}_{args.split}.json").resolve()
    results = []
    if args.resume and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        results = previous.get("results", [])
        done = {(x["strict_threshold"], x["relaxed_threshold"]) for x in results}
        pairs = [pair for pair in pairs if pair not in done]
        print(f"恢复扫描：跳过 {len(done)} 组，剩余 {len(pairs)} 组", flush=True)
    if not pairs:
        print("没有待扫描组合，保留已有结果。", flush=True)
        return

    cache_dir = Path(tempfile.mkdtemp(prefix="ai4_threshold_parts_"))
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = InteractionNetwork(*model_dimensions(args.features)).to(device)
        model.load_state_dict(torch.load(ROOT / f"best_interaction_network_{stem}.pth", map_location=device))
        model.eval()
        probability_parts = []
        seen = 0
        total_rounds = min(args.rounds, len(data_parts) * 100)
        print(f"GPU 分片推理：最多 {total_rounds} 张图", flush=True)
        for index, data_part in enumerate(data_parts):
            if seen >= total_rounds:
                break
            graphs = torch.load(data_part, weights_only=False)
            graphs = graphs[:max(0, total_rounds - seen)]
            values = [predict(graph, model, device, args.features) for graph in graphs]
            probability_path = cache_dir / f"prob_{index:05d}.pt"
            torch.save(values, probability_path)
            probability_parts.append((data_part, probability_path))
            seen += len(graphs)
            print(f"GPU 分片 {index + 1}: {seen}/{total_rounds}", flush=True)
        del model

        tasks = [(str(data_part), str(prob_path), strict, relaxed, args.features)
                 for strict, relaxed in pairs for data_part, prob_path in probability_parts]
        pending = {pair: len(probability_parts) for pair in pairs}
        accumulators = {pair: {"eff": 0.0, "fake": 0.0, "clone": 0.0, "trimmed": 0.0, "count": 0} for pair in pairs}
        completed_pairs = 0
        context = mp.get_context("spawn")
        print(f"开始扫描：{len(pairs)} 组，{len(tasks)} 个分片任务，CPU进程数={args.workers}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = [pool.submit(_scan_shard, task) for task in tasks]
            for future in as_completed(futures):
                strict, relaxed, sums = future.result()
                pair = (strict, relaxed)
                for key in accumulators[pair]:
                    accumulators[pair][key] += sums[key]
                pending[pair] -= 1
                if pending[pair] == 0:
                    result = aggregate(pair, accumulators[pair])
                    results.append(result)
                    completed_pairs += 1
                    print(f"[{completed_pairs}/{len(pairs)}] {result}", flush=True)
                    save_json(output, {
                        "features": args.features, "run_tag": args.run_tag, "split": args.split,
                        "data": str(data_path), "rounds": seen, "workers": args.workers,
                        "selection_metric": "efficiency - 0.01*fake - 0.02*clone",
                        "model_threshold": metadata.get("edge_threshold"), "partial": completed_pairs < len(pairs),
                        "results": sorted(results, key=lambda item: item["utility"], reverse=True),
                    })
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    best = max(results, key=lambda item: item["utility"])
    save_json(output, {
        "features": args.features, "run_tag": args.run_tag, "split": args.split,
        "data": str(data_path), "workers": args.workers, "rounds": seen,
        "selection_metric": "efficiency - 0.01*fake - 0.02*clone",
        "model_threshold": metadata.get("edge_threshold"), "best": best, "results": results,
    })
    print("推荐阈值：", best)
    print("报告：", output)


if __name__ == "__main__":
    main()
