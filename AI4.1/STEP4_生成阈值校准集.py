from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from common import DEFAULT_NOISE_RATE_HZ, DEFAULT_SIGMA_T_NS, RANDOM_SEED, ROOT, SPLIT_FILE, load_axial_hits, load_cell_map, save_json
from STEP1_生成AI4数据集 import make_dataset, summarize


def main():
    parser = argparse.ArgumentParser(description="生成独立的 STEP4 阈值校准集（不覆盖固有测试集）")
    parser.add_argument("--graphs", type=int, default=3000)
    parser.add_argument("--tag", default="threshold_calibration_v1")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED + 3_000_000)
    parser.add_argument("--mean-tracks", type=float, default=30.0)
    parser.add_argument("--min-tracks", type=int, default=5)
    parser.add_argument("--max-tracks", type=int, default=60)
    parser.add_argument("--noise-rate-hz", type=float, default=DEFAULT_NOISE_RATE_HZ)
    parser.add_argument("--sigma-t-ns", type=float, default=DEFAULT_SIGMA_T_NS)
    parser.add_argument("--workers", type=int, default=min(12, max(1, os.cpu_count() or 1)))
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()
    if args.graphs < 1 or args.workers < 1 or args.chunk_size < 1:
        raise ValueError("graphs、workers 和 chunk-size 必须 >= 1")

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    frame = load_axial_hits()
    cell_map = load_cell_map()
    # Calibration deliberately reuses only the validation event pool with a new
    # composition/noise seed. The original test event pool remains untouched.
    event_pool = np.asarray(split["val_evt_ids"], dtype=int)
    output = ROOT / f"dataset_calibration_{args.tag}.pt"
    config = {
        "seed": args.seed,
        "mean_tracks": args.mean_tracks,
        "min_tracks": args.min_tracks,
        "max_tracks": args.max_tracks,
        "noise_rate_hz": args.noise_rate_hz,
        "sigma_t_ns": args.sigma_t_ns,
    }
    stats = make_dataset(frame, cell_map, event_pool, args.graphs, config, "calibration", args.workers, output, args.chunk_size)
    metadata = ROOT / f"dataset_calibration_{args.tag}.json"
    save_json(metadata, {
        "purpose": "threshold calibration only; not an independent test set",
        "source_pool": "validation event IDs",
        "original_test_untouched": True,
        "seed": args.seed,
        "noise_rate_hz": args.noise_rate_hz,
        "sigma_t_ns": args.sigma_t_ns,
        "summary": summarize(stats),
        "dataset": str(output),
    })
    print("校准集：", output)
    print("元数据：", metadata)


if __name__ == "__main__":
    main()
