from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import (
    AI3_SPLIT_FILE, DATASET_META_FILE, DEFAULT_NOISE_RATE_HZ, DEFAULT_SIGMA_T_NS,
    RANDOM_SEED, SPLIT_FILE, TEST_DATA, TRAIN_DATA, VAL_DATA, build_graph,
    load_axial_hits, load_cell_map, prepare_trigger, save_json,
)


def sample_target(rng: np.random.Generator, mean_tracks: float, minimum: int, maximum: int) -> int:
    while True:
        value = int(rng.poisson(mean_tracks))
        if minimum <= value <= maximum:
            return value


def sample_events(event_pool, tracks_per_event, target, rng):
    selected = []
    actual = 0
    for evt in rng.permutation(event_pool):
        selected.append(int(evt))
        actual += int(tracks_per_event[int(evt)])
        if actual >= target:
            break
    return selected, actual


_WORKER = {}


def _init_worker(frame, cell_map, event_pool, config):
    # 每个进程只初始化一次；任务本身只传整数索引，避免反复序列化大表。
    # 多进程生成时关闭每个进程内部的线程扩张，避免 40 个进程相互争抢 CPU。
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER.update(
        frame=frame,
        cell_map=cell_map,
        event_pool=np.asarray(event_pool),
        tracks_per_event=frame.groupby("EvtID")["TrkID"].nunique().astype(int).to_dict(),
        config=config,
    )


def _make_one(sample_index):
    frame = _WORKER["frame"]
    cell_map = _WORKER["cell_map"]
    config = _WORKER["config"]
    rng = np.random.default_rng(int(config["seed"]) + int(sample_index) * 1009)
    target = sample_target(rng, config["mean_tracks"], config["min_tracks"], config["max_tracks"])
    events, actual = sample_events(_WORKER["event_pool"], _WORKER["tracks_per_event"], target, rng)
    source = frame[frame["EvtID"].isin(events)].reset_index(drop=True)
    trigger = prepare_trigger(source, cell_map, config["noise_rate_hz"], config["sigma_t_ns"], rng)
    graph = build_graph(trigger)
    if graph is None:
        return None
    graph.target_tracks = torch.tensor([target], dtype=torch.long)
    graph.actual_tracks = torch.tensor([actual], dtype=torch.long)
    graph.source_event_count = torch.tensor([len(events)], dtype=torch.long)
    return graph, {
        "tracks": actual,
        "nodes": graph.num_nodes,
        "noise_nodes": int(graph.is_noise.sum()),
        "shared_nodes": int(graph.shared_hit.sum()),
    }


def make_dataset(frame, cell_map, event_pool, count, config, label, workers, output_base, chunk_size):
    print(f"生成 {label}: {count} 张图，CPU进程数={workers}")
    parts_dir = output_base.with_suffix(output_base.suffix + ".parts")
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    chunk = []
    part_index = 0
    # Windows 使用 spawn；initializer 保证每个子进程只接收一次输入表。
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(frame, cell_map, event_pool, config)) as pool:
        for result in tqdm(pool.map(_make_one, range(count), chunksize=1), total=count):
            if result is not None:
                chunk.append(result[0])
                stats.append(result[1])
                if len(chunk) >= chunk_size:
                    torch.save(chunk, parts_dir / f"part_{part_index:05d}.pt")
                    chunk = []
                    part_index += 1
    if chunk:
        torch.save(chunk, parts_dir / f"part_{part_index:05d}.pt")
        part_index += 1
    # 主路径保留为分片目录的清晰提示，避免误把不完整单文件当成数据集。
    output_base.with_suffix(output_base.suffix + ".manifest.json").write_text(
        json.dumps({"parts_dir": str(parts_dir), "parts": part_index, "graphs": len(stats)}, indent=2),
        encoding="utf-8",
    )
    return stats


def summarize(stats):
    return {
        "graphs": len(stats),
        "mean_tracks": float(np.mean([x["tracks"] for x in stats])) if stats else 0.0,
        "mean_nodes": float(np.mean([x["nodes"] for x in stats])) if stats else 0.0,
        "mean_noise_nodes": float(np.mean([x["noise_nodes"] for x in stats])) if stats else 0.0,
        "shared_nodes": int(sum(x["shared_nodes"] for x in stats)),
    }


def main():
    parser = argparse.ArgumentParser(description="生成带死时间和噪声的 AI工程化4 泊松数据集")
    parser.add_argument("--train-graphs", type=int, default=4000)
    parser.add_argument("--val-graphs", type=int, default=1000)
    parser.add_argument("--test-graphs", type=int, default=1000)
    parser.add_argument("--mean-tracks", type=float, default=30.0)
    parser.add_argument("--min-tracks", type=int, default=5)
    parser.add_argument("--max-tracks", type=int, default=60)
    parser.add_argument("--noise-rate-hz", type=float, default=DEFAULT_NOISE_RATE_HZ)
    parser.add_argument("--sigma-t-ns", type=float, default=DEFAULT_SIGMA_T_NS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--tag", default="noise100k_sigma3", help="输出文件标签；扫描配置时用于避免覆盖")
    default_workers = min(12, max(1, os.cpu_count() or 1))
    parser.add_argument("--workers", type=int, default=int(os.getenv("MACE_DATA_WORKERS", default_workers)), help=f"数据生成进程数，默认 {default_workers}")
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("MACE_DATA_CHUNK_SIZE", "100")), help="每个分片保存的图数量，默认 100")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers 必须 >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size 必须 >= 1")

    if not SPLIT_FILE.exists():
        if not AI3_SPLIT_FILE.exists():
            raise FileNotFoundError("AI3 event_split.json is required to preserve the independent split")
        shutil.copyfile(AI3_SPLIT_FILE, SPLIT_FILE)
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    frame = load_axial_hits()
    cell_map = load_cell_map()
    pools = {
        "train": np.asarray(split["train_evt_ids"], dtype=int),
        "validation": np.asarray(split["val_evt_ids"], dtype=int),
        "test": np.asarray(split["test_evt_ids"], dtype=int),
    }
    counts = {"train": args.train_graphs, "validation": args.val_graphs, "test": args.test_graphs}
    if args.tag == "noise100k_sigma3":
        paths = {"train": TRAIN_DATA, "validation": VAL_DATA, "test": TEST_DATA}
        metadata_path = DATASET_META_FILE
    else:
        paths = {
            "train": Path(__file__).with_name(f"dataset_train_{args.tag}.pt"),
            "validation": Path(__file__).with_name(f"dataset_val_{args.tag}.pt"),
            "test": Path(__file__).with_name(f"dataset_test_{args.tag}.pt"),
        }
        metadata_path = Path(__file__).with_name(f"dataset_metadata_{args.tag}.json")
    summaries = {}
    for name in ("train", "validation", "test"):
        split_offset = {"train": 0, "validation": 1_000_000, "test": 2_000_000}[name]
        worker_config = {
            "seed": args.seed + split_offset,
            "mean_tracks": args.mean_tracks,
            "min_tracks": args.min_tracks,
            "max_tracks": args.max_tracks,
            "noise_rate_hz": args.noise_rate_hz,
            "sigma_t_ns": args.sigma_t_ns,
        }
        stats = make_dataset(frame, cell_map, pools[name], counts[name], worker_config, name, args.workers, paths[name], args.chunk_size)
        summaries[name] = summarize(stats)
    save_json(metadata_path, {
        "source_hits": str(frame.shape),
        "source_split": str(SPLIT_FILE),
        "seed": args.seed,
        "track_distribution": {"name": "truncated_poisson", "mean": args.mean_tracks, "min": args.min_tracks, "max": args.max_tracks},
        "detector": {"dead_time": "non-paralyzable per CellID, ordered by t_obs", "dead_time_ns": 100.0, "event_window_ns": 300.0},
        "noise": {"rate_hz": args.noise_rate_hz, "per_cell_mean": args.noise_rate_hz * 300e-9, "all_cells_compete": True, "distance_mm": "uniform(0,6)"},
        "timing": {"sigma_t_ns": args.sigma_t_ns, "internal_time": "t_window=t-t0", "measurement": "t_obs=t_window+Normal(0,sigma)", "model_inputs_exclude": ["t0", "tHit", "t_window", "t_rel", "dt_rel"]},
        "distance": {"measurement": "d_obs=abs(d_truth+Normal(0,0.075 mm))", "sigma_mm": 0.075},
        "features": {"time": {"node": 7, "edge": 6}, "no-time": {"node": 6, "edge": 4}, "excluded_truth_inputs": ["is_noise", "shared_hit", "t0", "tHit", "t_window", "t_rel", "dt_rel"]},
        "summary": summaries,
    })
    print("完成：", summaries)
    print("元数据：", metadata_path)


if __name__ == "__main__":
    main()
