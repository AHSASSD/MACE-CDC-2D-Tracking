from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
AI4 = ROOT.parent / "AI工程化4"
sys.path.insert(0, str(AI4))

from common import InteractionNetwork, fit_2d_circle, model_dimensions  # noqa: E402
from STEP4_两级寻迹 import predict  # noqa: E402
from step4_hit_audit.analyze_physics_algorithm_loss import CORE_PROFILE, truth_maps  # noqa: E402
from step4_short_rerun_experiment.run_experiment import run_with_short_rerun  # noqa: E402


def fit_rms(data, indices):
    indices = sorted(set(indices))
    if len(indices) < 5:
        return None
    x = data.wire_x[indices].numpy()
    y = data.wire_y[indices].numpy()
    d = data.drift_d[indices].numpy()
    params = fit_2d_circle(x, y, d)
    if params is None or not np.all(np.isfinite(params)):
        return None
    a, b, radius = params
    residual = np.abs(np.hypot(x - a, y - b) - radius) - d
    return {"params": np.asarray(params), "rms": float(np.sqrt(np.mean(residual**2))), "hits": len(indices)}


def main():
    parser = argparse.ArgumentParser(description="比较 shared 点对候选轨迹拟合的影响")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--features", choices=["time", "no-time"], default="time")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--rms-limit", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graphs = torch.load(args.audit_dir / "graphs.pt", weights_only=False)
    limit = min(args.rounds, len(graphs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionNetwork(*model_dimensions(args.features)).to(device)
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.eval()

    counts = Counter()
    rms_deltas = []
    parameter_deltas = []
    for data in graphs[:limit]:
        probabilities = predict(data, model, device, args.features)
        tracks, _ = run_with_short_rerun(
            data, model, device, args.features, probabilities,
            0.70, 0.575, 0.55, 0.20, CORE_PROFILE,
        )
        for track in tracks:
            indices = track["indices"]
            shared = [i for i in indices if bool(data.shared_hit[i])]
            if not shared:
                counts["tracks_without_shared"] += 1
                continue
            exclusive = [i for i in indices if not bool(data.shared_hit[i])]
            full = fit_rms(data, indices)
            excl = fit_rms(data, exclusive)
            counts["tracks_with_shared"] += 1
            counts["shared_nodes"] += len(shared)
            if excl is None:
                counts["exclusive_fit_unavailable"] += 1
                continue
            counts["exclusive_fit_available"] += 1
            if excl["rms"] <= args.rms_limit:
                counts["exclusive_fit_within_limit"] += 1
            if full is not None:
                rms_deltas.append(full["rms"] - excl["rms"])
                parameter_deltas.append(float(np.linalg.norm(full["params"] - excl["params"])))
                if abs(full["rms"] - excl["rms"]) > 0.1:
                    counts["rms_change_over_0_1"] += 1

    report = {
        "audit_dir": str(args.audit_dir.resolve()),
        "model_file": str(args.model_file.resolve()),
        "graphs": limit,
        "rms_limit": args.rms_limit,
        "counts": dict(counts),
        "mean_rms_delta_full_minus_exclusive": float(np.mean(rms_deltas)) if rms_deltas else 0.0,
        "median_abs_rms_delta": float(np.median(np.abs(rms_deltas))) if rms_deltas else 0.0,
        "mean_abs_parameter_delta": float(np.mean(parameter_deltas)) if parameter_deltas else 0.0,
        "p95_abs_parameter_delta": float(np.percentile(np.abs(parameter_deltas), 95)) if parameter_deltas else 0.0,
        "contract": "shared nodes are excluded only for this offline fit ablation; online candidate generation is unchanged",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
