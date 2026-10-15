"""按 EvtID 对实际 Excel hit 做 time/no-time 两级寻迹，保存每事件 PNG。"""
# Historical Excel timing contract t-t0; do not use this branch to evaluate the whole-event time33 model.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import torch

from common import ROOT, MAP_FILE, InteractionNetwork, build_graph, model_dimensions
from STEP4_两级寻迹 import predict, run_two_pass
from STEP4_剩余混轨恢复 import recover_remaining
from step4_short_core_experiment.run_experiment import run_with_short_core


def prepare_hits(hits, geometry):
    """保留原始行身份；筛选有效直丝及目标物理过程。"""
    hits = hits.copy()
    hits.columns = [str(c).strip() for c in hits.columns]
    aliases = {c.lower(): c for c in hits.columns}
    required = ["EvtID", "HitID", "CellID", "d", "WireType", "Good", "CreatProc"]
    if any(c.lower() not in aliases for c in required):
        raise ValueError(f"Excel 必须包含列：{required}")
    hits = hits.rename(columns={aliases[c.lower()]: c for c in required})
    hits["excel_row"] = np.arange(len(hits)) + 2
    original = hits.copy()
    hits = hits.loc[hits["Good"].eq(True)
                    & hits["WireType"].astype(str).str.strip().str.lower().eq("axial")
                    & hits["CreatProc"].eq("ExtendedDecayWithSpin")].copy()
    for column in ("EvtID", "HitID", "CellID", "d"):
        hits[column] = pd.to_numeric(hits[column], errors="raise")
        if not np.isfinite(hits[column]).all():
            raise ValueError(f"{column} 含缺失或非有限值")
    for column in ("EvtID", "HitID", "CellID"):
        if (hits[column] % 1 != 0).any():
            raise ValueError(f"{column} 必须是整数")
        hits[column] = hits[column].astype("int64")
    if hits["d"].lt(0).any():
        raise ValueError("漂移距离 d 不能为负数，单位应为 mm")
    wire_type = hits["WireType"].astype(str).str.strip().str.lower()
    if not wire_type.isin(["axial", "stereo"]).all():
        raise ValueError("WireType 只支持 Axial / Stereo")
    geometry = geometry.rename(columns={"x": "wire_x", "y": "wire_y"})
    geometry = geometry[["CellID", "wire_x", "wire_y", "u", "v"]]
    axial = hits.loc[wire_type.eq("axial")].merge(
        geometry, on="CellID", how="left", validate="many_to_one", sort=False,
    )
    if not np.isfinite(axial[["wire_x", "wire_y", "u", "v"]].to_numpy(dtype=float)).all():
        raise ValueError("直丝 hit 缺少有效几何坐标，请检查 CellID 和几何表")
    if not (axial["u"].eq(0) & axial["v"].eq(0)).all():
        raise ValueError("WireType=Axial 与几何表的 u=v=0 不一致")
    axial["wire_r"] = np.hypot(axial["wire_x"], axial["wire_y"])
    # no-time 不读取时间；占位字段仅用于复用原有建图接口，不添加噪声或真值。
    axial["t_obs"] = 0.0
    axial["truth_tracks"] = [[] for _ in range(len(axial))]
    axial["is_noise"] = False
    axial["shared_hit"] = False
    return original, axial


def track_event(frame, model, device, strict_threshold, relaxed_threshold, variant="no-time", recipe=None):
    frame = frame.reset_index(drop=True)
    graph = build_graph(frame)
    if graph is None:
        return [], "fewer_than_5_axial_hits" if len(frame) < 5 else "no_candidate_edges"
    recipe = recipe or {}
    probabilities = predict(graph, model, device, variant)
    if recipe.get("short_core"):
        profile = {"skip_parent_rms": .20, "max_core_rms": .80, "min_rms_gain": .10,
                   "min_probability_gain": .05, "min_coverage": .40}
        tracks, _ = run_with_short_core(graph, probabilities, strict_threshold, relaxed_threshold, profile)
    else:
        tracks, _ = run_two_pass(graph, None, None, variant, strict_threshold, relaxed_threshold, probabilities)
    if recipe.get("recover_remaining"):
        tracks = recover_remaining(graph, tracks, probabilities,
                                   min_probability=recipe.get("recover_min_probability", .92))
    return tracks, "ok" if tracks else "no_accepted_circles"


def draw_event(evtid, frame, tracks, output, variant="no-time"):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(frame.wire_x, frame.wire_y, c="0.65", s=15, label="Axial wire centers", zorder=3)
    for number, track in enumerate(tracks, 1):
        color = plt.get_cmap("tab20")((number - 1) % 20)
        members = frame.iloc[track["indices"]]
        a, b, radius = track["fit"]
        ax.add_patch(Circle((a, b), radius, fill=False, color=color, linewidth=1.5,
                            label=f"C{number}: {len(members)} hits, RMS={track['rms']:.3f} mm"))
        ax.scatter(members.wire_x, members.wire_y, color=color, s=20, zorder=4)
        for hit in members.itertuples():
            ax.add_patch(Circle((hit.wire_x, hit.wire_y), hit.d, fill=False,
                                color=color, linewidth=0.6, alpha=0.55))
    extent = max(1.0, float(frame.wire_r.max()) + 25.0) if len(frame) else 100.0
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel="x (mm)", ylabel="y (mm)",
           title=f"EvtID {evtid} | {variant} | {len(frame)} axial hits | {len(tracks)} circles")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def event_result(evtid, frame, tracks, status, input_count):
    circles = []
    used = set()
    for number, track in enumerate(tracks, 1):
        indices = track["indices"]
        used.update(indices)
        a, b, radius = track["fit"]
        circles.append({
            "circle_id": number, "center_x_mm": a, "center_y_mm": b, "radius_mm": radius,
            "rms_mm": track["rms"], "stage": track["stage"],
            "hits": frame.iloc[indices][["HitID", "CellID", "excel_row", "wire_x", "wire_y", "d"]].to_dict("records"),
        })
    return {"EvtID": int(evtid), "input_hits": int(input_count), "axial_hits": len(frame),
            "status": status, "circles": circles,
            "unassigned_hits": frame.iloc[[i for i in range(len(frame)) if i not in used]][
                ["HitID", "CellID", "excel_row", "wire_x", "wire_y", "d"]].to_dict("records")}


def main(export_parameters=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "test.xlsx")
    parser.add_argument("--features", choices=["time", "no-time"], default="no-time")
    parser.add_argument("--baseline", action="store_true", help="使用原模型 metadata 阈值，关闭改进")
    parser.add_argument("--event-ids", nargs="+", type=int)
    parser.add_argument("--cell-map", type=Path, default=MAP_FILE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--strict-threshold", type=float)
    parser.add_argument("--relaxed-threshold", type=float)
    args = parser.parse_args()
    stem = f"{args.features.replace('-', '_')}_clean_physics_v2"
    mode = "parameters" if export_parameters else "png"
    label = "notime" if args.features == "no-time" else "time"
    args.output_dir = args.output_dir or ROOT / f"test_{label}_{mode}{'_baseline' if args.baseline else ''}"
    metadata = json.loads((ROOT / f"model_metadata_{stem}.json").read_text(encoding="utf-8"))
    recipe = {}
    selected = ROOT / "real_data_evaluation" / "selected_config.json"
    if selected.exists() and not args.baseline:
        configuration = json.loads(selected.read_text(encoding="utf-8"))["models"][args.features]
        if configuration.get("holdout_accepted", False):
            recipe = configuration["recipe"]
    strict = recipe.get("strict", metadata["edge_threshold"]) if args.strict_threshold is None else args.strict_threshold
    relaxed = recipe.get("relaxed", max(.35, strict - .15)) if args.relaxed_threshold is None else args.relaxed_threshold
    if not 0 < relaxed <= strict < 1:
        raise ValueError("必须满足 0 < relaxed-threshold <= strict-threshold < 1")
    hits, axial = prepare_hits(pd.read_excel(args.input), pd.read_excel(args.cell_map))
    if args.event_ids is not None:
        hits = hits.loc[hits.EvtID.isin(args.event_ids)]
        axial = axial.loc[axial.EvtID.isin(args.event_ids)].copy()
    if args.features == "time":
        if not {"t", "t0"}.issubset(axial.columns):
            raise ValueError("time 模型需要 t 和 t0 列，按原约定逐行计算 t-t0")
        axial["t_obs"] = pd.to_numeric(axial.t, errors="raise") - pd.to_numeric(axial.t0, errors="raise")
        if not np.isfinite(axial.t_obs).all():
            raise ValueError("time 输入 t-t0 存在非有限值")
    if hits.empty:
        raise ValueError("输入工作表没有 hit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionNetwork(*model_dimensions(args.features)).to(device)
    model.load_state_dict(torch.load(ROOT / f"best_interaction_network_{stem}.pth",
                                    map_location=device, weights_only=True))
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for evtid, original in hits.groupby("EvtID", sort=False):
        frame = axial.loc[axial.EvtID.eq(evtid)].reset_index(drop=True)
        tracks, status = track_event(frame, model, device, strict, relaxed, args.features, recipe)
        draw_event(evtid, frame, tracks, args.output_dir / f"event_{evtid}.png", args.features)
        result = event_result(evtid, frame, tracks, status, len(original))
        results.append(result)
        print(f"EvtID={evtid}: {len(frame)} axial hits, {len(tracks)} circles ({status})", flush=True)
    report = {"input": str(args.input.resolve()), "cell_map": str(args.cell_map.resolve()),
              "model": stem, "features": args.features, "length_unit": "mm", "recipe": recipe,
              "strict_threshold": strict, "relaxed_threshold": relaxed, "events": results}
    if export_parameters:
        (args.output_dir / "circles_and_hits.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        circles, memberships = [], []
        for event in results:
            for circle in event["circles"]:
                circles.append({"EvtID": event["EvtID"], **{k: v for k, v in circle.items() if k != "hits"},
                                "hit_count": len(circle["hits"])})
                memberships.extend({"EvtID": event["EvtID"], "circle_id": circle["circle_id"], **hit}
                                   for hit in circle["hits"])
            memberships.extend({"EvtID": event["EvtID"], "circle_id": None, **hit}
                               for hit in event["unassigned_hits"])
        pd.DataFrame(circles, columns=["EvtID", "circle_id", "center_x_mm", "center_y_mm", "radius_mm",
                                      "rms_mm", "stage", "hit_count"]).to_csv(
            args.output_dir / "circles.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(memberships, columns=["EvtID", "circle_id", "HitID", "CellID", "excel_row",
                                          "wire_x", "wire_y", "d"]).to_csv(
            args.output_dir / "hit_membership.csv", index=False, encoding="utf-8-sig")
    else:
        summary = {k: v for k, v in report.items() if k != "events"}
        summary["events"] = [{"EvtID": r["EvtID"], "input_hits": r["input_hits"],
                              "axial_hits": r["axial_hits"], "circle_count": len(r["circles"]),
                              "unassigned_count": len(r["unassigned_hits"]), "status": r["status"]}
                             for r in results]
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
