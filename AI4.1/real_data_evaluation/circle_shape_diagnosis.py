"""已知真轨成员的圆形状诊断，比较旧验证粒子与新成对子集。"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import fit_2d_circle, load_cell_map

OUT = Path(__file__).resolve().parent / "paired_diagnosis"


def fit(task):
    batch, evt, trk, x, y, d, birth_radius = task
    result = {"batch": batch, "EvtID": evt, "TrkID": trk, "hits": len(x), "birth_radius_mm": birth_radius}
    try:
        circle = fit_2d_circle(x, y, d)
        if circle is None or not np.isfinite(circle).all():
            return {**result, "error": "no_finite_fit"}
        a, b, radius = circle
        rms = float(np.sqrt(np.mean((np.abs(np.hypot(x - a, y - b) - radius) - d) ** 2)))
        return {**result, "rms_mm": rms, "radius_mm": radius,
                "origin_impact_mm": abs(np.hypot(a, b) - radius)}
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return {**result, "error": str(error)}


def main():
    split = json.loads((ROOT.parent / "AI工程化4" / "event_split.json").read_text(encoding="utf-8"))
    new_ids = set(json.loads((OUT / "n200_single_seed20261013" / "time_raw.json").read_text(encoding="utf-8"))["source_event_ids"])
    geometry = load_cell_map()[["CellID", "wire_x", "wire_y"]]
    tasks = []
    for batch, ids in [("old_validation", set(split["val_evt_ids"])), ("new_paired_subset", new_ids)]:
        prefix = "old" if batch.startswith("old") else "new"
        source = pd.read_pickle(OUT / f"{prefix}_filtered_source.pkl")
        physics = pd.read_csv(OUT / f"{prefix}_particle_physics.csv").set_index(["EvtID", "TrkID"])
        source = source.loc[source.EvtID.isin(ids)].merge(geometry, on="CellID", how="left", validate="many_to_one")
        assert not source[["wire_x", "wire_y"]].isna().any().any()
        for (evt, trk), group in source.groupby(["EvtID", "TrkID"]):
            if 7 <= len(group) <= 20:
                tasks.append((batch, int(evt), int(trk), group.wire_x.to_numpy(dtype=np.float32),
                              group.wire_y.to_numpy(dtype=np.float32), group.d.to_numpy(dtype=np.float32),
                              float(physics.loc[(evt, trk), "birth_radius_mm"])))
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(fit, tasks), 1):
            rows.append(row)
            if i % 500 == 0:
                print("Circle truth fit", i, "/", len(tasks), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "circle_shapes.csv", index=False, encoding="utf-8-sig")
    report = {}
    for batch, group in frame.groupby("batch"):
        valid = group.loc[group.origin_impact_mm.notna()]
        report[batch] = {"tracks": len(group), "failed_fits": len(group) - len(valid),
                         "origin_impact_quantiles_mm": np.quantile(valid.origin_impact_mm, [0, .1, .5, .9, 1]).tolist(),
                         "median_rms_mm": float(valid.rms_mm.median()),
                         "fraction_rms_over_020": float(valid.rms_mm.gt(.2).mean()),
                         "fraction_impact_over_15": float(valid.origin_impact_mm.gt(15).mean())}
    (OUT / "circle_shape_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
