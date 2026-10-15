"""比较两批筛选后粒子的可观测几何和离线物理分布，不作模型输入。"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import HIT_FILE, load_cell_map

OUT = Path(__file__).resolve().parent / "paired_diagnosis"


def vector(text):
    values = np.fromstring(str(text).replace("f", "").replace("F", "").strip("{}[]() "), sep=",")
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("物理向量格式必须是有限三元组")
    return values


def particles(path):
    names = {"CellID", "EvtID", "TrkID", "Good", "CreatProc", "d", "Ek", "Ek0", "PDGID", "p", "p0", "x0", "t", "t0", "tHit"}
    frame = pd.read_excel(path, usecols=lambda c: c in names)
    axial_ids = set(load_cell_map().CellID)
    frame = frame.loc[frame.Good.eq(True) & frame.CreatProc.eq("ExtendedDecayWithSpin") & frame.CellID.isin(axial_ids)]
    groups = frame.sort_values("tHit", kind="stable").groupby(["EvtID", "TrkID"])
    truth = groups.first().copy()
    truth["hit_count"] = groups.size()
    p = np.array([vector(t) for t in truth.p0])
    x = np.array([vector(t) for t in truth.x0])
    truth["birth_radius_mm"] = np.hypot(x[:, 0], x[:, 1])
    truth["birth_z_mm"] = x[:, 2]
    truth["birth_phi"] = np.arctan2(x[:, 1], x[:, 0])
    truth["initial_cos_theta"] = p[:, 2] / np.maximum(np.linalg.norm(p, axis=1), 1e-12)
    truth["initial_phi"] = np.arctan2(p[:, 1], p[:, 0])
    truth["transverse_momentum"] = np.hypot(p[:, 0], p[:, 1])
    truth["mean_d_mm"] = groups.d.mean()
    truth["energy_loss_before_first_axial_MeV"] = truth.Ek0 - truth.Ek
    return frame, truth


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "quantiles": np.quantile(values, [0, .1, .5, .9, 1]).tolist()}


def main():
    OUT.mkdir(exist_ok=True)
    old_frame, old = particles(HIT_FILE)
    new_frame, new = particles(ROOT / "test.xlsx")
    report = {"old_rows": len(old_frame), "new_rows": len(new_frame),
              "old_tracks": len(old), "new_tracks": len(new),
              "pdg_counts": {"old": old.PDGID.value_counts().to_dict(), "new": new.PDGID.value_counts().to_dict()},
              "features": {}, "interpretation": "Distribution differences are observational, not causal attribution"}
    for feature in ["Ek0", "hit_count", "birth_radius_mm", "birth_z_mm", "birth_phi", "initial_cos_theta", "initial_phi", "transverse_momentum", "mean_d_mm", "energy_loss_before_first_axial_MeV"]:
        report["features"][feature] = {"old": describe(old[feature]), "new": describe(new[feature]),
                                        "ks_distance": float(ks_2samp(old[feature], new[feature]).statistic)}
    assert len(new_frame) == 116935
    report["old_source_shape_matches_split_metadata"] = len(old_frame) == 81883
    columns = ["Ek0", "PDGID", "hit_count", "birth_radius_mm", "birth_z_mm", "birth_phi", "initial_cos_theta", "initial_phi", "transverse_momentum", "mean_d_mm", "energy_loss_before_first_axial_MeV"]
    old[columns].to_csv(OUT / "old_particle_physics.csv", encoding="utf-8-sig")
    new[columns].to_csv(OUT / "new_particle_physics.csv", encoding="utf-8-sig")
    old_frame.to_pickle(OUT / "old_filtered_source.pkl")
    new_frame.to_pickle(OUT / "new_filtered_source.pkl")
    (OUT / "source_physics_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"old_tracks": len(old), "new_tracks": len(new), "ks": {k: v["ks_distance"] for k, v in report["features"].items()}}, flush=True)


if __name__ == "__main__":
    main()
