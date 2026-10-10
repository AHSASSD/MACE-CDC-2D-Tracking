from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix


NUMERIC_FEATURES = [
    "hits", "rms", "dca", "consensus_iou",
    "mean_edge_probability", "min_edge_probability", "min_internal_support",
    "mean_noise_probability", "max_noise_probability", "mean_uncertainty",
    "loo_mean_heldout_residual", "loo_max_heldout_residual",
    "nearest_baseline_mean_residual", "nearest_baseline_max_residual",
    "nearest_baseline_compatible_fraction", "nearest_baseline_track_hits",
]
LOG_FEATURES = {
    "dca", "loo_mean_heldout_residual", "loo_max_heldout_residual",
    "nearest_baseline_mean_residual", "nearest_baseline_max_residual",
}
SOURCES = ["classic_consensus", "classic_ca", "classic_hough"]
CLASSES = ["accept", "fragment", "reject"]


def target(row):
    if row["audit_category"] == "new_match":
        return "accept"
    if row["audit_category"] == "unmatched_fragment":
        return "fragment"
    return "reject"


def features(frame):
    columns = []
    for name in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=0.0)
        if name in LOG_FEATURES:
            values = np.log1p(np.maximum(values, 0.0))
        columns.append(values)
    for source in SOURCES:
        columns.append((frame["source"].to_numpy() == source).astype(np.float64))
    return np.column_stack(columns)


def binary_stats(labels, selected, positive):
    truth = labels == positive
    tp = int(np.sum(selected & truth))
    fp = int(np.sum(selected & ~truth))
    fn = int(np.sum(~selected & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "selected": int(np.sum(selected)), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def select_threshold(labels, probabilities, positive, policy):
    evaluated = []
    for threshold in np.arange(0.20, 0.951, 0.025):
        stats = binary_stats(labels, probabilities >= threshold, positive)
        evaluated.append({"threshold": float(round(threshold, 3)), **stats})
    if policy == "utility":
        eligible = [value for value in evaluated if value["selected"] >= 5]
        return max(eligible, key=lambda value: (value["tp"] - value["fp"], value["tp"], value["precision"]))
    return max(evaluated, key=lambda value: (value["f1"], value["precision"], value["recall"]))


def probability_column(model, probabilities, name):
    return probabilities[:, list(model.classes_).index(name)]


def evaluate(model, frame, accept_threshold, fragment_threshold):
    labels = frame["target"].to_numpy()
    probabilities = model.predict_proba(features(frame))
    accept_probability = probability_column(model, probabilities, "accept")
    fragment_probability = probability_column(model, probabilities, "fragment")
    accept = accept_probability >= accept_threshold
    fragment = ~accept & (fragment_probability >= fragment_threshold)
    decisions = np.full(len(frame), "reject", dtype=object)
    decisions[fragment] = "fragment"
    decisions[accept] = "accept"
    return {
        "graphs": int(frame["graph_index"].nunique()),
        "candidates": len(frame),
        "accept": binary_stats(labels, accept, "accept"),
        "fragment": binary_stats(labels, fragment, "fragment"),
        "confusion_labels": CLASSES,
        "confusion_matrix": confusion_matrix(labels, decisions, labels=CLASSES).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="训练经典残差候选三分类裁决器")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--train-events", type=int, default=2400)
    parser.add_argument("--calibration-events", type=int, default=600)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.candidate_csv)
    frame["target"] = frame.apply(target, axis=1)
    calibration_end = args.train_events + args.calibration_events
    train = frame[frame["graph_index"] < args.train_events].copy()
    calibration = frame[
        (frame["graph_index"] >= args.train_events) & (frame["graph_index"] < calibration_end)
    ].copy()
    locked = frame[frame["graph_index"] >= calibration_end].copy()
    if min(len(train), len(calibration), len(locked)) == 0:
        raise ValueError("train/calibration/locked split contains no candidates")

    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
        min_samples_leaf=12, l2_regularization=1.0,
        class_weight="balanced", random_state=20260901,
    )
    model.fit(features(train), train["target"].to_numpy())
    calibration_probability = model.predict_proba(features(calibration))
    calibration_labels = calibration["target"].to_numpy()
    accept_rule = select_threshold(
        calibration_labels, probability_column(model, calibration_probability, "accept"),
        "accept", "utility",
    )
    fragment_rule = select_threshold(
        calibration_labels, probability_column(model, calibration_probability, "fragment"),
        "fragment", "f1",
    )

    bundle = {
        "model": model, "numeric_features": NUMERIC_FEATURES, "log_features": sorted(LOG_FEATURES),
        "sources": SOURCES, "accept_threshold": accept_rule["threshold"],
        "fragment_threshold": fragment_rule["threshold"],
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_output)
    report = {
        "candidate_csv": str(args.candidate_csv.resolve()),
        "split": {
            "train_graphs": [0, args.train_events - 1],
            "calibration_graphs": [args.train_events, calibration_end - 1],
            "locked_graphs": [calibration_end, int(frame["graph_index"].max())],
        },
        "class_counts": {
            name: frame["target"].value_counts().to_dict()
            for name, frame in (("train", train), ("calibration", calibration), ("locked", locked))
        },
        "selected_rules": {"accept": accept_rule, "fragment": fragment_rule},
        "calibration": evaluate(
            model, calibration, accept_rule["threshold"], fragment_rule["threshold"]
        ),
        "locked": evaluate(model, locked, accept_rule["threshold"], fragment_rule["threshold"]),
        "model_file": str(args.model_output.resolve()),
        "contract": "truth defines offline targets only; model inputs are candidate-level online observables",
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
