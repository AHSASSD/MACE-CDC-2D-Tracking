from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report


class _TripletNet(nn.Module):
    def __init__(self, input_dim, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)


class TripletFocalClassifier:
    """Small predict_proba-compatible classifier trained with multiclass focal loss."""
    def __init__(self, input_dim, gamma=2.5, epochs=5, batch_size=8192, seed=20260904):
        self.input_dim = int(input_dim)
        self.gamma = float(gamma)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.classes_ = np.arange(3, dtype=np.int64)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = str(device)
        self.model_ = _TripletNet(self.input_dim).to(device)
        counts = np.bincount(y.astype(np.int64), minlength=3).astype(np.float64)
        # Alpha is inverse-frequency but clipped to avoid making rare labels
        # dominate completely when the generated graph distribution shifts.
        alpha = counts.sum() / np.maximum(counts, 1.0)
        alpha = alpha / alpha.mean()
        alpha = np.clip(alpha, 0.75, 8.0).astype(np.float32)
        self.alpha_ = alpha
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=2e-3, weight_decay=1e-4)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.model_.train()
        for epoch in range(self.epochs):
            order = rng.permutation(len(y))
            running = 0.0
            for start in range(0, len(y), self.batch_size):
                idx = order[start:start + self.batch_size]
                xb = torch.from_numpy(X[idx]).to(device)
                yb = torch.from_numpy(y[idx]).to(device)
                logits = self.model_(xb)
                logp = F.log_softmax(logits, dim=1)
                p = logp.exp()
                target_logp = logp.gather(1, yb[:, None]).squeeze(1)
                target_p = p.gather(1, yb[:, None]).squeeze(1)
                alpha_t = torch.as_tensor(self.alpha_, device=device)[yb]
                loss = (-alpha_t * (1.0 - target_p).pow(self.gamma) * target_logp).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += float(loss.detach()) * len(idx)
            print(f"epoch {epoch + 1}/{self.epochs} loss={running / max(len(y), 1):.6f}", flush=True)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        device = next(self.model_.parameters()).device
        values = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                logits = self.model_(torch.from_numpy(X[start:start + self.batch_size]).to(device))
                values.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(values, axis=0) if values else np.empty((0, 3), dtype=np.float32)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--report-output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--gamma", type=float, default=2.5)
    args = p.parse_args()
    payload = torch.load(args.data, weights_only=False)
    X = payload["X"].numpy(); y = payload["y"].numpy(); groups = payload["group"].numpy()
    cutoff = np.quantile(groups, 0.75)
    train = groups < cutoff
    clf = TripletFocalClassifier(X.shape[1], gamma=args.gamma, epochs=args.epochs)
    clf.fit(X[train], y[train])
    proba = clf.predict_proba(X[~train])
    pred = clf.classes_[np.argmax(proba, axis=1)]
    # Select a development operating point: maximize accepted true pairs while
    # keeping the accepted fake fraction below 5%.
    best = {"threshold": 0.80, "accepted": 0, "fake_fraction": 1.0}
    for threshold in np.arange(0.50, 0.96, 0.02):
        accepted = np.max(proba[:, :2], axis=1) >= threshold
        if not accepted.any():
            continue
        fake_fraction = float(np.mean(y[~train][accepted] == 2))
        true_count = int(np.sum(accepted & (y[~train] != 2)))
        if fake_fraction <= 0.05 and true_count > best["accepted"]:
            best = {"threshold": float(threshold), "accepted": true_count,
                    "fake_fraction": fake_fraction}
    report = {"classes": [int(v) for v in clf.classes_], "development": best,
              "classification_report": classification_report(y[~train], pred, output_dict=True),
              "samples": {"total": int(len(y)), "train": int(train.sum()), "development": int((~train).sum())}}
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "accept_threshold": best["threshold"],
                 "loss": "multiclass_focal", "gamma": args.gamma,
                 "alpha": clf.alpha_.tolist(),
                 "feature_names": payload.get("feature_names", [])}, args.model_output)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
