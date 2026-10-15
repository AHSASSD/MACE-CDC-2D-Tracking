from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parent
MAP_FILE = Path(os.environ.get("MACE_CDC_CELL_MAP", "cdc_cell_map.xlsx"))
HIT_FILE = Path(os.environ.get("MACE_CDC_HITS", "CDCSimHit.xlsx"))
AI3_SPLIT_FILE = ROOT.parent / "AI工程化3" / "event_split.json"
SPLIT_FILE = ROOT / "event_split.json"

TRAIN_DATA = ROOT / "dataset_train_noise100k_sigma3.pt"
VAL_DATA = ROOT / "dataset_val_noise100k_sigma3.pt"
TEST_DATA = ROOT / "dataset_test_noise100k_sigma3.pt"
DATASET_META_FILE = ROOT / "dataset_metadata_noise100k_sigma3.json"

RANDOM_SEED = 20260823
SEARCH_DIST_MM = 120.0
EVENT_WINDOW_NS = 300.0
DEFAULT_NOISE_RATE_HZ = 100_000.0
DEFAULT_SIGMA_T_NS = 3.0
DRIFT_DISTANCE_SIGMA_MM = 0.075
DEAD_TIME_NS = 100.0
# Only quantities available online may enter the GNN.  is_noise and
# shared_hit remain offline truth/evaluation fields, never model inputs.
NODE_TIME_DIM = 7
NODE_NO_TIME_DIM = 6
EDGE_TIME_DIM = 6
EDGE_NO_TIME_DIM = 4


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_graph_dataset(path: Path):
    """Load a legacy monolithic dataset or the memory-bounded STEP1 shards."""
    parts_dir = path.with_suffix(path.suffix + ".parts")
    if parts_dir.is_dir():
        parts = sorted(parts_dir.glob("part_*.pt"))
        if not parts:
            raise FileNotFoundError(f"Dataset shard directory is empty: {parts_dir}")
        graphs = []
        for part in parts:
            graphs.extend(torch.load(part, weights_only=False))
        return graphs
    return torch.load(path, weights_only=False)


def load_cell_map() -> pd.DataFrame:
    frame = pd.read_excel(MAP_FILE).rename(columns={"x": "wire_x", "y": "wire_y"})
    frame = frame[frame["u"].eq(0) & frame["v"].eq(0)].copy()
    frame["wire_r"] = np.hypot(frame["wire_x"], frame["wire_y"])
    return frame.reset_index(drop=True)


def load_axial_hits() -> pd.DataFrame:
    """Load the clean axial truth sample, including offline-only timing columns."""
    cell_map = load_cell_map()
    hits = pd.read_excel(HIT_FILE)
    hits = hits[hits["Good"].eq(True) & hits["CreatProc"].astype(str).eq("ExtendedDecayWithSpin")]
    columns = ["CellID", "wire_x", "wire_y", "wire_r"]
    merged = hits.merge(cell_map[columns], on="CellID", how="inner")
    required = ["EvtID", "TrkID", "CellID", "d", "t", "t0", "tHit", "wire_x", "wire_y", "wire_r"]
    if merged[required].isna().any().any():
        raise ValueError("AI4 source contains missing geometry, truth, distance, or timing values")
    return merged.reset_index(drop=True)


def apply_dead_time(frame: pd.DataFrame, dead_time_ns: float = DEAD_TIME_NS) -> pd.DataFrame:
    """Apply non-paralyzable per-cell dead time using observed rising-edge time."""
    if frame.empty:
        return frame.copy()
    rows = []
    for _, group in frame.groupby("CellID", sort=False):
        last_accepted = -np.inf
        for _, hit in group.sort_values("t_obs", kind="stable").iterrows():
            if float(hit["t_obs"]) - last_accepted >= float(dead_time_ns):
                rows.append(hit.copy())
                last_accepted = float(hit["t_obs"])
    return pd.DataFrame(rows).reset_index(drop=True)


def inject_noise(
    real_candidates: pd.DataFrame,
    cell_map: pd.DataFrame,
    noise_rate_hz: float,
    sigma_t_ns: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add Poisson noise candidates to every cell on the same observed time axis."""
    mean_noise = float(noise_rate_hz) * EVENT_WINDOW_NS * 1e-9
    multiplicity = rng.poisson(mean_noise, len(cell_map))
    repeated = np.repeat(np.arange(len(cell_map)), multiplicity)
    noise = cell_map.iloc[repeated][["CellID", "wire_x", "wire_y", "wire_r"]].copy().reset_index(drop=True)
    noise_count = int(multiplicity.sum())
    if noise_count:
        # Synthetic trigger has no source-event identity for noise.
        noise["EvtID"] = -1
        noise["TrkID"] = -1
        noise["d"] = rng.uniform(0.0, 6.0, noise_count)
        noise["t_window"] = rng.uniform(0.0, EVENT_WINDOW_NS, noise_count)
        noise["t_obs"] = noise["t_window"] + rng.normal(0.0, float(sigma_t_ns), noise_count)
        noise["t0"] = np.nan
        noise["tHit"] = np.nan
        noise["truth_tracks"] = [[] for _ in range(noise_count)]
        noise["shared_hit"] = False
        noise["is_noise"] = True
        noise["noise_multiplicity"] = 1
        combined = pd.concat([real_candidates, noise], ignore_index=True, sort=False)
    else:
        combined = real_candidates.copy().reset_index(drop=True)
    return combined[combined["t_obs"].between(0.0, EVENT_WINDOW_NS, inclusive="left")].reset_index(drop=True)


def prepare_trigger(
    source_hits: pd.DataFrame,
    cell_map: pd.DataFrame,
    noise_rate_hz: float = DEFAULT_NOISE_RATE_HZ,
    sigma_t_ns: float = DEFAULT_SIGMA_T_NS,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    rng = rng or np.random.default_rng(RANDOM_SEED)
    real = source_hits.copy().reset_index(drop=True)
    real["t_window"] = real["t"].to_numpy(dtype=float) - real["t0"].to_numpy(dtype=float)
    real["t_obs"] = real["t_window"] + rng.normal(0.0, float(sigma_t_ns), len(real))
    real["d"] = np.abs(real["d"].to_numpy(dtype=float) + rng.normal(0.0, DRIFT_DISTANCE_SIGMA_MM, len(real)))
    real["truth_tracks"] = [[(int(evt), int(trk))] for evt, trk in zip(real["EvtID"], real["TrkID"])]
    real["shared_hit"] = False
    real["is_noise"] = False
    real["noise_multiplicity"] = 0
    candidates = inject_noise(real, cell_map, noise_rate_hz, sigma_t_ns, rng)
    return apply_dead_time(candidates)


def node_features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    phi = np.arctan2(frame["wire_y"].to_numpy(), frame["wire_x"].to_numpy())
    geometry = np.column_stack([
        frame["wire_x"].to_numpy(dtype=float) / 800.0,
        frame["wire_y"].to_numpy(dtype=float) / 800.0,
        frame["wire_r"].to_numpy(dtype=float) / 800.0,
        frame["d"].to_numpy(dtype=float) / 10.0,
        np.sin(phi),
        np.cos(phi),
    ])
    no_time = geometry.astype(np.float32)
    time = np.column_stack([
        geometry,
        frame["t_obs"].to_numpy(dtype=float) / EVENT_WINDOW_NS,
    ]).astype(np.float32)
    return time, no_time


def candidate_edges(time_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords_mm = time_features[:, :2] * 800.0
    pairs = cKDTree(coords_mm).query_pairs(SEARCH_DIST_MM, output_type="ndarray")
    if len(pairs):
        separation = np.linalg.norm(coords_mm[pairs[:, 0]] - coords_mm[pairs[:, 1]], axis=1)
        pairs = pairs[separation > 0]
    if len(pairs) == 0:
        empty_index = np.empty((2, 0), dtype=np.int64)
        return empty_index, np.empty((0, EDGE_TIME_DIM), np.float32), np.empty((0, EDGE_NO_TIME_DIM), np.float32)
    directed = np.vstack([pairs, pairs[:, ::-1]])
    row, col = directed[:, 0], directed[:, 1]
    r = time_features[:, 2]
    phi = np.arctan2(time_features[:, 4], time_features[:, 5])
    dr = r[col] - r[row]
    dphi = (phi[col] - phi[row] + np.pi) % (2 * np.pi) - np.pi
    dx = time_features[col, 0] - time_features[row, 0]
    dy = time_features[col, 1] - time_features[row, 1]
    scaled_dist = np.hypot(dx, dy)
    geometry = np.column_stack([dr, scaled_dist, dphi, dphi / (scaled_dist + 1e-6)])
    dt_obs = time_features[col, 6] - time_features[row, 6]
    with_time = np.column_stack([geometry, dt_obs, np.abs(dt_obs)]).astype(np.float32)
    return directed.T.astype(np.int64), with_time, geometry.astype(np.float32)


def _truth_storage(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs: list[tuple[int, int]] = []
    ptr = [0]
    for tracks in frame["truth_tracks"]:
        pairs.extend((int(evt), int(trk)) for evt, trk in tracks)
        ptr.append(len(pairs))
    pair_tensor = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    all_pairs = sorted(set(pairs))
    all_tensor = torch.tensor(all_pairs, dtype=torch.long).reshape(-1, 2)
    return torch.tensor(ptr, dtype=torch.long), pair_tensor, all_tensor


def build_graph(frame: pd.DataFrame) -> Data | None:
    if len(frame) < 5:
        return None
    time_features, _ = node_features(frame)
    edge_index, edge_time, _ = candidate_edges(time_features)
    if edge_index.shape[1] == 0:
        return None
    truth_sets = [set(value) for value in frame["truth_tracks"]]
    labels = np.fromiter(
        (bool(truth_sets[int(u)] & truth_sets[int(v)]) for u, v in edge_index.T),
        dtype=np.bool_, count=edge_index.shape[1],
    )
    ptr, pairs, _ = _truth_storage(frame)
    return Data(
        x=torch.from_numpy(time_features),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_time),
        y=torch.from_numpy(labels.astype(np.float32)),
        wire_x=torch.tensor(frame["wire_x"].to_numpy(dtype=np.float32)),
        wire_y=torch.tensor(frame["wire_y"].to_numpy(dtype=np.float32)),
        drift_d=torch.tensor(frame["d"].to_numpy(dtype=np.float32)),
        cell_id=torch.tensor(frame["CellID"].to_numpy(dtype=np.int64)),
        is_noise=torch.tensor(frame["is_noise"].to_numpy(dtype=np.bool_)),
        shared_hit=torch.tensor(frame["shared_hit"].to_numpy(dtype=np.bool_)),
        truth_ptr=ptr,
        truth_pairs=pairs,
    )


def select_features(data: Data, variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    if variant == "time":
        return data.x, data.edge_attr
    if variant == "no-time":
        return data.x[:, :6], data.edge_attr[:, :4]
    raise ValueError(f"Unknown feature variant: {variant}")


class InteractionNetwork(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int = 256, n_passes: int = 8):
        super().__init__()
        self.n_passes = n_passes
        self.relational_model = nn.Sequential(
            nn.Linear(node_in * 2 + edge_in, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.object_model = nn.Sequential(
            nn.Linear(node_in + hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, node_in),
        )
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        effects = None
        for _ in range(self.n_passes):
            effects = self.relational_model(torch.cat([x[row], x[col], edge_attr], dim=-1))
            messages = torch.zeros(x.size(0), effects.size(1), device=x.device, dtype=effects.dtype)
            messages.index_add_(0, col, effects)
            x = self.object_model(torch.cat([x, messages], dim=-1)) + x
        return self.edge_classifier(effects).squeeze(-1)


def model_dimensions(variant: str) -> tuple[int, int]:
    return (NODE_TIME_DIM, EDGE_TIME_DIM) if variant == "time" else (NODE_NO_TIME_DIM, EDGE_NO_TIME_DIM)


def fit_2d_circle(x: np.ndarray, y: np.ndarray, d: np.ndarray) -> tuple[float, float, float] | None:
    if len(x) < 3:
        return None
    x_mean, y_mean = float(x.mean()), float(y.mean())
    covariance = np.cov(x - x_mean, y - y_mean)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, np.argmax(values)]
    normal = np.array([-direction[1], direction[0]])

    def residual(params):
        a, b, radius = params
        return np.abs(np.hypot(x - a, y - b) - radius) - d

    best_guess = [x_mean, y_mean, 1000.0]
    best_loss = float("inf")
    for curvature in [-0.02, -0.005, -0.001, -0.0001, 0.0001, 0.001, 0.005, 0.02]:
        radius = abs(1.0 / curvature)
        guess = [x_mean + normal[0] / curvature, y_mean + normal[1] / curvature, radius]
        candidate = least_squares(residual, guess, ftol=1e-2, xtol=1e-2)
        loss = float(np.sum(candidate.fun ** 2))
        if loss < best_loss:
            best_loss, best_guess = loss, candidate.x
    result = least_squares(
        residual, best_guess, loss="soft_l1",
        bounds=([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, np.inf]),
        ftol=1e-6, xtol=1e-6,
    )
    return tuple(float(v) for v in result.x)
