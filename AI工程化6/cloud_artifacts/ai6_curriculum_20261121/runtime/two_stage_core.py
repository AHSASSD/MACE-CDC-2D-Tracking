"""Frozen AI4 geometry/two-pass core, reused with AI6 axial measurements only.
Source SHA256: {"common.py":"876525b9113bb8421d04940598b9d31ec5b1e272f9e294b2172844db409af75b","STEP4_两级寻迹.py":"c5df7984e0aab7c21624773df6883da0577f93434ad3d0a2b8a1444290cf93e7"}
The AI6 caller supplies precomputed undirected probabilities; no legacy model,
time features, training data or truth are loaded by this module.
"""
import itertools
from collections import Counter
import networkx as nx
import numpy as np
from scipy.optimize import least_squares

MIN_HITS=5
STRICT_RMS=1.0
RELAXED_RMS=1.5
MAX_SPLIT_DEPTH=8
MAX_TRIMMED_HITS=2
MIN_HIERARCHY_COVERAGE=.70
MIN_HIERARCHY_RMS_GAIN=.15
MIN_HIERARCHY_SIZE=15

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


def components(data, probabilities, threshold, allowed=None):
    allowed = set(range(data.num_nodes)) if allowed is None else set(allowed)
    graph = nx.Graph()
    graph.add_nodes_from(allowed)
    edges = data.edge_index.t().cpu().numpy()
    keep = probabilities >= threshold
    if len(allowed) != data.num_nodes:
        allowed_mask = np.zeros(data.num_nodes, dtype=bool)
        allowed_mask[list(allowed)] = True
        keep &= allowed_mask[edges[:, 0]] & allowed_mask[edges[:, 1]]
    graph.add_edges_from((int(u), int(v), {"weight": float(p)}) for (u, v), p in zip(edges[keep], probabilities[keep]))
    return [sorted(value) for value in nx.connected_components(graph) if len(value) >= MIN_HITS]


def fit_candidate(data, indices, probabilities, threshold, rms_limit, stage):
    indices = sorted(set(indices))
    if len(indices) < MIN_HITS:
        return None
    x = data.wire_x[indices].numpy()
    y = data.wire_y[indices].numpy()
    d = data.drift_d[indices].numpy()
    fit = fit_2d_circle(x, y, d)
    if fit is None or not np.all(np.isfinite(fit)):
        return None
    a, b, radius = fit
    residuals = np.abs(np.hypot(x - a, y - b) - radius) - d
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    edges = data.edge_index.cpu().numpy()
    member_mask = np.zeros(data.num_nodes, dtype=bool)
    member_mask[indices] = True
    keep = member_mask[edges[0]] & member_mask[edges[1]] & (probabilities >= threshold)
    edge_values = probabilities[keep]
    return {
        "indices": indices,
        "fit": [float(v) for v in fit],
        "hits": len(indices),
        "rms": rms,
        "mean_probability": float(np.mean(edge_values)) if edge_values.size else 0.0,
        "stage": stage,
        "accepted_geometry": rms <= rms_limit,
    }


def trim_outliers(data, component, probabilities, threshold, rms_limit, stage):
    indices = sorted(set(component))
    removed = []
    for _ in range(MAX_TRIMMED_HITS + 1):
        candidate = fit_candidate(data, indices, probabilities, threshold, rms_limit, stage)
        if candidate is None:
            return None
        if candidate["accepted_geometry"]:
            candidate["removed_indices"] = removed
            return candidate
        if len(indices) <= MIN_HITS or len(removed) >= MAX_TRIMMED_HITS:
            return None
        a, b, radius = candidate["fit"]
        x = data.wire_x[indices].numpy()
        y = data.wire_y[indices].numpy()
        d = data.drift_d[indices].numpy()
        residuals = np.abs(np.abs(np.hypot(x - a, y - b) - radius) - d)
        worst = int(np.argmax(residuals))
        removed.append(indices.pop(worst))
    return None


def split_component(data, component, probabilities, threshold, rms_limit, depth=0):
    candidate = fit_candidate(data, component, probabilities, threshold, rms_limit, "strict")
    if candidate is not None and candidate["accepted_geometry"]:
        return [candidate]
    if candidate is None:
        if depth >= MAX_SPLIT_DEPTH or len(component) < 2 * MIN_HITS:
            return []
    if depth >= MAX_SPLIT_DEPTH or len(component) < 2 * MIN_HITS:
        return [candidate] if candidate is not None and candidate["accepted_geometry"] else []
    members = set(component)
    graph = nx.Graph()
    graph.add_nodes_from(component)
    for (u, v), probability in zip(data.edge_index.t().tolist(), probabilities):
        if u in members and v in members and probability >= threshold:
            graph.add_edge(u, v, weight=float(probability))
    if not graph.number_of_edges():
        trimmed = trim_outliers(data, component, probabilities, threshold, rms_limit, "trimmed")
        return [trimmed] if trimmed is not None else []
    tree = nx.maximum_spanning_tree(graph)
    weak_edges = sorted(tree.edges(data=True), key=lambda edge: edge[2]["weight"])[:5]
    original_score = -np.inf
    best_parts = None
    best_score = original_score
    # Try one- and two-edge cuts. The small cap keeps STEP4 bounded on dense graphs.
    for cut_count in (1, 2):
        for cuts in itertools.combinations(weak_edges, cut_count):
            trial = tree.copy()
            trial.remove_edges_from((u, v) for u, v, _ in cuts)
            parts = [sorted(value) for value in nx.connected_components(trial) if len(value) >= MIN_HITS]
            if len(parts) < 2:
                continue
            fitted = [fit_candidate(data, part, probabilities, threshold, rms_limit, "split") for part in parts]
            if any(value is None or not value["accepted_geometry"] for value in fitted):
                continue
            total = sum(score(value) for value in fitted)
            if total > best_score + 0.5:
                best_score = total
                best_parts = parts
    if best_parts is None:
        trimmed = trim_outliers(data, component, probabilities, threshold, rms_limit, "trimmed")
        return [trimmed] if trimmed is not None else []
    result = []
    for part in best_parts:
        result.extend(split_component(data, part, probabilities, threshold, rms_limit, depth + 1))
    return result


def score(candidate):
    return 2.0 * min(candidate["hits"] / 10.0, 2.0) + min(candidate["mean_probability"], 1.0) - min(candidate["rms"] / 2.0, 2.0)


def hierarchy_partition(data, component, probabilities, strict_threshold):
    parent = fit_candidate(data, component, probabilities, strict_threshold, STRICT_RMS, "strict")
    if len(component) < MIN_HIERARCHY_SIZE or (parent is not None and parent["accepted_geometry"] and parent["rms"] < 0.20):
        return None
    best = None
    start = int(np.floor(strict_threshold * 20.0)) + 1
    for value in range(start, 17):
        threshold = value / 20.0
        parts = components(data, probabilities, threshold, component)
        candidates = [trim_outliers(data, part, probabilities, threshold, STRICT_RMS, "hierarchy") for part in parts]
        candidates = [candidate for candidate in candidates if candidate is not None]
        if len(candidates) < 2:
            continue
        covered = len({index for candidate in candidates for index in candidate["indices"]}) / len(component)
        if covered < MIN_HIERARCHY_COVERAGE:
            continue
        weighted_rms = sum(candidate["rms"] * candidate["hits"] for candidate in candidates) / sum(candidate["hits"] for candidate in candidates)
        rms_gain = (parent["rms"] - weighted_rms) if parent is not None else float("inf")
        if parent is not None and parent["accepted_geometry"] and rms_gain < MIN_HIERARCHY_RMS_GAIN:
            continue
        mean_probability = float(np.mean([candidate["mean_probability"] for candidate in candidates]))
        utility = covered + 0.25 * mean_probability - 0.5 * weighted_rms
        if best is None or utility > best[0]:
            best = (utility, candidates, threshold, covered, rms_gain)
    if best is None:
        return None
    _, candidates, threshold, covered, rms_gain = best
    for candidate in candidates:
        candidate["hierarchy_threshold"] = threshold
        candidate["hierarchy_coverage"] = covered
        candidate["hierarchy_rms_gain"] = rms_gain
    return candidates


def run_two_pass(data, model, device, variant, strict_threshold, relaxed_threshold, probabilities=None):
    if probabilities is None:
        raise ValueError('AI6 adapter requires precomputed probabilities')
    strict_components = components(data, probabilities, strict_threshold)
    strict_candidates = []
    for component in strict_components:
        hierarchy = hierarchy_partition(data, component, probabilities, strict_threshold)
        strict_candidates.extend(hierarchy or split_component(data, component, probabilities, strict_threshold, STRICT_RMS))
    used = {index for candidate in strict_candidates for index in candidate["indices"]}
    leftover = set(range(data.num_nodes)) - used
    relaxed_candidates = [
        trim_outliers(data, component, probabilities, relaxed_threshold, RELAXED_RMS, "relaxed")
        for component in components(data, probabilities, relaxed_threshold, leftover)
    ]
    candidates = strict_candidates + [value for value in relaxed_candidates if value is not None]
    candidates.sort(key=score, reverse=True)
    final = []
    final_used = set()
    rejected = Counter()
    for candidate in candidates:
        if not candidate["accepted_geometry"]:
            rejected["rms_or_geometry"] += 1
            continue
        if final_used.intersection(candidate["indices"]):
            rejected["hit_conflict"] += 1
            continue
        candidate["score"] = score(candidate)
        final.append(candidate)
        final_used.update(candidate["indices"])
    return final, {
        "strict_components": len(strict_components),
        "candidate_count": len(candidates),
        "accepted_count": len(final),
        "split_candidates": sum(candidate["stage"] == "split" for candidate in candidates),
        "hierarchy_candidates": sum(candidate["stage"] == "hierarchy" for candidate in candidates),
        "trimmed_candidates": sum(bool(candidate.get("removed_indices")) for candidate in candidates),
        "rejected": dict(rejected),
    }

