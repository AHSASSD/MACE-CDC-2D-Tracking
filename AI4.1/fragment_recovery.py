"""Observable-only fragment recovery experiments; baseline tracking stays unchanged."""
import copy
from itertools import product

import numpy as np
from scipy.optimize import least_squares

from STEP4_两级寻迹 import score


def measurements(data):
    xy = np.column_stack([data.wire_x.numpy(), data.wire_y.numpy()]).astype(np.float64)
    d = data.drift_d.numpy().astype(np.float64)
    # Native signal time minus measured drift delay; only relative event time is needed.
    corrected_time = data.x[:, 6].numpy().astype(np.float64)*10000-d/.035
    return xy, d, corrected_time


def residual(fit, xy, d):
    return np.abs(np.hypot(xy[:, 0]-fit[0], xy[:, 1]-fit[1])-fit[2])-d


def wire_seed(xy):
    x, y = xy.T
    a, b, c = np.linalg.lstsq(np.column_stack([2*x, 2*y, np.ones(len(x))]), x*x+y*y, rcond=None)[0]
    return np.array([a, b, np.sqrt(max(c+a*a+b*b, .001))])


def fit_members(xy, d, indices, seeds):
    indices = sorted(set(indices))
    points, distances = xy[indices], d[indices]
    best = None
    for seed in seeds:
        if not np.isfinite(seed).all() or seed[2] <= .001:
            continue
        fit = least_squares(lambda p: residual(p, points, distances), seed,
                            bounds=([-np.inf, -np.inf, .001], [np.inf, np.inf, np.inf]),
                            loss='soft_l1', max_nfev=150, ftol=1e-6, xtol=1e-6)
        errors = residual(fit.x, points, distances)
        rms = float(np.sqrt(np.mean(errors**2)))
        if best is None or rms < best[1]:
            best = fit.x, rms
    return best


def candidate(data, probability, xy, d, indices, seeds, stage, rms_limit=1.):
    indices = sorted(set(indices))
    if len(indices) < 5:
        return None
    fitted = fit_members(xy, d, indices, seeds)
    if fitted is None or fitted[1] > rms_limit:
        return None
    fit, rms = fitted
    mask = np.zeros(data.num_nodes, dtype=bool)
    mask[indices] = True
    u, v = data.edge_index.numpy()
    values = probability[mask[u] & mask[v]]
    result = {'indices': indices, 'hits': len(indices), 'fit': fit.tolist(), 'rms': rms,
              'mean_probability': float(values.mean()) if len(values) else 0.,
              'stage': stage, 'accepted_geometry': True}
    result['score'] = score(result)
    return result


def recover_fragments(data, tracks, probability, use_time=False, merge=True):
    result = copy.deepcopy(tracks)
    xy, d, timing = measurements(data)
    adjacency = np.zeros((data.num_nodes, data.num_nodes), dtype=np.float32)
    u, v = data.edge_index.numpy()
    np.maximum.at(adjacency, (u, v), probability)
    adjacency = np.maximum(adjacency, adjacency.T)
    used = {i for t in result for i in t['indices']}
    # shortcut: capped local searches keep the experiment bounded; broaden after runtime checks.
    for number in sorted(range(len(result)), key=lambda i: result[i]['rms']):
        current = result[number]
        if current['mean_probability'] < .75 or current['rms'] > 1.:
            continue
        for _ in range(4):
            available = sorted(set(range(data.num_nodes))-used)
            if not available:
                break
            errors = np.abs(residual(current['fit'], xy[available], d[available]))
            evidence = adjacency[np.ix_(available, current['indices'])].max(axis=1)
            keep = (errors <= 1.5) & (evidence >= .35)
            if use_time:
                keep &= np.abs(timing[available]-np.median(timing[current['indices']])) <= 45
            options = np.array(available)[keep]
            if not len(options):
                break
            options = options[np.argsort(errors[keep], kind='stable')][:2]
            trial = candidate(data, probability, xy, d, current['indices']+options.tolist(), [current['fit']], 'fragment_extended')
            if trial is None or trial['rms'] > max(.65, current['rms']+.15):
                break
            used.update(options.tolist())
            current = result[number] = trial
    if merge:
        for _ in range(6):
            options = []
            for i, first in enumerate(result):
                if first['mean_probability'] < .75:
                    continue
                for j in range(i+1, len(result)):
                    second = result[j]
                    if second['mean_probability'] < .75:
                        continue
                    if use_time and abs(np.median(timing[first['indices']])-np.median(timing[second['indices']])) > 45:
                        continue
                    indices = first['indices']+second['indices']
                    seeds = [np.array(first['fit']), np.array(second['fit']), wire_seed(xy[indices])]
                    preliminary = [float(np.sqrt(np.mean(residual(seed, xy[indices], d[indices])**2))) for seed in seeds]
                    best_seed = seeds[int(np.argmin(preliminary))]
                    if min(preliminary) <= 3.:
                        options.append((min(preliminary), i, j, indices, best_seed))
            best = None
            for _, i, j, indices, seed in sorted(options, key=lambda value: value[0])[:8]:
                trial = candidate(data, probability, xy, d, indices, [seed], 'fragment_merged')
                if trial is None or trial['rms'] > max(.65, max(result[i]['rms'], result[j]['rms'])+.15):
                    continue
                utility = trial['hits']/(1+trial['rms'])
                if best is None or utility > best[0]:
                    best = utility, i, j, trial
            if best is None:
                break
            _, i, j, trial = best
            result = [t for index, t in enumerate(result) if index not in [i, j]]+[trial]
    result.sort(key=score, reverse=True)
    assert len({i for t in result for i in t['indices']}) == sum(t['hits'] for t in result)
    assert {i for t in tracks for i in t['indices']} <= {i for t in result for i in t['indices']}
    return result


def tangent_circles(xy, distances):
    """Apollonius hypotheses: a track circle tangent to three measured drift circles."""
    matrix = 2*(xy[1:]-xy[0])
    if np.linalg.cond(matrix) > 1e4:
        return []
    inverse = np.linalg.inv(matrix)
    constant = inverse @ (np.sum(xy[1:]**2, axis=1)-np.sum(xy[0]**2)-distances[1:]**2+distances[0]**2)
    offset = constant-xy[0]
    results = []
    for signs in product([-1, 1], repeat=3):
        signed = distances*np.array(signs)
        slope = inverse @ (-2*(signed[1:]-signed[0]))
        coefficients = [float(slope@slope-1), float(2*(offset@slope-signed[0])), float(offset@offset-distances[0]**2)]
        roots = np.roots(coefficients[1:] if abs(coefficients[0]) < 1e-12 else coefficients)
        for root in roots:
            if abs(root.imag) > 1e-7 or not .01 < root.real < 20000:
                continue
            radius = float(root.real)
            fit = np.r_[constant+radius*slope, radius]
            if np.all(radius+signed > 0) and np.isfinite(fit).all():
                results.append(fit)
    return results


def hough_remaining(data, tracks, probability, use_time=True, seed=0):
    result = copy.deepcopy(tracks)
    xy, d, timing = measurements(data)
    ambiguous = [t for t in result if np.ptp(np.quantile(timing[t['indices']], [.25, .75])) > 60]
    result = [t for t in result if t not in ambiguous]
    used = {i for t in result for i in t['indices']}
    available = np.array(sorted(set(range(data.num_nodes))-used), dtype=int)
    if len(available) < 7:
        return copy.deepcopy(tracks)
    rng = np.random.default_rng(seed)
    votes = {}
    # shortcut: randomized, coarse 3D Hough; it does not certify separation below parameter resolution.
    for _ in range(350):
        if use_time:
            first = int(rng.choice(available))
            nearby = available[(np.abs(timing[available]-timing[first]) <= 45) & (available != first)]
            if len(nearby) < 2:
                continue
            triple = np.r_[first, rng.choice(nearby, 2, replace=False)]
        else:
            triple = rng.choice(available, 3, replace=False)
        if use_time and np.ptp(timing[triple]) > 70:
            continue
        if len(np.unique(xy[triple], axis=0)) < 3:
            continue
        for fit in tangent_circles(xy[triple], d[triple]):
            center_radius = np.hypot(fit[0], fit[1])
            key = (int(np.floor(np.mod(np.arctan2(fit[1], fit[0]), 2*np.pi)/.03)),
                   int(np.floor((1/fit[2])/.0001)), int(np.floor((center_radius-fit[2])/5)))
            if use_time:
                key += (int(np.floor(np.median(timing[triple])/45)),)
            if key not in votes:
                votes[key] = [0, fit, float(np.median(timing[triple]))]
            votes[key][0] += 1
    proposals = []
    for count, fit, time_center in sorted(votes.values(), key=lambda value: value[0], reverse=True)[:12]:
        keep = np.abs(residual(fit, xy[available], d[available])) <= 1.5
        if use_time:
            keep &= np.abs(timing[available]-time_center) <= 45
        indices = available[keep].tolist()
        if count < 2 or len(indices) < 7:
            continue
        trial = candidate(data, probability, xy, d, indices, [fit], 'hough_residual')
        if trial is not None:
            proposals.append(trial)
    for track in sorted(proposals, key=lambda t: (t['hits'], -t['rms']), reverse=True):
        if not used.intersection(track['indices']):
            result.append(track)
            used.update(track['indices'])
    # Restore a disputed baseline candidate if replacement does not cover enough of its hits.
    restored = set()
    while True:
        covered = {i for t in result for i in t['indices']}
        unsafe = next((i for i, old in enumerate(ambiguous) if i not in restored and len(set(old['indices']) & covered) < .8*old['hits']), None)
        if unsafe is None:
            break
        old = ambiguous[unsafe]
        result = [t for t in result if not set(old['indices']).intersection(t['indices'])]+[old]
        restored.add(unsafe)
    result.sort(key=score, reverse=True)
    assert len({i for t in result for i in t['indices']}) == sum(t['hits'] for t in result)
    return result
