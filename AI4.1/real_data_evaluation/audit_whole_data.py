"""Audit complete-event CSV exports without adding errors or mixing events."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'whole data'
OUT = ROOT / 'whole_event_retraining'
sys.path.insert(0, str(ROOT))
from common import MAP_FILE


def vectors(series):
    result = series.str.strip('{} ').str.split(',', expand=True).astype(float).to_numpy()
    if result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError('Invalid vertex vectors')
    return result


def main():
    OUT.mkdir(exist_ok=True)
    event_tables, track_tables, files = [], [], []
    error_n = error_sum = error_sq = 0
    raw_rows = selected_rows = duplicate_ids = invalid_geometry = 0
    geometry = pd.read_excel(MAP_FILE)
    axial_ids = set(geometry.loc[geometry.u.eq(0) & geometry.v.eq(0), 'CellID'])
    columns = ['EvtID', 'TrkID', 'HitID', 'CellID', 'Good', 'CreatProc', 'WireType',
               'd', 'd_true', 'd_error_mm', 't', 't0', 'tHit', 'x0', 'Ek0',
               'NativeEvtID', 'SourceShard', 'PoolEvtID']
    paths = sorted(SOURCE.glob('CDCSimHit_*.csv'))
    if not paths:
        raise FileNotFoundError('No complete-event CSV files')
    for path in paths:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        frame = pd.read_csv(path, usecols=columns)
        required_numeric = ['EvtID', 'TrkID', 'HitID', 'CellID', 'd', 'd_true',
                            'd_error_mm', 't', 't0', 'tHit', 'Ek0',
                            'NativeEvtID', 'SourceShard', 'PoolEvtID']
        if not np.isfinite(frame[required_numeric].to_numpy()).all():
            raise ValueError(f'Nonfinite input: {path}')
        for column in ['EvtID', 'TrkID', 'HitID', 'CellID', 'NativeEvtID', 'SourceShard', 'PoolEvtID']:
            if (frame[column] % 1 != 0).any():
                raise ValueError(f'Noninteger identity: {column}')
        raw_rows += len(frame)
        duplicate_ids += int(frame.duplicated(['EvtID', 'HitID']).sum())
        identity = frame[['EvtID', 'SourceShard', 'NativeEvtID', 'PoolEvtID']].drop_duplicates()
        identity['filename'] = path.name
        counts = frame.groupby('EvtID').agg(raw_hits=('HitID', 'size'),
                                            all_tracks=('TrkID', 'nunique'))
        selected = frame.loc[frame.Good.eq(True) & frame.WireType.eq('Axial') &
                             frame.CreatProc.eq('ExtendedDecayWithSpin')].copy()
        invalid_geometry += int((~selected.CellID.isin(axial_ids)).sum())
        selected_rows += len(selected)
        counts['axial_hits'] = selected.groupby('EvtID').size()
        counts['axial_tracks'] = selected.groupby('EvtID').TrkID.nunique()
        counts['repeated_cells'] = selected.assign(repeated=selected.duplicated(['EvtID', 'CellID'])).groupby('EvtID').repeated.sum()
        times = selected.groupby('EvtID').t.agg(['min', 'max'])
        counts['observed_event_span_ns'] = times['max'] - times['min']
        event_tables.append(identity.merge(counts, left_on='EvtID', right_index=True))
        tracks = selected.groupby(['EvtID', 'TrkID'], sort=False).agg(
            hit_count=('HitID', 'size'), x0=('x0', 'first'), t0=('t0', 'first'),
            Ek0=('Ek0', 'first'), SourceShard=('SourceShard', 'first'),
            NativeEvtID=('NativeEvtID', 'first'), PoolEvtID=('PoolEvtID', 'first')).reset_index()
        xyz = vectors(tracks.x0)
        tracks['birth_radius_mm'] = np.hypot(xyz[:, 0], xyz[:, 1])
        tracks['birth_z_mm'] = xyz[:, 2]
        track_tables.append(tracks)
        error = selected.d - selected.d_true
        error_n += len(error)
        error_sum += float(error.sum())
        error_sq += float(np.square(error).sum())
        files.append({'file': path.name, 'bytes': path.stat().st_size,
                      'sha256': digest.hexdigest(), 'rows': len(frame),
                      'selected_rows': len(selected),
                      'd_error_column_max_difference': float(np.max(np.abs(error - selected.d_error_mm))),
                      'negative_d': int((selected.d < 0).sum()),
                      'timing_minus_true_drift_mean_ns': float((selected.t - selected.tHit - selected.d_true / .035).mean()),
                      'timing_minus_true_drift_std_ns': float((selected.t - selected.tHit - selected.d_true / .035).std())})
        print(path.name, 'rows', len(frame), 'selected', len(selected), flush=True)
    events = pd.concat(event_tables, ignore_index=True)
    tracks = pd.concat(track_tables, ignore_index=True)
    evt_duplicates = int(events.EvtID.duplicated().sum())
    source_duplicates = int(events.duplicated(['SourceShard', 'NativeEvtID']).sum())
    pool_duplicates = int(events.PoolEvtID.duplicated().sum())
    if duplicate_ids or evt_duplicates or source_duplicates or pool_duplicates or invalid_geometry:
        raise ValueError(f'Identity/geometry conflicts: hits={duplicate_ids}, events={evt_duplicates}, source={source_duplicates}, pool={pool_duplicates}, geometry={invalid_geometry}')
    mother = pd.read_csv(SOURCE / 'mother_decays.csv')
    if mother.duplicated(['EvtID', 'TrkID']).any():
        raise ValueError('Repeated mother identity')
    mother['x0'] = mother.x_mm
    mother['t0'] = mother.t_ns
    mother_xyz = vectors(mother.x_mm)
    mother['birth_radius_mm'] = np.hypot(mother_xyz[:, 0], mother_xyz[:, 1])
    # Birth times differ at floating-point roundoff after global-local subtraction.
    parent = mother[['EvtID', 'x0', 't0', 'TrkID', 'PDGID', 'Ek_MeV']].rename(
        columns={'x0': 'mother_x0', 't0': 'mother_t0', 'TrkID': 'TrkID_mother'})
    joined = pd.merge_asof(tracks.sort_values('t0'), parent.sort_values('mother_t0'),
                           left_on='t0', right_on='mother_t0', by='EvtID',
                           direction='nearest', tolerance=1e-5)
    present = joined.TrkID_mother.notna()
    vertex_error = np.linalg.norm(vectors(joined.loc[present, 'x0']) -
                                  vectors(joined.loc[present, 'mother_x0']), axis=1)
    valid = present.copy()
    valid.loc[present] = vertex_error < 1e-4
    joined.loc[~valid, ['TrkID_mother', 'PDGID', 'Ek_MeV']] = np.nan
    wall = joined.birth_radius_mm.between(70, 70.5)
    matched_wall = joined.loc[wall & joined.TrkID_mother.notna()]
    rng = np.random.default_rng(20261014)
    shuffled = rng.permutation(events.EvtID.to_numpy(dtype=int))
    n = len(shuffled)
    partitions = {int(evt): name for name, ids in [
        ('train', shuffled[:int(n * .7)]),
        ('validation', shuffled[int(n * .7):int(n * .85)]),
        ('test', shuffled[int(n * .85):])]
        for evt in ids}
    assert len(partitions) == n
    events['partition'] = events.EvtID.map(partitions)
    joined['partition'] = joined.EvtID.map(partitions)
    events.to_csv(OUT / 'event_inventory.csv', index=False)
    joined.to_csv(OUT / 'track_inventory.csv', index=False)
    sigma = float(np.sqrt(error_sq / error_n - (error_sum / error_n) ** 2))
    report = {'files': files, 'raw_rows': raw_rows, 'selected_rows': selected_rows,
              'events': len(events), 'event_min': int(events.EvtID.min()), 'event_max': int(events.EvtID.max()),
              'source_event_duplicates': source_duplicates, 'pool_event_duplicates': pool_duplicates,
              'partitions': events.partition.value_counts().to_dict(),
              'axial_tracks_per_event': events.axial_tracks.describe().to_dict(),
              'axial_hits_per_event': events.axial_hits.describe().to_dict(),
              'observed_event_span_ns': events.observed_event_span_ns.describe().to_dict(),
              'selected_tracks': len(tracks), 'eligible_tracks': int((tracks.hit_count >= 5).sum()),
              'observed_d_error_mean_mm': error_sum / error_n, 'observed_d_error_std_mm': sigma,
              'mother_rows': len(mother), 'mother_match_tracks': int(joined.TrkID_mother.notna().sum()),
              'mother_match_time_tolerance_ns': 1e-5, 'mother_match_vertex_tolerance_mm': 1e-4,
              'wall_tracks': int(wall.sum()), 'matched_wall_tracks': len(matched_wall),
              'wall_muplus_mothers': int(matched_wall.PDGID.eq(-13).sum()),
              'wall_stopped_mothers': int((matched_wall.Ek_MeV.abs() <= 1e-8).sum()),
              'wall_max_mother_kinetic_MeV': float(matched_wall.Ek_MeV.max()) if len(matched_wall) else None,
              'geometry_sha256': hashlib.sha256((SOURCE / 'SimMACE.yaml').read_bytes()).hexdigest(),
              'split_seed': 20261014, 'filter': 'Good==True AND Axial AND ExtendedDecayWithSpin',
              'added_smearing_noise_deadtime': False,
              'training_started': False}
    (OUT / 'data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
