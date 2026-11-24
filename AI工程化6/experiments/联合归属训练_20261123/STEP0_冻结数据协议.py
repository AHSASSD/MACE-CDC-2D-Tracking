"""Freeze a research plan's event identities and audit; never trains or submits jobs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = PROJECT / 'joint_event_dataset'
PREVIOUS = PROJECT / 'experiments' / '双路线寻迹_20261122' / 'split_manifest.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_or_verify(name, value, verify):
    path = HERE / name
    if verify or path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen artifact changed: {path}')
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    metadata = read(SOURCE / 'dataset_metadata_joint_no_time.json')
    old = read(PREVIOUS)
    plan = read(HERE / 'training_plan.json')
    digest = sha(SOURCE / 'dataset_metadata_joint_no_time.json')
    assert digest == old['dataset_sha256'] == '126015a11d8bed0f953b040af29dc512759308e308c685de893da04f914f7db2'
    ids = metadata['event_ids']
    for name in ['train', 'validation', 'test']:
        assert ids[name] == old[f'original_{name}_event_ids']
        assert len(ids[name]) == len(set(ids[name]))
    assert len(set(ids['train']) | set(ids['validation']) | set(ids['test'])) == 10000
    assert not (set(ids['train']) & set(ids['validation']) or set(ids['train']) & set(ids['test']) or set(ids['validation']) & set(ids['test']))

    member = old['periodic_member_validation_event_ids']
    tune = old['validation_tune_event_ids']
    assert len(member) == 200 and set(member) <= set(tune)
    remaining = sorted(set(tune) - set(member))
    rng = np.random.default_rng(plan['optimization']['seed_first'])
    calibration = sorted(rng.permutation(remaining)[:200].tolist())
    expansion = sorted(set(remaining) - set(calibration))
    assert len(expansion) == 600
    assert set(tune).isdisjoint(old['validation_confirm_event_ids'])
    assert set(tune) | set(old['validation_confirm_event_ids']) == set(ids['validation'])
    pilot = old['pilot_train_event_ids']
    assert len(pilot) == 1000 and set(pilot) <= set(ids['train'])
    inventory = {x['event_id']: x for x in read(SOURCE / 'event_inventory.json')}
    ranked = sorted(ids['train'], key=lambda eid: (inventory[eid]['hits'], eid))
    smoke = sorted(set(ranked[:8] + ranked[-8:] + pilot[:16]))
    for eid in pilot:
        if len(smoke) >= 32:
            break
        if eid not in smoke:
            smoke.append(eid)
    smoke = sorted(smoke[:32])
    assert len(smoke) == 32

    manifest = dict(
        schema='ai6-joint-membership-splits-v1', dataset_sha256=digest,
        previous_manifest_sha256=sha(PREVIOUS), original_train_event_ids=ids['train'],
        pilot_train_event_ids=pilot, smoke_train_event_ids=smoke,
        original_validation_event_ids=ids['validation'], validation_tune_event_ids=tune,
        member_selection_event_ids=member, decoder_calibration_event_ids=calibration,
        development_expansion_event_ids=expansion,
        historical_validation_recheck_event_ids=old['validation_confirm_event_ids'],
        historical_test_event_ids=ids['test'], known_debug_cases=old['known_debug_cases'],
        new_confirmation=dict(status='awaiting_independent_simulation_source', target_events=1000, event_ids=[], pristine_claim=False),
        historical_sets_are_pristine=False,
    )
    write_or_verify('split_manifest.json', manifest, args.verify)
    stats = {}
    for name, group in ids.items():
        hits = np.array([inventory[e]['hits'] for e in group])
        stereo = np.array([inventory[e]['stereo_hits'] for e in group])
        stats[name] = dict(events=len(group), hits_total=int(hits.sum()),
                           hits_quantiles_0_50_95_99_100=np.quantile(hits, [0, .5, .95, .99, 1]).tolist(),
                           stereo_fraction=float(stereo.sum() / hits.sum()))
    negative = exact = near = repeated_cell_hits = 0
    for eid in pilot:
        with np.load(SOURCE / 'events' / f'event_{eid:05d}.npz') as data:
            cell, drift, truth = data['cell_id'], data['d'], data['truth']
            hit_id = data['hit_id']
            assert len(hit_id) == len(np.unique(hit_id))
            assert np.isfinite(drift).all()
            negative += int((truth < 0).sum())
            repeated_cell_hits += len(cell) - len(np.unique(cell))
            for value, label in [(drift, 'exact'), (np.round(drift, 3), 'near')]:
                _, inverse = np.unique(np.rec.fromarrays([cell, value]), return_inverse=True)
                unique = np.unique(np.column_stack([inverse, truth]), axis=0)
                counts = np.bincount(unique[:, 0], minlength=int(inverse.max()) + 1)
                amount = int((counts[inverse] > 1).sum())
                if label == 'exact':
                    exact += amount
                else:
                    near += amount
    audit = dict(schema='ai6-joint-membership-data-audit-v1', source_metadata_sha256=digest,
                 inventory_sha256=sha(SOURCE / 'event_inventory.json'), event_size_statistics=stats,
                 pilot_observability=dict(events=len(pilot), negative_truth_hits=negative,
                     repeated_cell_hits=repeated_cell_hits,
                     cross_truth_hits_identical_cell_and_d=exact,
                     cross_truth_hits_same_cell_and_d_rounded_0p001mm=near,
                     rounding_is_not_detector_resolution=True,
                     coverage='pilot_train_only; full_train_audit_required_before_formal_training'),
                 splits_disjoint=True, selection_calibration_disjoint=True, baseline_unchanged=True)
    write_or_verify('data_audit.json', audit, args.verify)
    names = ['训练计划.md', 'training_plan.json', 'split_manifest.json', 'data_audit.json', Path(__file__).name]
    lock = dict(schema='ai6-joint-membership-plan-lock-v1', status='plan_only_not_training',
                file_sha256={name: sha(HERE / name) for name in names})
    write_or_verify('plan_lock.json', lock, args.verify)
    print(json.dumps(dict(status='verified' if args.verify else 'frozen', train=len(ids['train']),
        member_selection=len(member), calibration=len(calibration), development_expansion=len(expansion),
        exact_ambiguous_pilot_hits=exact, gpu_jobs_submitted=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
