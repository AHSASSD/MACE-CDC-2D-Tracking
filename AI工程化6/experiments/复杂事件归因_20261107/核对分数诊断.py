import experiment_common as e
import json
from pathlib import Path
import numpy as np


def main():
    folder=e.EXP/'edge_score_diagnosis_v1';r=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    assert e.sha(e.EXP/'STEP4_真边分数与连通性诊断.py')==r['plan']['source_sha256']
    missed=boost=unlock=tracks=0
    for eid in e.IDS:
        row=json.loads((folder/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
        hist=np.load(folder/'events'/f'hist_{eid:05d}.npy')
        with np.load(e.SNAPSHOT/('validation_prediction_cache_'+e.CONTRACT['weight_sha256'][:12])/f'event_{eid:05d}.npz') as f:edges=f['pairs'].shape[1]
        assert int(hist[:,:2].sum())==edges
        assert np.all(hist[:,2]+hist[:,3]<=hist[:,0])
        old=json.loads((e.EXP/'attribution_v1/events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
        assert {t['truth_id'] for t in row['tracks'] if not t['found']}=={t['truth_id'] for t in old['missed_tracks']}
        tracks+=len(row['tracks']);missed+=sum(not t['found'] for t in row['tracks']);boost+=sum(t['boost_sensitive'] for t in row['tracks']);unlock+=sum(t['restored_single_pass_denoised'] for t in row['tracks'])
    assert missed==r['missed_tracks']==223 and boost==r['boost_sensitive_tracks']==181
    assert unlock==r['restored_by_denoised_nonlocking']==24
    training=json.loads((e.EXP/'training_score_probe_v1/summary.json').read_text(encoding='utf-8'))
    assert training['plan']['optimization_steps']==0
    assert not set(training['plan']['event_ids']).intersection(e.IDS)
    original=Path(e.manifest['source'])
    for name,digest in e.manifest['deployment_pins'].items():assert e.sha(original/name)==digest,name
    e.verify_pins()
    result=dict(validation_events=200,validation_eligible_tracks=tracks,training_probe_events=200,
                histograms_cover_each_cached_edge_once=True,original_missed_identity_matches=True,
                counterfactual_count_verified=True,train_validation_disjoint=True,
                original_and_backup_deployment_pins_unchanged=True,gnn_optimization_steps=0)
    e.save_json(folder/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
