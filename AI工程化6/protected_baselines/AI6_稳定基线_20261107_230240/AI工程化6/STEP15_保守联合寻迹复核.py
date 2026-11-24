import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import pickle
from pathlib import Path
import time
import numpy as np
from common import ROOT,MODEL_ROOT,DATA_ROOT,JointEventDataset,save_json,sha
from STEP5_两级寻迹 import prepare_predictions,axial_two_pass,attach_stereo,score_tracks,summarize,DEFAULT_CONFIG
from hybrid_joint_tracker import track


def evaluate_event(eid,snapshot,split):
    snapshot=Path(snapshot);weight=sha(snapshot/'best_interaction_network_joint_no_time.pth')
    cache=snapshot/(('validation' if split=='validation' else 'test')+'_prediction_cache_'+weight[:12])
    with np.load(cache/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,drift,truth,hit_ids=f['cell_id'],f['d'],f['truth'],f['hit_id']
    geo=np.load(DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    if split=='test':
        old=json.loads((snapshot/'frozen_tracking_test_v1/events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
        baseline_joint=old['candidates'];baseline=[dict(t,indices=[i for i in t['indices'] if not stereo[i]]) for t in baseline_joint]
    else:
        baseline,_=axial_two_pass(geo,drift,pairs,p);baseline_joint,_=attach_stereo(baseline,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
    with (snapshot/'candidate_confidence_v2/candidate_confidence.pkl').open('rb') as f:model=pickle.load(f)
    selected=track(geo,drift,cells,pairs,p,baseline,baseline_joint,model)
    b,bids=score_tracks(baseline_joint,truth,stereo,DEFAULT_CONFIG);m,mids=score_tracks(selected,truth,stereo,DEFAULT_CONFIG)
    for t in selected:t['hit_ids']=hit_ids[t['indices']].tolist();t['cell_ids']=cells[t['indices']].tolist()
    return dict(event_id=eid,baseline=b,selected=m,baseline_matched=bids,selected_matched=mids,candidates=selected)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='local_evaluation_epoch095_20261107');parser.add_argument('--split',choices=['validation','test'],default='validation');parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot;split=args.split
    development=json.loads((snapshot/'hybrid_confidence_v1/summary.json').read_text(encoding='utf-8'))
    if development['selected_on_oof']!='base_0.02_new_0.2':raise ValueError('Unexpected calibration selection')
    old_ids=development['calibration_events']+development['check_events']
    if split=='validation':
        available=[eid for eid in JointEventDataset(DATA_ROOT,'validation').ids if eid not in old_ids]
        ids=list(map(int,np.random.default_rng(20261108).permutation(available)[:300]))
    else:
        verified=json.loads((snapshot/'hybrid_fresh_validation_v1/summary.json').read_text(encoding='utf-8'))
        if not verified['non_regression_counts']:raise ValueError('Fresh validation failed non-regression; do not proceed to test')
        ids=JointEventDataset(DATA_ROOT,'test').ids
    out=snapshot/('hybrid_fresh_validation_v1' if split=='validation' else 'hybrid_frozen_test_v1');out.mkdir(exist_ok=True)
    source_names=['hybrid_joint_tracker.py','hybrid_candidate_tracking.py','joint_candidate_tracking.py','STEP12_候选可靠性校准.py','STEP5_两级寻迹.py','two_stage_core.py']
    plan=dict(split=split,event_ids=ids,excluded_calibration_event_ids=old_ids if split=='validation' else [],
              base_cut=.02,new_cut=.2,confidence_model_sha256=sha(snapshot/'candidate_confidence_v2/candidate_confidence.pkl'),
              weight_sha256=sha(snapshot/'best_interaction_network_joint_no_time.pth'),source_hashes={name:sha(ROOT/name) for name in source_names},
              test_used_for_selection=False)
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text(encoding='utf-8'))!=plan:raise ValueError('Frozen plan changed')
    save_json(out/'contract.json',plan)
    if split=='validation':prepare_predictions(snapshot,ids,snapshot/('validation_prediction_cache_'+plan['weight_sha256'][:12]))
    event_dir=out/'events';event_dir.mkdir(exist_ok=True);rows={};started=time.time()
    for eid in ids:
        if (event_dir/f'event_{eid:05d}.json').exists():rows[eid]=json.loads((event_dir/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(evaluate_event,eid,str(snapshot),split) for eid in ids if eid not in rows]):
            row=future.result();rows[row['event_id']]=row;save_json(event_dir/f"event_{row['event_id']:05d}.json",row)
            if len(rows)%50==0:
                progress=dict(split=split,events=len(rows),total=len(ids),seconds=time.time()-started);save_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    b=summarize([rows[eid]['baseline'] for eid in ids]);m=summarize([rows[eid]['selected'] for eid in ids])
    gained=sum(len(set(r['selected_matched'])-set(r['baseline_matched'])) for r in rows.values());lost=sum(len(set(r['baseline_matched'])-set(r['selected_matched'])) for r in rows.values())
    denom=np.array([rows[eid]['baseline']['truth_tracks'] for eid in ids]);delta=np.array([rows[eid]['selected']['matched_tracks']-rows[eid]['baseline']['matched_tracks'] for eid in ids]);draws=np.random.default_rng(20261108).integers(len(ids),size=(2000,len(ids)))
    ci=np.quantile(delta[draws].sum(1)/denom[draws].sum(1),[.025,.975]).tolist()
    counts_ok=m['matched_tracks']>=b['matched_tracks'] and m.get('fake_candidates',0)<=b.get('fake_candidates',0) and m.get('clone_candidates',0)<=b.get('clone_candidates',0)
    rates_ok=m['track_efficiency']>=b['track_efficiency'] and m['fake_fraction']<=b['fake_fraction'] and m['clone_fraction']<=b['clone_fraction']
    report=dict(plan=plan,baseline=b,selected=m,gained=gained,lost=lost,efficiency_difference_95_ci=ci,
                non_regression_counts=counts_ok,non_regression_rates=rates_ok,seconds=time.time()-started)
    save_json(out/'summary.json',report);print(json.dumps(dict(split=split,baseline=b,selected=m,counts_ok=counts_ok,rates_ok=rates_ok,ci=ci)),flush=True)


if __name__=='__main__':main()
