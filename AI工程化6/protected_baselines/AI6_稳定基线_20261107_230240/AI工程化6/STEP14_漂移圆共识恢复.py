import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from common import ROOT,MODEL_ROOT,DATA_ROOT,save_json,sha
from STEP12_候选可靠性校准 import records,train_model,features
from drift_consensus_tracking import consensus_candidates
from joint_candidate_tracking import exclusive_selection
from STEP5_两级寻迹 import attach_stereo,score_tracks,summarize,DEFAULT_CONFIG
from STEP8_寻迹改进评估 import select_profile


def trial(eid,snapshot,fold):
    snapshot=Path(snapshot);old=json.loads((snapshot/'two_stage_validation_v1/events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    weight=json.loads((snapshot/'two_stage_validation_v1/experiment_contract.json').read_text(encoding='utf-8'))['weight_sha256']
    with np.load(snapshot/('validation_prediction_cache_'+weight[:12])/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,drift,truth=f['cell_id'],f['d'],f['truth']
    geo=np.load(DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    matrix=np.zeros((len(cells),len(cells)),dtype=np.float32);matrix[pairs[0],pairs[1]]=p;matrix[pairs[1],pairs[0]]=p
    proposals=consensus_candidates(geo,drift,cells,matrix,old['tracks']['axial_two_pass'],20261107+eid)
    with (snapshot/'drift_consensus_validation_v1'/f'confidence_fold_{fold}.pkl').open('rb') as f:model=pickle.load(f)
    prob=model.predict_proba(np.array([features(t,geo,cells,matrix) for t in proposals]))[:,1] if proposals else np.array([])
    methods={'baseline':old['methods']['stereo_assisted']};candidates={}
    for name,cut in [('consensus_raw',0.),('consensus_020',.2),('consensus_050',.5),('consensus_080',.8)]:
        keep=np.flatnonzero(prob>=cut);pool=[proposals[i] for i in keep]
        selected=exclusive_selection(pool,[len(proposals[i]['indices'])*(.5+float(prob[i]))-1 for i in keep],len(cells))
        tracks=old['tracks']['axial_two_pass']+selected
        joint,_=attach_stereo(tracks,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
        methods[name],_=score_tracks(joint,truth,stereo,DEFAULT_CONFIG);candidates[name]=joint
    return dict(event_id=eid,methods=methods,candidates=candidates,proposed=len(proposals))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='local_evaluation_epoch095_20261107');parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot;out=snapshot/'drift_consensus_validation_v1';out.mkdir(exist_ok=False)
    ids=json.loads((snapshot/'two_stage_validation_v1/experiment_contract.json').read_text(encoding='utf-8'))['event_ids']
    rows=[records(eid,snapshot) for eid in ids[:100]];x=np.concatenate([r['x'] for r in rows]);y=np.concatenate([r['y'] for r in rows]);groups=np.concatenate([np.full(len(r['x']),r['event_id']) for r in rows]);folds={eid:5 for eid in ids}
    for fold,(train,valid) in enumerate(GroupKFold(5).split(x,y,groups)):
        model=train_model(x[train],y[train]);pickle.dump(model,(out/f'confidence_fold_{fold}.pkl').open('wb'))
        for eid in np.unique(groups[valid]):folds[int(eid)]=fold
    with (snapshot/'candidate_confidence_v2/candidate_confidence.pkl').open('rb') as f:model=pickle.load(f)
    pickle.dump(model,(out/'confidence_fold_5.pkl').open('wb'));results={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(trial,eid,str(snapshot),folds[eid]) for eid in ids]):
            row=future.result();results[row['event_id']]=row;save_json(out/f"event_{row['event_id']:05d}.json",row)
            if len(results)%25==0:print(json.dumps(dict(events=len(results),total=len(ids))),flush=True)
    def aggregate(selected):return {name:summarize([results[eid]['methods'][name] for eid in selected]) for name in ['baseline','consensus_raw','consensus_020','consensus_050','consensus_080']}
    d=aggregate(ids[:100]);c=aggregate(ids[100:]);selected=select_profile(d)
    save_json(out/'summary.json',dict(dev_oof=d,check=c,all=aggregate(ids),selected_on_oof=selected,folds=folds,
              source_sha256=sha(ROOT/'drift_consensus_tracking.py'),calibration_events=ids[:100],check_events=ids[100:]))
    print(json.dumps(dict(selected=selected,dev=d,check=c)),flush=True)


if __name__=='__main__':main()
