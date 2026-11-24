import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from sklearn.model_selection import GroupKFold
from common import MODEL_ROOT,DATA_ROOT,save_json,sha
from STEP12_候选可靠性校准 import records,train_model
from hybrid_candidate_tracking import hybrid_select
from STEP5_两级寻迹 import attach_stereo,score_tracks,summarize,DEFAULT_CONFIG
from STEP8_寻迹改进评估 import select_profile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='local_evaluation_epoch095_20261107');args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot;out=snapshot/'hybrid_confidence_v1';out.mkdir(exist_ok=False)
    ids=json.loads((snapshot/'two_stage_validation_v1/experiment_contract.json').read_text(encoding='utf-8'))['event_ids']
    rows=[records(eid,snapshot) for eid in ids];dev=rows[:100]
    x=np.concatenate([r['x'] for r in dev]);y=np.concatenate([r['y'] for r in dev]);groups=np.concatenate([np.full(len(r['x']),r['event_id']) for r in dev]);oof=np.zeros(len(y))
    for train,valid in GroupKFold(5).split(x,y,groups):oof[valid]=train_model(x[train],y[train]).predict_proba(x[valid])[:,1]
    with (snapshot/'candidate_confidence_v2/candidate_confidence.pkl').open('rb') as f:model=pickle.load(f)
    profiles={f'base_{b:g}_new_{r:g}':(b,r) for b in [0.,.02,.05,.1] for r in [.2,.5,.8]}
    all_rows=[];offset=0
    for j,row in enumerate(rows):
        probability=oof[offset:offset+len(row['x'])] if j<100 else model.predict_proba(row['x'])[:,1]
        if j<100:offset+=len(row['x'])
        eid=row['event_id'];old=json.loads((snapshot/'two_stage_validation_v1/events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
        lookup={tuple(t['indices']):i for i,t in enumerate(row['candidates'])}
        baseline_indices=[lookup[tuple(sorted(t['indices']))] for t in old['tracks']['axial_two_pass']]
        with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:drift=f['d']
        methods={'baseline':row['baseline']}
        for name,(b,r) in profiles.items():
            picked=hybrid_select(row['candidates'],probability,baseline_indices,len(row['cells']),b,r)
            joint,_=attach_stereo(picked,row['geo'],drift,row['cells'],row['pairs'],row['p'],DEFAULT_CONFIG)
            methods[name],_=score_tracks(joint,row['truth'],row['stereo'],DEFAULT_CONFIG)
        result=dict(event_id=eid,methods=methods);all_rows.append(result);save_json(out/f'event_{eid:05d}.json',result)
        if (j+1)%25==0:print(json.dumps(dict(events=j+1,total=len(rows))),flush=True)
    def aggregate(subset):return {name:summarize([row['methods'][name] for row in subset]) for name in ['baseline']+list(profiles)}
    d=aggregate(all_rows[:100]);c=aggregate(all_rows[100:]);selected=select_profile(d)
    save_json(out/'summary.json',dict(profiles=profiles,dev_oof=d,check=c,selected_on_oof=selected,all=aggregate(all_rows),
              model_sha256=sha(snapshot/'candidate_confidence_v2/candidate_confidence.pkl'),source_sha256=sha(Path(__file__)),
              calibration_events=ids[:100],check_events=ids[100:],test_used_for_selection=False))
    print(json.dumps(dict(selected=selected,dev=d,check=c)),flush=True)


if __name__=='__main__':main()
