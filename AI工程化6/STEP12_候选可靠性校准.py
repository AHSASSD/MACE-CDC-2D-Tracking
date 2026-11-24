"""Event-group cross-validated candidate confidence, no test-set fitting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from common import MODEL_ROOT,DATA_ROOT,save_json,sha
from STEP5_两级寻迹 import score_tracks,summarize,attach_stereo,DEFAULT_CONFIG
from STEP8_寻迹改进评估 import select_profile
from joint_candidate_tracking import candidate_evidence,exclusive_selection

FEATURES=['axial_hits','aa_mean','stereo_support','ss_coherence','outside_continuations','rms',
          'aa_q10','aa_q90','internal_coverage','continuation_fraction','stereo_ratio','log_radius']


def features(track,geo,cells,matrix):
    aa=np.array(track['indices']);e=candidate_evidence(track,geo,cells,matrix)
    block=matrix[np.ix_(aa,aa)];v=block[np.triu_indices(len(aa),1)];nonzero=v[v>0]
    e.update(aa_q10=float(np.quantile(nonzero,.1)) if len(nonzero) else 0.,aa_q90=float(np.quantile(nonzero,.9)) if len(nonzero) else 0.,
             internal_coverage=float((v>=.7).mean()) if len(v) else 0.,
             continuation_fraction=e['outside_continuations']/max(len(aa)+e['outside_continuations'],1),
             stereo_ratio=e['stereo_support']/max(len(aa),1),log_radius=float(np.log1p(track['fit'][2])))
    return [e[k] for k in FEATURES]


def records(eid,snapshot):
    snapshot=Path(snapshot);base=snapshot/'two_stage_validation_v1';old=json.loads((base/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    pool=json.loads((snapshot/'joint_pool_validation_v1'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    w=json.loads((base/'experiment_contract.json').read_text(encoding='utf-8'))['weight_sha256']
    with np.load(snapshot/('validation_prediction_cache_'+w[:12])/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,truth=f['cell_id'],f['truth']
    geo=np.load(DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    candidates=[];seen=set()
    for source in [old['tracks']['axial_two_pass']]+[pool['candidates'][k] for k in ['joint_residual','joint_global']]:
        for t in source:
            aa=tuple(sorted(i for i in t['indices'] if not stereo[i]))
            if aa in seen:continue
            seen.add(aa);candidates.append(dict(t,indices=list(aa)))
    matrix=np.zeros((len(cells),len(cells)),dtype=np.float32);matrix[pairs[0],pairs[1]]=p;matrix[pairs[1],pairs[0]]=p
    x=np.array([features(t,geo,cells,matrix) for t in candidates]);sizes=Counter(truth[~stereo]);y=[]
    for t in candidates:
        tid,count=Counter(truth[t['indices']]).most_common(1)[0]
        y.append(int(tid>=0 and count/len(t['indices'])>=.8 and count/sizes[tid]>=.5))
    return dict(event_id=eid,x=x,y=np.array(y),candidates=candidates,geo=geo,cells=cells,truth=truth,stereo=stereo,pairs=pairs,p=p,baseline=old['methods']['stereo_assisted'])


def train_model(x,y):
    model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=10,random_state=20261107)
    weights=np.where(y==0,(y==1).sum()/max((y==0).sum(),1),1.)
    model.fit(x,y,sample_weight=weights)
    return model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='local_evaluation_epoch095_20261107');args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot;out=snapshot/'candidate_confidence_v2';out.mkdir(exist_ok=False)
    ids=json.loads((snapshot/'two_stage_validation_v1/experiment_contract.json').read_text(encoding='utf-8'))['event_ids']
    rows=[records(eid,snapshot) for eid in ids];dev=rows[:100];check=rows[100:]
    x=np.concatenate([r['x'] for r in dev]);y=np.concatenate([r['y'] for r in dev]);groups=np.concatenate([np.full(len(r['x']),r['event_id']) for r in dev]);oof=np.zeros(len(y))
    for train,valid in GroupKFold(5).split(x,y,groups):
        model=train_model(x[train],y[train]);oof[valid]=model.predict_proba(x[valid])[:,1]
    model=train_model(x,y)
    with (out/'candidate_confidence.pkl').open('wb') as f:pickle.dump(model,f)
    thresholds={'confidence_020':.2,'confidence_040':.4,'confidence_060':.6,'confidence_080':.8}
    results=[];offset=0
    for j,row in enumerate(rows):
        if j<100:prob=oof[offset:offset+len(row['x'])];offset+=len(row['x'])
        else:prob=model.predict_proba(row['x'])[:,1]
        with np.load(DATA_ROOT/'events'/f"event_{row['event_id']:05d}.npz") as f:drift=f['d']
        methods={'baseline':row['baseline']};saved={}
        for name,threshold in thresholds.items():
            keep=np.flatnonzero(prob>=threshold);pool=[row['candidates'][i] for i in keep]
            quality=[len(row['candidates'][i]['indices'])*(.5+float(prob[i]))-1 for i in keep]
            picked=exclusive_selection(pool,quality,len(row['cells']))
            joint,_=attach_stereo(picked,row['geo'],drift,row['cells'],row['pairs'],row['p'],DEFAULT_CONFIG)
            methods[name],_=score_tracks(joint,row['truth'],row['stereo'],DEFAULT_CONFIG);saved[name]=joint
        result=dict(event_id=row['event_id'],methods=methods,candidates=saved)
        results.append(result);save_json(out/f"event_{row['event_id']:05d}.json",result)
        if (j+1)%25==0:print(json.dumps(dict(events=j+1,total=len(rows))),flush=True)
    def aggregate(subset):return {name:summarize([r['methods'][name] for r in subset]) for name in ['baseline']+list(thresholds)}
    d=aggregate(results[:100]);c=aggregate(results[100:]);selected=select_profile(d)
    save_json(out/'summary.json',dict(features=FEATURES,thresholds=thresholds,dev_oof=d,check=c,selected_on_oof=selected,
              calibration_event_ids=ids[:100],check_event_ids=ids[100:],source_sha256=sha(Path(__file__)),
              model_sha256=sha(out/'candidate_confidence.pkl'),scope='calibrator fitted on first100 validation events; OOF grouped by event; no test fitting'))
    print(json.dumps(dict(selected=selected,dev=d,check=c)),flush=True)


if __name__=='__main__':main()
