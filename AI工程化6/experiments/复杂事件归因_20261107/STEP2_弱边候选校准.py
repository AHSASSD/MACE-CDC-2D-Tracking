"""Fit a small recovery scorer with event-group OOF on diagnosis/calibration events.

GNN and stable tracking weights stay frozen. Labels are read only after proposals.
"""
import experiment_common as e
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import pickle
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from STEP5_两级寻迹 import score_tracks,summarize,DEFAULT_CONFIG
from 生成粗三维成员报告 import seed_counts
from weak_edge_recovery import generate,reconstruct,CONFIG,FEATURES

CUTS=[.80,.90,.95,.975,.99,.995,1.1]
OUT=e.EXP/'weak_edge_recovery_v1'


def load_event(eid):
    with np.load(e.DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,drift,truth=f['cell_id'],f['d'],f['truth']
    geo=np.load(e.DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    with np.load(e.SNAPSHOT/('validation_prediction_cache_'+e.CONTRACT['weight_sha256'][:12])/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    baseline=json.loads((e.SNAPSHOT/'coarse_3d_inputs_validation'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))['tracks']
    coarse=json.loads((e.VALIDATION/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))['candidates']['guarded_loose']
    return geo,drift,cells,truth,stereo,pairs,p,baseline,coarse


def make_pool(eid):
    geo,drift,cells,truth,stereo,pairs,p,baseline,coarse=load_event(eid)
    occupied={i for t in coarse for i in t['indices'] if stereo[i]}
    candidates,x=generate(geo,drift,cells,pairs,p,baseline,occupied)
    _,found=score_tracks(baseline,truth,stereo,DEFAULT_CONFIG);sizes={int(t):int(((truth==t)&~stereo).sum()) for t in np.unique(truth) if t>=0}
    y=[];labels=[]
    for t in candidates:
        counts,matched=score_tracks([t],truth,stereo,DEFAULT_CONFIG)
        good=bool(matched);novel=bool(good and matched[0] not in found)
        y.append(int(novel));labels.append('novel_true_track' if novel else 'clone_of_found_truth' if good else 'fake')
    folder=OUT/'pool';folder.mkdir(exist_ok=True);path=folder/f'event_{eid:05d}.npz';temp=path.with_suffix('.npz.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,x=x,y=np.array(y,dtype=np.int8))
    temp.replace(path)
    row=dict(event_id=eid,candidates=candidates,diagnostic_labels=labels,arrays_sha256=e.sha(path),positives=sum(y),candidates_count=len(candidates))
    e.save_json(folder/f'event_{eid:05d}.json',row);return row


def evaluate(eid,probability):
    geo,drift,cells,truth,stereo,pairs,p,baseline,coarse=load_event(eid)
    row=json.loads((OUT/'pool'/f'event_{eid:05d}.json').read_text(encoding='utf-8'));methods={};seeds={};outputs={}
    for cut in CUTS:
        joint,ready,diag=reconstruct(geo,drift,cells,pairs,p,baseline,coarse,row['candidates'],probability,cut)
        name=str(cut);methods[name],_=score_tracks(joint,truth,stereo,DEFAULT_CONFIG);seeds[name]=seed_counts(ready,truth,stereo,cells)
        outputs[name]=dict(tracks=joint,coarse=ready,diagnostics=diag)
    return dict(event_id=eid,methods=methods,seeds=seeds,candidates=outputs)


def seed_summary(rows):
    from collections import Counter
    c=Counter()
    for row in rows:c.update(row)
    r=dict(c)
    r.update(efficiency=c['matched_tracks']/max(c['truth_tracks'],1),fake_fraction=c['fake_candidates']/max(c['output_candidates'],1),clone_fraction=c['clone_candidates']/max(c['output_candidates'],1))
    return r


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args();OUT.mkdir(exist_ok=True)
    contract=dict(event_ids=e.IDS,purpose='All 200 diagnostic validation events become dedicated recovery calibration; grouped OOF selects cut; separate new validation required',
                  baseline=e.PIN,gnn_weight_sha256=e.CONTRACT['weight_sha256'],proposal_config=CONFIG,features=FEATURES,cuts=CUTS,
                  source_sha256={p:e.sha(e.EXP/p) for p in ['weak_edge_recovery.py','STEP2_弱边候选校准.py','experiment_common.py']},
                  gnn_retrained=False,test_used=False)
    if (OUT/'contract.json').exists() and json.loads((OUT/'contract.json').read_text(encoding='utf-8'))!=contract:raise ValueError('Recovery calibration contract changed')
    e.save_json(OUT/'contract.json',contract);folder=OUT/'pool';folder.mkdir(exist_ok=True);rows={};started=time.perf_counter()
    for eid in e.IDS:
        path=folder/f'event_{eid:05d}.json'
        if path.exists():
            row=json.loads(path.read_text(encoding='utf-8'))
            if e.sha(folder/f'event_{eid:05d}.npz')!=row['arrays_sha256']:raise ValueError('Pool cache changed')
            rows[eid]=row
    with ProcessPoolExecutor(max_workers=args.workers,initializer=e.initialize_worker) as pool:
        for future in as_completed([pool.submit(make_pool,eid) for eid in e.IDS if eid not in rows]):
            row=future.result();rows[row['event_id']]=row
            if len(rows)%10==0:
                progress=dict(phase='candidate_pool',events=len(rows),total=len(e.IDS),seconds=time.perf_counter()-started,positives=sum(r['positives'] for r in rows.values()));e.save_json(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    matrices=[];labels=[];groups=[];slices={};offset=0
    for eid in e.IDS:
        with np.load(folder/f'event_{eid:05d}.npz') as f:x,y=f['x'],f['y']
        matrices.append(x);labels.append(y);groups.append(np.full(len(y),eid));slices[eid]=slice(offset,offset+len(y));offset+=len(y)
    x=np.concatenate(matrices);y=np.concatenate(labels);group=np.concatenate(groups)
    if len(np.unique(y))<2:raise ValueError('Candidate pool did not contain both usable new tracks and negatives')
    def model():return HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=15,l2_regularization=5.,early_stopping=False,random_state=20261110)
    oof=np.zeros(len(y))
    for train,valid in GroupKFold(5).split(x,y,group):
        m=model();m.fit(x[train],y[train]);oof[valid]=m.predict_proba(x[valid])[:,1]
    final=model();final.fit(x,y)
    with (OUT/'candidate_recovery_scorer.pkl').open('wb') as f:pickle.dump(final,f)
    np.savez_compressed(OUT/'event_group_oof.npz',probability=oof,y=y,event_id=group)
    result=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=e.initialize_worker) as pool:
        for future in as_completed([pool.submit(evaluate,eid,oof[slices[eid]]) for eid in e.IDS]):
            row=future.result();result.append(row)
            events=OUT/'oof_events';events.mkdir(exist_ok=True);e.save_json(events/f"event_{row['event_id']:05d}.json",row)
    metrics={str(c):dict(axial=summarize([r['methods'][str(c)] for r in result]),seed=seed_summary([r['seeds'][str(c)] for r in result])) for c in CUTS}
    baseline=metrics['1.1'];eligible=[]
    for name,m in metrics.items():
        a,b=m['axial'],baseline['axial'];s,t=m['seed'],baseline['seed']
        if a['track_efficiency']>=b['track_efficiency'] and a['fake_fraction']<=b['fake_fraction'] and a['clone_fraction']<=b['clone_fraction'] and s['efficiency']>=t['efficiency'] and s['fake_fraction']<=t['fake_fraction'] and s['clone_fraction']<=t['clone_fraction']:eligible.append(name)
    selected=max(eligible,key=lambda n:(metrics[n]['axial']['track_efficiency'],metrics[n]['seed']['efficiency'],-metrics[n]['axial']['fake_fraction']))
    summary=dict(contract=contract,pool_events=len(rows),candidates=len(y),positive_candidate_variants=int(y.sum()),
                 novel_truths_in_pool=sum(len({tuple(t['indices']) for t,label in zip(rows[eid]['candidates'],rows[eid]['diagnostic_labels']) if label=='novel_true_track'}) for eid in e.IDS),
                 metrics_oof=metrics,selected_cut=float(selected),scorer_sha256=e.sha(OUT/'candidate_recovery_scorer.pkl'),
                 selection='max axial efficiency, then coarse seed efficiency; neither A nor seed fake/clone fraction may worsen',seconds=time.perf_counter()-started)
    e.verify_pins();e.save_json(OUT/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
