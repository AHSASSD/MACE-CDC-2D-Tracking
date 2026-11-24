"""Revisit the exact old223 misses; labels are used only in offline attribution."""
from pathlib import Path
import json
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import numpy as np
import STEP7_训练结果与同流程复测 as r
from STEP4_真边分数与连通性诊断 import topology

ROOT=r.OUT/'case_attribution_20261109'
MODELS=['control_012','balanced_012','balanced_015']
CATEGORIES=['broadly_low_scores','pure_connected_at_existing_relaxed_cut','mixed_fragmentation','few_weak_connectors','candidate_graph_insufficient']
LABELS=['真边普遍低分','宽松阈值下纯真边已足够连通','混合碎裂','少数关键连接边低分','AA候选图不足']
r.MODELS['balanced_015']=('balanced',15)


def evaluate(key,eid):
    # The deployable reconstruction receives measurements and predictions only.
    reconstructed=r.event_evaluate(key,eid)
    old=json.loads((r.e.EXP/'edge_score_diagnosis_v1/events'/('event_%05d.json'%eid)).read_text(encoding='utf-8'))
    with np.load(r.e.DATA_ROOT/'events'/('event_%05d.npz'%eid)) as f:truth,cells=f['truth'],f['cell_id']
    geometry=np.load(r.e.DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geometry[:,2:4],axis=1)>1e-9
    with np.load(r.e.SNAPSHOT/('validation_prediction_cache_'+r.e.CONTRACT['weight_sha256'][:12])/('event_%05d.npz'%eid)) as f:pairs,old_p=f['pairs'],f['probabilities']
    with np.load(r.OUT/'prediction_cache'/key/('event_%05d.npz'%eid)) as f:
        assert np.array_equal(pairs,f['pairs']), 'Candidate topology changed'
        new_p=f['probabilities']
    same=(truth[pairs[0]]==truth[pairs[1]])&(truth[pairs[0]]>=0)
    aa=~stereo[pairs[0]]&~stereo[pairs[1]];found=set(reconstructed['matched'])
    old_found={t['truth_id'] for t in old['tracks'] if t['found']};rows=[]
    for t in old['tracks']:
        if t['found']:continue
        tid=t['truth_id'];hits=np.flatnonzero((truth==tid)&~stereo)
        mapping=np.full(len(truth),-1,dtype=int);mapping[hits]=np.arange(len(hits))
        true_mask=aa&same&(truth[pairs[0]]==tid)
        incident_false=aa&~same&((truth[pairs[0]]==tid)|(truth[pairs[1]]==tid))
        new=topology(len(hits),mapping[pairs[:,true_mask]],new_p[true_mask]);new.pop('repair_local_edge_indices')
        assert new['true_candidate_edges']==t['true_candidate_edges']
        old_connected=t['largest_relaxed_component']>=t['pure_match_hits_required']
        new_connected=new['largest_relaxed_component']>=new['pure_match_hits_required']
        rows.append(dict(truth_id=tid,old_category=t['category'],new_category=new['category'],recovered=tid in found,
                         old_connected=old_connected,new_connected=new_connected,
                         old_median=t['median_true_score'],new_median=new['median_true_score'],
                         old_below=t['below_relaxed_fraction'],new_below=new['below_relaxed_fraction'],
                         old_true_mean=float(old_p[true_mask].mean()) if true_mask.any() else None,
                         new_true_mean=float(new_p[true_mask].mean()) if true_mask.any() else None,
                         old_false_acceptance=float((old_p[incident_false]>=.575).mean()) if incident_false.any() else None,
                         new_false_acceptance=float((new_p[incident_false]>=.575).mean()) if incident_false.any() else None,
                         old_largest=t['largest_relaxed_component'],new_largest=new['largest_relaxed_component'],
                         old_match_connectivity=t['match_connectivity_score'],new_match_connectivity=new['match_connectivity_score']))
    missed={t['truth_id'] for t in old['tracks'] if not t['found']}
    assert set(found)-old_found<=missed
    return dict(event_id=eid,model=key,tracks=rows,counts=reconstructed['counts'],seed=reconstructed['seed'],
                recovered_old=sorted(found-old_found),lost_old=sorted(old_found-found),all_old_missed=sorted(missed))


def summarize(rows):
    groups={};tracks=[t for row in rows for t in row['tracks']]
    for category,label in zip(CATEGORIES,LABELS):
        ts=[t for t in tracks if t['old_category']==category]
        def delta(a,b):
            values=[t[b]-t[a] for t in ts if t[a] is not None and t[b] is not None]
            return float(np.mean(values)) if values else None
        groups[category]=dict(label=label,total=len(ts),recovered=sum(t['recovered'] for t in ts),
                             now_connected=sum(t['new_connected'] for t in ts),
                             became_connected=sum(not t['old_connected'] and t['new_connected'] for t in ts),
                             lost_connectivity=sum(t['old_connected'] and not t['new_connected'] for t in ts),
                             increased_median=sum(t['new_median'] is not None and t['old_median'] is not None and t['new_median']>t['old_median'] for t in ts),
                             mean_true_score_change=delta('old_true_mean','new_true_mean'),
                             mean_below_fraction_change=delta('old_below','new_below'),
                             mean_incident_false_acceptance_change=delta('old_false_acceptance','new_false_acceptance'),
                             transitions=dict(Counter(t['new_category'] for t in ts)))
    return dict(groups=groups,recovered=sum(len(r['recovered_old']) for r in rows),lost=sum(len(r['lost_old']) for r in rows),
                actual_tracks=r.summarize([row['counts'] for row in rows]),coarse_seed=r.seed_summary([row['seed'] for row in rows]))


def main():
    ROOT.mkdir(exist_ok=True);ids=r.e.IDS
    expected=Counter()
    for eid in ids:
        packet=json.loads((r.e.EXP/'edge_score_diagnosis_v1/events'/('event_%05d.json'%eid)).read_text(encoding='utf-8'))
        expected.update(t['category'] for t in packet['tracks'] if not t['found'])
    assert [expected[k] for k in CATEGORIES]==[83,88,48,2,2]
    plan=dict(event_ids=ids,models={key:dict(arm=r.MODELS[key][0],epoch=r.MODELS[key][1],weight_sha256=r.e.sha(r.OUT/'cloud'/r.MODELS[key][0]/('epoch_%03d.pth'%r.MODELS[key][1]))) for key in MODELS},
              old_categories=dict(expected),initial_weight_sha256=r.e.CONTRACT['weight_sha256'],
              source_sha256=r.e.sha(Path(__file__)),topology_source_sha256=r.e.sha(r.e.EXP/'STEP4_真边分数与连通性诊断.py'),
              scope='Exact frozen223 old misses in diagnostic200; scoring and truth attribution only, not independent generalization; no test selection or training',
              tracker='Stable hybrid .02/.2 + frozen candidate calibrator + guarded_loose coarse3D')
    if (ROOT/'contract.json').exists():assert json.loads((ROOT/'contract.json').read_text(encoding='utf-8'))==plan
    r.e.save_json(ROOT/'contract.json',plan);rows={key:{} for key in MODELS};tasks=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8,initializer=r.initialize) as pool:
        for key in MODELS:
            r.predict(key,ids)
            for eid in ids:
                path=ROOT/'events'/key/('event_%05d.json'%eid)
                if path.exists():rows[key][eid]=json.loads(path.read_text(encoding='utf-8'))
                else:tasks.append(pool.submit(evaluate,key,eid))
        for future in as_completed(tasks):
            row=future.result();key=row['model'];eid=row['event_id'];rows[key][eid]=row
            folder=ROOT/'events'/key;folder.mkdir(parents=True,exist_ok=True);r.e.save_json(folder/('event_%05d.json'%eid),row)
            n=sum(len(v) for v in rows.values())
            if n%20==0:
                progress=dict(events=n,total=len(ids)*len(MODELS),seconds=time.perf_counter()-started);r.e.save_json(ROOT/'progress.json',progress);print(json.dumps(progress),flush=True)
    result=dict(models={key:summarize([rows[key][i] for i in ids]) for key in MODELS},plan=plan,seconds=time.perf_counter()-started)
    r.e.verify_pins();r.e.save_json(ROOT/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
