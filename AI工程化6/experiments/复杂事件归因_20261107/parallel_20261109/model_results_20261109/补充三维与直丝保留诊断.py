"""Offline labels only: paired coarse-seed uncertainty and preservation of old correct A hits."""
from pathlib import Path
from collections import Counter
import json
import numpy as np

root=Path(__file__).resolve().parent
plan=json.loads((root/'evaluation_contract.json').read_text())
pin=json.loads((root.parent.parent/'experiment_config.json').read_text(encoding='utf-8'))
data=Path(pin['project'])/'joint_event_dataset';geometry=np.load(data/'wire_geometry.npy')
ids=plan['event_ids'];models=list(plan['models']);reference={};truths={};old_groups={}
for eid in ids:
    packet=json.loads((root.parent/'extractor_results/validation/events'/('event_%05d.json'%eid)).read_text())
    reference[eid]=packet;old_groups[eid]=packet['axial_members']['baseline']
    with np.load(data/'events'/('event_%05d.npz'%eid)) as f:
        truth=f['truth'];cells=f['cell_id']
    axial=np.linalg.norm(geometry[cells,2:4],axis=1)<=1e-9
    truths[eid]=(truth,Counter(truth[axial]))


def correct_hits(groups,truth,sizes):
    result=set()
    for group in groups:
        if not group:continue
        tid,count=Counter(truth[group]).most_common(1)[0]
        if tid>=0 and sizes[tid]>=5 and len(group)>=5 and count/len(group)>=.8 and count/sizes[tid]>=.5:
            result.update(i for i in group if truth[i]==tid)
    return result


samples=np.random.default_rng(20261109).integers(len(ids),size=(2000,len(ids)))
den=np.array([reference[i]['seeds']['baseline']['truth_tracks'] for i in ids]);results={}
for model in models:
    rows={r['event_id']:r for r in (json.loads(p.read_text()) for p in (root/'events'/model).glob('event_*.json'))}
    assert set(rows)==set(ids),model+' not complete'
    delta=np.array([rows[i]['seed']['matched_tracks']-reference[i]['seeds']['baseline']['matched_tracks'] for i in ids])
    ci=np.quantile(delta[samples].sum(1)/den[samples].sum(1),[.025,.975]).tolist()
    counts=Counter()
    for eid in ids:
        old={i for g in old_groups[eid] for i in g};new={i for g in rows[eid]['axial_members'] for i in g}
        truth,sizes=truths[eid];correct_old=correct_hits(old_groups[eid],truth,sizes);correct_new=correct_hits(rows[eid]['axial_members'],truth,sizes)
        counts.update(old_axial_hits=len(old),retained_old_axial_hits=len(old&new),new_axial_hits=len(new),
                      old_correct_axial_hits=len(correct_old),old_correct_axial_hits_still_correct=len(correct_old&correct_new))
    results[model]=dict(counts,old_axial_hit_retention=counts['retained_old_axial_hits']/counts['old_axial_hits'],
                       old_correct_hit_correct_assignment_retention=counts['old_correct_axial_hits_still_correct']/counts['old_correct_axial_hits'],
                       seed_efficiency_difference=float(delta.sum()/den.sum()),seed_efficiency_difference_95_ci=ci)
(root/'coarse_and_axial_diagnostics.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results))
