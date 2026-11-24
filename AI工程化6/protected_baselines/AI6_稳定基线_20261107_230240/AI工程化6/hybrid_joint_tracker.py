"""Frozen conservative joint tracker; inference accepts observations only."""
import numpy as np
from joint_candidate_tracking import reconstruct
from STEP12_候选可靠性校准 import features
from hybrid_candidate_tracking import hybrid_select
from STEP5_两级寻迹 import attach_stereo,DEFAULT_CONFIG


def track(geo,drift,cells,pairs,p,baseline,baseline_joint,model,base_cut=.02,new_cut=.2):
    stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    alternatives,_=reconstruct(geo,drift,cells,pairs,p,baseline,baseline_joint)
    candidates=[];seen=set()
    for group in [baseline,alternatives['joint_residual'],alternatives['joint_global']]:
        for t in group:
            aa=tuple(sorted(i for i in t['indices'] if not stereo[i]))
            if aa in seen:continue
            seen.add(aa);candidates.append(dict(t,indices=list(aa)))
    if not candidates:return []
    matrix=np.zeros((len(cells),len(cells)),dtype=np.float32);matrix[pairs[0],pairs[1]]=p;matrix[pairs[1],pairs[0]]=p
    probability=model.predict_proba(np.array([features(t,geo,cells,matrix) for t in candidates]))[:,1]
    lookup={tuple(t['indices']):i for i,t in enumerate(candidates)}
    protected=[lookup[tuple(sorted(t['indices']))] for t in baseline]
    selected=hybrid_select(candidates,probability,protected,len(cells),base_cut,new_cut)
    joint,_=attach_stereo(selected,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
    used=[i for t in joint for i in t['indices']];assert len(used)==len(set(used))
    return joint
