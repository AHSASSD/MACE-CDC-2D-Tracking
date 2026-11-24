"""Deployable observation-only candidate confidence and global selection."""
import numpy as np
from joint_candidate_tracking import reconstruct,exclusive_selection
from STEP12_候选可靠性校准 import features
from STEP5_两级寻迹 import attach_stereo,DEFAULT_CONFIG


def calibrated_reconstruct(geo,drift,cells,pairs,p,baseline,baseline_joint,model,threshold):
    stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    alternatives,_=reconstruct(geo,drift,cells,pairs,p,baseline,baseline_joint)
    candidates=[];seen=set()
    for group in [baseline,alternatives['joint_residual'],alternatives['joint_global']]:
        for track in group:
            aa=tuple(sorted(i for i in track['indices'] if not stereo[i]))
            if aa in seen:continue
            seen.add(aa);candidates.append(dict(track,indices=list(aa)))
    if not candidates:return [],dict(proposals=0,accepted=0)
    matrix=np.zeros((len(cells),len(cells)),dtype=np.float32)
    matrix[pairs[0],pairs[1]]=p;matrix[pairs[1],pairs[0]]=p
    x=np.array([features(t,geo,cells,matrix) for t in candidates])
    probability=model.predict_proba(x)[:,1];keep=np.flatnonzero(probability>=threshold)
    pool=[candidates[i] for i in keep];quality=[len(candidates[i]['indices'])*(.5+float(probability[i]))-1 for i in keep]
    selected=exclusive_selection(pool,quality,len(geo))
    joint,_=attach_stereo(selected,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
    used=[i for t in joint for i in t['indices']];assert len(used)==len(set(used))
    return joint,dict(proposals=len(candidates),eligible=len(pool),accepted=len(joint))
