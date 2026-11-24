"""Conservative confidence fusion: protect baseline, assess new alternatives."""
import numpy as np
from joint_candidate_tracking import exclusive_selection


def hybrid_select(candidates,probability,baseline_indices,n,baseline_cut,recovery_cut):
    baseline_indices=set(baseline_indices)
    protected=[candidates[i] for i in sorted(baseline_indices) if probability[i]>=baseline_cut]
    used={hit for t in protected for hit in t['indices']}
    additions=[];quality=[]
    for i,t in enumerate(candidates):
        if i in baseline_indices or probability[i]<recovery_cut or used.intersection(t['indices']):continue
        additions.append(t);quality.append(len(t['indices'])*(.5+float(probability[i]))-1)
    selected=protected+exclusive_selection(additions,quality,n)
    members=[i for t in selected for i in t['indices']];assert len(members)==len(set(members))
    return selected
