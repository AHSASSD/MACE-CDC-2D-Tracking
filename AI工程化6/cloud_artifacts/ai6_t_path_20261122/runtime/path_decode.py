"""Observable-only directed triplet paths, with deferred measurement assignment."""
import bootstrap
import math,time
from dataclasses import dataclass,asdict
import numpy as np
import torch
from scipy.sparse import csc_matrix
from scipy.optimize import milp,Bounds,LinearConstraint
from torch_geometric.data import Data
from two_stage_core import components,trim_outliers,hierarchy_partition,split_component,fit_candidate
from STEP5_两级寻迹 import circle_wire_lower_bound
from coarse_3d_membership import propose

@dataclass(frozen=True)
class Config:
    q_min:float=.2
    cosine_min:float=.0
    out_degree:int=3
    seed_limit:int=128
    beam_width:int=4
    max_hits:int=32
    fit_limit:int=48
    rms_limit:float=1.0
    solver_seconds:float=2.

def path_hypotheses(f,t,q,geo,config=Config()):
    """States contain entire ordered paths; alternatives may reuse observations here.
    No global used-hit set or connected-component union is applied to triplet links.
    Entry history across a shared segment survives in each beam hypothesis.
    """
    axial=~f['stereo'];a,c,b,i,j=t.T;xy=geo[:,:2]
    v1=xy[c]-xy[a];v2=xy[b]-xy[c]
    l1=np.linalg.norm(v1,axis=1);l2=np.linalg.norm(v2,axis=1)
    cosine=(v1*v2).sum(1)/np.maximum(l1*l2,1e-9)
    keep=axial[a]&axial[c]&axial[b]&(q>=config.q_min)&(cosine>=config.cosine_min)&(l1>1)&(l2>1)
    transitions={}
    for index in np.flatnonzero(keep):
        x,y,z=int(a[index]),int(c[index]),int(b[index]);prob=float(q[index])
        for u,v,w in [(x,y,z),(z,y,x)]:
            transitions.setdefault((u,v),[]).append((w,prob,float(np.linalg.norm(xy[w]-xy[v]))))
    # Keep close, plausible steps; preserve several exits rather than one max-q edge.
    for state,values in transitions.items():
        shortest=min(x[2] for x in values)
        values.sort(key=lambda x:-(np.log(max(x[1],1e-6))-.25*np.log1p(x[2]/max(shortest,1))))
        transitions[state]=values[:config.out_degree]
    # Seed diversity is a search budget only, never ownership of hits or regions.
    incident=np.zeros(len(geo),np.float32)
    np.maximum.at(incident,f['pairs'][0],f['p']);np.maximum.at(incident,f['pairs'][1],f['p'])
    ranked=sorted(transitions,key=lambda s:-(max(v[1] for v in transitions[s])+.15*float(incident[s[0]])))
    seeds=[];visits={}
    for state in ranked:
        if visits.get(state[0],0)>=1:continue
        seeds.append(state);visits[state[0]]=visits.get(state[0],0)+1
        if len(seeds)>=config.seed_limit:break
    saved={};expanded=0
    def key(path):return min(path,tuple(reversed(path)))
    def record(path,qs,log_sum):
        if len(path)<5:return
        value=.8*len(path)+.3*log_sum
        k=key(path)
        if k not in saved or saved[k][0]<value:saved[k]=(value,path,qs)
    for seed in seeds:
        beam=[(tuple(seed),tuple(),0.)]
        for depth in range(config.max_hits-2):
            branches={}
            for path,qs,log_sum in beam:
                record(path,qs,log_sum)
                for front,state in [(False,(path[-2],path[-1])),(True,(path[1],path[0]))]:
                    for node,prob,_ in transitions.get(state,[]):
                        if node in path:continue
                        new=(node,)+path if front else path+(node,);newq=qs+(prob,);new_log=log_sum+math.log(max(prob,1e-6))
                        k=key(new);value=.8*len(new)+.3*new_log
                        if k not in branches or value>branches[k][0]:branches[k]=(value,new,newq,new_log)
                        expanded+=1
            if not branches:break
            beam=[(x[1],x[2],x[3]) for x in sorted(branches.values(),reverse=True)[:config.beam_width]]
        for path,qs,log_sum in beam:record(path,qs,log_sum)
    # Keep distinct membership alternatives; suppress redundant prefixes only when
    # a higher-valued full path contains them. Competing branches remain separate.
    selected=[];sets=[]
    for value,path,qs in sorted(saved.values(),reverse=True):
        members=frozenset(path)
        if any(members<=s for s in sets):continue
        if sum(len(members&s)/len(members|s)>.8 for s in sets)>=2:continue
        selected.append(dict(path=list(path),triplet_probabilities=list(qs),search_utility=value));sets.append(members)
        if len(selected)>=config.fit_limit:break
    return selected,dict(transitions=len(transitions),seeds=len(seeds),expanded=expanded,paths=len(selected),config=asdict(config))

def no_lock_candidates(f,geo,d):
    aa=np.flatnonzero(~f['stereo']);mapping=np.full(len(d),-1,int);mapping[aa]=np.arange(len(aa))
    mask=~f['stereo'][f['pairs'][0]]&~f['stereo'][f['pairs'][1]]
    data=Data(wire_x=torch.tensor(geo[aa,0]),wire_y=torch.tensor(geo[aa,1]),drift_d=torch.tensor(d[aa]),edge_index=torch.tensor(mapping[f['pairs'][:,mask]]).long(),num_nodes=len(aa))
    candidates=[]
    # Both passes see all observations. Accepted strict candidates do not hide
    # measurements from relaxed candidate generation.
    for cut in [.7,.575]:
        for group in components(data,f['p'][mask],cut):
            if len(group)>80:continue
            values=hierarchy_partition(data,group,f['p'][mask],cut) or split_component(data,group,f['p'][mask],cut,1. if cut==.7 else 1.5)
            for c in values:c=dict(c,indices=aa[c['indices']].tolist(),origin='unlocked_geometry');candidates.append(c)
    return candidates

def decode(f,t,q,geo,d,baseline,config=Config()):
    started=time.perf_counter();paths,diag=path_hypotheses(f,t,q,geo,config);diag['path_search_seconds']=time.perf_counter()-started
    aa=np.flatnonzero(~f['stereo']);mapping=np.full(len(d),-1,int);mapping[aa]=np.arange(len(aa))
    mask=~f['stereo'][f['pairs'][0]]&~f['stereo'][f['pairs'][1]]
    data=Data(wire_x=torch.tensor(geo[aa,0]),wire_y=torch.tensor(geo[aa,1]),drift_d=torch.tensor(d[aa]),edge_index=torch.tensor(mapping[f['pairs'][:,mask]]).long(),num_nodes=len(aa))
    started=time.perf_counter();candidates=[dict(c,origin='baseline_proposal') for c in baseline]+no_lock_candidates(f,geo,d);diag['unlocked_candidate_seconds']=time.perf_counter()-started
    started=time.perf_counter()
    for path in paths:
        c=fit_candidate(data,mapping[path['path']].tolist(),f['p'][mask],.4,config.rms_limit,'triplet_path')
        if c and c['accepted_geometry']:
            c.update(indices=aa[c['indices']].tolist(),origin='directed_triplet_path',path=path['path'],path_q=path['triplet_probabilities']);candidates.append(c)
    unique={}
    for c in candidates:
        key=tuple(sorted(c['indices']))
        if key not in unique or c['rms']<unique[key]['rms']:unique[key]=c
    candidates=list(unique.values());diag['fitted_candidates']=len(candidates);diag['path_fit_seconds']=time.perf_counter()-started
    if not candidates:return [],diag
    # Candidate-local coarse 3D evidence is computed before observation ownership.
    # propose() does not perform stereo assignment and accepts overlapping paths.
    started=time.perf_counter();hypotheses=propose(candidates,geo,d,f['cells'],f['pairs'],f['p'],refine=False);diag['preselection_3d_seconds']=time.perf_counter()-started
    aa_idx=f['pairs'];pa=f['p'];ta,tb,tc=t[:,:3].T
    utilities=[]
    for c,h in zip(candidates,hypotheses):
        member=np.zeros(len(d),bool);member[c['indices']]=True
        internal=member[ta]&member[tb]&member[tc]&~f['stereo'][ta]&~f['stereo'][tb]&~f['stereo'][tc]
        support=np.zeros(len(d),float)
        if internal.any():
            for col in [ta,tb,tc]:np.maximum.at(support,col[internal],q[internal])
        coherence=float(np.mean(support[c['indices']]))
        evidence=float(np.sqrt(max(c['mean_probability'],0)*max(coherence,.05)))
        geometric=np.exp(-.5*(c['rms']/.7)**2)
        hs=h['hypotheses'];distinct=max([x['distinct_support'] for x in hs]+[0])
        stereo_bonus=.6*min(distinct/6.,1.)
        utility=c['hits']*(.35+.65*evidence)*geometric-2.+stereo_bonus
        c.update(path_state_utility=float(utility),triplet_coherence=coherence,preselection_stereo_support=int(distinct))
        utilities.append(utility)
    rr=[];cc=[]
    for j,c in enumerate(candidates):rr.extend(c['indices']);cc.extend([j]*len(c['indices']))
    A=csc_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(d),len(candidates)))
    solution=milp(-np.asarray(utilities),integrality=np.ones(len(candidates)),bounds=Bounds(0,1),constraints=LinearConstraint(A,0,1),options={'time_limit':config.solver_seconds,'mip_rel_gap':.001})
    # Fail conservatively rather than falling back to an early-locking heuristic.
    selected=[c for c,x in zip(candidates,solution.x) if x>.5] if solution.x is not None else list(baseline)
    assigned=[i for c in selected for i in c['indices']];assert len(assigned)==len(set(assigned))
    diag.update(solver_status=int(solution.status),solver_fallback=solution.x is None,output_candidates=len(selected),proposed_path_candidates=sum(c['origin']=='directed_triplet_path' for c in candidates),selected_path_candidates=sum(c.get('origin')=='directed_triplet_path' for c in selected))
    return selected,diag
