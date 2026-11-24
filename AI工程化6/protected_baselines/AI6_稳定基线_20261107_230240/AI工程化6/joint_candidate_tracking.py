"""Joint AA/AS/SS proposals and exclusive candidate selection, no truth inputs."""
import networkx as nx
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csc_matrix
from two_stage_core import fit_2d_circle
from STEP5_两级寻迹 import attach_stereo,DEFAULT_CONFIG


def fit_axial(indices,geo,drift):
    keep=np.array(sorted(set(indices)),dtype=int)
    if len(keep)<5:return None
    for attempt in range(3):
        xy=geo[keep,:2];d=drift[keep]
        fit=fit_2d_circle(xy[:,0],xy[:,1],d)
        if fit is None or not np.isfinite(fit).all():return None
        residual=np.abs(np.linalg.norm(xy-np.asarray(fit[:2]),axis=1)-fit[2])-d
        rms=float(np.sqrt(np.mean(residual**2)))
        if rms<=1.:return dict(indices=keep.tolist(),fit=list(fit),rms=rms,stage='joint_proposal')
        if len(keep)<=5 or attempt==2:return None
        keep=np.delete(keep,int(np.argmax(np.abs(residual))))


def proposal_pool(geometry,drift,cells,pairs,p,baseline,baseline_joint,full=False):
    stereo=np.linalg.norm(geometry[:,2:4],axis=1)>1e-9;n=len(cells)
    used_a={i for t in baseline for i in t['indices']}
    used_s={i for t in baseline_joint for i in t.get('stereo_indices',[])}
    matrix=np.zeros((n,n),dtype=np.float32);matrix[pairs[0],pairs[1]]=p;matrix[pairs[1],pairs[0]]=p
    allowed=np.array([i for i in range(n) if (stereo[i] and i not in used_s) or (not stereo[i] and (full or i not in used_a))],dtype=np.int64)
    pool=[dict(t,indices=list(t['indices']),origin='baseline') for t in baseline] if full else []
    seen={tuple(t['indices']) for t in pool}
    # A few fixed scales expose alternatives instead of trusting one component cut.
    for threshold,resolution in [(.55,1.),(.65,1.),(.65,2.),(.75,1.)]:
        permitted=np.zeros(n,dtype=bool);permitted[allowed]=True
        mask=(p>=threshold)&permitted[pairs[0]]&permitted[pairs[1]]
        graph=nx.Graph();graph.add_nodes_from(map(int,allowed))
        graph.add_weighted_edges_from((int(a),int(b),float(prob)) for a,b,prob in zip(pairs[0,mask],pairs[1,mask],p[mask]))
        if not graph.number_of_edges():continue
        communities=nx.community.louvain_communities(graph,weight='weight',resolution=resolution,seed=20261107)
        for group in communities:
            aa=sorted(i for i in group if not stereo[i])
            if len(aa)<5 or tuple(aa) in seen:continue
            candidate=fit_axial(aa,geometry,drift)
            if candidate is None:continue
            key=tuple(candidate['indices'])
            if key in seen:continue
            seen.add(key);candidate['origin']='joint_community';pool.append(candidate)
    return pool,matrix


def candidate_evidence(track,geo,cells,matrix):
    aa=np.array(track['indices']);stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9;ss=np.flatnonzero(stereo)
    internal=matrix[np.ix_(aa,aa)];observed=internal[np.triu_indices(len(aa),1)];observed=observed[observed>0]
    mean=float(observed.mean()) if len(observed) else 0.
    values=matrix[np.ix_(ss,aa)];votes=(values>=.7).sum(1)
    support=ss[votes>=2]
    # Distinct independent wires are supporting evidence, not duplicated CellID votes.
    if len(support):
        support=np.array([s for s in support if len(np.unique(cells[aa[matrix[s,aa]>=.7]]))>=2])
    if len(support)>=2:
        inter=matrix[np.ix_(support,support)];v=inter[np.triu_indices(len(support),1)];v=v[v>0]
        coherence=float(v.mean()) if len(v) else 0.
    else:coherence=0.
    outside=np.flatnonzero(~stereo & ~np.isin(np.arange(len(geo)),aa))
    out=matrix[np.ix_(outside,aa)]
    continuation=int(((out>=.75).sum(1)>=2).sum()) if len(outside) else 0
    return dict(axial_hits=len(aa),aa_mean=mean,stereo_support=len(np.unique(cells[support])) if len(support) else 0,
                ss_coherence=coherence,outside_continuations=continuation,rms=track['rms'])


def exclusive_selection(pool,quality,n):
    if not pool:return []
    rows=[];cols=[]
    for j,t in enumerate(pool):
        rows.extend(t['indices']);cols.extend([j]*len(t['indices']))
    incidence=csc_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,len(pool)))
    result=milp(c=-np.asarray(quality),integrality=np.ones(len(pool)),bounds=Bounds(0,1),
                constraints=LinearConstraint(incidence,0,1),options={'time_limit':5.,'mip_rel_gap':0.001})
    if result.x is None:
        order=np.argsort(quality)[::-1];used=set();selected=[]
        for j in order:
            if quality[j]>0 and not used.intersection(pool[j]['indices']):selected.append(pool[j]);used.update(pool[j]['indices'])
        return selected
    return [t for t,x in zip(pool,result.x) if x>.5]


def reconstruct(geo,drift,cells,pairs,p,baseline,baseline_joint):
    residual,matrix=proposal_pool(geo,drift,cells,pairs,p,baseline,baseline_joint,False)
    full,_=proposal_pool(geo,drift,cells,pairs,p,baseline,baseline_joint,True)
    variants={};diagnostics={}
    for name,pool,strict in [('joint_residual',residual,False),('joint_supported',residual,True),('joint_global',full,True)]:
        accepted=[];scores=[];features=[]
        for t in pool:
            e=candidate_evidence(t,geo,cells,matrix);features.append(dict(origin=t['origin'],**e))
            if t['origin']!='baseline' and (e['aa_mean']<.60 or (strict and (e['stereo_support']<4 or e['ss_coherence']<.70))):continue
            utility=e['axial_hits']*(.5+e['aa_mean'])+.15*min(e['stereo_support'],20)-2*e['rms']-2
            accepted.append(t);scores.append(utility)
        picked=exclusive_selection(accepted,scores,len(geo))
        if name!='joint_global':picked=[dict(t,indices=list(t['indices'])) for t in baseline]+picked
        used=[i for t in picked for i in t['indices']];assert len(used)==len(set(used))
        joint,_=attach_stereo(picked,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
        variants[name]=joint;diagnostics[name]=dict(proposed=len(pool),eligible=len(accepted),selected=len(picked),features=features)
    return variants,diagnostics
