"""Observable-only proposals for weak-edge residuals; protect the full baseline."""
import experiment_common as e
from itertools import combinations
import networkx as nx
import numpy as np
from scipy.optimize import least_squares
from drift_consensus_tracking import tangent_circles
from two_stage_core import fit_2d_circle
from STEP5_两级寻迹 import attach_stereo,DEFAULT_CONFIG
from STEP12_候选可靠性校准 import features as original_features,FEATURES as ORIGINAL_FEATURES
from coarse_3d_membership import propose,select
from joint_candidate_tracking import exclusive_selection

CONFIG=dict(thresholds=[.25,.40,.55],resolutions=[1.,2.],random_triplets=64,max_candidates=64,
            seed_link_threshold=.20,grow_residual_mm=1.,accept_rms_mm=1.25,seed=20261110)
FEATURES=ORIGINAL_FEATURES+['max_old_link_mean','max_old_circle_coverage','old_supported_fraction',
                          'weak_edge_coverage','median_edge_probability','free_stereo_support',
                          'coarse_distinct_support','coarse_quality_per_support','coarse_inlier_s',
                          'coarse_ss_mean','coarse_layer_count','coarse_ambiguous','coarse_ready']


def generate(geo,drift,cells,pairs,p,baseline,protected_stereo=()):
    """No label argument or label fields are read. All proposed A is unused."""
    n=len(cells);stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    occupied_a={i for t in baseline for i in t['indices'] if not stereo[i]}
    occupied_s={i for t in baseline for i in t['indices'] if stereo[i]}|set(protected_stereo)
    free=np.array([i for i in range(n) if not stereo[i] and i not in occupied_a],dtype=int)
    matrix=np.zeros((n,n),dtype=np.float32);matrix[pairs[0],pairs[1]]=p;matrix[pairs[1],pairs[0]]=p
    if len(free)<5:return [],np.empty((0,len(FEATURES)))
    raw={}
    def add(members,circle=None,origin='community'):
        members=np.array(sorted(set(map(int,members))),dtype=int)
        if len(members)<3:return
        if circle is None:
            xy=geo[members,:2];circle=fit_2d_circle(xy[:,0],xy[:,1],drift[members])
            if circle is None:return
        circle=np.asarray(circle)
        if not np.isfinite(circle).all() or circle[2]<=0:return
        error=np.abs(np.abs(np.linalg.norm(geo[free,:2]-circle[:2],axis=1)-circle[2])-drift[free])
        links=matrix[np.ix_(free,members)]
        support=(links>=CONFIG['seed_link_threshold']).sum(1)>=2
        support[np.isin(free,members)]=True
        grown=free[(error<=CONFIG['grow_residual_mm'])&support]
        for indices in [members,grown]:
            if len(indices)<5:continue
            key=tuple(indices.tolist());block=matrix[np.ix_(indices,indices)]
            values=block[np.triu_indices(len(indices),1)];observed=values[values>0]
            mean=float(observed.mean()) if len(observed) else 0.
            quality=len(indices)*(.25+mean)-float(error[np.isin(free,indices)].mean())
            if key not in raw or quality>raw[key][0]:raw[key]=(quality,circle,origin)
    for joint in [False,True]:
        allowed=np.array([i for i in range(n) if i in set(free) or (joint and stereo[i] and i not in occupied_s)],dtype=int)
        permitted=np.zeros(n,dtype=bool);permitted[allowed]=True
        for threshold in CONFIG['thresholds']:
            keep=(p>=threshold)&permitted[pairs[0]]&permitted[pairs[1]]
            graph=nx.Graph();graph.add_nodes_from(map(int,allowed))
            graph.add_weighted_edges_from((int(a),int(b),float(v)) for a,b,v in zip(pairs[0,keep],pairs[1,keep],p[keep]))
            if not graph.number_of_edges():continue
            for resolution in CONFIG['resolutions']:
                for group in nx.community.louvain_communities(graph,weight='weight',resolution=resolution,seed=CONFIG['seed']):
                    aa=sorted(i for i in group if not stereo[i])
                    if len(aa)>=3:add(aa,origin='weak_joint' if joint else 'weak_AA')
    blocks=matrix[np.ix_(free,free)];eligible=np.flatnonzero((blocks>=.25).sum(1)>=2)
    rng=np.random.default_rng(CONFIG['seed']);triples=set()
    if len(eligible):
        for _ in range(CONFIG['random_triplets']):
            center=int(rng.choice(eligible));neighbors=np.flatnonzero(blocks[center]>=.25)
            weights=blocks[center,neighbors]**3;other=rng.choice(neighbors,2,replace=False,p=weights/weights.sum())
            triple=tuple(sorted([int(free[center])]+free[other].tolist()))
            if len(np.unique(cells[list(triple)]))==3:triples.add(triple)
    for triple in sorted(triples):
        for circle in tangent_circles(geo[list(triple),:2],drift[list(triple)]):add(triple,circle,'weak_triplet')
    candidates=[];seen=set()
    for members,(_,initial,origin) in sorted(raw.items(),key=lambda item:item[1][0],reverse=True)[:CONFIG['max_candidates']*2]:
        aa=np.array(members,dtype=int);circle=initial
        for attempt in range(3):
            xy=geo[aa,:2];d=drift[aa]
            def residual(v):return np.abs(np.linalg.norm(xy-v[:2],axis=1)-v[2])-d
            fitted=least_squares(residual,circle,bounds=([-np.inf,-np.inf,.001],[np.inf,np.inf,np.inf]),
                                 loss='soft_l1',f_scale=.5,max_nfev=30)
            circle=fitted.x;r=residual(circle);rms=float(np.sqrt(np.mean(r*r)))
            if rms<=CONFIG['accept_rms_mm']:break
            if len(aa)<=5 or attempt==2:break
            aa=np.delete(aa,int(np.argmax(np.abs(r))))
        key=tuple(aa.tolist())
        if rms<=CONFIG['accept_rms_mm'] and key not in seen:
            seen.add(key);candidates.append(dict(indices=list(key),fit=circle.tolist(),rms=rms,stage=origin))
        if len(candidates)>=CONFIG['max_candidates']:break
    if not candidates:return [],np.empty((0,len(FEATURES)))
    protected=np.zeros(n,dtype=bool);protected[list(occupied_s)]=True
    available_p=np.where(protected[pairs[0]]|protected[pairs[1]],0.,p)
    witnessed=[attach_stereo([t],geo,drift,cells,pairs,available_p,DEFAULT_CONFIG)[0][0] for t in candidates]
    geometry=propose(witnessed,geo,drift,cells,pairs,available_p,refine=False)
    old_a=[np.array([i for i in t['indices'] if not stereo[i]],dtype=int) for t in baseline]
    rows=[]
    for t,witness,g in zip(candidates,witnessed,geometry):
        aa=np.array(t['indices']);internal=matrix[np.ix_(aa,aa)];v=internal[np.triu_indices(len(aa),1)];observed=v[v>0]
        cross_mean=0.;circle_coverage=0.;old_supported=0.
        for old in old_a:
            links=matrix[np.ix_(aa,old)];cross_mean=max(cross_mean,float(np.sort(links.ravel())[-min(4,links.size):].mean()))
            old_supported=max(old_supported,float(((links>=.5).sum(1)>=2).mean()))
            residual=np.abs(np.abs(np.linalg.norm(geo[old,:2]-np.array(t['fit'][:2]),axis=1)-t['fit'][2])-drift[old])
            circle_coverage=max(circle_coverage,float((residual<=1.5).mean()))
        h=g['hypotheses'][0] if g['hypotheses'] else None
        inliers=int((np.abs(h['residual'])<=1.5).sum()) if h else 0
        extras=[cross_mean,circle_coverage,old_supported,float((v>=.25).mean()) if len(v) else 0.,
                float(np.median(observed)) if len(observed) else 0.,len(witness.get('stereo_indices',[])),
                h['distinct_support'] if h else 0.,h['quality']/max(h['distinct_support'],1) if h else 0.,inliers,
                float(np.mean(g.get('ss_mean',[]))) if g.get('ss_mean') else 0.,h['independent_layers'] if h else 0.,
                float(g['status']=='ambiguous'),float(h is not None and inliers>=2)]
        rows.append(original_features(t,geo,cells,matrix)+extras)
    return candidates,np.asarray(rows,dtype=float)


def reconstruct(geo,drift,cells,pairs,p,baseline,baseline_coarse,candidates,probabilities,cut):
    keep=np.flatnonzero(probabilities>=cut)
    chosen=exclusive_selection([candidates[i] for i in keep],
                               [len(candidates[i]['indices'])*(.5+float(probabilities[i]))-1 for i in keep],len(cells))
    stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    occupied_s={i for t in baseline+baseline_coarse for i in t['indices'] if stereo[i]}
    protected=np.zeros(len(cells),dtype=bool);protected[list(occupied_s)]=True
    available_p=np.where(protected[pairs[0]]|protected[pairs[1]],0.,p)
    additions,_=attach_stereo(chosen,geo,drift,cells,pairs,available_p,DEFAULT_CONFIG)
    coarse,diag=select(additions,propose(additions,geo,drift,cells,pairs,available_p,refine=False),geo,cells,'guarded_loose') if additions else ([],{})
    joint=[dict(t,indices=list(t['indices'])) for t in baseline]+additions
    full_coarse=[dict(t,indices=list(t['indices'])) for t in baseline_coarse]+coarse
    for groups in [joint,full_coarse]:
        used=[i for t in groups for i in t['indices']];assert len(used)==len(set(used))
    return joint,full_coarse,dict(additions=len(additions),**diag)
