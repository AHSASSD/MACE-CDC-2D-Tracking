"""Protected axial cores, rough longitudinal hypotheses and stereo consensus.

This is a member finder for downstream GenFit, not a precision helix fit.
No truth, timing, charge or magnetic-field assumptions enter reconstruction.
An unresolved longitudinal hypothesis never deletes an axial member or track.
"""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from STEP5_两级寻迹 import circle_wire_lower_bound

_spec=importlib.util.spec_from_file_location('finite_wire_helix_geometry',Path(__file__).with_name('三维拟合草稿_暂未启用.py'))
_geometry=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_geometry)

PROFILES={
    'precision':dict(residual_mm=1.,recovery_as=.65,margin=.12,trim_existing=True),
    'balanced':dict(residual_mm=1.5,recovery_as=.60,margin=.10,trim_existing=True),
    'recovery':dict(residual_mm=2.,recovery_as=.55,margin=.10,trim_existing=True),
    'protect_existing':dict(residual_mm=1.5,recovery_as=.60,margin=.10,trim_existing=False),
    'guarded_recovery':dict(residual_mm=1.,recovery_as=.70,margin=.10,trim_existing=True,guarded=True,recovery_ss=.70),
    'guarded_tight':dict(residual_mm=1.,recovery_as=.75,margin=.10,trim_existing=True,guarded=True,recovery_ss=.75),
    'guarded_loose':dict(residual_mm=1.5,recovery_as=.70,margin=.10,trim_existing=True,guarded=True,recovery_ss=.75),
    'guarded_keep':dict(residual_mm=1.,recovery_as=.75,margin=.10,trim_existing=False,guarded=True,recovery_ss=.75),
}
SEARCH=dict(max_pool=48,pair_seeds=512,retained_seeds=5,seed_tolerance_mm=2.,
            min_distinct_support=3,min_phase_separation=.08,min_arc_separation_mm=10.,parameter_bound=1e7,
            refine_evaluations=8,near_best_fraction=.90,arc_margin_mm=80.,wire_end_margin_mm=5.,axial_z_margin_mm=5.)


def finite_cylinder_intersections(circle,wire,radius_offset=0.,end_margin_mm=0.):
    """Exact roots along a finite wire; direction is unit length, t is mm.

    Inspired by Yu fit_sz_with_geometry: R +/- d encodes drift-side alternatives.
    The nominal cylinder alone is not a hard cut for a nonzero drift radius.
    """
    g=np.asarray(wire,dtype=float);radius=float(circle[2])+radius_offset
    if radius<=0:return []
    xy=g[:2]-np.asarray(circle[:2]);direction=g[2:4]
    a=float(direction@direction);b=2*float(xy@direction);c=float(xy@xy-radius**2)
    if a<1e-15:return []
    discriminant=b*b-4*a*c
    if discriminant < -1e-10*max(b*b,abs(4*a*c),1.):return []
    root=np.sqrt(max(discriminant,0.));result=[]
    for along in [(-b-root)/(2*a),(-b+root)/(2*a)]:
        if abs(along)>g[5]/2+end_margin_mm:continue
        if result and abs(along-result[-1]['along_wire_mm'])<1e-7:continue
        result.append(dict(along_wire_mm=float(along),x=float(g[0]+along*g[2]),
                           y=float(g[1]+along*g[3]),z=float(along*g[4]),radius_offset_mm=float(radius_offset)))
    return result


def axial_arc(circle,xy):
    phase=np.sort(np.mod(np.arctan2(xy[:,1]-circle[1],xy[:,0]-circle[0]),2*np.pi))
    gap=np.diff(np.r_[phase,phase[0]+2*np.pi]);start=phase[(int(gap.argmax())+1)%len(phase)]
    pad=min(SEARCH['arc_margin_mm']/circle[2],np.pi)
    return float(start-pad),float(min(2*np.pi-gap.max()+2*pad,2*np.pi))


def axial_longitudinal_check(seeds,circle,arc,axial_geo):
    phase=np.mod(np.arctan2(axial_geo[:,1]-circle[1],axial_geo[:,0]-circle[0])-arc[0],2*np.pi)
    predicted=seeds[:,0,None]+seeds[:,1,None]*phase[None,:]
    halfz=axial_geo[:,5]*np.abs(axial_geo[:,4])/2+SEARCH['axial_z_margin_mm']
    return (np.abs(predicted)<=halfz[None,:]).all(1)


def branch_observations(circle,arc,geo,drift):
    """XY circle +/- drift intersections supply alternatives, never measured z."""
    result=np.full((len(geo),6,3),np.nan,dtype=float)
    cx,cy,radius=map(float,circle);origin,span=arc
    for j,(g,d) in enumerate(zip(geo,drift)):
        slope=g[2:4]/g[4];offset=g[:2]-[cx,cy];a=float(slope@slope)
        if a<1e-14:continue
        branches=[]
        for signed_drift in [-d,d,0.]:
            for intersection in finite_cylinder_intersections(circle,g,signed_drift,SEARCH['wire_end_margin_mm']):
                z=intersection['z'];point=np.array([intersection['x'],intersection['y']]);radial=point-[cx,cy]
                phase=float(np.mod(np.arctan2(radial[1],radial[0])-origin,2*np.pi))
                if phase>span:continue
                radial_slope=abs(float(radial@slope))/max(float(np.linalg.norm(radial)),1e-9)
                if radial_slope<.002:continue  # Tangent geometry gives almost no z information.
                branches.append((phase,float(z),radial_slope))
        if branches:result[j,:len(branches)]=branches
    return result


def _branch_residual(seeds,branches):
    z=branches[None,:,:,1];phase=branches[None,:,:,0];scale=branches[None,:,:,2]
    residual=np.abs(z-seeds[:,None,None,0]-seeds[:,None,None,1]*phase)*scale
    return np.min(np.where(np.isfinite(residual),residual,np.inf),axis=2)


def _linear_refine(seed,branches,weights):
    seed=np.array(seed,dtype=float)
    for _ in range(4):
        residual=np.abs(branches[:,:,1]-seed[0]-seed[1]*branches[:,:,0])*branches[:,:,2]
        residual=np.where(np.isfinite(residual),residual,np.inf);choice=residual.argmin(1)
        b=branches[np.arange(len(branches)),choice];r=residual[np.arange(len(branches)),choice]
        keep=np.isfinite(r)&(r<4.)
        if keep.sum()<2:break
        design=np.column_stack([np.ones(keep.sum()),b[keep,0]])
        w=weights[keep]*b[keep,2]**2/np.maximum(1.,r[keep]/1.)
        solution,_,rank,_=np.linalg.lstsq(design*np.sqrt(w[:,None]),b[keep,1]*np.sqrt(w),rcond=None)
        if rank<2 or np.max(np.abs(solution))>SEARCH['parameter_bound']:break
        seed=solution
    return seed


def longitudinal_hypotheses(circle,arc,geo,drift,cells,weights,refine=True,axial_geo=None):
    """Two-hit proposals; consensus over other independent wires validates them."""
    branches=branch_observations(circle,arc,geo,drift)
    finite=np.argwhere(np.isfinite(branches[:,:,0]))
    if len(finite)<2:return [],'insufficient_intersections'
    observations=branches[finite[:,0],finite[:,1]]
    rng=np.random.default_rng(20261109)
    sampled=rng.integers(len(finite),size=(SEARCH['pair_seeds']*8,2))
    a,b=sampled.T;first,second=observations[a],observations[b]
    phase_difference=second[:,0]-first[:,0]
    phase_leverage=min(SEARCH['min_phase_separation'],SEARCH['min_arc_separation_mm']/circle[2])
    valid=(cells[finite[a,0]]!=cells[finite[b,0]])&(np.abs(phase_difference)>=phase_leverage)
    a,b=a[valid],b[valid];first,second=observations[a],observations[b]
    pitch=(second[:,1]-first[:,1])/(second[:,0]-first[:,0]);intercept=first[:,1]-pitch*first[:,0]
    seeds=np.column_stack([intercept,pitch]);seeds=seeds[np.max(np.abs(seeds),axis=1)<=SEARCH['parameter_bound']][:SEARCH['pair_seeds']]
    if axial_geo is not None:seeds=seeds[axial_longitudinal_check(seeds,circle,arc,axial_geo)]
    if len(seeds)==0:return [],'insufficient_phase_leverage'
    residual=_branch_residual(seeds,branches)
    votes=(weights[None,:]*np.exp(-.5*(residual/SEARCH['seed_tolerance_mm'])**2)).sum(1)
    chosen=[];anchors=np.array([0.,arc[1]/2,arc[1]])
    for index in np.argsort(votes)[::-1]:
        seed=_linear_refine(seeds[index],branches,weights)
        if axial_geo is not None and not axial_longitudinal_check(seed[None,:],circle,arc,axial_geo)[0]:continue
        if all(np.sqrt(np.mean(((seed[0]-s[0])+(seed[1]-s[1])*anchors)**2))>15. for s in chosen):chosen.append(seed)
        if len(chosen)>=SEARCH['retained_seeds']:break
    hypotheses=[]
    for seed in chosen:
        residual=_geometry.distances(circle,seed,arc,geo)-drift
        keep=np.flatnonzero(np.abs(residual)<SEARCH['seed_tolerance_mm'])
        if refine and len(keep)>=3:
            # The circle and all axial members stay fixed. Refine z0/pitch only.
            keep=keep[np.argsort(weights[keep])[::-1][:16]]
            fitted=least_squares(lambda z:_geometry.distances(circle,z,arc,geo[keep])-drift[keep],seed,
                                 bounds=(-SEARCH['parameter_bound'],SEARCH['parameter_bound']),
                                 loss='soft_l1',f_scale=.75,max_nfev=SEARCH['refine_evaluations'],
                                 x_scale=[100.,max(circle[2],100.)],diff_step=1e-4,ftol=1e-3,xtol=1e-3)
            refined_residual=_geometry.distances(circle,fitted.x,arc,geo)-drift
            if np.sum(weights*np.exp(-.5*(refined_residual/1.5)**2))>=np.sum(weights*np.exp(-.5*(residual/1.5)**2)):
                if axial_geo is None or axial_longitudinal_check(fitted.x[None,:],circle,arc,axial_geo)[0]:
                    seed=fitted.x;residual=refined_residual
        support=np.abs(residual)<=1.5
        distinct=len(np.unique(cells[support]))
        if distinct<SEARCH['min_distinct_support']:continue
        # A phase lever arm is needed even when three entries repeat one wire.
        supported_branches=branches[support,:,0]
        phases=supported_branches[np.isfinite(supported_branches)]
        if not len(phases) or np.ptp(phases)<phase_leverage:continue
        quality=float(np.sum(weights*np.exp(-.5*(residual/1.5)**2)))
        supported_geo=geo[support]
        layer_count=len(np.unique(np.round(np.linalg.norm(supported_geo[:,:2],axis=1),3)))
        orientations=len(np.unique(np.sign(supported_geo[:,0]*supported_geo[:,3]-supported_geo[:,1]*supported_geo[:,2])))
        hypotheses.append(dict(z0=float(seed[0]),pitch=float(seed[1]),arc=list(map(float,arc)),
                               residual=residual,quality=quality,distinct_support=distinct,
                               independent_layers=layer_count,stereo_orientations=orientations,
                               stereo_phase_span_rad=float(np.ptp(phases))))
    hypotheses.sort(key=lambda h:h['quality'],reverse=True)
    if not hypotheses:return [],'no_stereo_consensus'
    best=hypotheses[0]['quality'];near=[h for h in hypotheses if h['quality']>=best*SEARCH['near_best_fraction']]
    # Different longitudinal parameters may still agree on the useful hit subset.
    status='ambiguous' if any(np.sqrt(np.mean(((h['z0']-near[0]['z0'])+(h['pitch']-near[0]['pitch'])*anchors)**2))>30 for h in near[1:]) else 'supported'
    return near,status


def propose(tracks,geo,drift,cells,pairs,p,refine=True):
    """Generate all profile-independent hypotheses once per event."""
    n=len(cells);stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9;ss=np.flatnonzero(stereo)
    matrix=np.zeros((n,n),dtype=np.float32);matrix[pairs[0],pairs[1]]=p;matrix[pairs[1],pairs[0]]=p
    proposals=[]
    for t in tracks:
        aa=np.array([i for i in t['indices'] if not stereo[i]],dtype=int)
        previous=np.array([i for i in t['indices'] if stereo[i]],dtype=int)
        if not len(ss) or len(aa)<5:
            proposals.append(dict(pool=[],as_mean=[],strong_support=[],hypotheses=[],status='insufficient_hits'));continue
        values=matrix[np.ix_(ss,aa)];mean=np.sort(values,axis=1)[:,-2:].mean(1)
        support=np.array([len(np.unique(cells[aa[row>=.45]])) for row in values])
        compatible=circle_wire_lower_bound(t['fit'],geo[ss])<=drift[ss]+2.
        eligible=compatible&((support>=2)|np.isin(ss,previous))
        ids=np.flatnonzero(eligible)
        rank=mean[ids]+.05*np.isin(ss[ids],previous)
        pool=np.union1d(ss[ids[np.argsort(rank)[::-1][:SEARCH['max_pool']]]],previous)
        chosen=np.array([np.flatnonzero(ss==s)[0] for s in pool],dtype=int)
        _,inverse,counts=np.unique(cells[pool],return_inverse=True,return_counts=True)
        weights=np.clip(mean[chosen],.1,1.)/counts[inverse]
        arc=axial_arc(t['fit'],geo[aa,:2])
        hypotheses,status=longitudinal_hypotheses(t['fit'],arc,geo[pool],drift[pool],cells[pool],weights,refine,geo[aa])
        anchor=np.zeros(len(pool),dtype=bool)
        if hypotheses:
            residual=np.array([np.abs(h['residual']) for h in hypotheses])
            anchor=np.isin(pool,previous)&((residual<=1.5).mean(0)>=.5)&(mean[chosen]>=.80)
        anchor_ids=pool[anchor];ss_mean=np.zeros(len(pool));ss_support=np.zeros(len(pool),dtype=int)
        if len(anchor_ids)>=2:
            values=matrix[np.ix_(pool,anchor_ids)];ss_mean=np.sort(values,axis=1)[:,-2:].mean(1)
            ss_support=np.array([len(np.unique(cells[anchor_ids[row>=.7]])) for row in values])
        proposals.append(dict(pool=pool.tolist(),as_mean=mean[chosen].tolist(),strong_support=support[chosen].tolist(),
                              ss_mean=ss_mean.tolist(),ss_support=ss_support.tolist(),hypotheses=hypotheses,status=status))
    return proposals


def select(tracks,proposals,geo,cells,profile):
    """Stereo-only competition; exact preservation of every axial membership."""
    config=PROFILES[profile];n=len(cells);stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    score=np.zeros((n,len(tracks)));confirmed=np.zeros_like(score,dtype=bool)
    previous_owner=np.full(n,-1,dtype=int)
    for j,(t,proposal) in enumerate(zip(tracks,proposals)):
        previous=[i for i in t['indices'] if stereo[i]];previous_owner[previous]=j
        hypotheses=proposal['hypotheses'];pool=np.asarray(proposal['pool'],dtype=int)
        if not hypotheses:
            score[previous,j]=.75  # Preserve unresolved associations, flagged below.
            continue
        residual=np.array([np.abs(h['residual']) for h in hypotheses]);inliers=residual<=config['residual_mm']
        consensus=inliers.mean(0);best=residual.min(0);as_mean=np.asarray(proposal['as_mean'])
        support=np.asarray(proposal['strong_support']);is_previous=np.isin(pool,previous)
        accepted=(consensus>=.5)&((as_mean>=config['recovery_as'])&(support>=2)|is_previous)
        if config.get('guarded'):
            ss_mean=np.asarray(proposal['ss_mean']);ss_support=np.asarray(proposal['ss_support'])
            accepted&=is_previous|((ss_mean>=config['recovery_ss'])&(ss_support>=2))
        confidence=.80*as_mean+.20*np.exp(-.5*(best/config['residual_mm'])**2)
        score[pool[accepted],j]=confidence[accepted]
        confirmed[pool[accepted],j]=True
        if config.get('guarded'):
            strong_consensus=max(h['distinct_support'] for h in hypotheses)>=6
            retain=is_previous&((best<=3.)|(as_mean>=.85)|(not strong_consensus))
            score[pool[retain],j]=np.maximum(score[pool[retain],j],.85)
        if not config['trim_existing']:score[previous,j]=np.maximum(score[previous,j],.85)
    if not tracks:return [],dict()
    best_owner=score.argmax(1);best_score=score[np.arange(n),best_owner]
    second=np.partition(score,-2,axis=1)[:,-2] if len(tracks)>1 else np.zeros(n)
    owner=np.where((best_score>=.5)&(best_score-second>=config['margin']),best_owner,-1)
    owner[~stereo]=-1
    result=[];diagnostics=dict(supported_tracks=0,ambiguous_tracks=0,unresolved_tracks=0,removed_stereo=0,recovered_stereo=0)
    for j,(t,proposal) in enumerate(zip(tracks,proposals)):
        aa=[i for i in t['indices'] if not stereo[i]];previous={i for i in t['indices'] if stereo[i]};accepted=np.flatnonzero(owner==j).tolist()
        verified=[i for i in accepted if confirmed[i,j]];pending=[i for i in accepted if not confirmed[i,j]]
        hypotheses=[{k:v for k,v in h.items() if k!='residual'} for h in proposal['hypotheses']]
        result.append(dict(t,indices=aa+accepted,stereo_indices=accepted,confirmed_stereo_indices=verified,
                           pending_stereo_indices=pending,rejected_stereo_indices=sorted(previous-set(accepted)),
                           coarse_3d=dict(status=proposal['status'],hypotheses=hypotheses,axial_members_unchanged=True,
                                          seed_ready=bool(hypotheses and len(np.unique(cells[verified]))>=2))))
        key='supported_tracks' if proposal['status']=='supported' else 'ambiguous_tracks' if proposal['status']=='ambiguous' else 'unresolved_tracks'
        diagnostics[key]+=1;diagnostics['removed_stereo']+=len(previous-set(accepted));diagnostics['recovered_stereo']+=len(set(accepted)-previous)
        assert [i for i in result[-1]['indices'] if not stereo[i]]==aa
    assigned=[i for t in result for i in t['indices']];assert len(assigned)==len(set(assigned))
    return result,diagnostics
