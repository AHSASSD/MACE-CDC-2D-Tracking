"""GNN-guided drift-circle consensus in unused axial measurements."""
from itertools import product
import numpy as np
from scipy.optimize import least_squares


def tangent_circles(xy,d):
    matrix=2*(xy[1:]-xy[0])
    if np.linalg.cond(matrix)>1e4:return []
    inverse=np.linalg.inv(matrix)
    constant=inverse@(np.sum(xy[1:]**2,axis=1)-np.sum(xy[0]**2)-d[1:]**2+d[0]**2)
    offset=constant-xy[0];results=[]
    for signs in product([-1,1],repeat=3):
        signed=d*np.array(signs);slope=inverse@(-2*(signed[1:]-signed[0]))
        coefficients=[float(slope@slope-1),float(2*(offset@slope-signed[0])),float(offset@offset-d[0]**2)]
        roots=np.roots(coefficients[1:] if abs(coefficients[0])<1e-12 else coefficients)
        for root in roots:
            if abs(root.imag)>1e-7 or not .01<root.real<20000:continue
            radius=float(root.real);fit=np.r_[constant+radius*slope,radius]
            if np.all(radius+signed>0) and np.isfinite(fit).all():results.append(fit)
    return results


def consensus_candidates(geo,drift,cells,matrix,baseline,seed=20261107):
    used={i for t in baseline for i in t['indices']};stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    free=np.array([i for i in range(len(geo)) if not stereo[i] and i not in used],dtype=int)
    if len(free)<5:return []
    rng=np.random.default_rng(seed);blocks=matrix[np.ix_(free,free)];eligible=np.flatnonzero((blocks>=.55).sum(1)>=2)
    if not len(eligible):return []
    triples=set()
    for _ in range(128):
        a=int(rng.choice(eligible));neighbors=np.flatnonzero(blocks[a]>=.55)
        if len(neighbors)<2:continue
        weights=blocks[a,neighbors]**4;neighbors=rng.choice(neighbors,2,replace=False,p=weights/weights.sum())
        trip=tuple(sorted([int(free[a])]+free[neighbors].tolist()))
        if len(np.unique(cells[list(trip)]))==3:triples.add(trip)
    hypotheses={}
    for triple in sorted(triples):
        for circle in tangent_circles(geo[list(triple),:2],drift[list(triple)]):
            residual=np.abs(np.linalg.norm(geo[free,:2]-circle[:2],axis=1)-circle[2])-drift[free]
            links=matrix[np.ix_(free,np.array(triple))]
            support=(links>=.50).sum(1)>=2
            support[np.isin(free,triple)]=True
            keep=free[(np.abs(residual)<=1.)&support]
            if len(keep)<5:continue
            block=matrix[np.ix_(keep,keep)];values=block[np.triu_indices(len(keep),1)];values=values[values>0]
            if not len(values) or values.mean()<.6:continue
            key=tuple(keep);quality=len(keep)*float(values.mean())-float(np.mean(np.abs(residual[np.isin(free,keep)])))
            if key not in hypotheses or quality>hypotheses[key][0]:hypotheses[key]=(quality,circle)
    candidates=[];seen=set()
    for members,(_,circle) in sorted(hypotheses.items(),key=lambda x:x[1][0],reverse=True)[:24]:
        aa=np.array(members);xy=geo[aa,:2];d=drift[aa]
        def residual(fit):return np.abs(np.linalg.norm(xy-fit[:2],axis=1)-fit[2])-d
        fit=least_squares(residual,circle,bounds=([-np.inf,-np.inf,.01],[np.inf,np.inf,np.inf]),loss='soft_l1',f_scale=.5,max_nfev=50)
        for _ in range(2):
            error=np.abs(residual(fit.x));keep=aa[error<=1.5]
            if len(keep)<5:break
            aa=keep;xy=geo[aa,:2];d=drift[aa]
            fit=least_squares(residual,fit.x,bounds=([-np.inf,-np.inf,.01],[np.inf,np.inf,np.inf]),loss='soft_l1',f_scale=.5,max_nfev=50)
        if len(aa)<5:continue
        rms=float(np.sqrt(np.mean(residual(fit.x)**2)));key=tuple(aa.tolist())
        if rms<=1. and key not in seen:
            seen.add(key);candidates.append(dict(indices=aa.tolist(),fit=fit.x.tolist(),rms=rms,stage='drift_consensus'))
    return candidates
