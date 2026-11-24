"""Training-only difficulty proxies from axial wire observations, not true hit positions."""
import numpy as np

RULES=dict(min_hits=8,min_unique_cells=5,blob_extent_mm=80.,blob_minor_major=.20,loop_coverage_degrees=210.,loop_max_radial_cv=.40,high_occupancy_quantile=.95)


def track_shape(cells,xy):
    _,index=np.unique(cells,return_index=True);points=np.asarray(xy,dtype=float)[index]
    n=len(points)
    if n<3:return dict(unique_cells=n,extent_mm=0.,minor_major=0.,coverage_deg=0.,radius_mm=None,radial_cv=None,blob=False,loop=False)
    center=points.mean(0);q=points-center
    eigen=np.linalg.eigvalsh(q.T@q/max(n-1,1));ratio=float(max(eigen[0],0)/max(eigen[-1],1e-12))
    extent=float(np.max(np.quantile(points,.95,axis=0)-np.quantile(points,.05,axis=0)))
    scale=max(float(np.sqrt(np.mean(q*q))),1e-9);u=q/scale
    design=np.column_stack([2*u,np.ones(n)])
    beta,_,rank,_=np.linalg.lstsq(design,(u*u).sum(1),rcond=None)
    radius=None;coverage=0.;cv=None
    if rank==3 and beta[2]+beta[0]**2+beta[1]**2>0:
        circle_center=center+scale*beta[:2];r=np.linalg.norm(points-circle_center,axis=1)
        radius=float(scale*np.sqrt(beta[2]+beta[0]**2+beta[1]**2));cv=float(np.std(r)/max(float(r.mean()),1e-9))
        angle=np.sort(np.mod(np.arctan2(points[:,1]-circle_center[1],points[:,0]-circle_center[0]),2*np.pi))
        coverage=float(np.degrees(2*np.pi-np.diff(np.r_[angle,angle[0]+2*np.pi]).max()))
    enough=len(cells)>=RULES['min_hits'] and n>=RULES['min_unique_cells']
    return dict(unique_cells=n,extent_mm=extent,minor_major=ratio,coverage_deg=coverage,radius_mm=radius,radial_cv=cv,
                blob=bool(enough and extent<RULES['blob_extent_mm'] and ratio>=RULES['blob_minor_major']),
                loop=bool(enough and coverage>=RULES['loop_coverage_degrees'] and cv is not None and cv<=RULES['loop_max_radial_cv']))
