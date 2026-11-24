"""EXPERIMENTAL: checked on synthetic geometry, loaded only by STEP7 trial.
The file retains its draft name; it is not used by the member-tracking baseline.
Constrained single-arc helix fitting against finite 3D wire segments.

Transverse circle is fixed by axial drift fits; z(phi)=z0+h*(phi-origin).
No truth or time is accepted. Multiple turns and energy loss are not modeled.
"""
import numpy as np
from scipy.optimize import least_squares


def arc_from_axial(circle, xy):
    angles = np.sort(np.mod(np.arctan2(xy[:, 1]-circle[1], xy[:, 0]-circle[0]), 2*np.pi))
    gaps = np.diff(np.r_[angles, angles[0]+2*np.pi])
    start = angles[(int(np.argmax(gaps))+1) % len(angles)]
    span = 2*np.pi-gaps.max()
    return float(start-.15), float(min(span+.30, 2*np.pi))


def distances(circle, longitudinal, arc, geometry, return_phi=False):
    """Coarse global search followed by safeguarded vectorized Newton steps."""
    cx, cy, radius = circle; z0, pitch = longitudinal; origin, span = arc
    geo = np.asarray(geometry, dtype=float)
    center = np.c_[geo[:, :2], np.zeros(len(geo))]
    direction = geo[:, 2:5]; half = geo[:, 5]/2
    def calculate(phi):
        q = np.stack([cx+radius*np.cos(phi), cy+radius*np.sin(phi), z0+pitch*(phi-origin)], axis=-1)
        along = np.sum((q-center[:, None, :])*direction[:, None, :], axis=-1)
        delta = q-center[:, None, :]-np.clip(along, -half[:, None], half[:, None])[..., None]*direction[:, None, :]
        return delta, along
    grid = np.broadcast_to(np.linspace(origin, origin+span, 65), (len(geo), 65))
    delta, _ = calculate(grid)
    best = np.argsort((delta*delta).sum(-1), axis=1)[:, :3]
    phi = np.take_along_axis(grid, best, axis=1).copy()
    for _ in range(12):
        delta, along = calculate(phi)
        velocity = np.stack([-radius*np.sin(phi), radius*np.cos(phi), np.full_like(phi, pitch)], axis=-1)
        acceleration = np.stack([-radius*np.cos(phi), -radius*np.sin(phi), np.zeros_like(phi)], axis=-1)
        inside = (np.abs(along)<half[:, None])[..., None]
        unit = direction[:, None, :]
        velocity = velocity-inside*np.sum(velocity*unit, axis=-1)[..., None]*unit
        acceleration = acceleration-inside*np.sum(acceleration*unit, axis=-1)[..., None]*unit
        grad = (delta*velocity).sum(-1)
        hess = (velocity*velocity).sum(-1)+(delta*acceleration).sum(-1)
        step = np.clip(grad/np.maximum(hess, 1e-6), -.2, .2)
        old = (delta*delta).sum(-1)
        for _ in range(5):
            trial = np.clip(phi-step, origin, origin+span)
            dd, _ = calculate(trial)
            bad = (dd*dd).sum(-1)>old
            step = np.where(bad, step*.5, step)
        trial = np.clip(phi-step, origin, origin+span)
        dd, _ = calculate(trial)
        phi = np.where((dd*dd).sum(-1)<=old, trial, phi)
    delta, _ = calculate(phi); squared=(delta*delta).sum(-1); choice=squared.argmin(1)
    result=np.sqrt(squared[np.arange(len(geo)), choice])
    return (result, phi[np.arange(len(geo)), choice]) if return_phi else result


def longitudinal_seeds(circle, arc, geometry, drift):
    """Generate finite-wire/circle intersection branches including +/- drift."""
    origin, span = arc; candidates=[]
    for index, (geo, d) in enumerate(zip(geometry, drift)):
        if np.linalg.norm(geo[2:4])<1e-9: continue
        slope=geo[2:4]/geo[4]; offset=geo[:2]-circle[:2]
        halfz=geo[5]*abs(geo[4])/2
        for radius in [circle[2], max(circle[2]-d, .01), circle[2]+d]:
            roots=np.roots([slope@slope, 2*(offset@slope), offset@offset-radius**2])
            for z in roots:
                if abs(z.imag)>1e-6 or abs(z.real)>halfz: continue
                z=float(z.real); point=geo[:2]+slope*z
                phi=origin+float(np.mod(np.arctan2(point[1]-circle[1],point[0]-circle[0])-origin,2*np.pi))
                if phi<=origin+span: candidates.append((index, phi-origin, z))
    if len(candidates)<2:return []
    rng=np.random.default_rng(20261106); seeds=[(0.,0.)]
    for _ in range(160):
        i,j=rng.choice(len(candidates),2,replace=False); a,b=candidates[i],candidates[j]
        if a[0]==b[0] or abs(a[1]-b[1])<.08:continue
        h=(b[2]-a[2])/(b[1]-a[1]); z0=a[2]-h*a[1]
        if abs(z0)<=2500 and abs(h)<=2500:seeds.append((z0,h))
    # Rank cheaply by the observed intersections before expensive 3D distances.
    grouped={i:np.array([(p,z) for k,p,z in candidates if k==i]) for i in set(c[0] for c in candidates)}
    def score(seed):return np.median([np.min(np.abs(v[:,1]-seed[0]-seed[1]*v[:,0])) for v in grouped.values()])
    ordered=sorted(seeds,key=score); unique=[]
    for seed in ordered:
        if all(np.linalg.norm(np.asarray(seed)-old)>10 for old in unique):unique.append(np.asarray(seed))
        if len(unique)==3:break
    return unique


def fit_helix(circle, indices, geometry, drift, rms_limit=1., trim=2):
    indices=list(indices); stereo=np.linalg.norm(geometry[:,2:4],axis=1)>1e-9
    aa=[i for i in indices if not stereo[i]]; ss=[i for i in indices if stereo[i]]
    if len(aa)<5 or len(ss)<4:return None, 'insufficient_hits'
    if len(np.unique(np.sign(geometry[ss,2]*geometry[ss,1]-geometry[ss,3]*geometry[ss,0])))<2:
        return None, 'insufficient_stereo_handedness'
    arc=arc_from_axial(circle, geometry[aa,:2]); seeds=longitudinal_seeds(circle,arc,geometry[ss],drift[ss])
    if not seeds:return None, 'no_longitudinal_seed'
    solutions=[]
    for seed in seeds:
        keep=np.array(indices,dtype=int); removed=[]; fitted=None
        for attempt in range(trim+1):
            fitted=least_squares(lambda z: distances(circle,z,arc,geometry[keep])-drift[keep],
                                 seed,bounds=([-2500,-2500],[2500,2500]),loss='soft_l1',f_scale=.5,
                                 diff_step=1e-4,max_nfev=35,ftol=1e-4,xtol=1e-4)
            residual=distances(circle,fitted.x,arc,geometry[keep])-drift[keep]
            rms=float(np.sqrt(np.mean(residual**2)))
            if rms<=rms_limit:break
            removable=np.flatnonzero(stereo[keep])
            if attempt==trim or len(removable)<=4:break
            worst=removable[np.argmax(np.abs(residual[removable]))]; removed.append(int(keep[worst]));keep=np.delete(keep,worst); seed=fitted.x
        jac=fitted.jac
        singular=np.linalg.svd(jac,compute_uv=False)
        identifiable=bool(len(singular)==2 and singular[-1]>1e-4 and singular[0]/singular[-1]<1e4)
        if rms<=rms_limit and identifiable and fitted.success:
            solutions.append(dict(indices=keep.tolist(),fit=list(map(float,circle)),helix_z0=float(fitted.x[0]),
                                  helix_pitch_mm_per_rad=float(fitted.x[1]),arc=list(arc),rms_3d_mm=rms,
                                  removed_stereo=removed,condition_number=float(singular[0]/singular[-1])))
    if not solutions:return None, 'residual_or_degenerate_fit'
    solutions.sort(key=lambda s:(-len(s['indices']),s['rms_3d_mm']))
    best=solutions[0]
    best['ambiguous']=any(abs(s['helix_z0']-best['helix_z0'])>20 and
                          s['rms_3d_mm']<=best['rms_3d_mm']+.1 and len(s['indices'])==len(best['indices']) for s in solutions[1:])
    if best['ambiguous']:return None, 'multiple_longitudinal_solutions'
    return best, 'accepted'
