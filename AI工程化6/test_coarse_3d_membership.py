"""Independent synthetic geometry, stereo outliers and protected axial cores."""
import numpy as np
from test_helix_geometry import synthetic,scalar_distance
from coarse_3d_membership import longitudinal_hypotheses,propose,select,PROFILES,finite_cylinder_intersections,axial_longitudinal_check


def main():
    circle,z,arc,geo,drift=synthetic();stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    ids=np.flatnonzero(stereo);cells=np.arange(len(ids))
    hypotheses,status=longitudinal_hypotheses(circle,arc,geo[ids],drift[ids],cells,np.ones(len(ids)),refine=True)
    assert hypotheses,status
    best=hypotheses[0]
    assert abs(best['z0']-z[0])<2 and abs(best['pitch']-z[1])<2,(best,z)
    reference=np.array([scalar_distance(circle,[best['z0'],best['pitch']],arc,g) for g in geo[ids]])-drift[ids]
    assert np.max(np.abs(reference-best['residual']))<1e-4
    # Arbitrarily many rejected S members must not trim the protected A core.
    axial=np.flatnonzero(~stereo).tolist();all_indices=list(range(len(geo)))
    track=dict(indices=all_indices,fit=circle.tolist(),rms=.1,stage='synthetic')
    residual=np.zeros(len(ids));residual[-4:]=10.
    proposal=dict(pool=ids.tolist(),as_mean=np.full(len(ids),.9).tolist(),strong_support=np.full(len(ids),3).tolist(),
                  hypotheses=[dict(z0=0.,pitch=1.,arc=list(arc),residual=residual,quality=10.,distinct_support=6)],status='supported')
    selected,diagnostics=select([track],[proposal],geo,np.arange(len(geo)),'balanced')
    assert [i for i in selected[0]['indices'] if not stereo[i]]==axial
    assert len(selected[0]['rejected_stereo_indices'])==4 and diagnostics['removed_stereo']==4
    # Degenerate/no-consensus cases keep the original A and flag pending S.
    proposal=dict(pool=[],as_mean=[],strong_support=[],hypotheses=[],status='insufficient_intersections')
    selected,_=select([track],[proposal],geo,np.arange(len(geo)),'balanced')
    assert set(selected[0]['indices'])==set(all_indices)
    assert selected[0]['confirmed_stereo_indices']==[]
    assert set(selected[0]['pending_stereo_indices'])==set(ids)
    # Duplicate CellIDs cannot manufacture longitudinal support.
    h,_=longitudinal_hypotheses(circle,arc,geo[ids],drift[ids],np.zeros(len(ids),dtype=int),np.ones(len(ids)),refine=False)
    assert not h
    assert select([],[],geo,np.arange(len(geo)),'balanced')[0]==[]
    # A wire can miss R and still intersect R+d; rejecting the nominal miss loses
    # a physically possible drift measurement. Along-wire length is not z length.
    wire=np.array([101.,0.,0.,.1,np.sqrt(.99),1000.]);cylinder=[0.,0.,100.]
    assert finite_cylinder_intersections(cylinder,wire)==[]
    intersections=finite_cylinder_intersections(cylinder,wire,2.)
    assert len(intersections)==2
    for point in intersections:
        assert abs(np.hypot(point['x'],point['y'])-102.)<1e-8
        assert abs(point['z']-point['along_wire_mm']*wire[4])<1e-8
    short=wire.copy();short[5]=100.
    assert finite_cylinder_intersections(cylinder,short,2.)==[]
    tangent=wire.copy();tangent[0]=100.
    assert len(finite_cylinder_intersections(cylinder,tangent))==1
    assert not axial_longitudinal_check(np.array([[10000.,0.]]),circle,arc,geo[~stereo])[0]
    print('Rough z recovery, independent finite-wire distances, unlimited stereo trimming, protected A and pending fallback passed')


if __name__=='__main__':main()
