"""Independent scalar minimization checks finite-wire helix distances."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

spec=importlib.util.spec_from_file_location('helix_prototype',Path(__file__).with_name('三维拟合草稿_暂未启用.py'))
helix=importlib.util.module_from_spec(spec);spec.loader.exec_module(helix)


def scalar_distance(circle,longitudinal,arc,geo):
    origin,span=arc;center=np.r_[geo[:2],0.];direction=geo[2:5]
    def squared(phi):
        q=np.array([circle[0]+circle[2]*np.cos(phi),circle[1]+circle[2]*np.sin(phi),longitudinal[0]+longitudinal[1]*(phi-origin)])
        delta=q-center;delta-=np.clip(delta@direction,-geo[5]/2,geo[5]/2)*direction
        return delta@delta
    grid=np.linspace(origin,origin+span,513);values=np.array([squared(phi) for phi in grid])
    minima=[values[0],values[-1]]
    for i in range(1,len(grid)-1):
        if values[i]<=values[i-1] and values[i]<=values[i+1]:
            minima.append(minimize_scalar(squared,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-12}).fun)
    return np.sqrt(min(minima))


def synthetic():
    circle=np.array([160.,80.,100.]);arc=(.2,1.8);longitudinal=np.array([-30.,40.])
    geo=[]
    for i,phi in enumerate(np.linspace(.35,1.8,16)):
        point=np.array([circle[0]+circle[2]*np.cos(phi),circle[1]+circle[2]*np.sin(phi),longitudinal[0]+longitudinal[1]*(phi-arc[0])])
        tilt=0. if i%3==0 else (.12 if i%2 else -.12)
        wire_phi=np.arctan2(point[1],point[0])
        direction=np.array([-np.sin(wire_phi)*tilt,np.cos(wire_phi)*tilt,np.sqrt(1-tilt**2)])
        normal=np.array([np.cos(phi),np.sin(phi),0.]);point+=normal*(.5+.1*(i%3))
        midpoint=point-direction*(point[2]/direction[2])
        geo.append(np.r_[midpoint[:2],direction,500.])
    geo=np.array(geo)
    drift=np.array([scalar_distance(circle,longitudinal,arc,g) for g in geo])
    return circle,longitudinal,arc,geo,drift


def main():
    circle,z,arc,geo,drift=synthetic()
    calculated=helix.distances(circle,z,arc,geo)
    assert np.max(np.abs(calculated-drift))<1e-4
    # Short segments force endpoint projection, unlike infinite-wire geometry.
    short=geo.copy();short[:,5]=10
    reference=np.array([scalar_distance(circle,z,arc,g) for g in short])
    assert np.max(np.abs(helix.distances(circle,z,arc,short)-reference))<1e-4
    assert np.all(reference>=drift-1e-6)
    result,reason=helix.fit_helix(circle,list(range(len(geo))),geo,drift,rms_limit=.05,trim=0)
    assert result is not None,reason
    recovered_at_original=result['helix_z0']+result['helix_pitch_mm_per_rad']*(arc[0]-result['arc'][0])
    assert abs(recovered_at_original-z[0])<1 and abs(result['helix_pitch_mm_per_rad']-z[1])<1
    print('finite-wire distance vs independent scalar solver and synthetic longitudinal recovery passed')


if __name__=='__main__':main()
