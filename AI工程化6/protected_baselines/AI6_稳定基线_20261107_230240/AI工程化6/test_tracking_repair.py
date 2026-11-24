import numpy as np
from STEP8_寻迹改进评估 import repair_tracks,select_profile


def main():
    phi=np.linspace(0,2*np.pi,11,endpoint=False);g=np.zeros((11,6));g[:,:2]=np.c_[100*np.cos(phi),100*np.sin(phi)];g[:,4]=1;g[:,5]=1000
    a,b=np.triu_indices(11,1);pairs=np.array([a,b]);p=np.full(len(a),.95)
    baseline=[dict(indices=list(range(5)),fit=[0.,0.,100.],rms=0.,stage='strict'),dict(indices=list(range(5,10)),fit=[0.,0.,100.],rms=0.,stage='strict')]
    repaired=repair_tracks(baseline,g,np.zeros(11),np.arange(11),pairs,p)
    assert len(repaired)==1 and repaired[0]['indices']==list(range(11))
    assert baseline[0]['indices']==list(range(5))
    base=dict(fake_fraction=.02,clone_fraction=.01,track_efficiency=.9,joint_track_efficiency=.8)
    bad=dict(base,track_efficiency=.95,fake_fraction=.03)
    good=dict(base,track_efficiency=.92,clone_fraction=0.)
    assert select_profile(dict(baseline=base,repair=bad,repair_recover=good))=='repair_recover'
    print('tracking merge, extension, exclusive membership and constrained selection checks passed')


if __name__=='__main__':main()
