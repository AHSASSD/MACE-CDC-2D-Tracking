"""Synthetic weak links, label-metadata isolation and exact baseline protection."""
import experiment_common as e
from copy import deepcopy
import numpy as np
from weak_edge_recovery import generate,reconstruct,FEATURES
from coarse_3d_membership import propose,select
from STEP1_复杂事件失败归因 import truth_components


def main():
    n=12;phi=np.linspace(.15,1.8,n);drift=np.full(n,.5)
    center=np.array([120.,70.]);radius=110.
    xy=center+(radius+drift)[:,None]*np.c_[np.cos(phi),np.sin(phi)]
    geo=np.c_[xy,np.zeros((n,2)),np.ones(n),np.full(n,1000.)];cells=np.arange(n)
    pairs=np.array([(a,b) for a in range(n) for b in range(a+1,n)]).T
    p=np.array([.35 if a>=5 and b>=5 else .01 for a,b in pairs.T])
    old=[dict(indices=list(range(5)),fit=[120.,70.,110.],rms=0.,stage='protected')]
    first,x=generate(geo,drift,cells,pairs,p,old)
    altered=deepcopy(old);altered[0]['truth']=-123;altered[0]['TrkID']=999999;altered[0]['time']=1e12
    second,y=generate(geo,drift,cells,pairs,p,altered)
    assert first and [t['indices'] for t in first]==[t['indices'] for t in second]
    assert np.allclose(x,y) and x.shape[1]==len(FEATURES) and np.isfinite(x).all()
    assert all(set(t['indices']).isdisjoint(range(5)) for t in first)
    baseline=deepcopy(old);coarse,_=select(old,propose(old,geo,drift,cells,pairs,p,refine=False),geo,cells,'guarded_loose')
    joint,ready,_=reconstruct(geo,drift,cells,pairs,p,old,coarse,first,np.ones(len(first)),.95)
    assert old==baseline and joint[0]['indices']==list(range(5))
    used=[i for t in joint for i in t['indices']];assert len(used)==len(set(used))
    # S witnesses can connect pure A islands even with no AA edge present.
    edges=np.array([[0,1,2,3,4],[5,5,5,5,5]])
    truth=np.zeros(6,dtype=int);stereo=np.array([False]*5+[True])
    largest=truth_components(6,edges,np.ones(5,dtype=bool),truth,stereo,{0:5})
    assert largest[0]==5
    print('Weak-score recovery, metadata isolation, exact A protection, exclusive assignment and joint-connectivity diagnosis passed')


if __name__=='__main__':main()
