import numpy as np
from joint_candidate_tracking import exclusive_selection,fit_axial,candidate_evidence,proposal_pool


def main():
    # Global selection must beat taking the highest individual conflicting item.
    pool=[dict(indices=[0,1,2,3]),dict(indices=[0,1]),dict(indices=[2,3])]
    chosen=exclusive_selection(pool,[6.,4.,4.],4)
    assert chosen==pool[1:]
    members=[i for t in chosen for i in t['indices']];assert len(members)==len(set(members))
    phi=np.linspace(0,2*np.pi,8,endpoint=False);g=np.zeros((8,6));g[:,:2]=np.c_[100*np.cos(phi),100*np.sin(phi)];g[:,4]=1;g[:,5]=1000
    t=fit_axial(list(range(8)),g,np.zeros(8));assert t is not None and t['rms']<.001
    e=candidate_evidence(t,g,np.arange(8),np.ones((8,8))*.9)
    assert e['axial_hits']==8 and e['stereo_support']==0
    pairs=np.array(np.triu_indices(8,1))
    remaining,_=proposal_pool(g,np.zeros(8),np.arange(8),pairs,np.full(pairs.shape[1],.9),[t],[t],False)
    assert remaining==[]  # Every hit already used: the residual index set is empty.
    print('joint candidate global objective, exclusive ownership, geometry and no-stereo checks passed')


if __name__=='__main__':main()
