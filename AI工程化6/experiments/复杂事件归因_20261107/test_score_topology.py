import experiment_common as e
import numpy as np
from STEP4_真边分数与连通性诊断 import topology,single_pass
from STEP5_两级寻迹 import axial_two_pass,score_tracks,DEFAULT_CONFIG


def main():
    edges=np.array([[0,1,0,3,4,3,2],[1,2,2,4,5,5,3]])
    t=topology(6,edges,np.array([.9]*6+[.4]))
    assert t['category']=='few_weak_connectors' and t['repair_edges_to_match']==1
    complete=np.array([(a,b) for a in range(6) for b in range(a+1,6)]).T
    assert topology(6,complete,np.full(len(complete.T),.3))['category']=='broadly_low_scores'
    assert topology(6,edges[:,:6],np.full(6,.9))['category']=='candidate_graph_insufficient'
    # Three geometrically sound fragments become irrevocably consumed in the
    # strict pass. A relaxed pass on the full graph can join them without any
    # increase in true-edge scores; this distinguishes extraction from learning.
    n=15;phi=np.linspace(.15,1.8,n);d=np.full(n,.5);xy=np.array([120.,70.])+(110.+d)[:,None]*np.c_[np.cos(phi),np.sin(phi)]
    geo=np.c_[xy,np.zeros((n,2)),np.ones(n),np.full(n,1000.)]
    pairs=np.array([(a,b) for a in range(n) for b in range(a+1,n)]).T
    p=np.array([.8 if a//5==b//5 else .6 for a,b in pairs.T]);truth=np.zeros(n,dtype=int);stereo=np.zeros(n,dtype=bool)
    old,_=axial_two_pass(geo,d,pairs,p);new=single_pass(geo,d,pairs,p)
    before,_=score_tracks(old,truth,stereo,DEFAULT_CONFIG);after,_=score_tracks(new,truth,stereo,DEFAULT_CONFIG)
    assert before['matched_tracks']==0 and after['matched_tracks']==1
    print('Few connectors, broadly weak scores, structural disconnection and strict-pass locking distinguished')


if __name__=='__main__':main()
