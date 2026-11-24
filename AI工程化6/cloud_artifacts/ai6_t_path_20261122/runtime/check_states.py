import bootstrap
import numpy as np
from path_decode import path_hypotheses,Config

def main():
    # Two entrances and exits across one common segment. Keep both complete
    # hypotheses even though temporary states share observations1and2.
    xy=np.array([[-10,-3],[0,0],[10,0],[20,3],[-10,3],[20,-3]],float)
    geo=np.c_[xy,np.zeros((6,4))]
    pairs=np.array([[0,1,2,4,2],[1,2,3,1,5]],int)
    t=np.array([[0,1,2,0,1],[1,2,3,1,2],[4,1,2,3,1],[1,2,5,1,4]],int)
    f=dict(stereo=np.zeros(6,bool),pairs=pairs,p=np.ones(5)*.9)
    paths,diag=path_hypotheses(f,t,np.ones(4)*.99,geo,Config(seed_limit=32,fit_limit=32))
    # Four-hit paths are intentionally below production minimum5: use a tail.
    xy=np.vstack([xy,[30,5],[30,-5]])
    geo=np.c_[xy,np.zeros((8,4))];pairs=np.c_[pairs,[[3,5],[6,7]]]
    t=np.vstack([t,[2,3,6,2,5],[2,5,7,4,6]])
    f=dict(stereo=np.zeros(8,bool),pairs=pairs,p=np.ones(7)*.9)
    paths,diag=path_hypotheses(f,t,np.ones(6)*.99,geo,Config(seed_limit=32,fit_limit=32))
    members=[set(x['path']) for x in paths]
    assert any({0,1,2,3,6}<=x for x in members),members
    assert any({4,1,2,5,7}<=x for x in members),members
    assert all(len(x['path'])==len(set(x['path'])) for x in paths)
    # Reverse every triplet: undirected measurement ordering must not remove paths.
    reversed_t=t[:,[2,1,0,4,3]]
    rev,_=path_hypotheses(f,reversed_t,np.ones(6)*.99,geo,Config(seed_limit=32,fit_limit=32))
    assert {frozenset(x['path']) for x in paths}=={frozenset(x['path']) for x in rev}
    print('PASS shared-segment alternatives survive, no global lock, no revisits, reversal invariant',diag)

if __name__=='__main__':main()
