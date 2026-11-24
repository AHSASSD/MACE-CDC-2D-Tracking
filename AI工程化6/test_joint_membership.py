import numpy as np
from STEP6_联合两级分组 import membership_groups


def main():
    n=12;matrix=np.zeros((n,n))
    matrix[:5,:5]=.95;matrix[5:10,5:10]=.95
    matrix[0,5]=matrix[5,0]=.99  # A single wrong bridge must not join tracks.
    matrix[10,:5]=matrix[:5,10]=.75
    matrix[11,:5]=matrix[:5,11]=.75
    matrix[11,5:10]=matrix[5:10,11]=.75  # Ambiguous hit stays unassigned.
    a,b=np.triu_indices(n,1);pairs=np.array([a,b]);p=matrix[a,b]
    cells=np.arange(n);stereo=np.zeros(n,dtype=bool);stereo[10:]=True
    tracks,diag=membership_groups(cells,stereo,pairs,p)
    assert len(tracks)==2
    assert any(10 in t['indices'] for t in tracks)
    assert all(11 not in t['indices'] for t in tracks)
    assert all(not(set(range(5))<=set(t['indices']) and set(range(5,10))<=set(t['indices'])) for t in tracks)
    all_ids=[i for t in tracks for i in t['indices']]
    assert len(all_ids)==len(set(all_ids))
    print('joint membership bridge and competition checks passed')


if __name__=='__main__':main()
