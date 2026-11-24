import numpy as np
from drift_consensus_tracking import tangent_circles,consensus_candidates

center=np.array([30.,-20.]);radius=100.;phi=np.array([.1,.8,1.6]);d=np.array([.2,.4,.6]);xy=center+(radius+d)[:,None]*np.c_[np.cos(phi),np.sin(phi)]
solutions=tangent_circles(xy,d)
assert min(np.linalg.norm(v-np.r_[center,radius]) for v in solutions)<1e-6
phi=np.linspace(.1,2.,16);g=np.zeros((16,6));g[:,:2]=center+100.5*np.c_[np.cos(phi),np.sin(phi)];g[:,4]=1.;g[:,5]=1000
matrix=np.full((16,16),.95);np.fill_diagonal(matrix,0)
pool=consensus_candidates(g,np.full(16,.5),np.arange(16),matrix,[],42)
assert any(len(t['indices'])>=14 for t in pool)
protected=[dict(indices=list(range(8)))]
pool=consensus_candidates(g,np.full(16,.5),np.arange(16),matrix,protected,42)
assert all(not set(t['indices']).intersection(range(8)) for t in pool)
print('analytical drift tangency and protected consensus recovery checks passed')
