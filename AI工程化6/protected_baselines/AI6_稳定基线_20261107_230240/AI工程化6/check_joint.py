"""Run with the PyTorch interpreter before using the experiment."""
import numpy as np
import pandas as pd
import torch
from joint_experiment import graph, legacy, MemoryEfficientInteractionNetwork

f = pd.DataFrame({'x':[160.,160.,170.,175.,180.,190.], 'y':[0.,0.,10.,20.,30.,40.],
                  'u':[0.,0.,.1,-.1,0.,.1], 'v':[0.]*6, 'l':[1200.]*6,
                  'd':[.5]*6,'TrkID':[1,1,1,2,2,2], 'HitID':range(6),
                  'CellID':[0,0,1,2,3,4], 'EvtID':[42]*6})
f['w']=np.sqrt(1-f.u**2)
g=graph(f)
assert g.x.shape==(6,12) and g.edge_attr.shape[1]==11
assert (g.edge_index[0]!=g.edge_index[1]).all()
assert ((g.edge_index[0]==0)&(g.edge_index[1]==1)).any(), 'Repeated-cell hits must survive'
order=np.array([3,0,5,1,4,2]); shuffled=graph(f.iloc[order])
def signatures(data):
    return sorted((int(data.hit_id[a]),int(data.hit_id[b]),tuple(np.round(e.numpy(),5)))
                  for (a,b),e in zip(data.edge_index.T,data.edge_attr))
assert signatures(g)==signatures(shuffled), 'Graph depends on CSV ordering'
altered=f.copy(); altered['TrkID']=999; altered['t']=np.arange(6)*1e12
altered['d_true']=999; altered['x0']='truth'; other=graph(altered)
assert torch.equal(g.x,other.x) and torch.equal(g.edge_attr,other.edge_attr)
assert torch.equal(g.edge_index,other.edge_index)
model=legacy.InteractionNetwork(12,11,hidden=64,n_passes=3)
loss=torch.nn.functional.binary_cross_entropy_with_logits(model(g.x,g.edge_index,g.edge_attr),g.y)
loss.backward(); assert torch.isfinite(loss)
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
print('PASS: duplicate hits; permutation invariance; no truth/time features; finite forward/backward')
reference=legacy.InteractionNetwork(12,11,hidden=64,n_passes=3)
recomputed=MemoryEfficientInteractionNetwork(12,11,hidden=64,n_passes=3)
recomputed.load_state_dict(reference.state_dict())
torch.manual_seed(17); expected=reference(g.x,g.edge_index,g.edge_attr)
torch.manual_seed(17); actual=recomputed(g.x,g.edge_index,g.edge_attr)
assert torch.allclose(actual,expected,atol=1e-6)
expected.sum().backward(); actual.sum().backward()
for a,b in zip(reference.parameters(),recomputed.parameters()):
    assert torch.allclose(a.grad,b.grad,atol=1e-5,rtol=1e-4)
print('PASS: activation recomputation preserves outputs and gradients')
