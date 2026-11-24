"""Structural checks on a small graph, including a real optimizer update."""
import json,torch
from torch_geometric.data import Data
from membership import Membership,sample,objectives,oc_loss
torch.set_num_threads(1);torch.manual_seed(20261123)
n=8;pairs=torch.triu_indices(n,n,1);ei=torch.cat([pairs,pairs.flip(0)],1)
x=torch.randn(n,12);x[:,10]=0
ea=torch.randn(len(ei.T),12);ea[:,9:11]=0
g=Data(x=x,edge_index=ei,edge_attr=ea,truth=torch.tensor([0,0,0,0,1,1,1,1]),stereo=torch.zeros(n,dtype=torch.bool),edge_kind=torch.zeros(len(ei.T),dtype=torch.long))
t=sample(g,1,128);model=Membership();result=model(g,t);losses=objectives(g,t,result)
loss=sum(losses);loss.backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
for name in ['encode','message','triplet','oc']:
 assert sum(float(p.grad.abs().sum()) for p in getattr(model,name).parameters() if p.grad is not None)>0,name
torch.optim.AdamW(model.parameters()).step();model.eval()
with torch.no_grad():
 a=model(g,t);g.triplet_edges=g.triplet_edges.flip(1);b=model(g,t[:,[2,1,0]])
 assert torch.allclose((a[1][0]+a[1][1])/2,(b[1][0]+b[1][1])/2,atol=1e-6)
 pred=torch.randn(8,4);p=torch.randperm(8)
 assert torch.allclose(oc_loss(pred,g.truth,g.stereo),oc_loss(pred[p],g.truth[p],g.stereo[p]),atol=1e-6)
print(json.dumps(dict(passed=True,parameters=sum(p.numel() for p in model.parameters()),losses=[float(v) for v in losses])))
