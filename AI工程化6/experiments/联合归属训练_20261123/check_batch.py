import json,torch
from torch_geometric.data import Data
from fast_model import FastMembership,pack,event_objectives
from repair_v2 import install,sample
torch.set_num_threads(1);torch.manual_seed(111);_,objective=install('v1')
graphs=[];ts=[]
for eid in range(2):
    x=torch.randn(8,12);x[:,10]=0;p=torch.triu_indices(8,8,1);ei=torch.cat([p,p.flip(0)],1);ea=torch.randn(ei.shape[1],12);ea[:,9:11]=0
    g=Data(x=x,edge_index=ei,edge_attr=ea,truth=torch.tensor([0,0,0,0,1,1,1,1]),stereo=torch.zeros(8,dtype=torch.bool),edge_kind=torch.zeros(ei.shape[1],dtype=torch.long))
    ts.append(sample(g,123+eid,128));graphs.append(g)
model=FastMembership().eval();individual=[model(g,t) for g,t in zip(graphs,ts)]
g,t,slices=pack(graphs,ts);combined=model(g,t)
for i,(ns,ne,es,ee,begin,end) in enumerate(slices):
    assert torch.allclose(combined[0][es:ee],individual[i][0],atol=2e-5)
    assert torch.allclose(combined[1][0][begin:end],individual[i][1][0],atol=2e-5)
    assert torch.allclose(combined[2][ns:ne],individual[i][2],atol=2e-5)
expected=torch.stack([torch.stack(objective(a,b,c)) for a,b,c in zip(graphs,ts,individual)]).mean(0)
actual=torch.stack(event_objectives(g,t,combined,slices,objective))
assert torch.allclose(expected,actual,atol=2e-5),(expected,actual)
parameters=[model.encode[0].weight,model.message[0].weight,model.oc[0].weight,model.triplet[0].weight]
ga=torch.autograd.grad(expected.sum(),parameters,retain_graph=True);gb=torch.autograd.grad(actual.sum(),parameters)
assert all(torch.allclose(a,b,atol=2e-4,rtol=2e-4) for a,b in zip(ga,gb))
print(json.dumps(dict(passed=True,batch_edge_direction_mapping=True,event_isolation_same_truth_ids=True,event_equal_loss=True,event_equal_gradients=True)))
