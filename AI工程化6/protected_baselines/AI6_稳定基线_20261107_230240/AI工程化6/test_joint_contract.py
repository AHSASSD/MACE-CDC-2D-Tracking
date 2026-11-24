import numpy as np
import torch
from common import *

torch.set_num_threads(2)
geo=np.array([[160.,0,0,0,1,1200],[170.,0,0,.1,np.sqrt(.99),1200],[180.,0,0,-.1,np.sqrt(.99),1200]])
cell=np.array([0,0,1,2]);d=np.array([.1,.2,.3,.4]);hit=np.arange(4);truth=np.array([1,1,2,2])
g=build_graph(cell,d,hit,geo,truth,7);other=build_graph(cell,d,hit,geo,np.zeros(4),7)
assert torch.equal(g.x,other.x) and torch.equal(g.edge_attr,other.edge_attr)
assert g.x.shape==(4,12) and g.edge_attr.shape[1]==12
assert g.x[0,-1]==0 and g.x[2,-1]>0 and g.x[3,-1]<0
assert ((g.edge_index[0]==0)&(g.edge_index[1]==1)).any()
order=np.array([3,1,2,0]);permuted=build_graph(cell[order],d[order],hit[order],geo,truth[order],7)
def signature(a):return sorted((int(a.hit_id[u]),int(a.hit_id[v]),tuple(np.round(e.numpy(),5))) for (u,v),e in zip(a.edge_index.T,a.edge_attr))
assert signature(g)==signature(permuted)
torch.manual_seed(SEED);model=JointInteractionNetwork(chunk_size=3)
loss=FocalLoss()(model(g.x,g.edge_index,g.edge_attr),g.y);loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
model.eval();large=JointInteractionNetwork(chunk_size=100);large.load_state_dict(model.state_dict());large.eval()
with torch.no_grad():assert torch.allclose(model(g.x,g.edge_index,g.edge_attr),large(g.x,g.edge_index,g.edge_attr),atol=1e-5)
assert model.hidden==256 and model.n_passes==8
print('PASS: explicit signed tilt, duplicate-hit identity, truth exclusion, permutation invariance, 256x8 forward/backward, chunk equivalence')
from torch_geometric.data import Batch
from STEP2_训练模型 import graph_balanced_loss
small=build_graph(cell[:3],d[:3],hit[:3],geo,truth[:3],8)
packed=Batch.from_data_list([g,small]);logits=torch.randn(packed.num_edges,requires_grad=True)
expected=(FocalLoss()(logits[:g.num_edges],g.y)+FocalLoss()(logits[g.num_edges:],small.y))/2
actual=graph_balanced_loss(logits,packed)
assert torch.allclose(actual,expected,atol=1e-7)
grad1=torch.autograd.grad(actual,logits,retain_graph=True)[0]
grad2=torch.autograd.grad(expected,logits)[0]
assert torch.allclose(grad1,grad2,atol=1e-7)
print('PASS: larger batching preserves equal-event loss and gradients across unequal edge counts')
for policy in ['one_direct','adaptive']:
    reference=JointInteractionNetwork(chunk_size=3,recompute_policy='all')
    candidate=JointInteractionNetwork(chunk_size=3,recompute_policy=policy);candidate.load_state_dict(reference.state_dict())
    torch.manual_seed(43);a=reference(g.x,g.edge_index,g.edge_attr)
    torch.manual_seed(43);b=candidate(g.x,g.edge_index,g.edge_attr)
    assert torch.allclose(a,b,atol=1e-5)
    a.sum().backward();b.sum().backward()
    for p,q in zip(reference.parameters(),candidate.parameters()):assert torch.allclose(p.grad,q.grad,atol=1e-4,rtol=1e-3)
print('PASS: partial recomputation preserves outputs and gradients')

import tempfile
from graph_cache import CachedGraphDataset,contract,NODE_KEYS,EDGE_KEYS
with tempfile.TemporaryDirectory() as tmp:
    base=Path(tmp);source=base/'source';source.mkdir();cache=base/'cache';cache.mkdir()
    ids={'train':[7,8],'validation':[],'test':[]};save_json(source/'dataset_metadata_joint_no_time.json',{'event_ids':ids})
    index={}
    for eid,value in [(7,g),(8,small)]:
        shard=cache/str(eid);shard.mkdir()
        for key in NODE_KEYS+EDGE_KEYS:
            arr=getattr(value,key).numpy()
            if key=='edge_index':arr=arr.astype(np.int32)
            if key in ['edge_kind','y']:arr=arr.astype(np.uint8)
            np.save(shard/(key+'.npy'),arr)
        index[str(eid)]={'shard':str(eid),'node_start':0,'node_end':value.num_nodes,'edge_start':0,'edge_end':value.num_edges}
    save_json(cache/'manifest.json',{'status':'complete','contract':contract(source),'event_ids':ids,'index':index})
    ds=CachedGraphDataset(cache,'train',source,mapped_shards=1);first=ds[0]
    mapping=ds._maps['7']['x']._mmap
    assert first.x.data_ptr()!=ds._maps['7']['x'].ctypes.data, 'Returned graph must own its data before mmap eviction'
    second=ds[1]
    assert mapping.closed, 'Eviction must release the mapping explicitly'
    assert len(ds._maps)==1 and not ds.__getstate__()['_maps']
    for key in NODE_KEYS+EDGE_KEYS:assert torch.equal(getattr(first,key),getattr(g,key)) and torch.equal(getattr(second,key),getattr(small,key))
    assert torch.equal(ds[0].edge_index,g.edge_index)
    del ds,first,second
    import gc
    gc.collect()
print('PASS: exact cached graph recovery, mapping LRU, spawn-safe state')
