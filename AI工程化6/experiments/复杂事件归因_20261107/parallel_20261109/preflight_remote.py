"""One real cached batch per loss, backward only, zero optimization updates."""
import json
import os
from pathlib import Path
import torch
from common import JointInteractionNetwork,sha
from server_runtime import dataset,loader
from paired_loss import objective

torch.set_num_threads(1)
root=Path.home()/'network_datasets/ai6'
weight=root/'joint_training/evaluation_snapshot_epoch126_1794052868/best_interaction_network_joint_no_time.pth'
assert sha(weight)=='5f3f02409399cd62efce5f254be02fa004237fb61a5b4717681cad1ad90cdcac'
data=dataset(root/'joint_event_dataset','train')
assert len(data)==7000
assert len(dataset(root/'joint_event_dataset','validation'))==1500
g=next(iter(loader(torch.utils.data.Subset(data,list(range(4))),4,workers=0))).to('cuda')
model=JointInteractionNetwork(chunk_size=131072,recompute_policy='adaptive').cuda()
model.load_state_dict(torch.load(weight,map_location='cuda',weights_only=True))
rows={}
for arm in ['control','balanced']:
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda'):z=model(g.x,g.edge_index,g.edge_attr)
    loss,terms=objective(z,g,arm);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    rows[arm]=dict(loss=float(loss),terms=terms)
torch.cuda.synchronize()
result=dict(status='passed',optimization_updates=0,gpu=torch.cuda.get_device_name(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),graphs=g.num_graphs,edges=g.edge_index.shape[1],weight_sha256=sha(weight),arms=rows)
Path('preflight_remote.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
