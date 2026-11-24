"""Measure a real dense-batch backward pass without updating the baseline weights."""
import argparse
import gc
import json
from pathlib import Path
import torch
from torch_geometric.data import Batch
from common import JointInteractionNetwork, sha, save_json
from server_runtime import dataset
from recall_loss import objective


def attempt(data, weight, micro):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = JointInteractionNetwork(chunk_size=131072 if micro == 4 else 65536,
                                    recompute_policy='adaptive').cuda().train()
    model.load_state_dict(torch.load(weight, map_location='cuda', weights_only=True))
    # Largest cached candidate graphs, rather than an unusually easy random batch.
    ordered = sorted(range(len(data)), key=lambda i: data.index[str(data.ids[i])]['edge_end']-
                     data.index[str(data.ids[i])]['edge_start'], reverse=True)[:micro]
    graph = Batch.from_data_list([data[i] for i in ordered]).cuda()
    with torch.autocast('cuda'):
        logits = model(graph.x, graph.edge_index, graph.edge_attr)
    loss, terms = objective(logits, graph, 5)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.cuda.synchronize()
    return dict(micro_batch=micro, edges=graph.num_edges,
                event_ids=[data.ids[i] for i in ordered], loss=float(loss), terms=terms,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), optimizer_updates=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weight',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    assert sha(args.weight)=='5f3f02409399cd62efce5f254be02fa004237fb61a5b4717681cad1ad90cdcac'
    data=dataset(args.data,'train');failures=[]
    candidates=[4,2,1] if torch.cuda.get_device_properties(0).total_memory>=24*1024**3 else [2,1]
    for micro in candidates:
        try:
            result=attempt(data,args.weight,micro)
            break
        except torch.cuda.OutOfMemoryError as exc:
            failures.append(dict(micro_batch=micro,error=str(exc)))
            gc.collect();torch.cuda.empty_cache()
    else:
        raise RuntimeError('Dense batch does not fit even at micro_batch=1')
    result.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
                  total_memory=torch.cuda.get_device_properties(0).total_memory,oom_fallbacks=failures)
    save_json(args.root/'runtime_config.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
