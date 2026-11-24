"""Full-step GPU benchmarks; exact-v1 control isolates speed from objective change."""
import argparse,json,sys,time,copy,gc
from pathlib import Path
import numpy as np
import torch
import membership as legacy
from repair_v2 import install,sample as fast_sample
from fast_model import FastMembership,pack,event_objectives

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--cache');p.add_argument('--out',required=True);args=p.parse_args()
    root=Path(__file__).parent;sys.path.insert(0,str(root/'runtime'))
    from common import JointEventDataset
    from graph_cache import CachedGraphDataset
    torch.set_num_threads(1);torch.manual_seed(123)
    ds=CachedGraphDataset(args.cache,'train',args.data) if args.cache else JointEventDataset(args.data,'train')
    s=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'));lookup={int(e):i for i,e in enumerate(ds.ids)}
    inventory={r['event_id']:r['hits'] for r in json.loads((Path(args.data)/'event_inventory.json').read_text())}
    order=sorted(ds.ids,key=lambda e:inventory[e]);selected=[order[len(order)//2],order[int(len(order)*.95)],order[-1]]
    original_balanced=legacy.balanced;original_oc=legacy.oc_loss;initial=FastMembership().state_dict();rows=[]
    for eid in selected:
        io=time.perf_counter();raw=ds[lookup[eid]];io=time.perf_counter()-io
        for name,sampler,chunk,direct,micro in [('legacy',legacy.sample,4096,0,1),('vectorized',fast_sample,4096,0,1),('chunk16384',fast_sample,16384,0,1),('partial16384',fast_sample,16384,2,1),('micro2',fast_sample,16384,2,2),('micro4',fast_sample,16384,2,4)]:
            try:
                if name=='legacy':legacy.balanced=original_balanced;legacy.oc_loss=original_oc
                else:install('v1')
                sample_start=time.perf_counter();graphs=[raw.clone() for _ in range(micro)];ts=[sampler(g,123+i) for i,g in enumerate(graphs)];g,t,slices=pack(graphs,ts);sample_seconds=time.perf_counter()-sample_start
                model=FastMembership(chunk,direct).cuda();model.load_state_dict(initial);model.train()
                opt=torch.optim.AdamW(model.parameters(),lr=3e-4);scaler=torch.amp.GradScaler('cuda',init_scale=1024)
                torch.cuda.reset_peak_memory_stats();measure=[];stages=[]
                for repeat in range(3):
                    torch.cuda.synchronize();start=time.perf_counter();gpu=g.clone().to('cuda');tt=t.cuda();torch.cuda.synchronize();transfer=time.perf_counter()-start
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda'):result=model(gpu,tt)
                    torch.cuda.synchronize();fwd=time.perf_counter()
                    le,lt,lo=event_objectives(gpu,tt,result,slices,legacy.objectives);loss=.5*le+.5*lt+lo
                    torch.cuda.synchronize();lossend=time.perf_counter();scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();torch.cuda.synchronize();end=time.perf_counter()
                    if repeat:measure.append(end-start);stages.append(dict(transfer=transfer,forward=fwd-start-transfer,objective=lossend-fwd,backward_update=end-lossend))
                    del result,loss,le,lt,lo,gpu,tt
                row=dict(event_id=eid,nodes=raw.num_nodes,edges=raw.num_edges,profile=name,microbatch=micro,io_seconds=io,sample_seconds=sample_seconds,
                    gpu_step_seconds=float(np.mean(measure)),total_step_including_io_sample=(float(np.mean(measure))+sample_seconds)/micro+io,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,stage_seconds=stages)
                del model,opt,scaler,g,t
            except torch.cuda.OutOfMemoryError:
                row=dict(event_id=eid,profile=name,status='oom')
            rows.append(row);print(json.dumps(row),flush=True)
            Path(args.out).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),rows=rows),indent=2))
            gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
