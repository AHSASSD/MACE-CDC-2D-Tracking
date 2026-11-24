"""End-to-end server benchmark: disk/CPU/H2D/forward/backward/optimizer."""
import gc
import json
import os
import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from common import *
from graph_cache import CachedGraphDataset
from server_runtime import loader,DevicePrefetch
from STEP2_训练模型 import graph_balanced_loss

CACHE=Path('/seu_nvme/home/zhangyongchao/213230965/ai6_graph_cache_v1')

def trial(config,indices,stress=False):
    for key,value in {'MACE_PIN_MEMORY':int(config['workers']>0),'MACE_ASYNC_TRANSFER':int(config['workers']>0)}.items():os.environ[key]=str(value)
    torch.set_num_threads(max(1,3-config['workers']))
    source=CachedGraphDataset(CACHE,'train') if config['cache'] else JointEventDataset(DATA_ROOT,'train')
    subset=torch.utils.data.Subset(source,indices)
    batches=loader(subset,config['batch'],workers=config['workers'])
    model=JointInteractionNetwork(chunk_size=config['chunk'],recompute_policy=config['policy']).cuda()
    state=torch.load(MODEL_ROOT/'joint_no_time_scratch_v1/training_state_joint_no_time.pth',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);opt=torch.optim.AdamW(model.parameters());opt.load_state_dict(state['optimizer']);scaler=torch.amp.GradScaler('cuda')
    elapsed=[];torch.cuda.reset_peak_memory_stats()
    try:
        for repeat in range(2 if stress else 3):
            model.train();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();begin=time.perf_counter();events=0
            for index,g in enumerate(DevicePrefetch(batches,torch.device('cuda'))):
                count=g.num_graphs;group_start=(events//32)*32;group_size=min(32,len(subset)-group_start)
                with torch.autocast('cuda'):pred=model(g.x,g.edge_index,g.edge_attr)
                loss=graph_balanced_loss(pred,g);scaler.scale(loss*count/group_size).backward();events+=count
                if events%32==0 or events==len(subset):
                    scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite benchmark loss')
                del g,pred,loss
            torch.cuda.synchronize()
            if repeat:elapsed.append(time.perf_counter()-begin)
        seconds=sum(elapsed)/len(elapsed)
        return {'status':'passed','seconds':seconds,'events_per_second':len(subset)/seconds,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
    except torch.cuda.OutOfMemoryError:return {'status':'oom'}
    finally:
        del model,opt,scaler,batches,source,subset;gc.collect();torch.cuda.empty_cache()

def main():
    manifest=json.loads((CACHE/'manifest.json').read_text());ids=manifest['event_ids']['train']
    indices=np.random.default_rng(20261106).choice(len(ids),64,replace=False).tolist()
    heavy=sorted(range(len(ids)),key=lambda i:manifest['index'][str(ids[i])]['edge_end']-manifest['index'][str(ids[i])]['edge_start'],reverse=True)
    rows=[]
    def measure(c):
        existing=next((r for r in rows if r['config']==c),None)
        if existing:return existing
        gc.collect();torch.cuda.empty_cache();result=trial(c,indices);gc.collect();torch.cuda.empty_cache()
        row={'config':c,'end_to_end':result}
        if result['status']=='passed':row['stress']=trial(c,heavy[:c['batch']],True)
        else:row['stress']={'status':'not_run'}
        gc.collect();torch.cuda.empty_cache();rows.append(row);save_json(ROOT/'server_pipeline_benchmark.json',{'sample_event_ids':[ids[i] for i in indices],'rows':rows});print(json.dumps(row),flush=True);return row
    def valid(row):return row['end_to_end']['status']=='passed' and row['stress']['status']=='passed' and row['stress']['peak_memory_bytes']<26*1024**3
    baseline={'cache':False,'workers':0,'batch':8,'chunk':65536,'policy':'all'};measure(baseline)
    stages=[]
    for w in [0,1,2]:
        r=measure({**baseline,'cache':True,'workers':w})
        if valid(r):stages.append(r)
    best=max(stages,key=lambda r:r['end_to_end']['events_per_second'])
    for policy in ['one_direct','adaptive']:
        r=measure({**best['config'],'policy':policy})
        if valid(r) and r['end_to_end']['events_per_second']>best['end_to_end']['events_per_second']:best=r
    selected=dict(best['config'])
    for batch in [4,8,16]:
        for chunk in [32768,65536,131072]:
            r=measure({**selected,'batch':batch,'chunk':chunk})
            if valid(r) and r['end_to_end']['events_per_second']>best['end_to_end']['events_per_second']:best=r
    save_json(ROOT/'server_runtime_selection.json',{'config':best['config'],'measurement':best,'baseline':rows[0],
        'speedup':best['end_to_end']['events_per_second']/rows[0]['end_to_end']['events_per_second'],'cache_root':str(CACHE),
        'effective_batch':32,'scope':'same64 training events, warmup plus two timed complete data-to-optimizer sweeps; stress uses largest edge-count graphs over entire training partition'})

if __name__=='__main__':main()
