"""Remote-only real-graph throughput and high-occupancy memory probes."""
import gc
import json
import time
import numpy as np
import torch
from torch_geometric.data import Batch
from common import *
from STEP2_训练模型 import graph_balanced_loss

def probe(graphs,batch_size,state):
    model=JointInteractionNetwork(chunk_size=65536).cuda();model.load_state_dict(state['model']);model.train()
    optimizer=torch.optim.AdamW(model.parameters());optimizer.load_state_dict(state['optimizer'])
    scaler=torch.amp.GradScaler('cuda');timings=[]
    torch.cuda.reset_peak_memory_stats()
    for repeat in range(3):
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
        g=Batch.from_data_list(graphs[:batch_size]).cuda()
        with torch.autocast('cuda'): pred=model(g.x,g.edge_index,g.edge_attr)
        loss=graph_balanced_loss(pred,g);scaler.scale(loss).backward();scaler.unscale_(optimizer)
        assert torch.isfinite(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        if repeat:timings.append(time.perf_counter()-start)
        del g,pred,loss
    return {'seconds_per_batch':sum(timings)/len(timings),'events_per_second':batch_size/(sum(timings)/len(timings)),'peak_memory_bytes':torch.cuda.max_memory_allocated()}

def main():
    torch.set_num_threads(3);dataset=JointEventDataset(DATA_ROOT,'train')
    state=torch.load(MODEL_ROOT/'joint_no_time_scratch_v1/training_state_joint_no_time.pth',map_location='cpu',weights_only=False)
    sample=[dataset[int(i)] for i in np.random.default_rng(20261106).choice(len(dataset),32,replace=False)]
    inventory=json.loads((DATA_ROOT/'event_inventory.json').read_text());train_ids=set(dataset.ids)
    heavy=sorted([v for v in inventory if v['event_id'] in train_ids],key=lambda v:v['hits'],reverse=True)[:64]
    heavy_graphs=sorted([dataset[dataset.ids.index(v['event_id'])] for v in heavy],key=lambda g:g.num_edges,reverse=True)[:32]
    results=[]
    for b in [1,4,8,16,32]:
        row={'batch_size':b}
        try:
            row['random']=probe(sample,b,state);gc.collect();torch.cuda.empty_cache()
            row['high_occupancy']=probe(heavy_graphs,b,state);row['status']='passed'
        except torch.cuda.OutOfMemoryError:
            row['status']='out_of_memory'
        gc.collect();torch.cuda.empty_cache();results.append(row);save_json(ROOT/'remote_batch_benchmark.json',results);print(json.dumps(row),flush=True)
    eligible=[r for r in results if r['status']=='passed' and r['high_occupancy']['peak_memory_bytes']<27*1024**3]
    best=max(eligible,key=lambda r:r['random']['events_per_second'])
    save_json(ROOT/'remote_batch_selection.json',{'batch_size':best['batch_size'],'accumulation_steps':32//best['batch_size'],'validation_batch_size':best['batch_size'],'criterion':'highest random throughput among tested batches passing high-occupancy probe below27GiB; 64 largest-hit events considered, not exhaustive graph edge ranking'})

def fair_comparison():
    torch.set_num_threads(3);dataset=JointEventDataset(DATA_ROOT,'train')
    state=torch.load(MODEL_ROOT/'joint_no_time_scratch_v1/training_state_joint_no_time.pth',map_location='cpu',weights_only=False)
    graphs=[dataset[int(i)] for i in np.random.default_rng(20261106).choice(len(dataset),32,replace=False)]
    rows=json.loads((ROOT/'remote_batch_benchmark.json').read_text())
    for row in rows:
        if row['status']!='passed':continue
        b=row['batch_size'];model=JointInteractionNetwork(chunk_size=65536).cuda();model.load_state_dict(state['model']);model.train()
        opt=torch.optim.AdamW(model.parameters());opt.load_state_dict(state['optimizer']);scaler=torch.amp.GradScaler('cuda');samples=[]
        for repeat in range(3):
            opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            for i in range(0,32,b):
                group=Batch.from_data_list(graphs[i:i+b]).cuda()
                with torch.autocast('cuda'):pred=model(group.x,group.edge_index,group.edge_attr)
                loss=graph_balanced_loss(pred,group);scaler.scale(loss*group.num_graphs/32).backward()
                del group,pred,loss
            scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();torch.cuda.synchronize()
            if repeat:samples.append(time.perf_counter()-start)
        row['same_32_events']={'seconds':sum(samples)/len(samples),'events_per_second':32/(sum(samples)/len(samples))}
        print(json.dumps({'batch':b,**row['same_32_events']}),flush=True);del model,opt,scaler;gc.collect();torch.cuda.empty_cache()
    save_json(ROOT/'remote_batch_benchmark.json',rows)
    valid=[r for r in rows if r['status']=='passed' and r['high_occupancy']['peak_memory_bytes']<27*1024**3]
    best=max(valid,key=lambda r:r['same_32_events']['events_per_second'])
    save_json(ROOT/'remote_batch_selection.json',{'batch_size':best['batch_size'],'accumulation_steps':32//best['batch_size'],'validation_batch_size':best['batch_size'],'criterion':'same32events, identical effective32 objective, max throughput with stress<27GiB','results':rows})

if __name__=='__main__':
    import sys
    fair_comparison() if '--fair' in sys.argv else main()
