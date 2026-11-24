"""Select validation batch independently using the validation partition only."""
import gc
import json
import os
import time
import numpy as np
import torch
from common import *
from graph_cache import CachedGraphDataset
from server_runtime import loader,DevicePrefetch
from STEP2_训练模型 import evaluate

def main():
    selection=json.loads((ROOT/'server_runtime_selection.json').read_text());c=selection['config'];workers=c['workers']
    os.environ['MACE_PIN_MEMORY']=str(int(workers>0));os.environ['MACE_ASYNC_TRANSFER']=str(int(workers>0));torch.set_num_threads(max(1,3-workers))
    data=CachedGraphDataset(selection['cache_root'],'validation');indices=np.random.default_rng(20261106).choice(len(data),64,replace=False).tolist()
    m=json.loads((Path(selection['cache_root'])/'manifest.json').read_text());heavy=sorted(range(len(data)),key=lambda i:m['index'][str(data.ids[i])]['edge_end']-m['index'][str(data.ids[i])]['edge_start'],reverse=True)
    state=torch.load(MODEL_ROOT/'joint_no_time_scratch_v1/training_state_joint_no_time.pth',map_location='cpu',weights_only=False)
    model=JointInteractionNetwork(chunk_size=c['chunk']).cuda();model.load_state_dict(state['model']);device=torch.device('cuda');rows=[]
    for batch in [8,16,32]:
        batches=loader(torch.utils.data.Subset(data,indices),batch,workers=workers);times=[];torch.cuda.reset_peak_memory_stats()
        for repeat in range(3):
            torch.cuda.synchronize();begin=time.perf_counter();evaluate(model,DevicePrefetch(batches,device),device,True);torch.cuda.synchronize()
            if repeat:times.append(time.perf_counter()-begin)
        del batches;gc.collect()
        pressure=loader(torch.utils.data.Subset(data,heavy[:batch]),batch,workers=workers)
        evaluate(model,DevicePrefetch(pressure,device),device,True);del pressure;gc.collect()
        row={'batch':batch,'events_per_second':64/(sum(times)/len(times)),'peak_memory_bytes':torch.cuda.max_memory_allocated()};rows.append(row);print(json.dumps(row),flush=True)
    best=max(rows,key=lambda row:row['events_per_second']);selection['validation_batch_size']=best['batch'];selection['validation_benchmark']=rows;save_json(ROOT/'server_runtime_selection.json',selection)

if __name__=='__main__':main()
