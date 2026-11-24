import json,sys,time
from pathlib import Path
import torch
from fast_model import FastMembership
from train_full_stage import evaluate
root=Path(__file__).parent;torch.set_num_threads(1);models={}
for arm,path in [('D10',root/'review_20261123/weights/D64_epoch_010.pth'),('control7',root/'review_20261123/end_to_end_probe/control_v1_epoch_07.pth')]:
    m=FastMembership(16384).cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True));models[arm]=m
s=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'));start=time.time()
result=evaluate(models,root.parents[1]/'joint_event_dataset',s)
out=dict(protocol='32 calibration +200 development selection; historical16 are included; no blind confirmation',seconds=time.time()-start,arms=result)
(root/'review_20261123/end_to_end_probe/expanded_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
