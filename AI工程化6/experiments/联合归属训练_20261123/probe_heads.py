"""Cheap paired loss-mechanism probe with frozen D10 encoder; not formal training."""
import copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
from fast_model import FastMembership
from repair_v2 import oc_v1,oc_v2,oc_hard
from review_pilot import decode,counts,aggregate

def main():
    torch.set_num_threads(1);torch.manual_seed(20261123);root=Path(__file__).parent
    sys.path.insert(0,str(root/'runtime'));from common import JointEventDataset
    source=root.parents[1]/'joint_event_dataset';s=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'))
    sets={'train':s['pilot_train_event_ids'][:32],'calibration':s['decoder_calibration_event_ids'][:8],'validation':s['member_selection_event_ids'][:16]}
    model=FastMembership(16384).cuda();model.load_state_dict(torch.load(root/'review_20261123/weights/D64_epoch_010.pth',weights_only=True));model.eval();cached={}
    for split,ids in sets.items():
        ds=JointEventDataset(source,'train' if split=='train' else 'validation');lookup={int(e):i for i,e in enumerate(ds.ids)}
        for eid in ids:
            g=ds[lookup[eid]].to('cuda')
            with torch.no_grad():
                h=model.encode(g.x)
                for _ in range(7):h=model.step(h,g.x,g.edge_index,g.edge_attr)
            cached[eid]=(h.detach(),g.truth,g.stereo)
    heads={name:copy.deepcopy(model.oc) for name in ['v1','v2','hard_v1']};opts={k:torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001) for k,h in heads.items()}
    losses={'v1':oc_v1,'v2':oc_v2,'hard_v1':oc_hard};history=[];start=time.time()
    for step in range(400):
        eid=sets['train'][step%len(sets['train'])];h,truth,stereo=cached[eid]
        for name,head in heads.items():
            opts[name].zero_grad(set_to_none=True);loss=losses[name](head(h).float(),truth,stereo);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);opts[name].step()
        if step%50==0:print(json.dumps(dict(step=step,seconds=time.time()-start)),flush=True)
    heads['initial']=copy.deepcopy(model.oc)
    result=dict(protocol='frozen_D10_encoder_head_only_400_updates_not_end_to_end',train_event_ids=sets['train'],arms={})
    for name,head in heads.items():
        predictions={}
        with torch.no_grad():
            for eid,(h,truth,stereo) in cached.items():
                output=head(h);predictions[eid]=(output[:,:3].cpu().numpy(),output[:,3].sigmoid().cpu().numpy(),truth.cpu().numpy(),stereo.cpu().numpy())
        grid=[]
        for cut in [.3,.5,.7]:
            for radius in [.25,.5,.75,1.]:
                rows=[]
                for eid in sets['calibration']:
                    z,b,t,sr=predictions[eid];rows.append(counts(decode(z,b,cut,radius),t,sr))
                grid.append(dict(cut=cut,radius=radius,metrics=aggregate(rows)))
        best=max(grid,key=lambda x:2*x['metrics']['correct']/max(x['metrics']['truth']+x['metrics']['outputs'],1))
        metrics={}
        for split in ['train','validation']:
            rows=[]
            for eid in sets[split]:
                z,b,t,sr=predictions[eid];rows.append(counts(decode(z,b,best['cut'],best['radius']),t,sr))
            metrics[split]=aggregate(rows)
        result['arms'][name]=dict(best_calibration=best,metrics=metrics)
    out=root/'review_20261123/head_probe';out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    for name,head in heads.items():torch.save(head.state_dict(),out/(name+'.pth'))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
