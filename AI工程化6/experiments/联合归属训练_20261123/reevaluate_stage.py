"""Scale sensitivity: same expanded grid for all weights; calibration only picks cuts."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from fast_model import FastMembership
from review_pilot import decode,counts,aggregate

@torch.no_grad()
def main():
    torch.set_num_threads(1);root=Path(__file__).parent;out=root/'review_20261124'
    sys.path.insert(0,str(root/'runtime'));from common import JointEventDataset
    s=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'));ds=JointEventDataset(root.parents[1]/'joint_event_dataset','validation');lookup={int(e):i for i,e in enumerate(ds.ids)}
    paths={'initial':root/'review_20261123/end_to_end_probe/control_v1_epoch_07.pth','epoch5':out/'OC_v1_full_epoch_05.pth','epoch9':out/'OC_v1_full_epoch_09.pth'}
    models={}
    for k,p in paths.items():
        m=FastMembership(16384).cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=True));m.eval();models[k]=m
    state=torch.load(out/'state.pth',map_location='cpu',weights_only=False);m=FastMembership(16384).cuda();m.load_state_dict(state['models']['OC_v1_full']);m.eval();models['epoch10_partial']=m
    cal=s['decoder_calibration_event_ids'][:32];val=s['member_selection_event_ids'];pred={k:{} for k in models};start=time.time()
    for i,eid in enumerate(cal+val):
        g=ds[lookup[eid]].to('cuda')
        for arm,m in models.items():
            h=m.encode(g.x)
            for _ in range(7):h=m.step(h,g.x,g.edge_index,g.edge_attr)
            oc=m.oc(h).float();pred[arm][eid]=(oc[:,:3].cpu().numpy(),oc[:,3].sigmoid().cpu().numpy(),g.truth.cpu().numpy(),g.stereo.cpu().numpy())
        if i%40==0:print(json.dumps(dict(phase='encode',events=i+1,total=len(cal+val),seconds=time.time()-start)),flush=True)
    result=dict(protocol='32 calibration,200 development events; same log-radius grid for all4 checkpoints; not blind',radius_grid=np.geomspace(.015625,1,13).tolist(),arms={})
    for arm,p in pred.items():
        grid=[]
        for cut in [.3,.5,.7]:
            for radius in result['radius_grid']:
                rows=[]
                for eid in cal:
                    z,b,t,sr=p[eid];rows.append(counts(decode(z,b,cut,radius),t,sr))
                c=aggregate(rows);grid.append(dict(cut=cut,radius=radius,metrics=c))
        best=max(grid,key=lambda x:2*x['metrics']['correct']/max(x['metrics']['truth']+x['metrics']['outputs'],1));rows=[];oracle=[];geometry=[]
        for eid in val:
            z,b,t,sr=p[eid];rows.append(dict(event_id=eid,**counts(decode(z,b,best['cut'],best['radius']),t,sr)))
            tids=np.unique(t[t>=0]);centers=[]
            for tid in tids:
                ix=np.flatnonzero(t==tid);centers.append(ix[b[ix].argmax()])
            dist=np.linalg.norm(z[:,None]-z[np.asarray(centers)][None,:],axis=2)
            oracle.append(counts(dist.argmin(1),t,sr));geometry.append(z.std(0).tolist())
        result['arms'][arm]=dict(calibration=best,validation=aggregate(rows),events=rows,truth_center_diagnostic=aggregate(oracle),mean_embedding_std=np.mean(geometry,0).tolist())
        print(json.dumps(dict(arm=arm,best=best,validation=result['arms'][arm]['validation'],oracle=result['arms'][arm]['truth_center_diagnostic'])),flush=True)
        (out/'radius_review.json').write_text(json.dumps(result,indent=2))
    result['seconds']=time.time()-start;(out/'radius_review.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
