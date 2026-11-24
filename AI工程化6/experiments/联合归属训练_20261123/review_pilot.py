"""Small, explicitly preliminary validation of epoch10; no formal 3D claims."""
import json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from membership import Membership,oc_loss

ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'runtime'))
from common import JointEventDataset

def decode(z,beta,cut,radius):
    centers=[]
    for i in np.argsort(-beta,kind='stable'):
        if beta[i]<cut:break
        if not centers or np.min(np.linalg.norm(z[centers]-z[i],axis=1))>radius:centers.append(int(i))
    if not centers:return np.full(len(z),-1)
    distances=np.linalg.norm(z[:,None,:]-z[np.asarray(centers)][None,:,:],axis=2)
    group=distances.argmin(1);group[distances.min(1)>radius]=-1
    return group

def counts(labels,truth,stereo):
    axial=~stereo;ids,n=np.unique(truth[axial&(truth>=0)],return_counts=True)
    eligible={int(k):int(c) for k,c in zip(ids,n) if c>=5};matched=set();outputs=fakes=clones=correct=0
    for label in np.unique(labels[labels>=0]):
        selected=(labels==label)&axial
        if selected.sum()<5:continue
        outputs+=1;tid,c=np.unique(truth[selected],return_counts=True);j=c.argmax();k=int(tid[j]);common=int(c[j])
        if k in eligible and common>=5 and common/selected.sum()>=.8 and common/eligible[k]>=.5:
            if k in matched:clones+=1
            else:matched.add(k);correct+=1
        else:fakes+=1
    return dict(truth=len(eligible),outputs=outputs,correct=correct,fakes=fakes,clones=clones)

def aggregate(rows):
    total={k:sum(r[k] for r in rows) for k in ['truth','outputs','correct','fakes','clones']}
    return dict(total,efficiency=total['correct']/max(total['truth'],1),fake_fraction=total['fakes']/max(total['outputs'],1),clone_fraction=total['clones']/max(total['outputs'],1))

@torch.no_grad()
def main():
    torch.set_num_threads(1);out=ROOT/'review_20261123';out.mkdir(exist_ok=True)
    s=json.loads((ROOT/'split_manifest.json').read_text(encoding='utf-8'))
    ds=JointEventDataset(ROOT.parents[1]/'joint_event_dataset','validation');lookup={int(e):i for i,e in enumerate(ds.ids)}
    calibration=s['decoder_calibration_event_ids'][:8];validation=s['member_selection_event_ids'][:16]
    assert not set(calibration)&set(validation) and not (set(calibration+validation)&set(s['original_train_event_ids']))
    device='cuda' if torch.cuda.is_available() else 'cpu';models={}
    for arm in ['C64','D64']:
        m=Membership().to(device);m.load_state_dict(torch.load(out/'weights'/f'{arm}_epoch_010.pth',map_location=device,weights_only=True));m.eval();models[arm]=m
    records=[];began=time.time()
    for eid in calibration+validation:
        g=ds[lookup[eid]].to(device);truth=g.truth.cpu().numpy();stereo=g.stereo.cpu().numpy();record=dict(event_id=eid,split='calibration' if eid in calibration else 'validation',truth=truth,stereo=stereo)
        for arm,m in models.items():
            edge,_,oc=m(g,torch.empty((0,3),device=device,dtype=torch.long));target=(g.truth[g.edge_index[0]]==g.truth[g.edge_index[1]])&(g.truth[g.edge_index[0]]>=0)
            pred=edge.sigmoid()>=.5;ec=[]
            for k in range(3):
                mask=g.edge_kind==k
                ec.append([int((pred&target&mask).sum()),int((pred&~target&mask).sum()),int((~pred&target&mask).sum())])
            record[arm]=dict(edge_counts=ec)
            if arm=='D64':
                raw=oc.cpu().numpy();record.update(z=raw[:,:3],beta=oc[:,3].sigmoid().cpu().numpy());record['oc_validation_loss']=float(oc_loss(oc,g.truth,g.stereo))
            del edge,oc
        records.append(record)
        print(json.dumps(dict(event=eid,completed=len(records),total=len(calibration+validation),seconds=time.time()-began)),flush=True)
    grid=[]
    for cut in [.3,.5,.7]:
        for radius in [.5,.75,1.]:
            metrics=aggregate([counts(decode(r['z'],r['beta'],cut,radius),r['truth'],r['stereo']) for r in records if r['split']=='calibration'])
            grid.append(dict(beta_threshold=cut,radius=radius,metrics=metrics))
    best=max(grid,key=lambda r:2*r['metrics']['correct']/max(r['metrics']['truth']+r['metrics']['outputs'],1))
    selected=[r for r in records if r['split']=='validation'];rows=[]
    for r in selected:
        labels=decode(r['z'],r['beta'],best['beta_threshold'],best['radius'])
        tids=np.unique(r['truth'][r['truth']>=0]);centers=[]
        for tid in tids:
            ix=np.flatnonzero(r['truth']==tid);centers.append(ix[r['beta'][ix].argmax()])
        dist=np.linalg.norm(r['z'][:,None,:]-r['z'][centers][None,:,:],axis=2)
        oracle=dist.argmin(1)
        oracle_counts=counts(oracle,r['truth'],r['stereo'])
        rows.append(dict(event_id=r['event_id'],**counts(labels,r['truth'],r['stereo']),
            truth_center_oracle_counts=oracle_counts,
            truth_center_oracle_hit_accuracy=float(np.mean(tids[oracle]==r['truth'])),
            assigned_fraction=float((labels>=0).mean()),beta_quantiles=np.quantile(r['beta'],[0,.5,.9,1]).tolist(),embedding_std=r['z'].std(0).tolist()))
    edges={}
    for arm in models:
        c=np.array([r[arm]['edge_counts'] for r in selected]).sum(0)
        edges[arm]={name:dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=float(tp/max(tp+fp,1)),recall=float(tp/max(tp+fn,1))) for name,(tp,fp,fn) in zip(['AA','AS','SS'],c)}
    serial=dict(status='preliminary_validation_completed',calibration_event_ids=calibration,validation_event_ids=validation,
        selected_decoder=best,calibration_grid=grid,validation_raw_oc=aggregate(rows),validation_events=rows,
        validation_oc_loss=float(np.mean([r['oc_validation_loss'] for r in selected])),edge_threshold_0p5=edges,
        truth_center_oracle=aggregate([r['truth_center_oracle_counts'] for r in rows]),
        limits=['8 calibration and16 validation only','raw learned-space clusters; no geometric refinement or stereo seed evaluation','C64 OC head is untrained and is not a clustering control','no stable-baseline efficiency comparison','not formal200-epoch training'],seconds=time.time()-began,
        weight_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'weights').glob('*.pth')})
    (out/'validation.json').write_text(json.dumps(serial,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(done=True,metrics=serial['validation_raw_oc'],oc_loss=serial['validation_oc_loss'],seconds=serial['seconds'])),flush=True)

if __name__=='__main__':main()
