"""Bounded paired end-to-end OC ablation, with server-measured runtime settings.
Warm-start D10; this is not the final from-scratch 200-epoch experiment.
"""
import argparse,hashlib,json,os,sys,time,copy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'runtime'))
from common import JointEventDataset
from graph_cache import CachedGraphDataset
import membership as legacy
from fast_model import FastMembership,pack,event_objectives
from repair_v2 import balanced,oc_v1,oc_hard,sample
from review_pilot import decode,counts,aggregate

class Prepared(Dataset):
    def __init__(self,data,cache,ids,epoch):
        self.base=CachedGraphDataset(cache,'train',data) if cache else JointEventDataset(data,'train')
        self.lookup={int(e):i for i,e in enumerate(self.base.ids)};self.ids=ids;self.epoch=epoch
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        eid=self.ids[i];g=self.base[self.lookup[eid]];t=sample(g,20261123+self.epoch*100003+eid)
        return g,t
def collate(rows):return pack([r[0] for r in rows],[r[1] for r in rows])
def atomic(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)

@torch.no_grad()
def evaluate(models,data,splits):
    ds=JointEventDataset(data,'validation');lookup={int(e):i for i,e in enumerate(ds.ids)}
    ids=splits['decoder_calibration_event_ids'][:8]+splits['member_selection_event_ids'][:16];predictions={k:{} for k in models}
    for m in models.values():m.eval()
    for eid in ids:
        g=ds[lookup[eid]].to('cuda')
        for arm,m in models.items():
            _,_,oc=m(g,torch.empty((0,3),device='cuda',dtype=torch.long))
            predictions[arm][eid]=(oc[:,:3].cpu().numpy(),oc[:,3].sigmoid().cpu().numpy(),g.truth.cpu().numpy(),g.stereo.cpu().numpy())
    result={}
    for arm,pred in predictions.items():
        grid=[]
        for cut in [.3,.5,.7]:
            for radius in [.25,.5,.75,1.]:
                rows=[]
                for eid in ids[:8]:
                    z,beta,truth,stereo=pred[eid];rows.append(counts(decode(z,beta,cut,radius),truth,stereo))
                metrics=aggregate(rows);grid.append(dict(cut=cut,radius=radius,metrics=metrics))
        best=max(grid,key=lambda x:2*x['metrics']['correct']/max(x['metrics']['truth']+x['metrics']['outputs'],1));rows=[]
        for eid in ids[8:]:
            z,beta,truth,stereo=pred[eid];rows.append(counts(decode(z,beta,best['cut'],best['radius']),truth,stereo))
        result[arm]=dict(calibration=best,validation=aggregate(rows))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--cache');p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=10);p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(20261123);legacy.balanced=balanced
    s=json.loads((ROOT/'split_manifest.json').read_text(encoding='utf-8'));report=json.loads((ROOT/'server_benchmark.json').read_text())
    rows=[r for r in report['rows'] if r.get('status')!='oom'];profiles={r['profile'] for r in rows};eligible=[]
    for profile in profiles:
        group=[r for r in rows if r['profile']==profile]
        if len(group)==3 and max(r['peak_allocated_GiB'] for r in group)<27.2:
            eligible.append((float(np.mean([r['total_step_including_io_sample'] for r in group])),profile,group[0]['microbatch']))
    speed,profile,micro=min(eligible);assert profile!='legacy','No speed improvement demonstrated'
    baseline=np.mean([r['total_step_including_io_sample'] for r in rows if r['profile']=='legacy']);assert baseline/speed>1.2
    chunk=4096 if profile=='vectorized' else 16384;direct=2 if profile in ['partial16384','micro2','micro4'] else 0
    models={arm:FastMembership(chunk,direct).cuda() for arm in ['control_v1','hard_v1']}
    initial=torch.load(ROOT/'initial_D10.pth',map_location='cuda',weights_only=True)
    for m in models.values():m.load_state_dict(initial)
    opts={k:torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4) for k,m in models.items()};scalers={k:torch.amp.GradScaler('cuda',init_scale=1024) for k in models}
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True);signature={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'train_repair_probe.py',ROOT/'membership.py',ROOT/'repair_v2.py',ROOT/'fast_model.py',ROOT/'review_pilot.py',ROOT/'split_manifest.json',ROOT/'initial_D10.pth']}
    epoch=1;cursor=updates=0;history=[];statepath=out/'state.pth'
    if statepath.exists():
        state=torch.load(statepath,map_location='cuda',weights_only=False);assert state['signature']==signature
        for k in models:models[k].load_state_dict(state['models'][k]);opts[k].load_state_dict(state['optimizers'][k]);scalers[k].load_state_dict(state['scalers'][k])
        epoch,cursor,updates,history=state['epoch'],state['cursor'],state['updates'],state['history'];torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    def save():
        atomic(statepath,dict(signature=signature,epoch=epoch,cursor=cursor,updates=updates,history=history,models={k:m.state_dict() for k,m in models.items()},optimizers={k:o.state_dict() for k,o in opts.items()},scalers={k:v.state_dict() for k,v in scalers.items()},cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    def log(row):
        print(json.dumps(row),flush=True);temp=out/'progress.tmp';temp.write_text(json.dumps(row));temp.replace(out/'progress.json')
    contract=dict(signature=signature,stage='end_to_end_mechanism_probe',warmstart='D10',events=256,effective_batch=32,microbatch=micro,chunk=chunk,direct_steps=direct,workers=args.workers,measured_step_speedup=baseline/speed,formal_200_epoch=False)
    (out/'contract.json').write_text(json.dumps(contract,indent=2));log(contract);save();deadline=int(os.getenv('SLURM_JOB_END_TIME','0') or 0)
    if not history and cursor==0:
        metrics=evaluate(models,args.data,s);history.append(dict(epoch=0,members=metrics));save();log(dict(phase='initial_validation',metrics=metrics))
    while epoch<=args.epochs:
        ids=np.random.default_rng(20261123+epoch).permutation(s['pilot_train_event_ids'][:256]).tolist()
        ds=Prepared(args.data,args.cache,ids[cursor:],epoch)
        workers=1 if epoch==1 else args.workers
        if epoch>2:
            runs=[h for h in history if h.get('epoch') in [1,2] and 'train_seconds' in h]
            if len(runs)==2:workers=min(runs,key=lambda h:h['train_seconds'])['workers']
        kw=dict(batch_size=micro,num_workers=workers,collate_fn=collate,pin_memory=True)
        if workers:kw.update(prefetch_factor=2,multiprocessing_context='spawn')
        loader=DataLoader(ds,**kw);pending=0;start=time.time();wait=0;last=time.time();losses={k:[] for k in models}
        for m in models.values():m.train()
        for o in opts.values():o.zero_grad(set_to_none=True)
        for cpu,t,slices in loader:
            wait+=time.time()-last
            if pending==0 and deadline and time.time()>deadline-240:save();log(dict(status='deadline_checkpoint',epoch=epoch,cursor=cursor,updates=updates));return
            group=min(32,256-cursor);n=len(slices);g=cpu.to('cuda',non_blocking=True);t=t.to('cuda',non_blocking=True)
            for arm,m in models.items():
                legacy.oc_loss=oc_v1 if arm=='control_v1' else oc_hard
                with torch.autocast('cuda'):result=m(g,t)
                le,lt,lo=event_objectives(g,t,result,slices,legacy.objectives);loss=.5*le+.5*lt+lo
                if not torch.isfinite(loss):raise FloatingPointError(arm)
                scalers[arm].scale(loss*n/group).backward();losses[arm].append([le.item(),lt.item(),lo.item()]);del result,loss,le,lt,lo
            pending+=n;del g,t,cpu
            if pending==group:
                for arm in models:
                    scalers[arm].unscale_(opts[arm]);norm=torch.nn.utils.clip_grad_norm_(models[arm].parameters(),5)
                    if not torch.isfinite(norm):raise FloatingPointError('gradient '+arm)
                    scalers[arm].step(opts[arm]);scalers[arm].update();opts[arm].zero_grad(set_to_none=True)
                cursor+=pending;pending=0;updates+=1;save();log(dict(phase='training',epoch=epoch,event=cursor,updates=updates,seconds=time.time()-start,data_wait=wait,unix=time.time()))
            last=time.time()
        assert pending==0 and cursor==256
        train_seconds=time.time()-start
        metrics=evaluate(models,args.data,s);history.append(dict(epoch=epoch,members=metrics,workers=workers,train_seconds=train_seconds,epoch_seconds_including_validation=time.time()-start,data_wait=wait,losses={k:np.mean(v,0).tolist() for k,v in losses.items()}))
        log(dict(phase='validation',epoch=epoch,metrics=metrics))
        for arm,m in models.items():atomic(out/(arm+'_epoch_%02d.pth'%epoch),m.state_dict())
        epoch+=1;cursor=0;save()
    log(dict(status='mechanism_probe_completed',history=history,formal_training_started=False))
if __name__=='__main__':main()
