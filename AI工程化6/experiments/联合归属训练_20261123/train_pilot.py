"""Paired, resumable S2 pilot only. Formal training requires member-evaluation gate."""
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from membership import Membership,sample,objectives

def atomic(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--cache');p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(20261123);np.random.seed(20261123);random.seed(20261123)
    root=Path(__file__).parent;out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'runtime'))
    from common import JointEventDataset
    from graph_cache import CachedGraphDataset
    splits=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((Path(args.data)/'dataset_metadata_joint_no_time.json').read_bytes()).hexdigest()==splits['dataset_sha256']
    ds=CachedGraphDataset(args.cache,'train',args.data) if args.cache else JointEventDataset(args.data,'train')
    lookup={int(e):i for i,e in enumerate(ds.ids)}
    ids=splits['smoke_train_event_ids'][:2] if args.smoke else splits['pilot_train_event_ids']
    device='cpu' if args.smoke else 'cuda'; maximum=1 if args.smoke else 10
    models={k:Membership().to(device) for k in ['C64','D64']};models['D64'].load_state_dict(models['C64'].state_dict())
    opts={k:torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=1e-4) for k,m in models.items()}
    scalers={k:torch.amp.GradScaler('cuda',enabled=device=='cuda') for k in models}
    signature={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'membership.py',root/'train_pilot.py',root/'split_manifest.json']}
    epoch=1;cursor=0;updates=0;pending=0;history=[]
    statepath=out/'state.pth'
    if statepath.exists():
        s=torch.load(statepath,map_location=device,weights_only=False);assert s['signature']==signature
        for k in models:models[k].load_state_dict(s['models'][k]);opts[k].load_state_dict(s['optimizers'][k]);scalers[k].load_state_dict(s['scalers'][k])
        epoch,cursor,updates=s['epoch'],s['cursor'],s['updates'];history=s['history']
        torch.set_rng_state(s['rng_cpu'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in s['rng_cuda']])
    def save():
        assert pending==0
        atomic(statepath,dict(signature=signature,models={k:m.state_dict() for k,m in models.items()},optimizers={k:o.state_dict() for k,o in opts.items()},scalers={k:s.state_dict() for k,s in scalers.items()},epoch=epoch,cursor=cursor,updates=updates,history=history,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if device=='cuda' else []))
    def status(value):
        tmp=out/'progress.tmp';tmp.write_text(json.dumps(value),encoding='utf-8');tmp.replace(out/'progress.json')
    save();lastsave=time.time();deadline=int(os.getenv('SLURM_JOB_END_TIME','0') or 0)
    print(json.dumps(dict(start=True,parameters=sum(p.numel() for p in models['D64'].parameters()),device=device,pilot_events=len(ids),signature=signature)),flush=True)
    while epoch<=maximum:
        order=np.random.default_rng(20261123+epoch).permutation(ids).tolist();began=time.time();totals={k:np.zeros(3) for k in models};seen=0
        lr=3e-5+(3e-4-3e-5)*min((epoch-1)/4,1)
        for k in models:
            models[k].train();opts[k].zero_grad(set_to_none=True)
            for group in opts[k].param_groups:group['lr']=lr
        while cursor<len(order):
            if pending==0 and deadline and time.time()+240>deadline:
                save();status(dict(status='allocation_time_low',epoch=epoch,cursor=cursor,updates=updates));return
            g=ds[lookup[order[cursor+pending]]]
            t=sample(g,20261123+epoch*100003+int(g.event_id),limit=256 if args.smoke else 8192)
            g=g.to(device);t=t.to(device);group=min(32,len(order)-cursor);pending+=1
            for k in models:
                with torch.autocast(device_type='cuda',enabled=device=='cuda'):
                    result=models[k](g,t);le,lt,lo=objectives(g,t,result)
                    loss=.5*le+.5*lt+(min(.1*epoch,1)*lo if k=='D64' else 0*lo)
                if not torch.isfinite(loss):raise FloatingPointError((k,epoch,cursor))
                scalers[k].scale(loss/group).backward();totals[k]+=[le.item(),lt.item(),lo.item()]
                del result,loss,le,lt,lo
            seen+=1;del g,t
            if pending==group:
                for k in models:
                    scalers[k].unscale_(opts[k]);norm=torch.nn.utils.clip_grad_norm_(models[k].parameters(),5)
                    if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradients '+k)
                    scalers[k].step(opts[k]);scalers[k].update();opts[k].zero_grad(set_to_none=True)
                cursor+=pending;pending=0;updates+=1
                if updates%3==0 or time.time()-lastsave>=300:save();lastsave=time.time()
                row=dict(status='training',stage='S2_pilot',epoch=epoch,cursor=cursor,total=len(order),updates=updates,losses={k:(totals[k]/seen).tolist() for k in totals},seconds=time.time()-began,unix=time.time())
                status(row);print(json.dumps(row),flush=True)
        history.append(dict(epoch=epoch,losses={k:(totals[k]/max(seen,1)).tolist() for k in totals},seconds=time.time()-began))
        for k in models:atomic(out/(k+'_epoch_%03d.pth'%epoch),models[k].state_dict())
        epoch+=1;cursor=0;save()
    status(dict(status='pilot_completed_requires_validation',epochs=maximum,history=history,formal_training_started=False))

if __name__=='__main__':main()
