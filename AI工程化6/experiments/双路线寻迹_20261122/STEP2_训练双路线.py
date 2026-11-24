"""Train E and T independently while sharing frozen input I/O. Resumable per100 events."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,os,time,multiprocessing
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import sha,save_json
from et_data import Features,to_device,atomic_npz
from et_models import EHead,THead,predict
from et_decode import evaluate_event
from STEP5_两级寻迹 import summarize


def atomic(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def seed_summary(rows):
    c=Counter()
    for r in rows:c.update(r)
    return dict(c,f1=2*c['matched_tracks']/max(c['truth_tracks']+c['output_candidates'],1),
        efficiency=c['matched_tracks']/max(c['truth_tracks'],1),fake_fraction=c['fake_candidates']/max(c['output_candidates'],1),
        clone_fraction=c['clone_candidates']/max(c['output_candidates'],1))


def loaders(ds,workers=2):
    kwargs=dict(batch_size=None,num_workers=workers,pin_memory=True)
    if workers:kwargs.update(multiprocessing_context='spawn',prefetch_factor=2)
    return DataLoader(ds,**kwargs)


def class_weights(cache,ids):
    counts=np.zeros(4)
    for eid in ids:counts+=json.loads((cache/'stats'/('event_%05d.json'%eid)).read_text())['E_class_counts']
    weights=np.sqrt(max(counts.sum(),1)/(4*np.maximum(counts,1)))
    return torch.tensor(np.clip(weights,.5,10),dtype=torch.float32,device='cuda'),counts.tolist()


@torch.no_grad()
def validation(models,args,ids,epoch,member_ids):
    out=args.out/'validation'/('epoch_%03d'%epoch);out.mkdir(parents=True,exist_ok=True)
    confusion=np.zeros((4,4),dtype=np.int64);tc=np.zeros(3,dtype=np.int64);tasks=[];events=[];started=time.time()
    for model in models.values():model.eval()
    source=loaders(Features(args.cache,ids),workers=2)
    with ProcessPoolExecutor(max_workers=min(3,int(os.environ.get('SLURM_CPUS_PER_TASK','3'))),mp_context=multiprocessing.get_context('spawn')) as pool:
        for j,cpu in enumerate(source):
            b=to_device(cpu,'cuda')
            with torch.autocast('cuda'):
                ep=predict(models['E'],b,'E');tq=predict(models['T'],b,'T')
            ey=cpu['ey'].numpy();ty=cpu['ty'].numpy().astype(bool);tp=tq>=.5
            confusion+=np.bincount(ey*4+ep.argmax(1),minlength=16).reshape(4,4)
            tc+=[int((tp&ty).sum()),int((tp&~ty).sum()),int((~tp&ty).sum())]
            eid=int(cpu['eid'])
            if eid in member_ids:
                prediction=out/('prediction_%05d.npz'%eid);atomic_npz(prediction,compress=True,E=ep,T=tq)
                target=out/('member_%05d.json'%eid)
                if target.exists():events.append(json.loads(target.read_text(encoding='utf-8')))
                else:tasks.append(pool.submit(evaluate_event,(eid,str(args.cache),str(args.data),str(prediction),str(target),True)))
            if j%10==0:save_json(args.out/'progress.json',dict(phase='validation_scoring',epoch=epoch,events=j+1,total=len(ids),unix=time.time()))
        for future in as_completed(tasks):
            events.append(future.result());save_json(args.out/'progress.json',dict(phase='member_validation',epoch=epoch,events=len(events),total=len(member_ids),unix=time.time()))
    e_f1=[]
    for k in range(4):
        tp=confusion[k,k];e_f1.append(float(2*tp/max(confusion[k].sum()+confusion[:,k].sum(),1)))
    a,b,c=tc;result=dict(E=dict(confusion=confusion.tolist(),macro_f1=float(np.mean(e_f1)),class_f1=e_f1),
        T=dict(tp=int(a),fp=int(b),fn=int(c),precision=int(a)/max(int(a+b),1),recall=int(a)/max(int(a+c),1),f1=2*int(a)/max(int(2*a+b+c),1)),
        classification_events=len(ids),member_events=len(events),seconds=time.time()-started)
    if events:
        result['members']={key:dict(tracks=summarize([x['methods'][key]['counts'] for x in events]),
                         seeds=seed_summary([x['methods'][key]['seeds'] for x in events])) for key in ['raw_baseline','E','T']}
    save_json(out/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--splits',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['pilot','full'],default='pilot');p.add_argument('--epochs',type=int)
    p.add_argument('--member-count',type=int,default=200);p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(20261122);torch.cuda.manual_seed_all(20261122)
    args.out.mkdir(parents=True,exist_ok=True)
    if os.name=='posix':
        import fcntl
        lock=(args.out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    s=json.loads(args.splits.read_text(encoding='utf-8'));train=s['pilot_train_event_ids'] if args.mode=='pilot' else s['original_train_event_ids']
    valid=s['periodic_member_validation_event_ids'];maximum=args.epochs or (10 if args.mode=='pilot' else 60)
    weights,counts=class_weights(args.cache,train)
    contract=dict(mode=args.mode,epochs=maximum,train_ids=train,validation_ids=valid,member_count=args.member_count,
        source_hashes={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},cache_contract=sha(args.cache/'contract.json'),
        source_split_sha256=sha(args.splits),independent_routes=True,shared_trainable_parameters=False,E_class_counts=counts,
        hyperparameters=dict(lr=.001,eta_min=.0001,weight_decay=.0001,event_accumulation=4,seed=20261122),
        prototype_limits='E max3 lanes and new geometric groups <=80 axial; preserves baseline soft candidates; original HitIDs retained',auto_deploy=False)
    if (args.out/'contract.json').exists():assert json.loads((args.out/'contract.json').read_text(encoding='utf-8'))==contract
    save_json(args.out/'contract.json',contract)
    models={'E':EHead().cuda(),'T':THead().cuda()};optimizers={k:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for k,m in models.items()}
    schedules={k:torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=maximum,eta_min=.0001) for k,o in optimizers.items()}
    scalers={k:torch.amp.GradScaler('cuda',init_scale=1024.) for k in models};history=[];epoch=1;cursor=0;phase='train';updates=0
    if (args.out/'state.pth').exists():
        state=torch.load(args.out/'state.pth',map_location='cuda',weights_only=False)
        for k in models:models[k].load_state_dict(state['models'][k]);optimizers[k].load_state_dict(state['optimizers'][k]);schedules[k].load_state_dict(state['schedulers'][k]);scalers[k].load_state_dict(state['scalers'][k])
        epoch=state['epoch'];cursor=state['cursor'];phase=state['phase'];history=state['history'];updates=state['updates']
        torch.set_rng_state(state['rng_cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['rng_cuda']])
    def checkpoint():
        atomic(args.out/'state.pth',dict(models={k:m.state_dict() for k,m in models.items()},optimizers={k:o.state_dict() for k,o in optimizers.items()},
            schedulers={k:x.state_dict() for k,x in schedules.items()},scalers={k:x.state_dict() for k,x in scalers.items()},
            epoch=epoch,cursor=cursor,phase=phase,history=history,updates=updates,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()))
        save_json(args.out/'checkpoint_meta.json',dict(epoch=epoch,cursor=cursor,phase=phase,updates=updates,unix=time.time()))
    checkpoint();deadline=int(os.environ.get('SLURM_JOB_END_TIME','0') or 0)
    while epoch<=maximum:
        if deadline and time.time()+240>deadline:checkpoint();save_json(args.out/'status.json',dict(status='allocation_time_low',completed_epoch=epoch-1));return
        order=np.random.default_rng(20261122+epoch).permutation(train).tolist();loss_sum=Counter();events=0;started=time.time()
        if phase=='train':
            for m in models.values():m.train()
            for o in optimizers.values():o.zero_grad(set_to_none=True)
            pending=0;source=loaders(Features(args.cache,order[cursor:],epoch,training=True),workers=args.workers)
            for cpu in source:
                b=to_device(cpu,'cuda');group=min(4,len(train)-cursor);pending+=1;events+=1
                for key,model in models.items():
                    with torch.autocast('cuda'):
                        a,c=model(b)
                        if key=='E':loss=(torch.nn.functional.cross_entropy(a.float(),b['ey'].long(),weight=weights)+torch.nn.functional.cross_entropy(c.float(),b['ey'].long(),weight=weights))/2
                        else:loss=(torch.nn.functional.binary_cross_entropy_with_logits(a.float(),b['ty'])+torch.nn.functional.binary_cross_entropy_with_logits(c.float(),b['ty']))/2
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite '+key+' loss')
                    scalers[key].scale(loss/group).backward();loss_sum[key]+=float(loss.detach())
                if pending==group:
                    for k in models:
                        scalers[k].unscale_(optimizers[k]);torch.nn.utils.clip_grad_norm_(models[k].parameters(),5.)
                        scalers[k].step(optimizers[k]);scalers[k].update();optimizers[k].zero_grad(set_to_none=True)
                    cursor+=pending;pending=0;updates+=1
                    if updates%25==0:checkpoint()
                    save_json(args.out/'progress.json',dict(phase='training',mode=args.mode,epoch=epoch,events=cursor,total=len(train),updates=updates,
                        losses={k:loss_sum[k]/events for k in models},seconds=time.time()-started,unix=time.time()))
                    if updates%25==0:print(json.dumps(dict(mode=args.mode,epoch=epoch,events=cursor,total=len(train),updates=updates,losses={k:loss_sum[k]/events for k in models})),flush=True)
                    if deadline and time.time()+240>deadline:checkpoint();save_json(args.out/'status.json',dict(status='allocation_time_low',completed_epoch=epoch-1));return
            assert cursor==len(train) and pending==0;phase='validation';checkpoint()
        members=set(valid[:min(args.member_count,len(valid))]) if epoch%5==0 or epoch==maximum else set()
        check=validation(models,args,valid,epoch,members)
        row=dict(epoch=epoch,validation=check,updates=updates,train_seconds=time.time()-started)
        history.append(row);save_json(args.out/'history.json',history)
        for k in models:
            atomic(args.out/('%s_epoch_%03d.pth'%(k,epoch)),models[k].state_dict())
            if members:
                best=max([x for x in history if 'members' in x['validation']],key=lambda x:x['validation']['members'][k]['seeds']['f1'])
                if best['epoch']==epoch:atomic(args.out/('%s_best_members.pth'%k),models[k].state_dict())
        print(json.dumps(row),flush=True)
        checked=[x for x in history if 'members' in x['validation']]
        if args.mode=='full' and epoch>=20 and len(checked)>=5:
            best_epochs=[max(checked,key=lambda x:x['validation']['members'][k]['seeds']['f1'])['epoch'] for k in models]
            if all(x<=checked[-5]['epoch'] for x in best_epochs):
                epoch+=1;cursor=0;phase='train';checkpoint();save_json(args.out/'status.json',dict(status='early_stopped',completed_epoch=epoch-1,reason='Both routes without member F1 gain for4 checks'));return
        for schedule in schedules.values():schedule.step()
        epoch+=1;cursor=0;phase='train';checkpoint();save_json(args.out/'status.json',dict(status='running',completed_epoch=epoch-1,unix=time.time()))
    save_json(args.out/'status.json',dict(status='completed',completed_epoch=maximum,unix=time.time()))


if __name__=='__main__':main()
