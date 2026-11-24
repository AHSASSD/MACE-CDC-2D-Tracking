"""Resume T25 with focused junction positives and hard negatives. Frozen GNN cache."""
import bootstrap
from bootstrap import HERE,BASE
from pathlib import Path
import os,json,time,argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import save_json,sha
from et_data import Features,to_device
from et_models import THead,predict
from focus_data import FocusDataset,strata

def atomic(path,state):
    temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)

@torch.no_grad()
def validate(model,args,ids,epoch):
    model.eval();counts={key:np.zeros(3,np.int64) for key in ['all05','all09','weak05','weak09']};started=time.time()
    for j,cpu in enumerate(DataLoader(Features(args.cache,ids),batch_size=None,num_workers=2,multiprocessing_context='spawn')):
        b=to_device(cpu,'cuda')
        with torch.autocast('cuda'):q=predict(model,b,'T')
        f={k:v.numpy() if torch.is_tensor(v) else v for k,v in cpu.items()};y=f['ty'].astype(bool)
        cat,_=strata(f,f['t'],y);weak=(cat==1)
        for key,threshold,mask in [('all05',.5,np.ones(len(q),bool)),('all09',.9,np.ones(len(q),bool)),('weak05',.5,weak),('weak09',.9,weak)]:
            yes=q>=threshold;counts[key]+=[int((yes&y&mask).sum()),int((yes&~y&mask).sum()),int((~yes&y&mask).sum())]
        if j%10==0:save_json(args.out/'progress.json',dict(phase='validation',epoch=epoch,events=j+1,total=len(ids),unix=time.time()))
    result={}
    for key,(a,b,c) in counts.items():result[key]=dict(tp=int(a),fp=int(b),fn=int(c),precision=float(a/max(a+b,1)),recall=float(a/max(a+c,1)),f1=float(2*a/max(2*a+b+c,1)))
    result['seconds']=time.time()-started;return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--splits',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--limit-train',type=int);args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(20261123);torch.cuda.manual_seed_all(20261123)
    args.out.mkdir(parents=True,exist_ok=True)
    if os.name=='posix':
        import fcntl
        lock=(args.out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    s=json.loads(args.splits.read_text(encoding='utf-8'));train=s['original_train_event_ids'];valid=s['periodic_member_validation_event_ids']
    if args.limit_train:train=train[:args.limit_train];valid=valid[:8]
    contract=dict(initial_sha256=sha(args.initial),cache_sha256=sha(args.cache/'contract.json'),train_ids=train,validation_ids=valid,epochs=args.epochs,
       sources={f.name:sha(f) for f in HERE.glob('*.py')},base_sources={f.name:sha(f) for f in BASE.glob('*.py')},lr=.0001,eta_min=.00001,accumulation=4,seed=20261123,auto_deploy=False)
    assert contract['initial_sha256']=='46a9d457a19e3e81112534f52e5a76e0437baa26b06c84f98fcd29f0e3b3eed2'
    if (args.out/'contract.json').exists():assert json.loads((args.out/'contract.json').read_text(encoding='utf-8'))==contract
    else:save_json(args.out/'contract.json',contract)
    model=THead().cuda();model.load_state_dict(torch.load(args.initial,map_location='cuda',weights_only=True));teacher=THead().cuda();teacher.load_state_dict(model.state_dict());teacher.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=.00001);scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    epoch=1;cursor=0;updates=0;history=[];phase='train'
    if (args.out/'state.pth').exists():
        state=torch.load(args.out/'state.pth',map_location='cuda',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler']);epoch=state['epoch'];cursor=state['cursor'];updates=state['updates'];history=state['history'];phase=state['phase']
        torch.set_rng_state(state['rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    def checkpoint():
        atomic(args.out/'state.pth',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch,cursor=cursor,updates=updates,phase=phase,history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        save_json(args.out/'checkpoint_meta.json',dict(epoch=epoch,cursor=cursor,updates=updates,phase=phase,unix=time.time()))
    if not history:
        baseline=validate(model,args,valid,0);history=[dict(epoch=0,validation=baseline)];save_json(args.out/'history.json',history);checkpoint()
        atomic(args.out/'T_best_junction.pth',model.state_dict());save_json(args.out/'best.json',dict(epoch=0,validation=baseline,selection='InitialT25 retained until precision-guarded weak-positive recall improves'))
    deadline=int(os.environ.get('SLURM_JOB_END_TIME','0') or 0)
    while epoch<=args.epochs:
        if deadline and time.time()+240>deadline:checkpoint();save_json(args.out/'status.json',dict(status='allocation_time_low'));return
        order=np.random.default_rng(20261123+epoch).permutation(train).tolist();model.train();total=0.;seen=0
        if phase=='train':
            optimizer.zero_grad(set_to_none=True);pending=0
            ds=FocusDataset(args.cache,args.out/'training_packs',order[cursor:],epoch)
            loader=DataLoader(ds,batch_size=None,num_workers=2,multiprocessing_context='spawn',pin_memory=True,prefetch_factor=2)
            for cpu in loader:
                b=to_device(cpu,'cuda');group=min(4,len(train)-cursor);pending+=1
                with torch.autocast('cuda'):
                    a,c=model(b);weight=b['loss_weight'].float();target=b['ty'];losses=(torch.nn.functional.binary_cross_entropy_with_logits(a.float(),target,reduction='none')+torch.nn.functional.binary_cross_entropy_with_logits(c.float(),target,reduction='none'))/2
                    loss=(losses*weight).sum()/weight.sum()
                    normal=(b['category']==0)|(b['category']==4)
                    if normal.any():
                        with torch.no_grad():ta,tc=teacher(b);soft=(ta.sigmoid()+tc.sigmoid()).float()/2
                        preservation=(torch.nn.functional.binary_cross_entropy_with_logits(a[normal].float(),soft[normal])+torch.nn.functional.binary_cross_entropy_with_logits(c[normal].float(),soft[normal]))/2
                        loss=loss+.1*preservation
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite focused loss')
                scaler.scale(loss/group).backward();total+=float(loss.detach());seen+=1
                if pending==group:
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);cursor+=pending;pending=0;updates+=1
                    if updates%25==0:checkpoint();print(json.dumps(dict(epoch=epoch,events=cursor,total=len(train),updates=updates,loss=total/seen)),flush=True)
                    save_json(args.out/'progress.json',dict(phase='training',epoch=epoch,events=cursor,total=len(train),updates=updates,loss=total/seen,unix=time.time()))
                    if deadline and time.time()+240>deadline:checkpoint();save_json(args.out/'status.json',dict(status='allocation_time_low'));return
            assert cursor==len(train) and pending==0;phase='validation';checkpoint()
        check=validate(model,args,valid,epoch);history.append(dict(epoch=epoch,validation=check,updates=updates));save_json(args.out/'history.json',history)
        atomic(args.out/('T_epoch_%03d.pth'%epoch),model.state_dict())
        feasible=[r for r in history if r['validation']['all05']['precision']>=history[0]['validation']['all05']['precision']-.003]
        best=max(feasible,key=lambda r:r['validation']['weak05']['recall'])
        if best['epoch']==epoch:atomic(args.out/'T_best_junction.pth',model.state_dict());save_json(args.out/'best.json',dict(epoch=epoch,validation=check,selection='weak-positive recall@0.5 with global precision drop<=0.003; classifier selection only, member validation required'))
        print(json.dumps(history[-1]),flush=True);finished=epoch;scheduler.step();epoch+=1;cursor=0;phase='train';checkpoint()
        if finished>=10 and best['epoch']<=finished-8:
            save_json(args.out/'status.json',dict(status='early_stopped',completed_epoch=finished,reason='No precision-guarded weak-positive recall gain for8 epochs'));return
    save_json(args.out/'status.json',dict(status='completed',completed_epoch=args.epochs))

if __name__=='__main__':main()
