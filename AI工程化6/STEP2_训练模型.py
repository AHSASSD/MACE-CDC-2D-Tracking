"""AI6 random initialization, focal loss, effective batch 32, complete epoch resume."""
import argparse
import json
import random
import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from common import *
from server_runtime import dataset as runtime_dataset,loader as runtime_loader,DevicePrefetch

def atomic_torch(path,obj):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)

def graph_balanced_loss(logits,data):
    # Preserve the former batch=1 objective: each event has equal weight,
    # even when candidate edge counts differ or the last batch is smaller.
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),data.y,reduction='none')
    values=.5*(1-torch.exp(-bce)).square()*bce
    group=data.batch[data.edge_index[0]]
    sums=values.new_zeros(data.num_graphs).index_add(0,group,values)
    count=torch.bincount(group,minlength=data.num_graphs).clamp_min(1)
    return (sums/count).mean()

@torch.no_grad()
def evaluate(model,loader,device,amp):
    model.eval();loss_sum=edges=0;counts=np.zeros((3,3),np.int64)
    criterion=FocalLoss()
    for g in loader:
        g=g.to(device,non_blocking=True)
        with torch.autocast(device_type=device.type,enabled=amp): logits=model(g.x,g.edge_index,g.edge_attr)
        loss=criterion(logits,g.y);loss_sum+=loss.item()*len(g.y);edges+=len(g.y)
        pred=logits.sigmoid()>=.5;target=g.y.bool()
        for k in range(3):
            mask=g.edge_kind==k
            counts[k]+=[int((pred&target&mask).sum()),int((pred&~target&mask).sum()),int((~pred&target&mask).sum())]
    types={k:{'precision':int(tp)/max(int(tp+fp),1),'recall':int(tp)/max(int(tp+fn),1),'f1':2*int(tp)/max(int(2*tp+fp+fn),1)} for k,(tp,fp,fn) in zip(['AA','AS','SS'],counts)}
    return {'loss':loss_sum/max(edges,1),'edges':edges,'types':types,'macro_f1':sum(t['f1'] for t in types.values())/3}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,default=DATA_ROOT);p.add_argument('--run-tag',default='joint_no_time_scratch_v1');p.add_argument('--epochs',type=int,default=200);p.add_argument('--resume',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cuda');p.add_argument('--chunk-size',type=int,default=4096);args=p.parse_args()
    import os
    batch_size=int(os.getenv('MACE_BATCH_SIZE','1'));val_batch=int(os.getenv('MACE_VAL_BATCH_SIZE','1'))
    if batch_size not in [1,2,4,8,16,32] or val_batch<1:raise ValueError('Batch must divide effective batch32')
    accumulation=32//batch_size
    torch.set_num_threads(int(os.getenv('OMP_NUM_THREADS','2')));random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    device=torch.device(args.device);amp=device.type=='cuda'
    train=runtime_dataset(args.dataset_root,'train');val=runtime_dataset(args.dataset_root,'validation')
    if args.smoke:
        train=torch.utils.data.Subset(train,range(2));val=torch.utils.data.Subset(val,range(2))
    out=MODEL_ROOT/args.run_tag
    if out.exists() and not args.resume: raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=bool(args.resume))
    config={'profile':'AI6-joint-no-time-v1','hidden':256,'n_passes':8,'node_features':NODE_FEATURES,'edge_features':EDGE_FEATURES,'batch_size':1,'accumulation_steps':32,'amp':amp,'seed':SEED,'initial_lr':.001,'loss':'FocalLoss alpha=.5 gamma=2','weight_decay':.0001,'scheduler':'ReduceLROnPlateau val loss patience=10 factor=.5 min_lr=1e-6','dataset_metadata_sha256':sha(args.dataset_root/'dataset_metadata_joint_no_time.json'),'train_events':len(train),'validation_events':len(val),'smoke':args.smoke,'chunk_size':args.chunk_size}
    config['batch_size']=batch_size;config['accumulation_steps']=accumulation
    model=JointInteractionNetwork(chunk_size=args.chunk_size,recompute_policy=os.getenv('MACE_RECOMPUTE','all')).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,factor=.5,min_lr=1e-6)
    scaler=torch.amp.GradScaler(device.type,enabled=amp);generator=torch.Generator().manual_seed(SEED)
    criterion=FocalLoss();start_epoch=1;best=float('inf');history=[]
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False)
        if state['config']!=config: raise ValueError('Resume configuration mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        start_epoch=state['epoch']+1;best=state['best_validation_loss'];history=state['history']
        torch.set_rng_state(state['torch_rng'].cpu());generator.set_state(state['loader_rng'].cpu());random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        if amp:torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
    save_json(out/'model_metadata_joint_no_time.json',config)
    train_loader=runtime_loader(train,batch_size,shuffle=True,generator=generator)
    val_loader=runtime_loader(val,val_batch)
    save_json(out/'server_runtime.json',{'cache':os.getenv('MACE_GRAPH_CACHE'),'workers':int(os.getenv('MACE_WORKERS','0')),'prefetch_factor':2,'pin_memory':os.getenv('MACE_PIN_MEMORY','0'),'async_transfer':os.getenv('MACE_ASYNC_TRANSFER','0'),'recompute_policy':model.recompute_policy,'slurm_job':os.getenv('SLURM_JOB_ID'),'slurm_step':os.getenv('SLURM_STEP_ID')})
    if not args.resume:
        atomic_torch(out/'training_state_joint_no_time.pth',{'config':config,'epoch':0,'best_validation_loss':best,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'history':history,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if amp else [],'loader_rng':generator.get_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()})
    print('START',json.dumps(config,ensure_ascii=False),flush=True)
    try:
        for epoch in range(start_epoch,args.epochs+1):
            model.train();optimizer.zero_grad(set_to_none=True);total=edges=0;begin=time.perf_counter()
            for index,g in enumerate(DevicePrefetch(train_loader,device)):
                event_start=index*batch_size;group_start=(event_start//32)*32;group_size=min(32,len(train)-group_start)
                with torch.autocast(device_type=device.type,enabled=amp): logits=model(g.x,g.edge_index,g.edge_attr)
                loss=graph_balanced_loss(logits,g)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                scaler.scale(loss*g.num_graphs/group_size).backward()
                if (index+1)%accumulation==0 or index+1==len(train_loader):
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                total+=loss.item()*len(g.y);edges+=len(g.y)
                if index==0 or (index+1)%max(1,100//batch_size)==0:
                    event=min((index+1)*batch_size,len(train))
                    save_json(out/'training_progress.json',{'status':'training','epoch':epoch,'target_epochs':args.epochs,'event':event,'train_events':len(train),'batch_size':batch_size,'accumulation_steps':accumulation,'train_loss':total/max(edges,1),'elapsed_seconds':time.perf_counter()-begin,'updated_unix':time.time()})
                    print(f'epoch={epoch}/{args.epochs} event={event}/{len(train)} loss={total/max(edges,1):.6f}',flush=True)
                del logits,loss,g
            validation=evaluate(model,DevicePrefetch(val_loader,device),device,amp);scheduler.step(validation['loss'])
            row={'epoch':epoch,'train_loss':total/max(edges,1),'validation':validation,'lr':optimizer.param_groups[0]['lr'],'seconds':time.perf_counter()-begin};history.append(row)
            improved=validation['loss']<best
            if improved: best=validation['loss'];atomic_torch(out/'best_interaction_network_joint_no_time.pth',model.state_dict());save_json(out/'best_model_selection.json',{'epoch':epoch,'validation':validation,'threshold':.5,'test_used':False})
            state={'config':config,'epoch':epoch,'best_validation_loss':best,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'history':history,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if amp else [],'loader_rng':generator.get_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
            atomic_torch(out/'training_state_joint_no_time.pth',state);save_json(out/'training_history.json',history);print(json.dumps(row),flush=True)
        save_json(out/'training_progress.json',{'status':'complete','epoch':args.epochs,'updated_unix':time.time()})
    except BaseException as exc:
        save_json(out/'training_error.json',{'type':type(exc).__name__,'message':str(exc),'updated_unix':time.time()});raise

if __name__=='__main__':main()
