"""Alternating paired epochs inside one GPU allocation, independent full checkpoints."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from common import JointInteractionNetwork,sha,save_json
from server_runtime import dataset,loader,DevicePrefetch
from paired_loss import objective


def atomic(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),python=random.getstate(),numpy=np.random.get_state())


@torch.no_grad()
def validate(model,val,device):
    model.eval();loss=0.;edges=0;stats=np.zeros((3,3),dtype=np.int64)
    for g in DevicePrefetch(loader(val,8),device):
        with torch.autocast('cuda'):z=model(g.x,g.edge_index,g.edge_attr)
        bce=torch.nn.functional.binary_cross_entropy_with_logits(z.float(),g.y,reduction='none')
        f=.5*(1-torch.exp(-bce)).square()*bce;loss+=float(f.sum());edges+=len(f)
        p=z.sigmoid()>=.5;y=g.y.bool()
        for k in range(3):
            m=g.edge_kind==k;stats[k]+=[int((p&y&m).sum()),int((p&~y&m).sum()),int((~p&y&m).sum())]
    return dict(loss=loss/max(edges,1),edges=edges,types={name:dict(tp=int(a),fp=int(b),fn=int(c),recall=int(a)/max(int(a+c),1),precision=int(a)/max(int(a+b),1),f1=2*int(a)/max(int(2*a+b+c),1)) for name,(a,b,c) in zip(['AA','AS','SS'],stats)})


@torch.no_grad()
def member_validation(model,data,device):
    from STEP5_两级寻迹 import undirected_predictions,axial_two_pass,attach_stereo,score_tracks,summarize,DEFAULT_CONFIG
    geometry=np.load(Path(os.environ['PAIR_DATA_ROOT'])/'wire_geometry.npy');model.eval();rows=[]
    for g in DevicePrefetch(loader(data,1,workers=0),device):
        with torch.autocast('cuda'):p=model(g.x,g.edge_index,g.edge_attr).sigmoid().float().cpu().numpy()
        pairs,p=undirected_predictions(g.edge_index.cpu().numpy(),p)
        cells=g.cell_id.cpu().numpy()
        eid=int(g.event_id.item())
        with np.load(Path(os.environ['PAIR_DATA_ROOT'])/'events'/('event_%05d.npz'%eid)) as raw:
            assert np.array_equal(cells,raw['cell_id'])
            drift=raw['d'].copy()
        geo=geometry[cells];stereo=g.stereo.cpu().numpy();truth=g.truth.cpu().numpy()
        a,_=axial_two_pass(geo,drift,pairs,p);joint,_=attach_stereo(a,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
        counts,_=score_tracks(joint,truth,stereo,DEFAULT_CONFIG);rows.append(counts)
    return summarize(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--weight',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--member-ids',type=Path,required=True);parser.add_argument('--epochs',type=int,default=18);args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True);lock=(out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    device=torch.device('cuda');torch.set_num_threads(1);os.environ['PAIR_DATA_ROOT']=str(args.data)
    train=dataset(args.data,'train');val=dataset(args.data,'validation');ids=json.loads(args.member_ids.read_text())
    index={eid:i for i,eid in enumerate(val.ids)};member_data=torch.utils.data.Subset(val,[index[eid] for eid in ids])
    config=dict(initial_weight_sha256=sha(args.weight),dataset_sha256=sha(args.data/'dataset_metadata_joint_no_time.json'),
                epochs=args.epochs,batch=4,effective_batch=32,lr=1e-4,weight_decay=1e-4,
                arms=['control','balanced'],positive_aux_coefficient=.1,negative_aux_coefficient=.05,
                hard_negative_fraction=.1,hidden=256,n_passes=8,member_event_ids=ids,
                optimizer_policy='Both arms start fresh AdamW from identical epoch126 weights; not a full optimizer resume',
                ordering='control epoch1, balanced epoch1, control epoch2, balanced epoch2; seed identical within each pair',
                auto_cancel_allocation=False,source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=config:raise ValueError('Paired contract changed')
    save_json(out/'contract.json',config);deadline=int(os.environ.get('SLURM_JOB_END_TIME','0') or 0);durations=[]
    completed={arm:0 for arm in config['arms']}
    for arm in completed:
        state_path=out/arm/'state.pth'
        if state_path.exists():completed[arm]=int(torch.load(state_path,map_location='cpu',weights_only=False)['epoch'])
    for epoch in range(1,args.epochs+1):
        if all(completed[a]>=epoch for a in completed):continue
        remaining_arms=sum(completed[a]<epoch for a in completed)
        estimated=remaining_arms*(sum(durations[-4:])/max(len(durations[-4:]),1) if durations else 520)+600
        if deadline and time.time()+estimated>deadline:
            save_json(out/'status.json',dict(status='stopped_before_incomplete_pair',completed=completed,unix=time.time()));return
        for arm in config['arms']:
            if completed[arm]>=epoch:continue
            root=out/arm;root.mkdir(exist_ok=True);started=time.time();seed=20261109+epoch
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=JointInteractionNetwork(chunk_size=131072,recompute_policy='adaptive').to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            scaler=torch.amp.GradScaler('cuda',init_scale=1024.);history=[];updates=0;skips=0
            if (root/'state.pth').exists():
                state=torch.load(root/'state.pth',map_location=device,weights_only=False)
                if state['epoch']!=epoch-1:raise ValueError('Unexpected arm epoch')
                model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);history=state['history'];updates=state['updates'];skips=state['skips']
            else:model.load_state_dict(torch.load(args.weight,map_location=device,weights_only=True))
            # Module construction consumes RNG; reset after loading for equal dropout streams.
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            permutation=np.random.default_rng(seed).permutation(len(train)).tolist()
            source=loader(torch.utils.data.Subset(train,permutation),4,generator=torch.Generator().manual_seed(seed))
            model.train();optimizer.zero_grad(set_to_none=True);total=0.;events=0
            for step,g in enumerate(DevicePrefetch(source,device)):
                event_start=step*4;group_start=(event_start//32)*32;group_size=min(32,len(train)-group_start)
                with torch.autocast('cuda'):z=model(g.x,g.edge_index,g.edge_attr)
                loss,terms=objective(z,g,arm)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
                scaler.scale(loss*g.num_graphs/group_size).backward();total+=float(loss)*g.num_graphs;events+=g.num_graphs
                if (step+1)%8==0 or step+1==len(source):
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()<previous:skips+=1
                    else:updates+=1
                    optimizer.zero_grad(set_to_none=True)
                if step%25==0:
                    progress=dict(status='training',arm=arm,epoch=epoch,event=events,total=len(train),terms=terms,completed=completed,unix=time.time())
                    save_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
                del g,z,loss
            del source
            save_json(out/'progress.json',dict(status='validation',arm=arm,epoch=epoch,completed=completed,unix=time.time()))
            validation=validate(model,val,device)
            membership=member_validation(model,member_data,device) if epoch%3==0 or epoch==args.epochs else None
            elapsed=time.time()-started;durations.append(elapsed)
            row=dict(epoch=epoch,objective=total/max(events,1),validation=validation,membership=membership,seconds=elapsed,updates=updates,amp_skips=skips)
            history.append(row)
            state=dict(epoch=epoch,arm=arm,model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),history=history,updates=updates,skips=skips,rng=rng_state())
            atomic(root/'state.pth',state);atomic(root/('epoch_%03d.pth'%epoch),model.state_dict());save_json(root/'history.json',history)
            if validation['loss']<=min(r['validation']['loss'] for r in history):atomic(root/'best_validation_loss.pth',model.state_dict())
            completed[arm]=epoch;save_json(out/'status.json',dict(status='running',completed=completed,paired_epochs=min(completed.values()),unix=time.time()))
            print(json.dumps(dict(arm=arm,**row)),flush=True)
            del model,optimizer,scaler,state;torch.cuda.empty_cache()
    save_json(out/'status.json',dict(status='complete',completed=completed,paired_epochs=min(completed.values()),unix=time.time()))


if __name__=='__main__':main()
