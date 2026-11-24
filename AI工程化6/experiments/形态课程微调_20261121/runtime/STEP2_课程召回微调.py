"""30-epoch joint GNN fine-tuning, resumable at optimizer boundaries and validation events."""
import argparse,fcntl,json,os,time,random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from common import JointInteractionNetwork,sha,save_json
from server_runtime import dataset,loader,DevicePrefetch
from recall_loss import objective
from curriculum import epoch_ids
from membership_cpu import reconstruct
from STEP5_两级寻迹 import axial_two_pass,attach_stereo,undirected_predictions,score_tracks,summarize,DEFAULT_CONFIG
from coarse_3d_membership import propose,select

ROOT=Path(__file__).resolve().parent
DEVICE=torch.device('cuda')


def atomic(path,value):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)


def aggregate_seeds(rows):
    c=Counter()
    for row in rows:c.update(row)
    return dict(c,efficiency=c['matched_tracks']/max(c['truth_tracks'],1),fake_fraction=c['fake_candidates']/max(c['output_candidates'],1),clone_fraction=c['clone_candidates']/max(c['output_candidates'],1),f1=2*c['matched_tracks']/max(c['truth_tracks']+c['output_candidates'],1))


@torch.no_grad()
def validate(model,val,out,epoch,micro):
    model.eval();total=0.;edges=0;stats=np.zeros((3,3),dtype=np.int64)
    for j,g in enumerate(DevicePrefetch(loader(val,micro),DEVICE)):
        with torch.autocast('cuda'):z=model(g.x,g.edge_index,g.edge_attr)
        b=torch.nn.functional.binary_cross_entropy_with_logits(z.float(),g.y,reduction='none');total+=float((.5*(1-torch.exp(-b)).square()*b).sum());edges+=len(b)
        p=z.sigmoid()>=.5;y=g.y.bool()
        for k in range(3):
            mask=g.edge_kind==k;stats[k]+=[int((p&y&mask).sum()),int((p&~y&mask).sum()),int((~p&y&mask).sum())]
        if j%25==0:save_json(out/'progress.json',dict(phase='edge_validation',epoch=epoch,batch=j,unix=time.time()))
    return dict(loss=total/max(edges,1),edges=edges,types={name:dict(tp=int(a),fp=int(b),fn=int(c),precision=int(a)/max(int(a+b),1),recall=int(a)/max(int(a+c),1),f1=2*int(a)/max(int(2*a+b+c),1)) for name,(a,b,c) in zip(['AA','AS','SS'],stats)})


@torch.no_grad()
def members(model,val,args,out,epoch,selection,defer=False):
    ids=json.loads((ROOT/'member_ids.json').read_text());lookup={eid:i for i,eid in enumerate(val.ids)}
    folder=out/'member_metrics'/('epoch_%03d'%epoch);folder.mkdir(parents=True,exist_ok=True);rows=[];tasks=[];model.eval()
    predictions=out/'member_predictions'/('epoch_%03d'%epoch);predictions.mkdir(parents=True,exist_ok=True)
    for j,eid in enumerate(ids):
        path=folder/('event_%05d.json'%eid)
        if path.exists():rows.append(json.loads(path.read_text()));continue
        prediction=predictions/('event_%05d.npz'%eid)
        if not prediction.exists():
            g=val[lookup[eid]].to(DEVICE)
            with torch.autocast('cuda'):p=model(g.x,g.edge_index,g.edge_attr).sigmoid().float().cpu().numpy()
            pairs,p=undirected_predictions(g.edge_index.cpu().numpy(),p)
            temporary=prediction.with_suffix('.tmp')
            with temporary.open('wb') as stream:np.savez(stream,pairs=pairs,probabilities=p)
            temporary.replace(prediction)
        tasks.append((eid,str(args.data),str(prediction),str(path),selection['hit_cutoff']))
        save_json(out/'progress.json',dict(phase='member_scoring',epoch=epoch,events=j+1,total=len(ids),unix=time.time()))
    pool=None;futures=[]
    if tasks:
        workers=2 if defer else 3
        pool=ProcessPoolExecutor(max_workers=min(workers,int(os.environ.get('SLURM_CPUS_PER_TASK','3'))),mp_context=multiprocessing.get_context('spawn'))
        futures=[pool.submit(reconstruct,task) for task in tasks]
    def finish():
        try:
            for future in as_completed(futures):
                rows.append(future.result())
                save_json(out/'progress.json',dict(phase='member_reconstruction',epoch=epoch,events=len(rows),total=len(ids),unix=time.time()))
        finally:
            if pool:pool.shutdown(wait=True)
        return {name:dict(events=len(sub),tracks=summarize([r['counts'] for r in sub]),seed=aggregate_seeds([r['seed'] for r in sub])) for name in ['all','ordinary','high_hits','blob','loop'] if (sub:=[r for r in rows if name in r['strata']])}
    return finish if defer else finish()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--weight',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True);lock=(out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(1)
    selection=json.loads((ROOT/'selection.json').read_text());assert selection['dataset_sha256']==sha(args.data/'dataset_metadata_joint_no_time.json')
    train=dataset(args.data,'train');val=dataset(args.data,'validation');lookup={eid:i for i,eid in enumerate(train.ids)}
    runtime=json.loads((args.out.parent/'runtime_config.json').read_text())
    micro=runtime['micro_batch']
    assert set(selection['clean_event_ids'])|set(selection['hard_event_ids'])|set(selection['invalid_event_ids'])==set(train.ids)
    assert sha(args.weight)=='5f3f02409399cd62efce5f254be02fa004237fb61a5b4717681cad1ad90cdcac'
    contract=dict(initial_weight_sha256=sha(args.weight),selection_sha256=sha(ROOT/'selection.json'),source_hashes={p.name:sha(p) for p in ROOT.glob('*.py')},epochs=30,min_epochs=20,lr=5e-5,effective_batch=32,micro_batch=micro,gpu=torch.cuda.get_device_name(0),model='joint A/S hidden256 passes8',pipeline='Original two-pass + AS/SS attachment + guarded_loose coarse3D, no legacy learned candidate calibrator',loss='Event focal + track x edge-type true p<0.7 margin (lambda .02 to .08) + type-balanced hard false p>0.3 margin .02',curriculum='epochs1-5 ordinary +25% hard;6-15 linearly return hard;16-30 original valid train',early_stop='after20, 3 consecutive member checks without coarse seed F1 improvement',optimizer='Fresh AdamW, cosine LR 5e-5 to1e-5',auto_deploy=False)
    if (out/'contract.json').exists():assert json.loads((out/'contract.json').read_text())==contract
    save_json(out/'contract.json',contract)
    model=JointInteractionNetwork(chunk_size=131072 if micro==4 else 65536,recompute_policy='adaptive').to(DEVICE);optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30,eta_min=1e-5);scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    epoch=1;cursor=0;phase='train';history=[];groups=0;updates=0;skips=0
    if (out/'state.pth').exists():
        state=torch.load(out/'state.pth',map_location=DEVICE,weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        epoch=state['epoch'];cursor=state['cursor'];phase=state['phase'];history=state['history'];groups=state['groups'];updates=state['updates'];skips=state['skips']
        torch.set_rng_state(state['rng_cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['rng_cuda']])
    else:model.load_state_dict(torch.load(args.weight,map_location=DEVICE,weights_only=True))
    def checkpoint():
        atomic(out/'state.pth',dict(epoch=epoch,cursor=cursor,phase=phase,history=history,groups=groups,updates=updates,skips=skips,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()))
        save_json(out/'checkpoint_meta.json',dict(epoch=epoch,cursor=cursor,phase=phase,groups=groups,updates=updates,unix=time.time()))
    baseline_task=None;initial=None
    if not (out/'initial_members.json').exists():
        # All initial probabilities are frozen before the first optimizer update.
        # On recovery after updates, missing initial predictions must never use new weights.
        if updates:
            assert all((out/'member_predictions/epoch_000'/('event_%05d.npz'%eid)).exists() or (out/'member_metrics/epoch_000'/('event_%05d.json'%eid)).exists() for eid in json.loads((ROOT/'member_ids.json').read_text()))
        checkpoint();print('Freeze initial predictions; CPU membership overlaps training',flush=True);baseline_task=members(model,val,args,out,0,selection,defer=True)
    else:initial=json.loads((out/'initial_members.json').read_text())
    deadline=int(os.environ.get('SLURM_JOB_END_TIME','0') or 0)
    while epoch<=30:
        if deadline and time.time()+180>deadline:checkpoint();save_json(out/'status.json',dict(status='allocation_time_low',epoch=epoch,completed_epoch=epoch-1,unix=time.time()));return
        ids=epoch_ids(selection,epoch);assert len(set(ids))==len(ids)
        if cursor==0 and phase=='train':torch.manual_seed(20261121+epoch);torch.cuda.manual_seed_all(20261121+epoch)
        if phase=='train':
            model.train();optimizer.zero_grad(set_to_none=True);source=loader(torch.utils.data.Subset(train,[lookup[e] for e in ids[cursor:]]),micro,generator=torch.Generator().manual_seed(20261121+epoch));pending=0
            for j,g in enumerate(DevicePrefetch(source,DEVICE)):
                remaining=len(ids)-cursor;group_size=min(32,remaining);pending+=g.num_graphs
                with torch.autocast('cuda'):z=model(g.x,g.edge_index,g.edge_attr)
                loss,terms=objective(z,g,epoch)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss*g.num_graphs/group_size).backward()
                if pending==group_size:
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    previous=scaler.get_scale();scaler.step(optimizer);scaler.update();skips+=int(scaler.get_scale()<previous);updates+=int(scaler.get_scale()>=previous);optimizer.zero_grad(set_to_none=True)
                    cursor+=pending;pending=0;groups+=1
                    if groups%25==0:checkpoint()
                    save_json(out/'progress.json',dict(phase='training',epoch=epoch,events=cursor,total=len(ids),terms=terms,lr=optimizer.param_groups[0]['lr'],updates=updates,unix=time.time()))
                    if groups%10==0:print(json.dumps(dict(epoch=epoch,events=cursor,total=len(ids),updates=updates,loss=float(loss),terms=terms)),flush=True)
            assert cursor==len(ids) and pending==0
            phase='validation';checkpoint()
        validation=validate(model,val,out,epoch,micro)
        if epoch%5==0 and initial is None:
            initial=baseline_task();baseline_task=None;save_json(out/'initial_members.json',initial)
        membership=members(model,val,args,out,epoch,selection) if epoch%5==0 or epoch==30 else None
        row=dict(epoch=epoch,training_events=len(ids),validation=validation,membership=membership,updates=updates,amp_skips=skips)
        history.append(row);atomic(out/('epoch_%03d.pth'%epoch),model.state_dict());save_json(out/'history.json',history)
        print(json.dumps(row),flush=True)
        if membership:
            checks=[r for r in history if r['membership']];best=max(checks,key=lambda r:r['membership']['all']['seed']['f1'])
            if best['epoch']==epoch:atomic(out/'best_member_f1.pth',model.state_dict())
            a=membership['all'];b=initial['all'];gate=a['tracks']['track_efficiency']>=b['tracks']['track_efficiency'] and a['tracks']['fake_fraction']<=b['tracks']['fake_fraction'] and a['tracks']['clone_fraction']<=b['tracks']['clone_fraction'] and a['seed']['efficiency']>=b['seed']['efficiency'] and a['seed']['fake_fraction']<=b['seed']['fake_fraction'] and a['seed']['clone_fraction']<=b['seed']['clone_fraction']
            save_json(out/'last_member_gate.json',dict(epoch=epoch,non_regression=gate,selected_for_deployment=False))
            if epoch>=20 and len(checks)>=4 and best['epoch']<=checks[-4]['epoch']:
                epoch+=1;cursor=0;phase='train';checkpoint();save_json(out/'status.json',dict(status='early_stopped',completed_epoch=epoch-1,reason='three member checks without F1 gain'));return
        scheduler.step();epoch+=1;cursor=0;phase='train';checkpoint();save_json(out/'status.json',dict(status='running',completed_epoch=epoch-1,unix=time.time()))
    save_json(out/'status.json',dict(status='completed',completed_epoch=30,unix=time.time()))


if __name__=='__main__':main()
