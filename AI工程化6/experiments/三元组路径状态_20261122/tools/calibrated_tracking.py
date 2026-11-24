"""Frozen development score alignment, then paired reserved confirmation."""
from pathlib import Path
import sys,os,json,time,hashlib,argparse
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
import bootstrap
from bootstrap import BASE
os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from common import save_json,sha
from et_data import Features,to_device,atomic_npz,prepare
from et_models import THead,predict
OUT=ROOT/'calibrated_tracking_20261122';REVIEW=ROOT/'training_review_20261122';DATA=BASE.parent.parent/'joint_event_dataset'

def logit(x):
    x=np.clip(x,1e-6,1-1e-6);return np.log(x/(1-x))

def make_alignment():
    hist=np.load(REVIEW/'pr_histograms.npz');result={}
    for name in ['AA','other']:
        a=hist['T25_axial'] if name=='AA' else hist['T25_all']-hist['T25_axial']
        b=hist['focused10_axial'] if name=='AA' else hist['focused10_all']-hist['focused10_axial']
        a=np.cumsum(a[:,::-1],1)[:,::-1];b=np.cumsum(b[:,::-1],1)[:,::-1];precision=b[0]/np.maximum(b.sum(0),1);anchors=[]
        for target in [.2,.5,.9]:
            k=round(target*10000);p=a[0,k]/max(a[:,k].sum(),1);valid=np.flatnonzero((precision>=p)&(b[0]>0));assert len(valid)
            chosen=int(valid[np.argmax(b[0,valid])]);anchors.append(dict(old_score=target,new_score=chosen/10000,teacher_precision=float(p)))
        assert all(anchors[j]['new_score']<anchors[j+1]['new_score'] for j in range(2)),anchors
        result[name]=anchors
    return dict(method='Piecewise linear logit score transport at teacher precision targets; not posterior probability calibration',groups=result,source='Development200 only',source_hist_sha256=sha(REVIEW/'pr_histograms.npz'))

def align(q,t,stereo,cal):
    aa=~stereo[t[:,:3]].any(1);result=np.zeros_like(q)
    for name,mask in [('AA',aa),('other',~aa)]:
        anchor=cal['groups'][name];x=logit(np.array([a['new_score'] for a in anchor]));y=logit(np.array([a['old_score'] for a in anchor]));z=logit(q[mask]);v=np.interp(z,x,y)
        for left,right,condition in [(0,1,z<x[0]),(-2,-1,z>x[-1])]:v[condition]=y[left]+(z[condition]-x[left])*(y[right]-y[left])/(x[right]-x[left])
        result[mask]=1/(1+np.exp(-np.clip(v,-30,30)))
    return result

def event(task):
    eid,cache,folder=task;cache=Path(cache);folder=Path(folder);torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    limiter=threadpool_limits(limits=1)
    from et_decode import T_groups,geometry_extract,seeds
    from path_decode import decode
    from STEP5_两级寻迹 import axial_two_pass,attach_stereo,score_tracks,DEFAULT_CONFIG
    import coarse_3d_membership as coarse
    with np.load(cache/'features'/('event_%05d.npz'%eid)) as s:f={k:s[k] for k in s.files}
    with np.load(cache/'indices'/('event_%05d.npz'%eid)) as s:t=s['t']
    with np.load(DATA/'events'/('event_%05d.npz'%eid)) as s:d=s['d'];truth=s['truth']
    geo=np.load(DATA/'wire_geometry.npy')[f['cells']];pred=np.load(folder/'predictions'/('event_%05d.npz'%eid));started=time.perf_counter();base,_=axial_two_pass(geo,d,f['pairs'],f['p']);base_time=time.perf_counter()-started
    methods=['baseline','T25_old','T10_cal_old','T10_cal_path'];rows={};diagnostics={}
    if folder.name=='development':methods.insert(2,'T25_path')
    for key in methods:
        started=time.perf_counter();diag={};q=pred['old'] if key.startswith('T25') else pred['calibrated']
        if key=='baseline':tracks=base
        elif key.endswith('old'):groups,w=T_groups(f,t,q);tracks=geometry_extract(groups,w,f,geo,d,base)
        else:
            saved=coarse.SEARCH.copy()
            try:
                # Candidate-ranking evidence only; final coarse evaluation remains512.
                coarse.SEARCH.update(pair_seeds=64,retained_seeds=3,max_pool=32)
                tracks,diag=decode(f,t,q,geo,d,base)
            finally:coarse.SEARCH.clear();coarse.SEARCH.update(saved)
        joint,_=attach_stereo(tracks,geo,d,f['cells'],f['pairs'],f['p'],DEFAULT_CONFIG);counts,found=score_tracks(joint,truth,f['stereo'],DEFAULT_CONFIG)
        final,_=coarse.select(joint,coarse.propose(joint,geo,d,f['cells'],f['pairs'],f['p'],refine=False),geo,f['cells'],'guarded_loose')
        rows[key]=dict(counts=counts,found=found,seeds=seeds(final,truth,f['stereo'],f['cells']),seconds=base_time+time.perf_counter()-started);diagnostics[key]=diag
    save_json(folder/'events'/('event_%05d.json'%eid),dict(event_id=eid,methods=rows,diagnostics=diagnostics));return eid

def summarize(folder,ids):
    from STEP5_两级寻迹 import summarize as tracks
    from STEP2_训练双路线 import seed_summary
    packets=[json.loads((folder/'events'/('event_%05d.json'%eid)).read_text(encoding='utf-8')) for eid in ids]
    result=dict(events=len(ids),scope='Locked reserved confirmation, historically used original validation' if folder.name=='confirmation' else 'Development-only calibration/decoder check',methods={},paired={})
    for key in packets[0]['methods']:
        rows=[r['methods'][key] for r in packets];result['methods'][key]=dict(tracks=tracks([r['counts'] for r in rows]),seeds=seed_summary([r['seeds'] for r in rows]),mean_seconds=float(np.mean([r['seconds'] for r in rows])))
        if key=='T25_old':continue
        gain=[];lost=[]
        for r in packets:
            a=set(r['methods']['T25_old']['found']);b=set(r['methods'][key]['found']);gain.extend([[r['event_id'],int(t)] for t in b-a]);lost.extend([[r['event_id'],int(t)] for t in a-b])
        result['paired'][key]=dict(gained=gain,lost=lost,net=len(gain)-len(lost))
    save_json(folder/'summary.json',result);print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','confirmation'],required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args();torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    alignment=make_alignment();calpath=OUT/'alignment.json'
    if calpath.exists():assert json.loads(calpath.read_text(encoding='utf-8'))==alignment
    else:save_json(calpath,alignment)
    contract=json.loads((REVIEW/'download/training/contract.json').read_text(encoding='utf-8'));dev=contract['validation_ids'][:16]
    confirm=json.loads((ROOT/'next_member_manifest.json').read_text(encoding='utf-8'))['ids'];assert len(confirm)==200 and not set(confirm)&set(contract['validation_ids'])
    ids=dev if args.stage=='development' else confirm;folder=OUT/args.stage
    for sub in ['events','predictions']:(folder/sub).mkdir(parents=True,exist_ok=True)
    old=REVIEW/'download/initial_T25.pth';new=REVIEW/'download/training/T_epoch_010.pth'
    protocol=dict(ids=ids,initial_sha256=sha(old),focused_sha256=sha(new),alignment_sha256=sha(calpath),decoder_sha256=sha(ROOT/'path_decode.py'),runner_sha256=sha(Path(__file__)),preselection_budget=dict(pair_seeds=64,retained_seeds=3,max_pool=32),final_coarse_pair_seeds=512,no_training=True)
    if (folder/'contract.json').exists():assert json.loads((folder/'contract.json').read_text(encoding='utf-8'))==protocol
    else:save_json(folder/'contract.json',protocol)
    cache=REVIEW/'download/cache' if args.stage=='development' else folder/'cache'
    if args.stage=='confirmation':
        assert (OUT/'development/summary.json').exists()
        weight=ROOT.parent.parent.parent/'AI工程化6_稳定基线备份/AI6_稳定基线_20261107_230240/AI工程化6/joint_training/local_evaluation_epoch126_20261107/best_interaction_network_joint_no_time.pth'
        prepare(DATA,weight,cache,ids)
    models=[]
    for weight in [old,new]:
        model=THead().cuda();model.load_state_dict(torch.load(weight,map_location='cuda',weights_only=True));model.eval();models.append(model)
    pending=set();completed=sum((folder/'events'/('event_%05d.json'%eid)).exists() for eid in ids)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        def drain(all_done=False):
            nonlocal completed,pending
            if not pending:return
            done,pending=wait(pending,return_when='ALL_COMPLETED' if all_done else FIRST_COMPLETED)
            for f in done:f.result();completed+=1
            save_json(folder/'progress.json',dict(events=completed,total=len(ids),unix=time.time()));print(args.stage,completed,'/',len(ids),flush=True)
        for eid in ids:
            if (folder/'events'/('event_%05d.json'%eid)).exists():continue
            target=folder/'predictions'/('event_%05d.npz'%eid)
            if not target.exists():
                b=Features(cache,[eid])[0];gpu=to_device(b,'cuda');scores=[]
                for m in models:
                    with torch.no_grad(),torch.autocast('cuda'):scores.append(predict(m,gpu,'T'))
                atomic_npz(target,compress=True,old=scores[0],raw=scores[1],calibrated=align(scores[1],b['t'].numpy(),b['stereo'].numpy(),alignment))
            pending.add(pool.submit(event,(eid,str(cache),str(folder))))
            if len(pending)>=args.workers*2:drain()
        drain(True)
    summarize(folder,ids)

if __name__=='__main__':main()
