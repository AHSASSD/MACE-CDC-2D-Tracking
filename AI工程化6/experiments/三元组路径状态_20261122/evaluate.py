import bootstrap
from bootstrap import HERE,BASE
import os,json,time,argparse
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor,as_completed
from common import save_json
from et_data import Features,to_device,atomic_npz
from et_models import THead,predict
from path_decode import decode,Config

DATA=Path(os.environ.get('AI6_DATA',str(BASE.parent.parent/'joint_event_dataset')))

def event(task):
    eid,cache,pred,out=task;torch.set_num_threads(1);cache=Path(cache);out=Path(out)
    from threadpoolctl import threadpool_limits
    blas_controller=threadpool_limits(limits=1)
    from et_decode import T_groups,geometry_extract,seeds
    from STEP5_两级寻迹 import axial_two_pass,attach_stereo,score_tracks,DEFAULT_CONFIG
    from coarse_3d_membership import propose,select
    with np.load(cache/'features'/('event_%05d.npz'%eid)) as s:f={k:s[k] for k in s.files}
    with np.load(cache/'indices'/('event_%05d.npz'%eid)) as s:t=s['t']
    with np.load(DATA/'events'/('event_%05d.npz'%eid)) as s:d=s['d'];truth=s['truth']
    geo=np.load(DATA/'wire_geometry.npy')[f['cells']];q=np.load(pred)['T'];started=time.perf_counter()
    baseline,_=axial_two_pass(geo,d,f['pairs'],f['p']);base_time=time.perf_counter()-started
    result={};diag={}
    for key in ['raw_baseline','T_old','path_state']:
        started=time.perf_counter()
        if key=='raw_baseline':tracks=baseline
        elif key=='T_old':groups,w=T_groups(f,t,q);tracks=geometry_extract(groups,w,f,geo,d,baseline)
        else:tracks,diag=decode(f,t,q,geo,d,baseline)
        joint,_=attach_stereo(tracks,geo,d,f['cells'],f['pairs'],f['p'],DEFAULT_CONFIG)
        counts,found=score_tracks(joint,truth,f['stereo'],DEFAULT_CONFIG)
        confirmed,_=select(joint,propose(joint,geo,d,f['cells'],f['pairs'],f['p'],refine=False),geo,f['cells'],'guarded_loose')
        result[key]=dict(counts=counts,found=found,seeds=seeds(confirmed,truth,f['stereo'],f['cells']),seconds=base_time+time.perf_counter()-started,
            candidates=[dict(indices=c['indices'],origin=c.get('origin'),rms=c['rms']) for c in tracks])
    save_json(out/('event_%05d.json'%eid),dict(event_id=eid,methods=result,path_diagnostics=diag));return eid

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--workers',type=int,default=6);p.add_argument('--out',type=Path,default=HERE/'development_T25');args=p.parse_args()
    torch.set_num_threads(1);source=json.loads((BASE/'evaluation_download/full/contract.json').read_text(encoding='utf-8'))
    ids=source['validation_ids'][:16]+[1500,1501,1503,1504,4309,5477,9555]
    if args.limit:ids=ids[:args.limit]
    out=args.out;out.mkdir(exist_ok=True)
    model=THead().cuda();model.load_state_dict(torch.load(BASE/'evaluation_download/full/T_best_members.pth',map_location='cuda',weights_only=True));model.eval();tasks=[]
    for eid in ids:
        if (out/('event_%05d.json'%eid)).exists():continue
        cache=BASE/('junction_audit_20261122/cache' if eid in [1500,1501,1503,1504] else 'evaluation_download/cache')
        cpu=Features(cache,[eid])[0]
        with torch.no_grad(),torch.autocast('cuda'):q=predict(model,to_device(cpu,'cuda'),'T')
        prediction=out/('prediction_%05d.npz'%eid);atomic_npz(prediction,compress=True,T=q)
        tasks.append((eid,str(cache),str(prediction),str(out)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done in as_completed([pool.submit(event,t) for t in tasks]):print('evaluated',done.result(),flush=True)
    from STEP5_两级寻迹 import summarize
    from STEP2_训练双路线 import seed_summary
    packets=[json.loads((out/('event_%05d.json'%eid)).read_text(encoding='utf-8')) for eid in ids]
    result=dict(events=len(ids),scope='Development16 plus4 representative events and3 selected failure events; not independent confirmation; same frozenT25 for decoder comparison',config=Config().__dict__,methods={k:dict(tracks=summarize([r['methods'][k]['counts'] for r in packets]),seeds=seed_summary([r['methods'][k]['seeds'] for r in packets]),mean_seconds=float(np.mean([r['methods'][k]['seconds'] for r in packets]))) for k in packets[0]['methods']})
    save_json(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
