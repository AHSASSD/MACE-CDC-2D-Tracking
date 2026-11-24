"""Development-only set-packing of baseline + already proposed candidates."""
from pathlib import Path
import sys,os,json,time
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import torch
from common import save_json
from et_decode import T_groups,E_edges,geometry_extract,seeds
from STEP5_两级寻迹 import axial_two_pass,attach_stereo,score_tracks,DEFAULT_CONFIG,circle_wire_lower_bound
from two_stage_core import score
from coarse_3d_membership import propose,select
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csc_matrix
P=ROOT/'evaluation_download';DATA=ROOT.parent.parent/'joint_event_dataset';OUT=ROOT/'decoder_diagnosis_20261122/set_packing'

def choose(base,proposal,f,geo,d):
    # Uses measurements only. Keeps the original utility and changes conflict resolution.
    candidates={tuple(sorted(c['indices'])):dict(c) for c in [*base,*proposal]}
    candidates=list(candidates.values());ss=np.flatnonzero(f['stereo']);matrix=np.zeros((len(f['stereo']),)*2,np.float32)
    matrix[f['pairs'][0],f['pairs'][1]]=f['p'];matrix[f['pairs'][1],f['pairs'][0]]=f['p']
    weights=[]
    for c in candidates:
        ix=np.asarray(c['indices']);support=((matrix[np.ix_(ss,ix)]>=.7).sum(1)>=2)&(circle_wire_lower_bound(c['fit'],geo[ss])<=d[ss]+2.)
        weights.append(float(score(c)+.05*min(len(np.unique(f['cells'][ss[support]])),15)))
    if not candidates:return [],{}
    rr=[];cc=[]
    for j,c in enumerate(candidates):rr.extend(c['indices']);cc.extend([j]*len(c['indices']))
    A=csc_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(geo),len(candidates)))
    solution=milp(-np.asarray(weights),integrality=np.ones(len(candidates)),bounds=Bounds(0,1),constraints=LinearConstraint(A,0,1),options={'time_limit':2.,'mip_rel_gap':0.0001,'presolve':True})
    if solution.x is None:return proposal,dict(status=int(solution.status),fallback=True)
    chosen=[c for c,x in zip(candidates,solution.x) if x>.5]
    assert len({i for c in chosen for i in c['indices']})==sum(len(c['indices']) for c in chosen)
    return chosen,dict(status=int(solution.status),fallback=False,candidates=len(candidates),selected=len(chosen),utility=float(-solution.fun))

def one(eid):
    torch.set_num_threads(1)
    with np.load(P/'cache/features'/('event_%05d.npz'%eid)) as s:f={k:s[k] for k in s.files}
    with np.load(P/'cache/indices'/('event_%05d.npz'%eid)) as s:t=s['t']
    with np.load(DATA/'events'/('event_%05d.npz'%eid)) as s:d=s['d'];truth=s['truth']
    geo=np.load(DATA/'wire_geometry.npy')[f['cells']];q=np.load(P/'full/validation/epoch_025'/('prediction_%05d.npz'%eid))['T']
    base,_=axial_two_pass(geo,d,f['pairs'],f['p']);groups,w=T_groups(f,t,q);greedy=geometry_extract(groups,w,f,geo,d,base)
    started=time.perf_counter();packed,solver=choose(base,greedy,f,geo,d);solver['seconds']=time.perf_counter()-started
    result={}
    for k,tracks in [('baseline',base),('T_greedy',greedy),('T_packing',packed)]:
        joint,_=attach_stereo(tracks,geo,d,f['cells'],f['pairs'],f['p'],DEFAULT_CONFIG);counts,found=score_tracks(joint,truth,f['stereo'],DEFAULT_CONFIG)
        confirmed,_=select(joint,propose(joint,geo,d,f['cells'],f['pairs'],f['p'],refine=False),geo,f['cells'],'guarded_loose')
        result[k]=dict(counts=counts,found=found,seeds=seeds(confirmed,truth,f['stereo'],f['cells']))
    save_json(OUT/('event_%05d.json'%eid),dict(event_id=eid,methods=result,solver=solver));return eid

def main():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from STEP5_两级寻迹 import summarize
    from STEP2_训练双路线 import seed_summary
    OUT.mkdir(parents=True,exist_ok=True);ids=json.loads((P/'full/contract.json').read_text(encoding='utf-8'))['validation_ids'][:16]+[4309,5477,9555]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(one,eid) for eid in ids]):print('packed',future.result(),flush=True)
    rows=[json.loads((OUT/('event_%05d.json'%eid)).read_text(encoding='utf-8')) for eid in ids]
    result=dict(events=len(rows),scope='Development set: fixed16 plus3 selected failure events; not confirmation evidence',methods={k:dict(tracks=summarize([r['methods'][k]['counts'] for r in rows]),seeds=seed_summary([r['methods'][k]['seeds'] for r in rows])) for k in rows[0]['methods']},solver=[r['solver'] for r in rows])
    save_json(OUT/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
