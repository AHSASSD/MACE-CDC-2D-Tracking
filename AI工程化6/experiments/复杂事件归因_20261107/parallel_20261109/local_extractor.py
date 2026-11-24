"""Frozen126 proposals before locking, with exact preservation of baseline axial hits."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import experiment_common as e
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import pickle
import time
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csc_matrix
from two_stage_core import components,hierarchy_partition,split_component,fit_candidate
from STEP5_两级寻迹 import axial_two_pass,attach_stereo,score_tracks,summarize,DEFAULT_CONFIG
from STEP12_候选可靠性校准 import features
from hybrid_joint_tracker import track
from coarse_3d_membership import propose,select
from 生成粗三维成员报告 import seed_counts
from STEP2_弱边候选校准 import seed_summary
from STEP3_独立验证与固定测试 import prepare_predictions

OUT=HERE/'extractor_results'
PROFILES={('protect%03d_new%03d'%(round(protect*100),round(cut*100))):dict(protect=protect,cut=cut)
          for protect in [.8,.95,1.01] for cut in [.8,.95]}
_MODEL=None
_PHASE=None
_PROFILES=None


def initialize(phase,profiles):
    global _MODEL,_PHASE,_PROFILES
    e.initialize_worker();_PHASE=phase;_PROFILES=profiles
    with (e.MODEL_ROOT/'local_evaluation_epoch095_20261107/candidate_confidence_v2/candidate_confidence.pkl').open('rb') as f:_MODEL=pickle.load(f)


def proposals(geo,drift,pairs,p):
    """Measurements and frozen edge probabilities only; no truth argument."""
    axial=np.flatnonzero(np.linalg.norm(geo[:,2:4],axis=1)<=1e-9)
    mapping=np.full(len(geo),-1,dtype=int);mapping[axial]=np.arange(len(axial))
    keep=(mapping[pairs[0]]>=0)&(mapping[pairs[1]]>=0);prob=p[keep]
    data=Data(wire_x=torch.tensor(geo[axial,0],dtype=torch.float64),wire_y=torch.tensor(geo[axial,1],dtype=torch.float64),
              drift_d=torch.tensor(drift[axial],dtype=torch.float64),edge_index=torch.tensor(mapping[pairs[:,keep]]),num_nodes=len(axial))
    result=[]
    for cut,rms in [(.70,1.),(.575,1.5)]:
        for component in components(data,prob,cut):
            parent=fit_candidate(data,component,prob,cut,rms,'unlocked_parent')
            if parent is not None and parent['accepted_geometry']:result.append(parent)
            if cut==.70:result.extend(hierarchy_partition(data,component,prob,cut) or split_component(data,component,prob,cut,rms))
            else:result.extend(split_component(data,component,prob,cut,rms))
    return [dict(t,indices=axial[t['indices']].tolist(),stage='proposal_before_lock') for t in result]


def resolve(pool,confidence,old_count,owned,n,config):
    """Set packing: every old hit exactly once; every free hit at most once."""
    old_sets=[set(t['indices']) for t in pool[:old_count]]
    allowed=list(range(old_count))
    for j,t in enumerate(pool[old_count:],old_count):
        members=set(t['indices'])
        if confidence[j]<config['cut']:continue
        overlap=[i for i,g in enumerate(old_sets) if members&g]
        if any(not old_sets[i]<=members or confidence[i]>=config['protect'] for i in overlap):continue
        allowed.append(j)
    candidates=[pool[i] for i in allowed]
    if len(candidates)==old_count:return list(range(old_count)),dict(allowed=old_count,status='baseline_only')
    row=[];col=[]
    for j,t in enumerate(candidates):
        row.extend(t['indices']);col.extend([j]*len(t['indices']))
    matrix=csc_matrix((np.ones(len(row)),(row,col)),shape=(n,len(candidates)))
    lower=np.zeros(n);lower[list(owned)]=1
    quality=np.array([len(pool[i]['indices'])*(.5+confidence[i])-.5 for i in allowed])
    answer=milp(-quality,integrality=np.ones(len(candidates)),bounds=Bounds(0,1),constraints=LinearConstraint(matrix,lower,np.ones(n)),options={'time_limit':2.})
    chosen=np.flatnonzero(answer.x>.5).tolist() if answer.x is not None else list(range(old_count))
    coverage=np.asarray(matrix[:,chosen].sum(axis=1)).ravel()
    if np.any(coverage>1) or np.any(coverage[list(owned)]!=1) or quality[chosen].sum()+1e-7<quality[:old_count].sum():chosen=list(range(old_count))
    return [allowed[i] for i in chosen],dict(allowed=len(allowed),status=int(answer.status))


def materialize(chosen,pool,baseline,coarse,geo,drift,cells,pairs,p):
    """Unchanged groups keep their S; changed groups reattach only available S."""
    stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9;old_count=len(baseline)
    retained=sorted(i for i in chosen if i<old_count);new=[pool[i] for i in chosen if i>=old_count]
    protected={j for i in retained for t in [baseline[i],coarse[i]] for j in t['indices'] if stereo[j]}
    mask=np.zeros(len(cells),dtype=bool);mask[list(protected)]=True
    probability=np.where(mask[pairs[0]]|mask[pairs[1]],0.,p)
    additions,_=attach_stereo(new,geo,drift,cells,pairs,probability,DEFAULT_CONFIG)
    ready,_=select(additions,propose(additions,geo,drift,cells,pairs,probability,refine=False),geo,cells,'guarded_loose') if additions else ([],{})
    joint=[baseline[i] for i in retained]+additions;full=[coarse[i] for i in retained]+ready
    owned={j for t in baseline for j in t['indices'] if not stereo[j]}
    for output in [joint,full]:
        used=[j for t in output for j in t['indices']]
        assert len(used)==len(set(used))
        assert owned<={j for j in used if not stereo[j]}, 'Baseline axial hit lost'
    return joint,full


def evaluate(eid):
    start=time.perf_counter()
    with np.load(e.DATA_ROOT/'events'/('event_%05d.npz'%eid)) as f:cells,drift,truth=f['cell_id'],f['d'],f['truth']
    geo=np.load(e.DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    cache=e.SNAPSHOT/('validation_prediction_cache_'+e.CONTRACT['weight_sha256'][:12]) if _PHASE=='dev' else e.EXP/'evaluation_v1/validation_prediction_cache'
    with np.load(cache/('event_%05d.npz'%eid)) as f:pairs,p=f['pairs'],f['probabilities']
    if _PHASE=='dev':
        baseline=json.loads((e.SNAPSHOT/'coarse_3d_inputs_validation'/('event_%05d.json'%eid)).read_text(encoding='utf-8'))['tracks']
        coarse=json.loads((e.VALIDATION/'events'/('event_%05d.json'%eid)).read_text(encoding='utf-8'))['candidates']['guarded_loose']
    else:
        axial,_=axial_two_pass(geo,drift,pairs,p);joint,_=attach_stereo(axial,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
        baseline=track(geo,drift,cells,pairs,p,axial,joint,_MODEL,.02,.2)
        coarse,_=select(baseline,propose(baseline,geo,drift,cells,pairs,p,refine=False),geo,cells,'guarded_loose')
    pool=[dict(t,indices=[j for j in t['indices'] if not stereo[j]]) for t in baseline]
    keys={tuple(sorted(t['indices'])) for t in pool};old_count=len(pool)
    for t in proposals(geo,drift,pairs,p):
        key=tuple(sorted(t['indices']))
        if key not in keys:pool.append(t);keys.add(key)
    matrix=np.zeros((len(cells),len(cells)),dtype=np.float32);matrix[pairs[0],pairs[1]]=p;matrix[pairs[1],pairs[0]]=p
    confidence=_MODEL.predict_proba(np.array([features(t,geo,cells,matrix) for t in pool]))[:,1] if pool else np.empty(0)
    owned={j for t in pool[:old_count] for j in t['indices']}
    methods={'baseline':baseline};seeds={'baseline':coarse};diag={};reconstruction_seconds={}
    for name,config in _PROFILES.items():
        begin=time.perf_counter();chosen,d=resolve(pool,confidence,old_count,owned,len(cells),config)
        methods[name],seeds[name]=materialize(chosen,pool,baseline,coarse,geo,drift,cells,pairs,p)
        diag[name]=dict(d,changed_groups=sum(j>=old_count for j in chosen));reconstruction_seconds[name]=time.perf_counter()-begin
    counts={};matched={}
    for name,tracks in methods.items():counts[name],matched[name]=score_tracks(tracks,truth,stereo,DEFAULT_CONFIG)
    return dict(event_id=eid,methods=counts,matched=matched,seeds={k:seed_counts(v,truth,stereo,cells) for k,v in seeds.items()},
                diagnostics=diag,proposals=len(pool)-old_count,all_baseline_axial_hits_preserved=True,
                total_seconds=time.perf_counter()-start,profile_reconstruction_seconds=reconstruction_seconds,
                axial_members={k:[[i for i in t['indices'] if not stereo[i]] for t in v] for k,v in methods.items()})


def aggregate(rows):
    methods={k:summarize([r['methods'][k] for r in rows]) for k in rows[0]['methods']}
    seeds={k:seed_summary([r['seeds'][k] for r in rows]) for k in rows[0]['seeds']}
    metrics={}
    for name in methods:
        gains=sum(len(set(r['matched'][name])-set(r['matched']['baseline'])) for r in rows)
        lost=sum(len(set(r['matched']['baseline'])-set(r['matched'][name])) for r in rows)
        a,b=methods[name],methods['baseline'];s,t=seeds[name],seeds['baseline']
        passed=lost==0 and a['track_efficiency']>=b['track_efficiency'] and a['fake_fraction']<=b['fake_fraction'] and a['clone_fraction']<=b['clone_fraction'] and s['efficiency']>=t['efficiency'] and s['fake_fraction']<=t['fake_fraction'] and s['clone_fraction']<=t['clone_fraction']
        metrics[name]=dict(gained=gains,lost=lost,non_regression=passed)
    return dict(events=len(rows),methods=methods,seeds=seeds,paired=metrics)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['dev','validation'],default='dev');parser.add_argument('--workers',type=int,default=8);parser.add_argument('--limit',type=int);args=parser.parse_args()
    if args.phase=='dev':ids=e.IDS;profiles=PROFILES
    else:
        dev=json.loads((OUT/'dev/summary.json').read_text());selected=dev['selected']
        if selected=='baseline' or not dev['check']['paired'][selected]['non_regression']:
            e.save_json(OUT/'validation_skipped.json',dict(reason='No improvement passing development and check gates'));return
        meta=json.loads((e.DATA_ROOT/'dataset_metadata_joint_no_time.json').read_text(encoding='utf-8'))
        historical=json.loads((e.MODEL_ROOT/'local_evaluation_epoch095_20261107/hybrid_fresh_validation_v1/contract.json').read_text(encoding='utf-8'))['event_ids']
        previous=json.loads((e.EXP/'evaluation_v1/validation/contract.json').read_text(encoding='utf-8'))['event_ids']
        excluded=set(e.IDS)|set(historical)|set(previous)
        ids=list(map(int,np.random.default_rng(20261113).permutation([i for i in meta['event_ids']['validation'] if i not in excluded])[:300]));profiles={selected:PROFILES[selected]}
    if args.limit:ids=ids[:args.limit]
    out=OUT/(args.phase+('_smoke' if args.limit else ''));folder=out/'events';folder.mkdir(parents=True,exist_ok=True)
    plan=dict(event_ids=ids,profiles=profiles,weight_sha256=e.CONTRACT['weight_sha256'],source_sha256=e.sha(Path(__file__)),all_old_axial_hits_preserved=True,test_used=False)
    if (out/'contract.json').exists():assert json.loads((out/'contract.json').read_text())==plan
    e.save_json(out/'contract.json',plan)
    if args.phase=='validation':prepare_predictions(ids)
    rows={};started=time.perf_counter()
    for eid in ids:
        path=folder/('event_%05d.json'%eid)
        if path.exists():rows[eid]=json.loads(path.read_text())
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(args.phase,profiles)) as pool:
        for future in as_completed([pool.submit(evaluate,eid) for eid in ids if eid not in rows]):
            row=future.result();eid=row['event_id'];rows[eid]=row;e.save_json(folder/('event_%05d.json'%eid),row)
            if len(rows)%10==0 or len(rows)==len(ids):
                progress=dict(phase=args.phase,events=len(rows),total=len(ids),seconds=time.perf_counter()-started);e.save_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    ordered=[rows[i] for i in ids];result=aggregate(ordered)
    if args.phase=='dev' and not args.limit:
        dev=aggregate(ordered[:100]);check=aggregate(ordered[100:]);allowed=[k for k in profiles if dev['paired'][k]['non_regression'] and dev['paired'][k]['gained']>0]
        selected=max(allowed,key=lambda k:(dev['methods'][k]['track_efficiency'],dev['seeds'][k]['efficiency'],-dev['methods'][k]['fake_fraction'])) if allowed else 'baseline'
        result.update(dev=dev,check=check,selected=selected)
    result.update(seconds=time.perf_counter()-started,plan=plan);e.verify_pins();e.save_json(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
