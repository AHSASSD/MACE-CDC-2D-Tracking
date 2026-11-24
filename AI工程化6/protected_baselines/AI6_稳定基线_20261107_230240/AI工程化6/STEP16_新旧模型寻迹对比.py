"""Frozen paired comparison: raw components, two-pass, conservative joint tracking.

Only the GNN weights change. The epoch-95 calibrator and all decoding/scoring
parameters remain frozen. Test labels are read only by score_tracks.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import pickle
import time
import numpy as np
from threadpoolctl import threadpool_limits
from common import ROOT, MODEL_ROOT, DATA_ROOT, JointEventDataset, sha, save_json
from STEP5_两级寻迹 import simple_components, axial_two_pass, attach_stereo, score_tracks, summarize, DEFAULT_CONFIG
from STEP9_固定方案测试 import cache_predictions
from hybrid_joint_tracker import track

METHODS = ['direct_components','two_pass','conservative_joint']
SOURCES = ['STEP16_新旧模型寻迹对比.py','common.py','server_runtime.py','STEP9_固定方案测试.py',
           'STEP5_两级寻迹.py','two_stage_core.py','hybrid_joint_tracker.py',
           'joint_candidate_tracking.py','hybrid_candidate_tracking.py','STEP12_候选可靠性校准.py']
_CALIBRATOR = None


def initialize_worker(confidence):
    global _CALIBRATOR
    import torch
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    with open(confidence,'rb') as f:_CALIBRATOR=pickle.load(f)


def evaluate_event(eid, snapshot, reference, reuse_reference):
    started=time.perf_counter();snapshot=Path(snapshot);reference=Path(reference)
    cache=snapshot/('test_prediction_cache_'+sha(snapshot/'best_interaction_network_joint_no_time.pth')[:12])
    with np.load(cache/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:
        cells,drift,truth,hit_ids=f['cell_id'],f['d'],f['truth'],f['hit_id']
    geo=np.load(DATA_ROOT/'wire_geometry.npy')[cells]
    stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    direct=[t for t in simple_components(len(cells),pairs,p) if sum(not stereo[i] for i in t['indices'])>=5]
    if reuse_reference:
        old=json.loads((reference/'frozen_tracking_test_v1/events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
        improved=json.loads((reference/'hybrid_frozen_test_v1/events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
        two_pass=old['candidates'];joint=improved['candidates']
    else:
        axial,_=axial_two_pass(geo,drift,pairs,p)
        two_pass,_=attach_stereo(axial,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
        joint=track(geo,drift,cells,pairs,p,axial,two_pass,_CALIBRATOR,.02,.2)
    counts={};matched={};groups={}
    for name,candidates in zip(METHODS,[direct,two_pass,joint]):
        indices=[i for t in candidates for i in t['indices']]
        if len(indices)!=len(set(indices)):raise ValueError('Hit reused within method')
        counts[name],matched[name]=score_tracks(candidates,truth,stereo,DEFAULT_CONFIG)
        groups[name]=[dict(indices=t['indices'],hit_ids=hit_ids[t['indices']].tolist(),
                           axial_hit_ids=hit_ids[[i for i in t['indices'] if not stereo[i]]].tolist()) for t in candidates]
    if reuse_reference:
        if counts['two_pass']!=old['selected'] or counts['conservative_joint']!=improved['selected']:
            raise ValueError('Reference replay scoring changed')
    return dict(event_id=eid,methods=counts,matched_truth_ids=matched,candidates=groups,
                seconds=time.perf_counter()-started,reused_reference_memberships=reuse_reference)


def interval(left, right, ids):
    # Candidate denominators differ, so resample events and sum counts first.
    rng=np.random.default_rng(20261109)
    draws=rng.integers(len(ids),size=(2000,len(ids)))
    result={}
    for key,num,den in [('track_efficiency','matched_tracks','truth_tracks'),
                        ('fake_fraction','fake_candidates','output_candidates'),
                        ('clone_fraction','clone_candidates','output_candidates')]:
        def sample(rows):
            a=np.array([rows[eid].get(num,0) for eid in ids]);b=np.array([rows[eid].get(den,0) for eid in ids])
            return a[draws].sum(1)/np.maximum(b[draws].sum(1),1)
        result[key]=np.quantile(sample(right)-sample(left),[.025,.975]).tolist()
    return result


def run(snapshot,reference,ids,workers):
    out=snapshot/'model_comparison_v1';out.mkdir(exist_ok=True)
    confidence=reference/'candidate_confidence_v2/candidate_confidence.pkl'
    metadata=json.loads((snapshot/'model_metadata_joint_no_time.json').read_text(encoding='utf-8'))
    if metadata['smoke'] or metadata['dataset_metadata_sha256']!=sha(DATA_ROOT/'dataset_metadata_joint_no_time.json'):
        raise ValueError('Snapshot/dataset mismatch')
    plan=dict(snapshot=snapshot.name,epoch=json.loads((snapshot/'best_model_selection.json').read_text(encoding='utf-8'))['epoch'],
              event_ids=ids,weight_sha256=sha(snapshot/'best_interaction_network_joint_no_time.pth'),
              dataset_metadata_sha256=metadata['dataset_metadata_sha256'],geometry_sha256=sha(DATA_ROOT/'wire_geometry.npy'),
              confidence_snapshot=reference.name,confidence_model_sha256=sha(confidence),base_cut=.02,new_cut=.2,
              direct_edge_threshold=.5,edge_symmetrization='mean of both directions',scoring_config=DEFAULT_CONFIG,
              source_hashes={name:sha(ROOT/name) for name in SOURCES},test_used_for_selection=False,
              calibration_policy='Frozen epoch-95 calibrator transferred without refitting; no test-based parameter changes')
    contract=out/'contract.json'
    if contract.exists() and json.loads(contract.read_text(encoding='utf-8'))!=plan:raise ValueError('Comparison contract changed')
    save_json(contract,plan)
    for eid in ids:
        # Validate fixed data identities through the original dataset contract.
        if not (DATA_ROOT/'events'/f'event_{eid:05d}.npz').exists():raise FileNotFoundError(eid)
    cache=snapshot/('test_prediction_cache_'+plan['weight_sha256'][:12]);cache.mkdir(exist_ok=True)
    cache_plan={k:plan[k] for k in ['weight_sha256','dataset_metadata_sha256','edge_symmetrization']}
    cache_plan['graph_builder_sha256']=sha(ROOT/'common.py')
    if (cache/'contract.json').exists() and json.loads((cache/'contract.json').read_text(encoding='utf-8'))!=cache_plan:
        raise ValueError('Prediction cache contract mismatch')
    save_json(cache/'contract.json',cache_plan)
    predict_started=time.perf_counter();cache_predictions(snapshot,cache,ids)
    prediction_seconds=time.perf_counter()-predict_started
    events=out/'events';events.mkdir(exist_ok=True);rows={};started=time.perf_counter()
    for eid in ids:
        if (events/f'event_{eid:05d}.json').exists():rows[eid]=json.loads((events/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,initargs=(str(confidence),)) as pool:
        pending=[pool.submit(evaluate_event,eid,str(snapshot),str(reference),snapshot==reference) for eid in ids if eid not in rows]
        for future in as_completed(pending):
            row=future.result();rows[row['event_id']]=row;save_json(events/f"event_{row['event_id']:05d}.json",row)
            if len(rows)%50==0:
                progress=dict(snapshot=snapshot.name,phase='tracking',events=len(rows),total=len(ids),seconds=time.perf_counter()-started)
                save_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    summary=dict(plan=plan,events=len(ids),methods={m:summarize([rows[eid]['methods'][m] for eid in ids]) for m in METHODS},
                 seconds_this_invocation=time.perf_counter()-started,prediction_seconds_this_invocation=prediction_seconds,
                 scope='Fixed previously exposed test set; hit membership, not fitted 3D parameters')
    save_json(out/'summary.json',summary)
    print(json.dumps(dict(snapshot=snapshot.name,methods=summary['methods'])),flush=True)
    return rows,summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True)
    parser.add_argument('--reference',default='local_evaluation_epoch095_20261107');parser.add_argument('--workers',type=int,default=10)
    args=parser.parse_args();snapshot=MODEL_ROOT/args.snapshot;reference=MODEL_ROOT/args.reference
    previous=json.loads((reference/'hybrid_frozen_test_v1/contract.json').read_text(encoding='utf-8'))
    for name,digest in previous['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Reference reconstruction source changed: '+name)
    if sha(reference/'best_interaction_network_joint_no_time.pth')!=previous['weight_sha256']:
        raise ValueError('Reference weight changed')
    if sha(reference/'candidate_confidence_v2/candidate_confidence.pkl')!=previous['confidence_model_sha256']:
        raise ValueError('Reference confidence model changed')
    ids=sorted(JointEventDataset(DATA_ROOT,'test').ids)
    if ids!=sorted(previous['event_ids']):raise ValueError('Different test event identities')
    # Freeze BOTH model comparisons before computing any new test results.
    reference_rows,old=run(reference,reference,ids,args.workers)
    new_rows,new=run(snapshot,reference,ids,args.workers)
    comparisons={}
    for method in METHODS:
        left={eid:reference_rows[eid]['methods'][method] for eid in ids};right={eid:new_rows[eid]['methods'][method] for eid in ids}
        gained=sum(len(set(new_rows[eid]['matched_truth_ids'][method])-set(reference_rows[eid]['matched_truth_ids'][method])) for eid in ids)
        lost=sum(len(set(reference_rows[eid]['matched_truth_ids'][method])-set(new_rows[eid]['matched_truth_ids'][method])) for eid in ids)
        comparisons[method]=dict(gained=gained,lost=lost,paired_difference_95_ci=interval(left,right,ids))
    within={}
    for left_method,right_method in [('direct_components','two_pass'),('two_pass','conservative_joint')]:
        left={eid:new_rows[eid]['methods'][left_method] for eid in ids};right={eid:new_rows[eid]['methods'][right_method] for eid in ids}
        within[left_method+'_to_'+right_method]=interval(left,right,ids)
    for summary,source in [(old,reference),(new,snapshot)]:
        if sha(source/'best_interaction_network_joint_no_time.pth')!=summary['plan']['weight_sha256']:raise ValueError('Weight changed during test')
    report=dict(reference=old,latest=new,model_differences=comparisons,latest_method_difference_95_ci=within)
    save_json(snapshot/'model_comparison_v1/paired_comparison.json',report)
    with (MODEL_ROOT/'evaluation_exposure_history.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(run_tag=args.snapshot,assessment='fixed three-method paired comparison',
                                weight_sha256=new['plan']['weight_sha256'],event_ids=ids,unix=time.time()))+'\n')


if __name__=='__main__':main()
