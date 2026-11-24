"""Validation-first rough 3D stereo membership for downstream precision fitting."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'1')
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import pickle
import time
import numpy as np
from threadpoolctl import threadpool_limits
from common import ROOT,MODEL_ROOT,DATA_ROOT,sha,save_json
from STEP5_两级寻迹 import prepare_predictions,axial_two_pass,attach_stereo,score_tracks,summarize,DEFAULT_CONFIG
from hybrid_joint_tracker import track
from coarse_3d_membership import propose,select,PROFILES,SEARCH
from two_stage_core import fit_2d_circle

_MODEL=None


def initialize(confidence):
    global _MODEL
    import torch
    torch.set_num_threads(1);threadpool_limits(limits=1)
    with open(confidence,'rb') as f:_MODEL=pickle.load(f)


def stereo_score(tracks,truth,stereo,cells):
    eligible={int(t):int(((truth==t)&~stereo).sum()) for t in np.unique(truth) if t>=0 and ((truth==t)&~stereo).sum()>=5}
    seeded_truth={t for t in eligible if len(np.unique(cells[(truth==t)&stereo]))>=2}
    counts=Counter(truth_stereo_hits=sum(int(((truth==t)&stereo).sum()) for t in eligible),seed_eligible_tracks=len(seeded_truth))
    confirmed_found=set();associated_found=set()
    for t in tracks:
        indices=np.asarray(t['indices'],dtype=int);aa=indices[~stereo[indices]];ss=indices[stereo[indices]]
        verified=np.asarray(t.get('confirmed_stereo_indices',[]),dtype=int)
        counts['associated_stereo_hits']+=len(ss);counts['confirmed_stereo_hits']+=len(verified)
        tids,ns=np.unique(truth[aa],return_counts=True);tid=int(tids[ns.argmax()]);size=int(ns.max())
        valid=tid in eligible and size/len(aa)>=.8 and size/eligible[tid]>=.5
        if not valid:continue
        counts['matched_associated_stereo_hits']+=len(ss);counts['matched_confirmed_stereo_hits']+=len(verified)
        correct=ss[truth[ss]==tid];correct_verified=verified[truth[verified]==tid]
        counts['correct_associated_stereo_hits']+=len(correct);counts['correct_confirmed_stereo_hits']+=len(correct_verified)
        if tid in seeded_truth and len(np.unique(cells[correct]))>=2 and len(correct)/max(len(ss),1)>=.8:associated_found.add(tid)
        if tid in seeded_truth and len(np.unique(cells[correct_verified]))>=2 and len(correct_verified)/max(len(verified),1)>=.8:confirmed_found.add(tid)
    counts['two_correct_associated_tracks']=len(associated_found);counts['two_correct_confirmed_tracks']=len(confirmed_found)
    return dict(counts)


def stereo_summary(rows):
    counts=Counter()
    for row in rows:counts.update(row)
    result=dict(counts)
    for name,num,den in [('associated_precision','correct_associated_stereo_hits','associated_stereo_hits'),
                         ('associated_recall','correct_associated_stereo_hits','truth_stereo_hits'),
                         ('confirmed_precision','correct_confirmed_stereo_hits','confirmed_stereo_hits'),
                         ('confirmed_recall','correct_confirmed_stereo_hits','truth_stereo_hits'),
                         ('matched_associated_precision','correct_associated_stereo_hits','matched_associated_stereo_hits'),
                         ('matched_confirmed_precision','correct_confirmed_stereo_hits','matched_confirmed_stereo_hits'),
                         ('two_correct_associated_efficiency','two_correct_associated_tracks','seed_eligible_tracks'),
                         ('two_correct_confirmed_efficiency','two_correct_confirmed_tracks','seed_eligible_tracks')]:
        result[name]=counts[num]/max(counts[den],1)
    return result


def inputs(eid,snapshot,split,geo,drift,cells,pairs,p,stereo):
    folder=snapshot/f'coarse_3d_inputs_{split}';folder.mkdir(exist_ok=True);path=folder/f'event_{eid:05d}.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))['tracks']
    axial,_=axial_two_pass(geo,drift,pairs,p)
    if split=='test':
        # Replay existing frozen memberships without re-running joint proposals.
        old=json.loads((snapshot/'model_comparison_v1/events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
        groups=old['candidates']['conservative_joint'];lookup={tuple(sorted(t['indices'])):t for t in axial};result=[]
        for group in groups:
            aa=sorted(i for i in group['indices'] if not stereo[i]);key=tuple(aa)
            if key in lookup:t=dict(lookup[key])
            else:
                xy=geo[aa,:2];fit=fit_2d_circle(xy[:,0],xy[:,1],drift[aa])
                if fit is None:raise ValueError('Saved member set has no transverse fit')
                residual=np.abs(np.linalg.norm(xy-np.asarray(fit[:2]),axis=1)-fit[2])-drift[aa]
                t=dict(fit=list(fit),rms=float(np.sqrt(np.mean(residual**2))),stage='frozen_joint_member_replay')
            t['indices']=list(group['indices']);t['stereo_indices']=[i for i in group['indices'] if stereo[i]];result.append(t)
    else:
        joint,_=attach_stereo(axial,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
        result=track(geo,drift,cells,pairs,p,axial,joint,_MODEL,.02,.2)
    save_json(path,dict(event_id=eid,tracks=result));return result


def evaluate(eid,snapshot,split,refine,profiles):
    started=time.perf_counter();snapshot=Path(snapshot);weight=sha(snapshot/'best_interaction_network_joint_no_time.pth')
    cache=snapshot/(split+'_prediction_cache_'+weight[:12])
    with np.load(cache/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,drift,truth,hit_ids=f['cell_id'],f['d'],f['truth'],f['hit_id']
    geo=np.load(DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    baseline=inputs(eid,snapshot,split,geo,drift,cells,pairs,p,stereo)
    baseline_count,baseline_matched=score_tracks(baseline,truth,stereo,DEFAULT_CONFIG)
    baseline_s=stereo_score(baseline,truth,stereo,cells)
    proposals=propose(baseline,geo,drift,cells,pairs,p,refine=refine)
    methods={};stereo_metrics={};diagnostics={};saved={}
    for profile in profiles:
        result,diag=select(baseline,proposals,geo,cells,profile)
        count,matched=score_tracks(result,truth,stereo,DEFAULT_CONFIG)
        for name in ['truth_tracks','output_candidates','matched_tracks','fake_candidates','clone_candidates']:
            if count.get(name,0)!=baseline_count.get(name,0):raise AssertionError('Protected A count changed: '+name)
        if matched!=baseline_matched:raise AssertionError('A truth identity changed')
        for before,after in zip(baseline,result):
            if [i for i in before['indices'] if not stereo[i]]!=[i for i in after['indices'] if not stereo[i]]:raise AssertionError('A member changed')
        for t in result:
            t['hit_ids']=hit_ids[t['indices']].tolist()
            t['confirmed_stereo_hit_ids']=hit_ids[t['confirmed_stereo_indices']].tolist()
            t['pending_stereo_hit_ids']=hit_ids[t['pending_stereo_indices']].tolist()
        methods[profile]=count;stereo_metrics[profile]=stereo_score(result,truth,stereo,cells);diagnostics[profile]=diag;saved[profile]=result
    return dict(event_id=eid,baseline=baseline_count,baseline_stereo=baseline_s,methods=methods,stereo_metrics=stereo_metrics,
                diagnostics=diagnostics,candidates=saved,seconds=time.perf_counter()-started)


def aggregate(rows,profiles):
    result=dict(baseline=summarize([r['baseline'] for r in rows]),baseline_stereo=stereo_summary([r['baseline_stereo'] for r in rows]),methods={})
    for profile in profiles:
        diag=Counter()
        for r in rows:diag.update(r['diagnostics'][profile])
        result['methods'][profile]=dict(tracking=summarize([r['methods'][profile] for r in rows]),
                                       stereo=stereo_summary([r['stereo_metrics'][profile] for r in rows]),diagnostics=dict(diag))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='local_evaluation_epoch126_20261107')
    parser.add_argument('--phase',choices=['pilot','validation','test'],default='pilot');parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--refine',action='store_true');parser.add_argument('--run-tag',default='coarse_3d_v1');args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot;reference=MODEL_ROOT/'local_evaluation_epoch095_20261107'
    confidence=reference/'candidate_confidence_v2/candidate_confidence.pkl'
    original=json.loads((reference/'hybrid_confidence_v1/summary.json').read_text(encoding='utf-8'))
    ids=original['calibration_events']+original['check_events'];split='validation';profiles=list(PROFILES)
    if args.phase=='pilot':ids=ids[:20]
    elif args.phase=='test':
        validation=json.loads((snapshot/(args.run_tag+'_validation')/'summary.json').read_text(encoding='utf-8'))
        selected=validation['selected_on_development']
        if not selected:raise ValueError('No validation-selected profile; do not evaluate test')
        if not validation['check_passed']:raise ValueError('Held validation failed; do not evaluate test')
        profiles=[selected];all_test=json.loads((snapshot/'model_comparison_v1/contract.json').read_text(encoding='utf-8'))['event_ids']
        ids=list(map(int,np.random.default_rng(20261109).permutation(all_test)[:300]));split='test'
    out=snapshot/(args.run_tag+'_'+args.phase);out.mkdir(exist_ok=True)
    names=['coarse_3d_membership.py','STEP17_保护直丝的三维辅助寻迹.py','三维拟合草稿_暂未启用.py','STEP5_两级寻迹.py',
           'hybrid_joint_tracker.py','joint_candidate_tracking.py','hybrid_candidate_tracking.py','STEP12_候选可靠性校准.py','two_stage_core.py','common.py']
    contract=dict(event_ids=ids,split=split,phase=args.phase,profiles={k:PROFILES[k] for k in profiles},search=SEARCH,refine=args.refine,
                  weight_sha256=sha(snapshot/'best_interaction_network_joint_no_time.pth'),confidence_sha256=sha(confidence),
                  dataset_metadata_sha256=sha(DATA_ROOT/'dataset_metadata_joint_no_time.json'),source_sha256={n:sha(ROOT/n) for n in names},
                  purpose='Rough 3D stereo membership; exact A protection; precision fit delegated to downstream GenFit',test_used_for_selection=False)
    yu=ROOT.parent/'AI工程化4 - 副本'/'Z.L. YU元胞自动机'/'positron_momentum_analysis_multitrack_2dbranch_plotstats.py'
    contract['geometry_reference']=dict(path=str(yu),sha256=sha(yu),function='fit_sz_with_geometry',
                                       adopted=['finite R+/-d cylinder intersections','arc-length compatible branches','finite axial z range','independent wire evidence'],
                                       not_required=['six stereo layers','both stereo orientations','precision momentum fit'])
    path=out/'contract.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=contract:raise ValueError('Contract changed; use a new run-tag')
    save_json(path,contract)
    cache=snapshot/(split+'_prediction_cache_'+contract['weight_sha256'][:12]);cache.mkdir(exist_ok=True)
    if split=='validation':prepare_predictions(snapshot,ids,cache)
    events=out/'events';events.mkdir(exist_ok=True);rows={};started=time.perf_counter()
    for eid in ids:
        if (events/f'event_{eid:05d}.json').exists():rows[eid]=json.loads((events/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(str(confidence),)) as pool:
        futures=[pool.submit(evaluate,eid,str(snapshot),split,args.refine,profiles) for eid in ids if eid not in rows]
        for future in as_completed(futures):
            row=future.result();eid=row['event_id'];rows[eid]=row;save_json(events/f'event_{eid:05d}.json',row)
            if len(rows)%10==0:
                progress=dict(events=len(rows),total=len(ids),phase=args.phase,seconds=time.perf_counter()-started);save_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    ordered=[rows[eid] for eid in ids];summary=dict(contract=contract,all=aggregate(ordered,profiles),seconds=time.perf_counter()-started)
    if args.phase=='validation':
        summary['development']=aggregate(ordered[:100],profiles);summary['check']=aggregate(ordered[100:],profiles)
        eligible=[]
        # Preserve all A, prefer usable clean 3D seeds, then retain more correct S.
        # S completeness is intentionally not a hard constraint: GenFit receives
        # a reliable subset, while uncertain associations remain separately saved.
        for name,result in summary['development']['methods'].items():
            s=result['stereo']
            if s['matched_confirmed_precision']>=.995:
                eligible.append(name)
        summary['selection_policy']=dict(min_matched_confirmed_stereo_precision=.995,priority=['two_correct_confirmed_tracks','correct_confirmed_stereo_hits'],all_axial_members_protected=True)
        summary['selected_on_development']=max(eligible,key=lambda n:(summary['development']['methods'][n]['stereo']['two_correct_confirmed_tracks'],summary['development']['methods'][n]['stereo']['correct_confirmed_stereo_hits'])) if eligible else None
        selected=summary['selected_on_development']
        summary['check_passed']=bool(selected and summary['check']['methods'][selected]['stereo']['matched_confirmed_precision']>=.995)
    save_json(out/'summary.json',summary);print(json.dumps(dict(phase=args.phase,all=summary['all'])),flush=True)


if __name__=='__main__':main()
