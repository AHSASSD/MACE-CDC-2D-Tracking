"""Frozen recovery scorer: extra validation first, then the existing fixed test."""
import experiment_common as e
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import pickle
from pathlib import Path
import time
import numpy as np
import torch
from common import JointEventDataset,JointInteractionNetwork
from server_runtime import loader,DevicePrefetch
from STEP5_两级寻迹 import axial_two_pass,attach_stereo,score_tracks,summarize,undirected_predictions,DEFAULT_CONFIG
from hybrid_joint_tracker import track
from coarse_3d_membership import propose,select
from 生成粗三维成员报告 import seed_counts
from STEP2_弱边候选校准 import OUT as CALIBRATION,seed_summary
from weak_edge_recovery import generate,reconstruct

_OLD=None
_NEW=None
_PHASE=None
_CUT=None


def initialize(phase,cut):
    global _OLD,_NEW,_PHASE,_CUT
    e.initialize_worker();_PHASE=phase;_CUT=cut
    with (e.MODEL_ROOT/'local_evaluation_epoch095_20261107/candidate_confidence_v2/candidate_confidence.pkl').open('rb') as f:_OLD=pickle.load(f)
    with (CALIBRATION/'candidate_recovery_scorer.pkl').open('rb') as f:_NEW=pickle.load(f)


def prediction_folder(phase):
    return e.EXP/'evaluation_v1/validation_prediction_cache' if phase=='validation' else e.SNAPSHOT/('test_prediction_cache_'+e.CONTRACT['weight_sha256'][:12])


@torch.no_grad()
def prepare_predictions(ids):
    cache=prediction_folder('validation');cache.mkdir(parents=True,exist_ok=True)
    data=JointEventDataset(e.DATA_ROOT,'validation');lookup={eid:i for i,eid in enumerate(data.ids)}
    remaining=[eid for eid in ids if not (cache/f'event_{eid:05d}.npz').exists()]
    if not remaining:return
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=JointInteractionNetwork(chunk_size=65536).to(device)
    model.load_state_dict(torch.load(e.SNAPSHOT/'best_interaction_network_joint_no_time.pth',map_location=device,weights_only=True));model.eval()
    for j,g in enumerate(DevicePrefetch(loader(torch.utils.data.Subset(data,[lookup[eid] for eid in remaining]),1,workers=2),device)):
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):p=model(g.x,g.edge_index,g.edge_attr).sigmoid().float().cpu().numpy()
        pairs,p=undirected_predictions(g.edge_index.cpu().numpy(),p);eid=int(g.event_id.item())
        np.savez_compressed(cache/f'event_{eid:05d}.npz',pairs=pairs.astype(np.int32),probabilities=p)
        if (j+1)%50==0:print(json.dumps(dict(phase='prediction',events=j+1,total=len(remaining))),flush=True)


def event_evaluation(eid):
    started=time.perf_counter();phase=_PHASE
    with np.load(e.DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,drift,truth,hit_ids=f['cell_id'],f['d'],f['truth'],f['hit_id']
    geo=np.load(e.DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    with np.load(prediction_folder(phase)/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    base_cache=e.EXP/'evaluation_v1'/phase/'baseline';base_cache.mkdir(parents=True,exist_ok=True);path=base_cache/f'event_{eid:05d}.json'
    if path.exists():
        packet=json.loads(path.read_text(encoding='utf-8'));baseline,coarse=packet['tracks'],packet['coarse']
    elif phase=='test':
        baseline=json.loads((e.SNAPSHOT/'coarse_3d_inputs_test'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))['tracks']
        coarse=json.loads((e.SNAPSHOT/'coarse_3d_v4_fast_test/events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))['candidates']['guarded_loose']
    else:
        axial,_=axial_two_pass(geo,drift,pairs,p);joint,_=attach_stereo(axial,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
        baseline=track(geo,drift,cells,pairs,p,axial,joint,_OLD,.02,.2)
        coarse,_=select(baseline,propose(baseline,geo,drift,cells,pairs,p,refine=False),geo,cells,'guarded_loose')
    if not path.exists():e.save_json(path,dict(event_id=eid,tracks=baseline,coarse=coarse))
    protected_s={i for t in coarse for i in t['indices'] if stereo[i]}
    candidates,x=generate(geo,drift,cells,pairs,p,baseline,protected_s)
    probability=_NEW.predict_proba(x)[:,1] if len(x) else np.empty(0)
    selected,ready,diag=reconstruct(geo,drift,cells,pairs,p,baseline,coarse,candidates,probability,_CUT)
    assert [t['indices'] for t in selected[:len(baseline)]]==[t['indices'] for t in baseline]
    assert [t['indices'] for t in ready[:len(coarse)]]==[t['indices'] for t in coarse]
    b,bids=score_tracks(baseline,truth,stereo,DEFAULT_CONFIG);m,mids=score_tracks(selected,truth,stereo,DEFAULT_CONFIG)
    for t in selected:t['hit_ids']=hit_ids[t['indices']].tolist()
    for t in ready:
        t['hit_ids']=hit_ids[t['indices']].tolist();t['confirmed_stereo_hit_ids']=hit_ids[t['confirmed_stereo_indices']].tolist()
    return dict(event_id=eid,baseline=b,selected=m,baseline_matched=bids,selected_matched=mids,
                baseline_seed=seed_counts(coarse,truth,stereo,cells),selected_seed=seed_counts(ready,truth,stereo,cells),
                candidates=selected,coarse=ready,proposed=len(candidates),diagnostics=diag,old_candidate_memberships_unchanged=True,
                observable=dict(hits=len(cells),mean_graph_degree=2*len(p)/len(cells),duplicate_cell_fraction=1-len(np.unique(cells))/len(cells),
                                longitudinal_ambiguity_fraction=sum(t['coarse_3d']['status']=='ambiguous' for t in coarse)/max(len(coarse),1)),
                seconds=time.perf_counter()-started)


def aggregate(rows):
    b=summarize([r['baseline'] for r in rows]);m=summarize([r['selected'] for r in rows])
    bs=seed_summary([r['baseline_seed'] for r in rows]);ms=seed_summary([r['selected_seed'] for r in rows])
    a_ok=m['track_efficiency']>=b['track_efficiency'] and m['fake_fraction']<=b['fake_fraction'] and m['clone_fraction']<=b['clone_fraction']
    s_ok=ms['efficiency']>=bs['efficiency'] and ms['fake_fraction']<=bs['fake_fraction'] and ms['clone_fraction']<=bs['clone_fraction']
    den=np.array([r['baseline']['truth_tracks'] for r in rows]);delta=np.array([r['selected']['matched_tracks']-r['baseline']['matched_tracks'] for r in rows])
    draws=np.random.default_rng(20261111).integers(len(rows),size=(2000,len(rows)))
    ci=np.quantile(delta[draws].sum(1)/den[draws].sum(1),[.025,.975]).tolist()
    return dict(events=len(rows),baseline=b,selected=m,baseline_seed=bs,selected_seed=ms,
                axial_non_regression=bool(a_ok),seed_non_regression=bool(s_ok),paired_efficiency_difference_95_ci=ci,
                gained=sum(len(set(r['selected_matched'])-set(r['baseline_matched'])) for r in rows),
                lost=sum(len(set(r['baseline_matched'])-set(r['selected_matched'])) for r in rows))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['validation','test'],default='validation');parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    calibration=json.loads((CALIBRATION/'summary.json').read_text(encoding='utf-8'));cut=calibration['selected_cut']
    if cut>1:raise ValueError('OOF selection kept baseline; no supported recovery configuration')
    if e.sha(CALIBRATION/'candidate_recovery_scorer.pkl')!=calibration['scorer_sha256']:raise ValueError('Recovery scorer changed')
    if args.phase=='validation':
        all_ids=JointEventDataset(e.DATA_ROOT,'validation').ids
        historical=json.loads((e.MODEL_ROOT/'local_evaluation_epoch095_20261107/hybrid_fresh_validation_v1/contract.json').read_text(encoding='utf-8'))['event_ids']
        excluded=set(e.IDS)|set(historical);available=[eid for eid in all_ids if eid not in excluded]
        ids=list(map(int,np.random.default_rng(20261111).permutation(available)[:300]))
    else:
        check=json.loads((e.EXP/'evaluation_v1/validation/summary.json').read_text(encoding='utf-8'))
        if not check['axial_non_regression'] or not check['seed_non_regression']:raise ValueError('Extra validation did not pass both gates; do not proceed to test')
        ids=json.loads((e.SNAPSHOT/'coarse_3d_v4_fast_test/contract.json').read_text(encoding='utf-8'))['event_ids']
    out=e.EXP/'evaluation_v1'/args.phase;out.mkdir(parents=True,exist_ok=True)
    plan=dict(phase=args.phase,event_ids=ids,cut=cut,scorer_sha256=calibration['scorer_sha256'],baseline=e.PIN,
              source_sha256={p:e.sha(e.EXP/p) for p in ['weak_edge_recovery.py','STEP3_独立验证与固定测试.py','experiment_common.py']},
              calibration_contract_sha256=e.sha(CALIBRATION/'contract.json'),test_used_for_selection=False,
              validation_policy='300 additional validation events excluding 200 diagnosis/calibration and 300 earlier historical validation events')
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text(encoding='utf-8'))!=plan:raise ValueError('Frozen evaluation plan changed')
    e.save_json(out/'contract.json',plan)
    if args.phase=='validation':prepare_predictions(ids)
    folder=out/'events';folder.mkdir(exist_ok=True);rows={};started=time.perf_counter()
    for eid in ids:
        if (folder/f'event_{eid:05d}.json').exists():rows[eid]=json.loads((folder/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(args.phase,cut)) as pool:
        for future in as_completed([pool.submit(event_evaluation,eid) for eid in ids if eid not in rows]):
            row=future.result();rows[row['event_id']]=row;e.save_json(folder/f"event_{row['event_id']:05d}.json",row)
            if len(rows)%20==0:
                progress=dict(phase=args.phase,events=len(rows),total=len(ids),seconds=time.perf_counter()-started);e.save_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    ordered=[rows[eid] for eid in ids];summary=aggregate(ordered)
    diagnosis=json.loads((e.EXP/'attribution_v1/summary.json').read_text(encoding='utf-8'))
    reference=[json.loads((e.EXP/'attribution_v1/events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))['observable'] for eid in e.IDS]
    features=diagnosis['high_occupancy_proxy']['features'];cutoff=diagnosis['high_occupancy_proxy']['cutoff'];hard=[]
    for row in ordered:
        proxy=sum(np.mean([v[name]<row['observable'][name] for v in reference]) for name in features)/len(features)
        if proxy>=cutoff:hard.append(row)
    summary.update(plan=plan,high_occupancy_proxy=aggregate(hard) if hard else None,seconds=time.perf_counter()-started)
    e.verify_pins();e.save_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
