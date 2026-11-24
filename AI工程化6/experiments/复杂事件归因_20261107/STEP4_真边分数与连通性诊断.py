"""Frozen-score study: separation, pure connectivity and two-stage locking.
Truth interventions are offline diagnosis only; no GNN training or deployment edit.
"""
import experiment_common as e
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import time
import numpy as np
import torch
from torch_geometric.data import Data
from two_stage_core import components,trim_outliers
from STEP5_两级寻迹 import score_tracks,DEFAULT_CONFIG

BINS=4096
COHORTS=['all_true','all_false','found_true','missed_true','found_incident_false','missed_incident_false']


def topology(n,edges,score,cut=.575):
    """Maximum-spanning order quantifies how many low edges a pure group needs."""
    parent=np.arange(n);size=np.ones(n,dtype=int);largest=1;components_count=n;need=max(5,int(np.ceil(.5*n)))
    def root(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    order=np.argsort(-score,kind='stable');match_score=None;repair=[];low_unions=0
    for index in order:
        a,b=map(root,edges[:,index])
        if a==b:continue
        parent[b]=a;size[a]+=size[b];largest=max(largest,int(size[a]));components_count-=1
        if score[index]<cut:
            low_unions+=1
            if match_score is None:repair.append(int(index))
        if largest>=need and match_score is None:match_score=float(score[index])
    def count_at(threshold):
        pp=np.arange(n);ss=np.ones(n,dtype=int)
        def rr(a):
            while pp[a]!=a:pp[a]=pp[pp[a]];a=pp[a]
            return a
        for a,b in edges[:,score>=threshold].T:
            a,b=rr(a),rr(b)
            if a!=b:pp[b]=a;ss[a]+=ss[b]
        return int(ss.max())
    below=float((score<cut).mean()) if len(score) else 1.
    relaxed=count_at(cut);strict=count_at(.7)
    if largest<need:category='candidate_graph_insufficient'
    elif relaxed>=need:category='pure_connected_at_existing_relaxed_cut'
    elif below>=.8:category='broadly_low_scores'
    elif len(repair)<=2 and below<.5:category='few_weak_connectors'
    else:category='mixed_fragmentation'
    return dict(category=category,axial_hits=n,true_candidate_edges=len(score),below_relaxed_fraction=below,
                median_true_score=float(np.median(score)) if len(score) else None,
                p10_true_score=float(np.quantile(score,.1)) if len(score) else None,
                largest_relaxed_component=relaxed,largest_strict_component=strict,largest_all_component=largest,
                pure_match_hits_required=need,match_connectivity_score=match_score,
                repair_edges_to_match=len(repair) if match_score is not None else None,
                repair_edges_to_full=low_unions if components_count==1 else None,
                repair_local_edge_indices=repair)


def single_pass(geo,drift,pairs,p):
    axial=np.flatnonzero(np.linalg.norm(geo[:,2:4],axis=1)<=1e-9);mapping=np.full(len(geo),-1,dtype=int);mapping[axial]=np.arange(len(axial))
    keep=(mapping[pairs[0]]>=0)&(mapping[pairs[1]]>=0)
    data=Data(wire_x=torch.tensor(geo[axial,0],dtype=torch.float64),wire_y=torch.tensor(geo[axial,1],dtype=torch.float64),
              drift_d=torch.tensor(drift[axial],dtype=torch.float64),edge_index=torch.tensor(mapping[pairs[:,keep]]),num_nodes=len(axial))
    result=[]
    for group in components(data,p[keep],.575):
        t=trim_outliers(data,group,p[keep],.575,1.5,'diagnostic_single_pass')
        if t is not None:t['indices']=axial[t['indices']].tolist();result.append(t)
    return result


def analyze(eid):
    with np.load(e.DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,drift,truth=f['cell_id'],f['d'],f['truth']
    geo=np.load(e.DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    with np.load(e.SNAPSHOT/('validation_prediction_cache_'+e.CONTRACT['weight_sha256'][:12])/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    old=json.loads((e.EXP/'attribution_v1/events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    found=set(old['baseline_matched']);missed={t['truth_id'] for t in old['missed_tracks']};boost={t['truth_id'] for t in old['missed_tracks'] if t['restored_by_boost_FN']}
    denoised_found=set(old['oracle_matched']['oracle_remove_false_edges'])
    same=(truth[pairs[0]]==truth[pairs[1]])&(truth[pairs[0]]>=0);kind=stereo[pairs[0]].astype(int)+stereo[pairs[1]].astype(int)
    bins=np.minimum((p*BINS).astype(int),BINS-1);hist=np.zeros((3,len(COHORTS),BINS),dtype=np.int64)
    in_found=np.isin(truth[pairs[0]],list(found))|np.isin(truth[pairs[1]],list(found))
    in_missed=np.isin(truth[pairs[0]],list(missed))|np.isin(truth[pairs[1]],list(missed))
    masks=[same,~same,same&in_found,same&in_missed,~same&in_found,~same&in_missed]
    for k in range(3):
        for j,mask in enumerate(masks):hist[k,j]=np.bincount(bins[(kind==k)&mask],minlength=BINS)
    ideal=np.where(same,p,0.);tracks=single_pass(geo,drift,pairs,ideal)
    _,single_found=score_tracks(tracks,truth,stereo,DEFAULT_CONFIG);single_found=set(single_found)
    real_tracks=single_pass(geo,drift,pairs,p);real_count,real_found=score_tracks(real_tracks,truth,stereo,DEFAULT_CONFIG)
    members=[]
    ids,counts=np.unique(truth[(truth>=0)&~stereo],return_counts=True)
    for tid,n in zip(ids,counts):
        if n<5:continue
        hits=np.flatnonzero((truth==tid)&~stereo);mapping=np.full(len(truth),-1,dtype=int);mapping[hits]=np.arange(len(hits))
        keep=same&(kind==0)&(truth[pairs[0]]==tid)
        detail=topology(len(hits),mapping[pairs[:,keep]],p[keep]);detail.pop('repair_local_edge_indices')
        detail.update(truth_id=int(tid),found=int(tid) in found,boost_sensitive=int(tid) in boost,
                      restored_single_pass_denoised=int(tid) in missed and int(tid) not in denoised_found and int(tid) in single_found,
                      present_in_denoised_two_pass=int(tid) in denoised_found,present_in_denoised_single_pass=int(tid) in single_found)
        members.append(detail)
    return dict(event_id=eid,histogram=hist,tracks=members,real_single_pass_counts=real_count,
                real_single_pass_matched=real_found,diagnostic_only=True)


def distribution(hist):
    total=int(hist.sum());centers=(np.arange(BINS)+.5)/BINS
    if not total:return dict(n=0,mean=None,median=None,recall_at_0575=None)
    return dict(n=total,mean=float(hist@centers/total),median=float((np.searchsorted(np.cumsum(hist),total/2)+.5)/BINS),
                fraction_above_0575=float(hist[int(np.ceil(.575*BINS)):].sum()/total),fraction_above_070=float(hist[int(np.ceil(.7*BINS)):].sum()/total))


def operating_point(train_negative,held_positive,held_negative,budget):
    survival=np.r_[np.cumsum(train_negative[::-1])[::-1],0]
    threshold_bin=int(np.flatnonzero(survival<=budget*max(int(train_negative.sum()),1))[0])
    return dict(negative_edge_budget=budget,threshold=threshold_bin/BINS,
                held_true_edge_recall=float(held_positive[threshold_bin:].sum()/max(held_positive.sum(),1)),
                held_false_edge_acceptance=float(held_negative[threshold_bin:].sum()/max(held_negative.sum(),1)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    out=e.EXP/'edge_score_diagnosis_v1';out.mkdir(exist_ok=True);folder=out/'events';folder.mkdir(exist_ok=True)
    plan=dict(event_ids=e.IDS,model_sha256=e.CONTRACT['weight_sha256'],baseline=e.PIN,source_sha256=e.sha(__file__),
              bins=BINS,cohorts=COHORTS,false_edge_budgets=[.01,.02,.05],calibration_ids=e.IDS[:100],held_ids=e.IDS[100:],
              truth_only_diagnosis=True,gnn_training_started=False)
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text(encoding='utf-8'))!=plan:raise ValueError('Frozen score diagnosis changed')
    e.save_json(out/'contract.json',plan);rows={};histograms={};started=time.time()
    for eid in e.IDS:
        path=folder/f'event_{eid:05d}.json'
        if path.exists():rows[eid]=json.loads(path.read_text(encoding='utf-8'));histograms[eid]=np.load(folder/f'hist_{eid:05d}.npy')
    with ProcessPoolExecutor(max_workers=args.workers,initializer=e.initialize_worker) as pool:
        for future in as_completed([pool.submit(analyze,eid) for eid in e.IDS if eid not in rows]):
            row=future.result();eid=row['event_id'];hist=row.pop('histogram');histograms[eid]=hist;rows[eid]=row
            e.save_json(folder/f'event_{eid:05d}.json',row);np.save(folder/f'hist_{eid:05d}.npy',hist)
            if len(rows)%20==0:
                progress=dict(events=len(rows),total=len(e.IDS),seconds=time.time()-started);e.save_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    combined=sum(histograms.values());train=sum(histograms[eid] for eid in e.IDS[:100]);held=sum(histograms[eid] for eid in e.IDS[100:])
    stats={kind:{cohort:distribution(combined[k,j]) for j,cohort in enumerate(COHORTS)} for k,kind in enumerate(['AA','AS','SS'])}
    operating={}
    for k,kind in enumerate(['AA','AS','SS']):
        operating[kind]={}
        for name,pos,neg in [('all',0,1),('found',2,4),('missed',3,5)]:
            operating[kind][name]=[operating_point(train[k,neg],held[k,pos],held[k,neg],budget) for budget in plan['false_edge_budgets']]
    tracks=[dict(t,event_id=eid) for eid in e.IDS for t in rows[eid]['tracks']];missed=[t for t in tracks if not t['found']];sensitive=[t for t in missed if t['boost_sensitive']]
    summary=dict(plan=plan,score_distributions=stats,held_operating_points=operating,
                 missed_categories=dict(Counter(t['category'] for t in missed)),boost_sensitive_categories=dict(Counter(t['category'] for t in sensitive)),
                 missed_tracks=len(missed),boost_sensitive_tracks=len(sensitive),
                 restored_by_denoised_nonlocking=sum(t['restored_single_pass_denoised'] for t in missed),
                 restored_by_denoised_nonlocking_among_boost_sensitive=sum(t['restored_single_pass_denoised'] for t in sensitive),
                 seconds=time.time()-started)
    e.verify_pins();e.save_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
