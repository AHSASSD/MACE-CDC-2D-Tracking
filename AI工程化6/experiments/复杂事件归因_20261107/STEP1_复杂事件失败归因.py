"""Label-assisted diagnosis only. Oracle scores never enter deployment code."""
import experiment_common as e
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from STEP5_两级寻迹 import axial_two_pass,score_tracks,summarize,DEFAULT_CONFIG
from two_stage_core import fit_2d_circle
from coarse_3d_membership import propose,select,branch_observations


def truth_components(n,pairs,mask,truth,stereo,eligible):
    selected=pairs[:,mask];graph=coo_matrix((np.ones(len(selected.T)),selected),shape=(n,n))
    _,labels=connected_components(graph,directed=False)
    return {int(t):int(np.unique(labels[(truth==t)&~stereo],return_counts=True)[1].max()) for t in eligible}


def circle_evidence(indices,geo,drift):
    keep=np.array(indices,dtype=int);history=[];fit=None
    for attempt in range(3):
        xy=geo[keep,:2];fit=fit_2d_circle(xy[:,0],xy[:,1],drift[keep])
        if fit is None or not np.isfinite(fit).all():return dict(rms=None,rms_after_up_to_two_trims=None)
        residual=np.abs(np.linalg.norm(xy-np.asarray(fit[:2]),axis=1)-fit[2])-drift[keep]
        rms=float(np.sqrt(np.mean(residual**2)));history.append(rms)
        if rms<=1.5 or len(keep)<=5:break
        keep=np.delete(keep,int(np.argmax(np.abs(residual))))
    return dict(rms=history[0],rms_after_up_to_two_trims=history[-1],trimmed=len(indices)-len(keep),
                fitted_circle=list(fit),retained_indices=keep.tolist())


def event_diagnosis(eid):
    started=time.perf_counter();snapshot=e.SNAPSHOT
    with np.load(e.DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,drift,truth=f['cell_id'],f['d'],f['truth']
    geo=np.load(e.DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    with np.load(snapshot/('validation_prediction_cache_'+e.CONTRACT['weight_sha256'][:12])/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    saved=json.loads((snapshot/'coarse_3d_inputs_validation'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))['tracks']
    coarse=json.loads((e.VALIDATION/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    baseline_counts,baseline_matched=score_tracks(saved,truth,stereo,DEFAULT_CONFIG)
    assert baseline_counts==coarse['baseline']
    tids,ns=np.unique(truth[(truth>=0)&~stereo],return_counts=True);eligible={int(t):int(n) for t,n in zip(tids,ns) if n>=5}
    same=(truth[pairs[0]]==truth[pairs[1]])&(truth[pairs[0]]>=0)
    aa_mask=~stereo[pairs[0]]&~stereo[pairs[1]]
    connectivity={}
    for name,mask in [('all_candidate_edges',np.ones(len(p),dtype=bool)),('p050',p>=.5),('p0575',p>=.575),('p070',p>=.7)]:
        connectivity[name]=dict(AA=truth_components(len(cells),pairs,same&mask&aa_mask,truth,stereo,eligible),
                                joint=truth_components(len(cells),pairs,same&mask,truth,stereo,eligible))
    interventions={'actual_axial_stage':p,'oracle_remove_false_edges':np.where(same,p,0.),
                   'oracle_boost_true_edges':np.where(same,1.,p),'oracle_perfect_edges':same.astype(float)}
    methods={};matched={};diags={}
    for name,probability in interventions.items():
        tracks,diag=axial_two_pass(geo,drift,pairs,probability);counts,found=score_tracks(tracks,truth,stereo,DEFAULT_CONFIG)
        methods[name]=counts;matched[name]=found;diags[name]=diag
    missed=[]
    for tid,n in eligible.items():
        if tid in baseline_matched:continue
        indices=np.flatnonzero((truth==tid)&~stereo);geometry=circle_evidence(indices,geo,drift)
        remove=tid in matched['oracle_remove_false_edges'];boost=tid in matched['oracle_boost_true_edges'];perfect=tid in matched['oracle_perfect_edges']
        pure_connected=connectivity['all_candidate_edges']['AA'][tid]>=max(5,int(np.ceil(n*.5)))
        if not pure_connected:bucket='pure_true_graph_fragmented'
        elif not perfect:
            rms=geometry['rms_after_up_to_two_trims']
            bucket='single_circle_or_fitter_limit' if rms is None or rms>1.5 else 'extractor_organization_limit'
        elif remove and boost:bucket='sensitive_to_either_FP_or_FN'
        elif remove:bucket='false_edge_interference'
        elif boost:bucket='weak_true_edge_scores'
        else:bucket='combined_score_interference'
        coverage=[]
        for t in saved:
            aa=[i for i in t['indices'] if not stereo[i]];correct=sum(truth[i]==tid for i in aa)
            if correct:coverage.append(dict(correct=int(correct),purity=correct/len(aa),completeness=correct/n))
        used={i for t in saved for i in t['indices'] if not stereo[i]}
        missed.append(dict(truth_id=tid,axial_hits=n,geometry=geometry,bucket=bucket,
                           restored_by_remove_FP=remove,restored_by_boost_FN=boost,restored_by_perfect_edges=perfect,
                           largest_AA_component={k:v['AA'][tid] for k,v in connectivity.items()},
                           largest_joint_component={k:v['joint'][tid] for k,v in connectivity.items()},
                           unused_true_axial_hits=sum(int(i not in used) for i in indices),existing_candidate_coverage=coverage))
    fake=[]
    for j,t in enumerate(saved):
        aa=np.array([i for i in t['indices'] if not stereo[i]],dtype=int);ids,counts=np.unique(truth[aa],return_counts=True)
        tid=int(ids[counts.argmax()]);correct=int(counts.max());purity=correct/len(aa);completeness=correct/max(eligible.get(tid,0),1)
        reason='ineligible_truth' if tid not in eligible else 'mixed_hits' if purity<.8 else 'incomplete_fragment' if completeness<.5 else None
        if reason:fake.append(dict(candidate=j,dominant_truth=tid,axial_hits=len(aa),purity=purity,completeness=completeness,
                                   reason=reason,truth_already_found=tid in baseline_matched))
    proposals=propose(saved,geo,drift,cells,pairs,p,refine=False)
    selected,_=select(saved,proposals,geo,cells,'guarded_loose')
    expected=coarse['candidates']['guarded_loose']
    assert all(set(a['indices'])==set(b['indices']) and set(a['confirmed_stereo_indices'])==set(b['confirmed_stereo_indices']) for a,b in zip(selected,expected))
    stereo_losses=Counter();lost_s_examples=[]
    for j,(old,new,proposal) in enumerate(zip(saved,selected,proposals)):
        aa=np.array([i for i in old['indices'] if not stereo[i]],dtype=int);ids,counts=np.unique(truth[aa],return_counts=True);tid=int(ids[counts.argmax()])
        if tid not in eligible or counts.max()/len(aa)<.8 or counts.max()/eligible[tid]<.5:continue
        previous={i for i in old['indices'] if stereo[i] and truth[i]==tid};confirmed=set(new['confirmed_stereo_indices'])
        pool=np.asarray(proposal['pool'],dtype=int);lookup={int(i):k for k,i in enumerate(pool)}
        branches=branch_observations(old['fit'],proposal['hypotheses'][0]['arc'],geo[pool],drift[pool]) if proposal['hypotheses'] else None
        for hit in previous-confirmed:
            if hit not in lookup:reason='not_in_stereo_pool'
            elif not proposal['hypotheses']:reason='no_supported_longitudinal_hypothesis'
            else:
                pos=lookup[hit];res=np.array([abs(h['residual'][pos]) for h in proposal['hypotheses']])
                if branches is not None and not np.isfinite(branches[pos,:,0]).any():reason='no_branch_inside_finite_arc'
                elif (res<=1.5).mean()<.5:reason='longitudinal_consensus_residual'
                else:reason='stereo_competition_or_margin'
            stereo_losses[reason]+=1
            if len(lost_s_examples)<12:lost_s_examples.append(dict(candidate=j,hit_index=int(hit),truth_id=tid,reason=reason))
    observable=dict(hits=len(cells),axial_hits=int((~stereo).sum()),stereo_hits=int(stereo.sum()),
                    mean_graph_degree=2*len(p)/max(len(cells),1),duplicate_cell_fraction=1-len(np.unique(cells))/max(len(cells),1),
                    longitudinal_ambiguity_fraction=sum(x['status']=='ambiguous' for x in proposals)/max(len(proposals),1))
    return dict(event_id=eid,observable=observable,baseline=baseline_counts,baseline_matched=baseline_matched,
                oracle_methods=methods,oracle_matched=matched,axial_stage_diagnostics=diags,missed_tracks=missed,
                fake_candidates=fake,correct_stereo_not_confirmed=dict(stereo_losses),stereo_loss_examples=lost_s_examples,
                seconds=time.perf_counter()-started,truth_assisted_diagnostic_only=True)


def aggregate(rows):
    buckets=Counter();fakes=Counter();losses=Counter();recoveries=Counter();modes=list(rows[0]['oracle_methods'])
    for row in rows:
        buckets.update(t['bucket'] for t in row['missed_tracks']);fakes.update(t['reason'] for t in row['fake_candidates']);losses.update(row['correct_stereo_not_confirmed'])
        for t in row['missed_tracks']:
            recoveries.update(remove_FP=int(t['restored_by_remove_FP']),boost_FN=int(t['restored_by_boost_FN']),perfect_edges=int(t['restored_by_perfect_edges']))
    return dict(events=len(rows),baseline=summarize([r['baseline'] for r in rows]),missed_tracks=sum(buckets.values()),
                diagnostic_buckets=dict(buckets),overlapping_oracle_recoveries=dict(recoveries),fake_causes=dict(fakes),
                correct_stereo_not_confirmed=dict(losses),oracle_axial_stage={m:summarize([r['oracle_methods'][m] for r in rows]) for m in modes})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    out=e.EXP/'attribution_v1';out.mkdir(exist_ok=True)
    plan=dict(event_ids=e.IDS,split='validation',baseline=e.PIN,weight_sha256=e.CONTRACT['weight_sha256'],
              scoring=DEFAULT_CONFIG,diagnostic_interventions=['remove false edges','boost true edges','perfect edges'],
              source_sha256=e.sha(__file__),truth_assisted_diagnostic_only=True,test_used=False)
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text(encoding='utf-8'))!=plan:raise ValueError('Attribution contract changed')
    e.save_json(out/'contract.json',plan);folder=out/'events';folder.mkdir(exist_ok=True);rows={};started=time.perf_counter()
    for eid in e.IDS:
        if (folder/f'event_{eid:05d}.json').exists():rows[eid]=json.loads((folder/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    with ProcessPoolExecutor(max_workers=args.workers,initializer=e.initialize_worker) as pool:
        futures=[pool.submit(event_diagnosis,eid) for eid in e.IDS if eid not in rows]
        for future in as_completed(futures):
            row=future.result();rows[row['event_id']]=row;e.save_json(folder/f"event_{row['event_id']:05d}.json",row)
            if len(rows)%10==0:
                progress=dict(events=len(rows),total=len(e.IDS),seconds=time.perf_counter()-started);e.save_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    ordered=[rows[eid] for eid in e.IDS];summary=aggregate(ordered)
    features=['hits','mean_graph_degree','duplicate_cell_fraction','longitudinal_ambiguity_fraction']
    complexity=np.zeros(len(ordered))
    for name in features:
        values=np.array([r['observable'][name] for r in ordered]);complexity+=np.array([(values<v).mean() for v in values])/len(features)
    cutoff=float(np.quantile(complexity,.75));complex_ids=[r['event_id'] for r,c in zip(ordered,complexity) if c>=cutoff]
    summary.update(development=aggregate(ordered[:100]),check=aggregate(ordered[100:]),
                   high_occupancy_proxy=dict(features=features,cutoff=cutoff,event_ids=complex_ids,results=aggregate([r for r in ordered if r['event_id'] in complex_ids])),
                   seconds=time.perf_counter()-started,plan=plan,
                   interpretation='Oracle interventions measure sensitivity of the fixed axial extractor; not achievable performance or an independent causal partition.')
    e.verify_pins();e.save_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
