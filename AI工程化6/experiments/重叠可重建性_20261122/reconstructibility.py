"""Offline overlap/observability witnesses. Oracle labels are NEVER deployment inputs."""
from pathlib import Path
from collections import Counter
import json
import sys
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linear_sum_assignment

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'复杂事件归因_20261107'))
import experiment_common as e
sys.path.insert(0,str(HERE.parent/'复杂事件归因_20261107/parallel_20261109'))
import STEP7_训练结果与同流程复测 as review
from two_stage_core import fit_2d_circle
from coarse_3d_membership import axial_arc,longitudinal_hypotheses,_geometry
from STEP4_真边分数与连通性诊断 import single_pass

BATCH=HERE.parent.parent/'visual_diagnosis_20261121/joint_events_1500_2000'
OUT=HERE/'results'
GEOMETRY=None


def initialize():
    global GEOMETRY
    e.initialize_worker();GEOMETRY=np.load(e.DATA_ROOT/'wire_geometry.npy')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load(eid):
    with np.load(e.DATA_ROOT/'events'/('event_%05d.npz'%eid)) as f:
        cells,d,truth,hit=f['cell_id'],f['d'],f['truth'],f['hit_id']
    geo=GEOMETRY[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    with np.load(BATCH/'prediction_cache'/('event_%05d.npz'%eid)) as f:pairs,p=f['pairs'],f['probabilities']
    packet=read(next(BATCH.glob('event_%05d_T*_baseline.json'%eid)))
    return cells,d,truth,hit,geo,stereo,pairs,p,packet


def overlap_features(cells,drift,truth,xy,axial):
    aa=np.flatnonzero(axial);t=truth[aa];distance=cdist(xy[aa],xy[aa]);foreign=(t[:,None]!=t[None,:])&(t[None,:]>=0)
    distance=np.where(foreign,distance,np.inf);nearest=distance.min(1)
    shared=(cells[aa,None]==cells[None,aa])&foreign
    delta=np.abs(drift[aa,None]-drift[None,aa])
    float_equal=(np.float32(drift[aa,None]/10)==np.float32(drift[None,aa]/10))&shared
    result={}
    for tid in np.unique(t[t>=0]):
        local=np.flatnonzero(t==tid);indices=aa[local]
        candidates=[]
        for other in np.unique(t[(t>=0)&(t!=tid)]):
            separation=distance[np.ix_(local,np.flatnonzero(t==other))].min(1)
            candidates.append((float((separation<=10).mean()),-float(np.median(separation)),int(other)))
        competitor=max(candidates)[2] if candidates else None
        result[int(tid)]=dict(axial_hits=len(indices),unique_axial_cells=len(np.unique(cells[indices])),
            nearest_foreign_median_mm=float(np.median(nearest[local])),primary_competitor=competitor,
            **{'foreign_within_%dmm_fraction'%r:float((nearest[local]<=r).mean()) for r in [2,5,10,20]},
            shared_cell_fraction=float(shared[local].any(1).mean()),
            exact_input_equivalent_fraction=float(float_equal[local].any(1).mean()),
            **{'shared_cell_drift_within_%smm_fraction'%str(r).replace('.','p'):float((shared[local]&(delta[local]<=r)).any(1).mean()) for r in [.1,.5]})
    return result


def labels_at(n,pairs,keep):
    selected=pairs[:,keep]
    adjacency=coo_matrix((np.ones(selected.shape[1]),selected),shape=(n,n))
    return connected_components(adjacency,directed=False,return_labels=True)[1]


def light(eid):
    cells,d,truth,hit,geo,stereo,pairs,p,packet=load(eid)
    features=overlap_features(cells,d,truth,geo[:,:2],~stereo)
    same=(truth[pairs[0]]==truth[pairs[1]])&(truth[pairs[0]]>=0)
    aa=~stereo[pairs[0]]&~stereo[pairs[1]]
    labelsets={name:labels_at(len(cells),pairs,same&aa&(p>=cut)) for name,cut in [('all',0.),('relaxed',.575),('strict',.7)]}
    missed=set(packet['missed_axial_truth_ids']);seedmiss=set(packet['missing_3d_only_truth_ids']);rows=[]
    for tid,feature in features.items():
        if feature['axial_hits']<5:continue
        ix=np.flatnonzero((truth==tid)&~stereo);true_edges=same&aa&(truth[pairs[0]]==tid)
        wrong=aa&~same&((truth[pairs[0]]==tid)|(truth[pairs[1]]==tid));scores=p[true_edges]
        need=max(5,int(np.ceil(len(ix)*.5)))
        components={name:max(Counter(labels[ix]).values()) for name,labels in labelsets.items()}
        status='candidate_graph_insufficient' if components['all']<need else 'pure_connected_at_relaxed_cut' if components['relaxed']>=need else 'score_fragmented'
        touching=[g for g in packet['groups'] if set(g['indices'])&set(ix.tolist())]
        best=max(touching,key=lambda g:len(set(g['indices'])&set(ix.tolist()))) if touching else None
        rows.append(dict(event_id=eid,truth_id=tid,partition=packet['partition'],found=tid not in missed,
            seed_only_miss=tid in seedmiss,event_hits=len(cells),**feature,true_AA_edges=int(true_edges.sum()),
            true_AA_mean=float(scores.mean()) if len(scores) else None,
            true_AA_recall_0575=float((scores>=.575).mean()) if len(scores) else 0.,
            false_AA_above_0575=int((wrong&(p>=.575)).sum()),false_AA_above_070=int((wrong&(p>=.7)).sum()),
            largest_true_component=components,needed_true_hits=need,graph_diagnosis=status,
            best_overlap_group=({k:best[k] for k in ['number','truth','purity','completeness','status','rms']} if best else None)))
    result=dict(event_id=eid,tracks=rows);e.save_json(OUT/'survey'/('event_%05d.json'%eid),result);return result


def reconstruction(geo,d,cells,pairs,prob,truth,stereo):
    aa,_=review.axial_two_pass(geo,d,pairs,prob)
    attached,_=review.attach_stereo(aa,geo,d,cells,pairs,prob,review.DEFAULT_CONFIG)
    final=review.track(geo,d,cells,pairs,prob,aa,attached,review._CALIBRATOR,.02,.2)
    counts,found=review.score_tracks(final,truth,stereo,review.DEFAULT_CONFIG)
    return dict(counts=counts,found=found,groups=[dict(indices=t['indices'],fit=t['fit'],stage=t.get('stage')) for t in final])


def physical_model(tid,cells,d,truth,geo,stereo):
    aa=np.flatnonzero((truth==tid)&~stereo);ss=np.flatnonzero((truth==tid)&stereo)
    fit=fit_2d_circle(geo[aa,0],geo[aa,1],d[aa])
    result=dict(truth_id=int(tid),axial_hits=len(aa),stereo_hits=len(ss),distinct_stereo_cells=len(np.unique(cells[ss])),fit=fit)
    if fit is None:return dict(result,status='no_circle',hypotheses=[])
    residual=np.abs(np.linalg.norm(geo[aa,:2]-fit[:2],axis=1)-fit[2])-d[aa]
    result.update(axial_rms_mm=float(np.sqrt(np.mean(residual**2))),axial_inlier_1p5_fraction=float((np.abs(residual)<=1.5).mean()))
    arc=axial_arc(fit,geo[aa,:2]);result['arc']=arc
    if len(ss)<2:return dict(result,status='fewer_than_two_stereo_hits',hypotheses=[])
    hypotheses,status=longitudinal_hypotheses(fit,arc,geo[ss],d[ss],cells[ss],np.ones(len(ss)),refine=False,axial_geo=geo[aa])
    result['status']=status;result['hypotheses']=[{k:v for k,v in h.items() if k!='residual'} for h in hypotheses]
    if hypotheses:
        best=max(hypotheses,key=lambda h:h['quality']);result['best_hypothesis']={k:v for k,v in best.items() if k!='residual'}
        inlier=np.abs(best['residual'])<=1.5
        result.update(stereo_inlier_1p5_fraction=float(inlier.mean()),stereo_median_abs_residual_mm=float(np.median(np.abs(best['residual']))),
                      supported_distinct_stereo_cells=len(np.unique(cells[ss[inlier]])))
    return result


def metrics_for_assignment(matrix,model_ids,truth,indices,tid):
    k=model_ids.index(tid);own=matrix[indices,k];others=np.delete(matrix[indices],k,axis=1).min(1)
    finite=np.isfinite(own);margin=others-own;pred=np.argmin(matrix[indices],axis=1)
    return dict(n=len(indices),finite_own=int(finite.sum()),own_best_fraction=float(((pred==k)&finite).mean()) if len(indices) else None,
                median_competitor_minus_own_mm=float(np.median(margin[finite])) if finite.any() else None,
                own_better_by_0p2mm_fraction=float(((margin>.2)&finite).mean()) if len(indices) else None)


def detail(eid):
    started=time.time();review.initialize()
    cells,d,truth,hit,geo,stereo,pairs,p,packet=load(eid)
    rows=read(OUT/'survey'/('event_%05d.json'%eid))['tracks'];model_ids=sorted(r['truth_id'] for r in rows)
    same=(truth[pairs[0]]==truth[pairs[1]])&(truth[pairs[0]]>=0)
    trials={}
    for name,prob in [('remove_false_edges',np.where(same,p,0.)),('perfect_candidate_edge_labels',same.astype(np.float32))]:
        trials[name]=reconstruction(geo,d,cells,pairs,prob,truth,stereo)
    pure_single=single_pass(geo,d,pairs,np.where(same,p,0.))
    counts,found=review.score_tracks(pure_single,truth,stereo,review.DEFAULT_CONFIG)
    trials['remove_false_and_nonlocking_single_pass']=dict(counts=counts,found=found)
    models=[physical_model(t,cells,d,truth,geo,stereo) for t in model_ids]
    circle=np.full((len(cells),len(models)),np.inf);three=np.full_like(circle,np.inf)
    for j,model in enumerate(models):
        fit=model.get('fit')
        if fit is None:continue
        circle[~stereo,j]=np.abs(np.abs(np.linalg.norm(geo[~stereo,:2]-fit[:2],axis=1)-fit[2])-d[~stereo])
        if model.get('best_hypothesis'):
            h=model['best_hypothesis'];three[:,j]=np.abs(_geometry.distances(fit,[h['z0'],h['pitch']],model['arc'],geo)-d)
    # Templates are truth-seeded, but this assignment itself sees only measured residuals.
    # It is a conditional separability probe, not a blind track-finder or a test score.
    mixed=three.copy();fallback=~np.isfinite(mixed)&~stereo[:,None];mixed[fallback]=circle[fallback]
    for name,matrix in [('oracle_circle_template_assignment',circle),('oracle_3d_template_assignment',mixed)]:
        owner=matrix.argmin(1);minimum=matrix[np.arange(len(cells)),owner]
        groups=[dict(indices=np.flatnonzero((owner==j)&(minimum<=1.5)).tolist()) for j in range(len(models))]
        counts,found=review.score_tracks(groups,truth,stereo,review.DEFAULT_CONFIG)
        trials[name]=dict(counts=counts,found=found,truth_seeded_templates=True,per_hit_truth_used_for_assignment=False)
    for name,matrix in [('oracle_circle_cell_matching',circle),('oracle_3d_cell_matching',mixed)]:
        # Conditional single-arc probe: within one cell, each template has one slot.
        # This constraint is explicitly invalid for a track re-entering the same wire.
        owner=np.full(len(cells),-1,dtype=int)
        for cell in np.unique(cells):
            ix=np.flatnonzero(cells==cell);cost=np.minimum(matrix[ix]**2,1e12)
            cost=np.column_stack([cost,np.full((len(ix),len(ix)),1.5**2)])
            rr,cc=linear_sum_assignment(cost)
            for a,b in zip(rr,cc):
                if b<len(models) and cost[a,b]<1.5**2:owner[ix[a]]=b
        groups=[dict(indices=np.flatnonzero(owner==j).tolist()) for j in range(len(models))]
        counts,found=review.score_tracks(groups,truth,stereo,review.DEFAULT_CONFIG)
        repeated=sum(len(np.unique(cells[truth==tid]))<int((truth==tid).sum()) for tid in model_ids)
        trials[name]=dict(counts=counts,found=found,truth_seeded_templates=True,per_hit_truth_used_for_assignment=False,
                         assumption='At most one hit of a single-arc track per cell',truth_tracks_violating_assumption=repeated)
    focuses=[r['truth_id'] for r in rows if not r['found'] or r['seed_only_miss']]
    focusrows=[]
    for tid in focuses:
        row=next(r for r in rows if r['truth_id']==tid);model=models[model_ids.index(tid)]
        aa=np.flatnonzero((truth==tid)&~stereo);ss=np.flatnonzero((truth==tid)&stereo)
        result=dict(row,physical=model,oracle_remove_false_found=tid in trials['remove_false_edges']['found'],
                    oracle_perfect_graph_found=tid in trials['perfect_candidate_edge_labels']['found'],
                    oracle_denoised_nonlocking_found=tid in trials['remove_false_and_nonlocking_single_pass']['found'],
                    oracle_circle_assignment_found=tid in trials['oracle_circle_template_assignment']['found'],
                    oracle_3d_assignment_found=tid in trials['oracle_3d_template_assignment']['found'],
                    oracle_3d_cell_matching_found=tid in trials['oracle_3d_cell_matching']['found'],
                    axial_circle_template_assignment=metrics_for_assignment(circle,model_ids,truth,aa,tid),
                    stereo_3d_template_assignment=metrics_for_assignment(three,model_ids,truth,ss,tid))
        if model.get('best_hypothesis'):
            result['all_hit_3d_template_assignment']=metrics_for_assignment(three,model_ids,truth,np.r_[aa,ss],tid)
        # These are witnesses under the current geometry assumptions, not a proof of identifiability.
        result['has_axial_reconstruction_witness']=bool(result['oracle_perfect_graph_found'] or (model.get('axial_rms_mm',np.inf)<=1.5 and len(aa)>=5))
        result['has_3d_geometry_witness']=bool(model.get('supported_distinct_stereo_cells',0)>=3 and model.get('axial_inlier_1p5_fraction',0)>=.5)
        focusrows.append(result)
    np.savez_compressed(OUT/'details'/('event_%05d_residuals.npz'%eid),circle=circle,three_d=three,model_ids=model_ids,truth=truth,stereo=stereo,cells=cells,drift=d)
    result=dict(event_id=eid,partition=packet['partition'],baseline_counts=packet['joint_metrics'],missed_axial=packet['missed_axial_truth_ids'],
        seed_only_miss=packet['missing_3d_only_truth_ids'],focus=focusrows,models=models,oracle_trials=trials,seconds=time.time()-started,
        limitations=['Truth-grouped fitted templates and oracle edge labels are offline witnesses, not operational reconstruction efficiency.',
                    'No exact detector resolution/noise model was supplied; 0.1/0.2/0.5/1.5mm values are diagnostic thresholds.',
                    'Failure of the single-circle and bounded longitudinal search is not proof that a physical track is irreconstructible.',
                    'Minimum3 distinct stereo supports checks the current seed validation; two hits can give multiple left/right solutions.'])
    e.save_json(OUT/'details'/('event_%05d.json'%eid),result);return result
