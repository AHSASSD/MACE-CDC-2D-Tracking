"""Development diagnostic: scores at shared-wire/branch interfaces and T bottlenecks.

Truth is used only to label audit categories and connectivity upper bounds.
No training or stable deployment change.
"""
from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DATA=ROOT.parent.parent/'joint_event_dataset'
OLD=ROOT.parent.parent/'visual_diagnosis_20261121/joint_events_1500_2000'
OUT=ROOT/'junction_audit_20261122'
IDS=[1500,1501,1503,1504]

def components(n,pairs,keep):
    x=pairs[:,keep]
    return connected_components(coo_matrix((np.ones(x.shape[1]),x),shape=(n,n)),directed=False)[1]

def largest(labels,ix):return int(np.unique(labels[ix],return_counts=True)[1].max())

def stats(values):
    a=np.asarray(values)
    return dict(edges=len(a),median=float(np.median(a)) if len(a) else None,below04=int((a<.4).sum()),below0575=int((a<.575).sum()),above07=int((a>=.7).sum()))

def categories(cells,truth,geo,stereo):
    aa=np.flatnonzero(~stereo);shared_cells=[]
    for cell in np.unique(cells[aa]):
        tids=np.unique(truth[aa[cells[aa]==cell]]);tids=tids[tids>=0]
        if len(tids)>=2:shared_cells.append(int(cell))
    exact=np.isin(cells,shared_cells)&~stereo
    pp=cKDTree(geo[aa,:2]).query_pairs(10.,output_type='ndarray')
    pp=pp[(truth[aa[pp[:,0]]]!=truth[aa[pp[:,1]]])&(truth[aa[pp[:,0]]]>=0)&(truth[aa[pp[:,1]]]>=0)]
    near=np.zeros(len(cells),bool);near[aa[pp.ravel()]]=True
    return dict(shared_CellID=exact,foreign_within10mm=near)

def main():
    import torch
    from common import save_json
    from et_data import prepare,Features,to_device
    from et_models import THead,predict
    from et_decode import T_groups
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    weight=ROOT.parent.parent.parent/'AI工程化6_稳定基线备份/AI6_稳定基线_20261107_230240/AI工程化6/joint_training/local_evaluation_epoch126_20261107/best_interaction_network_joint_no_time.pth'
    prepare(DATA,weight,OUT/'cache',IDS)
    model=THead().cuda();model.load_state_dict(torch.load(ROOT/'evaluation_download/full/T_best_members.pth',map_location='cuda',weights_only=True));model.eval()
    ds=Features(OUT/'cache',IDS);geo_all=np.load(DATA/'wire_geometry.npy');rows=[];scores={}
    for j,eid in enumerate(IDS):
        with np.load(DATA/'events'/('event_%05d.npz'%eid)) as s:cells=s['cell_id'];truth=s['truth']
        geo=geo_all[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9;n=len(cells)
        with np.load(OLD/'prediction_cache'/('event_%05d.npz'%eid)) as s:pairs=s['pairs'];p=s['probabilities']
        aa=~stereo[pairs[0]]&~stereo[pairs[1]];same=(truth[pairs[0]]==truth[pairs[1]])&(truth[pairs[0]]>=0);true=aa&same
        missed=json.loads(next(OLD.glob('event_%05d_T*_baseline.json'%eid)).read_text(encoding='utf-8'))['missed_axial_truth_ids']
        b=ds[j]
        with torch.no_grad(),torch.autocast('cuda'):q=predict(model,to_device(b,'cuda'),'T')
        f={k:v.numpy() if torch.is_tensor(v) else v for k,v in b.items()};t=f['t'];tg,confidence=T_groups(f,t,q)
        # Actual T edge-graph preserves triplet links here; selected true-only links audit conditional continuity.
        tq_true=(truth[t[:,0]]==truth[t[:,1]])&(truth[t[:,1]]==truth[t[:,2]])&(truth[t[:,1]]>=0)
        links=t[(q>=.9)&tq_true,3:5].T;ne=f['pairs'].shape[1]
        edge_labels=connected_components(coo_matrix((np.ones(links.shape[1]),links),shape=(ne,ne)),directed=False)[1]
        definitions=categories(cells,truth,geo,stereo)
        supported_by_cut={}
        for cut in [.5,.7,.8,.9,.95]:
            choose=q>=cut;ll=t[choose&tq_true,3:5].T
            labels=connected_components(coo_matrix((np.ones(ll.shape[1]),ll),shape=(ne,ne)),directed=False)[1]
            enough=[]
            for tid in missed:
                own=np.flatnonzero((truth[f['pairs'][0]]==tid)&(truth[f['pairs'][1]]==tid));best_size=0
                for label in np.unique(labels[own]):
                    nodes=np.unique(f['pairs'][:,own[labels[own]==label]])
                    best_size=max(best_size,int(((truth[nodes]==tid)&~stereo[nodes]).sum()))
                if best_size>=max(5,int(np.ceil(.5*((truth==tid)&~stereo).sum()))):enough.append(int(tid))
            supported_by_cut[str(cut)]=dict(enough_missed_truth_ids=enough,selected_triplets=int(choose.sum()),false_selected_triplets=int((choose&~tq_true).sum()))
        retained_true=(truth[f['pairs'][0]]==truth[f['pairs'][1]])&(truth[f['pairs'][0]]>=0)&~stereo[f['pairs'][0]]&~stereo[f['pairs'][1]]
        cache_labels=components(n,f['pairs'],retained_true)
        trackrows=[];definition_rows={}
        for name,node_mask in definitions.items():
            u=node_mask[pairs[0]];v=node_mask[pairs[1]]
            masks=dict(shared_core=u&v,interface=u^v,nonshared_core=~u&~v)
            definition_rows[name]={k:stats(p[true&m]) for k,m in masks.items()}
            scores.setdefault(name,{k:[] for k in masks})
            for k,m in masks.items():scores[name][k].extend(p[true&m].tolist())
            native=components(n,pairs,true&(p>=.575));all04=components(n,pairs,true&(p>=.4));allgraph=components(n,pairs,true)
            promoted=components(n,pairs,true&((p>=.575)|masks['interface']))
            core=components(n,pairs,true&((p>=.575)|masks['shared_core']))
            for tid in missed:
                ix=np.flatnonzero((truth==tid)&~stereo);need=max(5,int(np.ceil(.5*len(ix))))
                groups=[g for g in tg if np.intersect1d(g,ix).size]
                best=max(groups,key=lambda g:np.intersect1d(g,ix).size) if groups else []
                true_edges=np.flatnonzero((truth[f['pairs'][0]]==tid)&(truth[f['pairs'][1]]==tid))
                best_true_T=0
                for label in np.unique(edge_labels[true_edges]):
                    nodes=np.unique(f['pairs'][:,true_edges[edge_labels[true_edges]==label]])
                    best_true_T=max(best_true_T,int(((truth[nodes]==tid)&~stereo[nodes]).sum()))
                local=true&(truth[pairs[0]]==tid)
                trackrows.append(dict(definition=name,truth_id=int(tid),axial_hits=len(ix),shared_hits=int(node_mask[ix].sum()),need=need,
                    pure_native0575=largest(native,ix),pure_native04=largest(all04,ix),pure_all_candidate=largest(allgraph,ix),
                    oracle_promote_interface=largest(promoted,ix),oracle_promote_shared_core=largest(core,ix),
                    true_interface_edges=stats(p[local&masks['interface']]),
                    T_cache_pure_edge_largest=largest(cache_labels,ix),T_true_triplet_largest=best_true_T,T_group_best_true_hits=int(np.intersect1d(best,ix).size),
                    T_group_best_total_axial=int((~stereo[best]).sum()) if len(best) else 0,
                    T_group_skipped_above80=bool(len(best) and (~stereo[best]).sum()>80)))
        rows.append(dict(event_id=eid,missed_truth_ids=missed,edge_categories=definition_rows,tracks=trackrows,triplet_threshold_audit=supported_by_cut))
        print('audited',eid,flush=True)
    pooled={name:{k:stats(v) for k,v in values.items()} for name,values in scores.items()}
    summary={}
    for name in scores:
        tracks=[dict(r,event_id=x['event_id']) for x in rows for r in x['tracks'] if r['definition']==name]
        fragmented=[r for r in tracks if r['pure_native0575']<r['need']<=r['pure_all_candidate']]
        summary[name]=dict(missed=len(tracks),pure_graph_fragmented_at0575=len(fragmented),
            rescued_by_oracle_interface_only=[[r['event_id'],r['truth_id']] for r in fragmented if r['oracle_promote_interface']>=r['need']],
            rescued_by_oracle_shared_core_only=[[r['event_id'],r['truth_id']] for r in fragmented if r['oracle_promote_shared_core']>=r['need']],
            still_insufficient_at04=[[r['event_id'],r['truth_id']] for r in tracks if r['pure_native04']<r['need']],
            T_true_triplet_insufficient=[[r['event_id'],r['truth_id']] for r in tracks if r['T_true_triplet_largest']<r['need']],
            T_best_group_skipped80=[[r['event_id'],r['truth_id']] for r in tracks if r['T_group_skipped_above80']])
    threshold_summary={}
    for cut in ['0.5','0.7','0.8','0.9','0.95']:
        entries=[r['triplet_threshold_audit'][cut] for r in rows]
        threshold_summary[cut]=dict(enough_missed_tracks=sum(len(e['enough_missed_truth_ids']) for e in entries),selected_triplets=sum(e['selected_triplets'] for e in entries),false_selected_triplets=sum(e['false_selected_triplets'] for e in entries))
    result=dict(scope='4 representative development events; audit categories and score promotions use truth and are not inference results. Shared wire and 10mm proximity are proxies, not ordered shared-path boundary labels.',pooled_scores=pooled,summary=summary,triplet_threshold_summary=threshold_summary,events=rows)
    save_json(OUT/'audit.json',result);print(json.dumps(dict(pooled=pooled,summary=summary,triplet_threshold_summary=threshold_summary),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
