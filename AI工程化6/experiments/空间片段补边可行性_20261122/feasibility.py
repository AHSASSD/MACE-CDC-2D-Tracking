"""Observable-only segment-link proposals followed by frozen T25 audit.
Truth is confined to reporting purity/connectivity upper bounds, never proposals.
"""
from pathlib import Path
import sys,os,json,time
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;BASE=EXP/'双路线寻迹_20261122';PATHS=EXP/'三元组路径状态_20261122'
sys.path[:0]=[str(PATHS),str(BASE)]
os.environ['MKL_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2
from torch_geometric.data import Data
from common import JointInteractionNetwork,JointEventDataset,save_json,sha
from et_models import THead,predict
from path_decode import fit_path,Config

DATA=EXP.parent/'joint_event_dataset';OUT=HERE/'results_v4';NMAX=128

def labels(n,pairs,keep):
    e=pairs[:,keep];return connected_components(coo_matrix((np.ones(e.shape[1]),e),shape=(n,n)),directed=False)[1]

def endpoints(ix,xy,fit=None):
    if len(ix)<=2:return list(ix)
    if fit is not None:
        angle=np.mod(np.arctan2(xy[ix,1]-fit[1],xy[ix,0]-fit[0]),2*np.pi);ordered=np.argsort(angle);a=angle[ordered]
        split=int(np.argmax(np.diff(np.r_[a,a[0]+2*np.pi])));order=np.r_[ordered[split+1:],ordered[:split+1]]
    else:
        v=np.linalg.eigh(np.cov(xy[ix].T))[1][:,-1];order=np.argsort(xy[ix]@v)
    return list(np.unique(np.r_[ix[order[:2]],ix[order[-2:]]]))

def spatial_candidates(cells,xy,aa,relaxed,cluster=None):
    candidates={}
    neighborhoods=cKDTree(xy[aa]).query_ball_point(xy[aa],80.)
    for i,near in enumerate(neighborhoods):
        u=int(aa[i]);vv=aa[np.asarray(near,int)]
        keep=(relaxed[vv]!=relaxed[u])&(cells[vv]!=cells[u])
        if cluster is not None:keep&=cluster[vv]==cluster[u]
        vv=vv[keep];dist=np.linalg.norm(xy[vv]-xy[u],axis=1);order=np.argsort(dist)
        for j in order[:2]:
            if dist[j]<=1:continue
            v=int(vv[j]);key=tuple(sorted([u,v]));candidates[key]=min(candidates.get(key,np.inf),float(dist[j]))
    return [e for e,c in sorted(candidates.items(),key=lambda x:(x[1],x[0]))[:NMAX]]

def geometric_candidates(cells,xy,aa,strict,relaxed,pairs,p,d,max_short=7,interior=False):
    groups=[np.flatnonzero((strict==k)&np.isin(np.arange(len(cells)),aa)) for k in np.unique(strict[aa])]
    groups=[g for g in groups if len(g)];cores=[]
    data=Data(wire_x=torch.tensor(xy[:,0]),wire_y=torch.tensor(xy[:,1]),drift_d=torch.tensor(d),edge_index=torch.tensor(pairs).long(),num_nodes=len(cells))
    for g in groups:
        if not 5<=len(g)<=80:continue
        c=fit_path(data,g.tolist(),p,Config(max_fit_evaluations=60))
        if c and c['accepted_geometry']:cores.append((g,c['fit'],endpoints(g,xy,c['fit'])))
    candidates={}
    for g in groups:
        if len(g)>max_short:continue
        proposals=[]
        for core,fit,ends in cores:
            if len(core)<len(g) or strict[core[0]]==strict[g[0]] or relaxed[core[0]]==relaxed[g[0]]:continue
            residual=np.abs(np.abs(np.linalg.norm(xy[g]-fit[:2],axis=1)-fit[2])-d[g])
            if (residual<=1.5).sum()<min(len(g),max(2,int(np.ceil(.6*len(g))))):continue
            se=endpoints(g,xy);pool=[];targets=core if interior else ends
            for u in se:
                for v in targets:
                    distance=float(np.linalg.norm(xy[u]-xy[v]))
                    if not 1<distance<=80 or cells[u]==cells[v]:continue
                    radial=xy[v]-fit[:2];tangent=np.array([-radial[1],radial[0]]);gap=xy[u]-xy[v]
                    alignment=abs(float(tangent@gap))/max(np.linalg.norm(tangent)*distance,1e-9)
                    if distance>15 and alignment<.6:continue
                    cost=distance/80+float(np.median(residual))/1.5+.3*(1-alignment)
                    pool.append((cost,tuple(sorted([int(u),int(v)]))))
            if pool:proposals.append((min(x[0] for x in pool),sorted(pool)[:2]))
        # Retain alternatives rather than assigning each short segment one parent.
        for _,pool in sorted(proposals,key=lambda x:x[0])[:2]:
            for cost,e in pool:candidates[e]=min(candidates.get(e,np.inf),cost)
    return [e for e,c in sorted(candidates.items(),key=lambda x:(x[1],x[0]))[:NMAX]],len(groups),len(cores)

def aux_triplets(proposals,pairs,p,xy):
    lookup={tuple(map(int,e)):k for k,e in enumerate(pairs.T)};incident=[[] for _ in range(len(xy))]
    for k,(a,b) in enumerate(pairs.T):
        if p[k]>=.7:incident[a].append(k);incident[b].append(k)
    rows=[];owners=[]
    for owner,(u,v) in enumerate(proposals):
        e=lookup.get((u,v))
        if e is None:continue
        for a,c in [(u,v),(v,u)]:
            anchors=sorted(incident[c],key=lambda k:-p[k])[:4]
            for j in anchors:
                b=int(pairs[1,j] if pairs[0,j]==c else pairs[0,j])
                if b==a:continue
                delta1=xy[c]-xy[a];delta2=xy[b]-xy[c]
                if delta1@delta2<0:continue
                rows.append([a,c,b,e,j]);owners.append(owner)
    return np.asarray(rows,np.int32).reshape(-1,5),np.asarray(owners,int)

def joint_segment_candidates(cells,xy,aa,strict,relaxed,pairs,p,d):
    groups=[np.flatnonzero((strict==k)&np.isin(np.arange(len(cells)),aa)) for k in np.unique(strict[aa])];groups=[g for g in groups if 2<=len(g)<=30]
    data=Data(wire_x=torch.tensor(xy[:,0]),wire_y=torch.tensor(xy[:,1]),drift_d=torch.tensor(d),edge_index=torch.tensor(pairs).long(),num_nodes=len(cells));ranked=[]
    for i,a in enumerate(groups):
        if len(a)>10:continue
        for j,b in enumerate(groups):
            if i==j or len(b)<max(3,len(a)) or len(a)+len(b)<5 or relaxed[a[0]]==relaxed[b[0]]:continue
            distances=np.linalg.norm(xy[a,None]-xy[None,b],axis=2);u,v=np.unravel_index(distances.argmin(),distances.shape);gap=float(distances[u,v])
            if not 1<gap<=80:continue
            ta=np.linalg.eigh(np.cov(xy[a].T))[1][:,-1];tb=np.linalg.eigh(np.cov(xy[b].T))[1][:,-1];cos=abs(float(ta@tb))
            if cos<.75:continue
            ranked.append((gap/80+.3*(1-cos),i,j))
    candidates={};seen=set();fits=0
    for cost,i,j in sorted(ranked):
        if tuple(sorted([i,j])) in seen:continue
        seen.add(tuple(sorted([i,j])))
        if fits>=64:break
        a,b=groups[i],groups[j];fits+=1;fit=fit_path(data,np.r_[a,b].tolist(),p,Config(max_fit_evaluations=60))
        if not fit or not fit['accepted_geometry']:continue
        pairs_here=[]
        for u in a:
            for v in b:
                dist=np.linalg.norm(xy[u]-xy[v])
                if cells[u]!=cells[v] and 1<dist<=80:pairs_here.append((float(dist),tuple(sorted([int(u),int(v)]))))
        for dist,e in sorted(pairs_here)[:2]:candidates[e]=min(candidates.get(e,np.inf),cost+.2*fit['rms'])
    return [e for e,c in sorted(candidates.items(),key=lambda x:(x[1],x[0]))[:NMAX]],fits

def measure(cells,truth,stereo,pairs,p,proposals,qmax,missed):
    n=len(cells);same=(truth[pairs[0]]==truth[pairs[1]])&(truth[pairs[0]]>=0)
    lookup={tuple(map(int,e)):float(s) for e,s in zip(pairs.T,p)};prop=np.asarray(proposals,int).reshape(-1,2)
    good=(truth[prop[:,0]]==truth[prop[:,1]])&(truth[prop[:,0]]>=0) if len(prop) else np.zeros(0,bool)
    scores=np.array([lookup.get(tuple(e),-1.) for e in prop]);base=labels(n,pairs,same&(p>=.575));statistics={}
    for key,keep in [('proposals',np.ones(len(prop),bool)),('T09',qmax>=.9),('T05',qmax>=.5),('T02',qmax>=.2)]:
        edges=prop[keep&good].T;union=np.c_[pairs,edges]
        select=np.r_[same&(p>=.575),np.ones(edges.shape[1],bool)];cc=labels(n,union,select);restored=[];full=[]
        for tid in missed:
            ix=np.flatnonzero((truth==tid)&~stereo);need=max(5,int(np.ceil(len(ix)*.5)))
            before=np.unique(base[ix],return_counts=True)[1].max();after=np.unique(cc[ix],return_counts=True)[1].max()
            if before<need<=after:restored.append(int(tid))
            if after==len(ix):full.append(int(tid))
        statistics[key]=dict(edges=int(keep.sum()),true_edges=int((keep&good).sum()),false_edges=int((keep&~good).sum()),restored_match_connectivity=restored,fully_connected_missed=full)
    return dict(candidate_scores=dict(in_existing_T_cache=int((scores>=.4).sum()),below_T_cache04=int(((scores>=0)&(scores<.4)).sum()),absent_from_original_graph=int((scores<0).sum())),steps=statistics,
                proposal_records=[dict(u=int(e[0]),v=int(e[1]),truth_same=bool(ok),original_score=float(s),max_T=float(q)) for e,ok,s,q in zip(prop,good,scores,qmax)])

@torch.no_grad()
def main():
    from threadpoolctl import threadpool_limits
    guard=threadpool_limits(1);torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    development=json.loads((BASE/'evaluation_download/full/contract.json').read_text(encoding='utf-8'))['validation_ids'][:16]
    ids=[1500,1501,1503,1504]+development
    meta=json.loads((DATA/'dataset_metadata_joint_no_time.json').read_text(encoding='utf-8'));sources={s:JointEventDataset(DATA,s) for s in meta['event_ids']};lookup={eid:(ds,j) for ds in sources.values() for j,eid in enumerate(ds.ids)}
    weight=EXP.parent.parent/'AI工程化6_稳定基线备份/AI6_稳定基线_20261107_230240/AI工程化6/joint_training/local_evaluation_epoch126_20261107/best_interaction_network_joint_no_time.pth'
    assert sha(weight)=='5f3f02409399cd62efce5f254be02fa004237fb61a5b4717681cad1ad90cdcac'
    model=JointInteractionNetwork(chunk_size=65536).cuda();model.load_state_dict(torch.load(weight,map_location='cuda',weights_only=True));model.eval()
    teacher=THead().cuda();teacher.load_state_dict(torch.load(BASE/'evaluation_download/full/T_best_members.pth',map_location='cuda',weights_only=True));teacher.eval()
    geometry=np.load(DATA/'wire_geometry.npy');summary=[]
    for eid in ids:
        started=time.perf_counter();ds,j=lookup[eid];g=ds[j];truth=g.truth.numpy();cells=g.cell_id.numpy();stereo=g.stereo.numpy();geo=geometry[cells];xy=geo[:,:2]
        with np.load(DATA/'events'/('event_%05d.npz'%eid)) as s:d=s['d']
        gpu=g.to('cuda');half=gpu.num_edges//2
        with torch.autocast('cuda'):
            h=gpu.x
            for _ in range(model.n_passes-1):h=model._message(h,gpu.edge_index,gpu.edge_attr)
            pieces=[]
            for begin in range(0,gpu.num_edges,65536):pieces.append(model._classify(h,gpu.edge_index[:,begin:begin+65536],gpu.edge_attr[begin:begin+65536]).sigmoid().float().cpu().numpy())
        directed=np.concatenate(pieces);allp=(directed[:half]+directed[half:])/2;allpairs=gpu.edge_index[:,:half].cpu().numpy();aa=~stereo[allpairs[0]]&~stereo[allpairs[1]];pair=allpairs[:,aa];prob=allp[aa];nodes=np.flatnonzero(~stereo)
        strict=labels(len(cells),pair,prob>=.7);relaxed=labels(len(cells),pair,prob>=.575)
        k=max(2,min(64,int(np.ceil(len(nodes)/10))))
        _,km=kmeans2(xy[nodes],k,minit='++',iter=30,seed=20261122);cluster=np.full(len(cells),-1);cluster[nodes]=km
        methods=dict(kmeans=spatial_candidates(cells,xy,nodes,relaxed,cluster),radius=spatial_candidates(cells,xy,nodes,relaxed))
        methods['geometry'],segments,cores=geometric_candidates(cells,xy,nodes,strict,relaxed,pair,prob,d)
        methods['geometry_interior'],_,_=geometric_candidates(cells,xy,nodes,strict,relaxed,pair,prob,d,max_short=80,interior=True)
        methods['geometry_joint'],joint_fits=joint_segment_candidates(cells,xy,nodes,strict,relaxed,pair,prob,d)
        # Coverage reference, NOT subject to128-edge budget and NOT a proposed
        # final policy: retain all original weakAA candidates0.4..0.575.
        methods['weak_retained_reference']=[tuple(map(int,e)) for e in pair[:,(prob>=.4)&(prob<.575)].T]
        if eid in [1500,1501,1503,1504]:
            old=EXP.parent/'visual_diagnosis_20261121/joint_events_1500_2000';missed=json.loads(next(old.glob('event_%05d_T*_baseline.json'%eid)).read_text(encoding='utf-8'))['missed_axial_truth_ids']
        else:
            packet=json.loads((BASE/'evaluation_download/full/validation/epoch_025'/('member_%05d.json'%eid)).read_text(encoding='utf-8'));found=set(packet['methods']['raw_baseline']['found']);missed=[int(t) for t in np.unique(truth[~stereo]) if t>=0 and ((truth==t)&~stereo).sum()>=5 and t not in found]
        triples={};own={}
        for key,edges in methods.items():triples[key],own[key]=aux_triplets(edges,pair,prob,xy)
        # Every<=80mm axial proposal is within the original120mm candidate graph.
        full_lookup={tuple(map(int,e)):int(i) for i,e in enumerate(allpairs.T)}
        needed=set()
        for rows in triples.values():needed.update(rows[:,3:5].ravel().tolist())
        edgeids=np.array(sorted(needed),int);original=np.array([full_lookup[tuple(map(int,pair[:,i]))] for i in edgeids],int);effects=[]
        for sign in [0,half]:
            ix=torch.tensor(original+sign,device='cuda');ei=gpu.edge_index[:,ix]
            with torch.autocast('cuda'):effect=model.relational_model(torch.cat([h[ei[0]],h[ei[1]],gpu.edge_attr[ix]],-1))
            effects.append(effect.float().cpu().numpy())
        latent=(effects[0]+effects[1])/2;mapping=np.full(pair.shape[1],-1,int);mapping[edgeids]=np.arange(len(edgeids))
        f=dict(raw=gpu.x,h=h,p=torch.tensor(prob[edgeids],device='cuda'),latent=torch.tensor(latent,device='cuda'))
        result={}
        for key,rows in triples.items():
            scores=np.zeros(len(methods[key]),np.float32)
            if len(rows):
                indexed=rows.copy();indexed[:,3:5]=mapping[indexed[:,3:5]];f['t']=torch.tensor(indexed,device='cuda')
                with torch.autocast('cuda'):q=predict(teacher,f,'T')
                np.maximum.at(scores,own[key],q)
            result[key]=measure(cells,truth,stereo,pair,prob,methods[key],scores,missed)
            result[key]['generated_triplets']=len(rows)
        record=dict(event_id=eid,scope='Representative development' if eid in [1500,1501,1503,1504] else 'Development control',missed=missed,segments=segments,cores=cores,kmeans_k=k,joint_fits=joint_fits,methods=result,seconds=time.perf_counter()-started)
        save_json(OUT/('event_%05d.json'%eid),record);summary.append(record);print('audited',eid,round(record['seconds'],2),flush=True)
    aggregate={}
    for key in methods:
        aggregate[key]={}
        for step in ['proposals','T09','T05','T02']:
            a=[r['methods'][key]['steps'][step] for r in summary]
            aggregate[key][step]=dict(edges=sum(x['edges'] for x in a),true_edges=sum(x['true_edges'] for x in a),false_edges=sum(x['false_edges'] for x in a),restored_representative=sum(len(r['methods'][key]['steps'][step]['restored_match_connectivity']) for r in summary[:4]),restored_total=sum(len(x['restored_match_connectivity']) for x in a))
        aggregate[key]['edge_provenance']={name:sum(r['methods'][key]['candidate_scores'][name] for r in summary) for name in ['in_existing_T_cache','below_T_cache04','absent_from_original_graph']}
        aggregate[key]['new_triplets']=sum(r['methods'][key]['generated_triplets'] for r in summary)
    save_json(OUT/'summary.json',dict(events=len(ids),representative_missed=sum(len(r['missed']) for r in summary[:4]),total_missed=sum(len(r['missed']) for r in summary),scope='Feasibility only: proposal precision and oracle-pure connectivity; NOT tracking efficiency; no training/deployment change',source_sha256=sha(Path(__file__)),max_aux_edges=NMAX,aggregate=aggregate));print(json.dumps(aggregate),flush=True)

if __name__=='__main__':main()
