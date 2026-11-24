"""Independent E lane expansion and T edge graph; common bounded geometric extraction."""
from pathlib import Path
from collections import Counter
from functools import lru_cache
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Data
from common import save_json
from two_stage_core import fit_candidate,split_component,hierarchy_partition,score
from STEP5_两级寻迹 import axial_two_pass,attach_stereo,score_tracks,DEFAULT_CONFIG,circle_wire_lower_bound
from coarse_3d_membership import propose,select


def E_edges(f,prob,copy=True):
    """Color->lane count->distinct original hit-pairs; no T score is used."""
    endpoints=f['bundle_endpoints'];mapping=f['edge_bundle'];pairs=f['pairs'];p=f['p'];weights=np.zeros(len(p),np.float32)
    order=np.argsort(mapping,kind='stable');unique,starts=np.unique(mapping[order],return_index=True);parts=np.split(order,starts[1:])
    for bundle,indices in zip(unique,parts):
        if bundle<0:continue
        label=int(prob[bundle].argmax());validity=1-float(prob[bundle,0])
        if label==0 or validity<.5:continue
        lanes=min(label if copy else 1,3)
        u,v=endpoints[:,bundle];family=f['family'];left=np.flatnonzero(family==u);right=np.flatnonzero(family==v)
        a={int(node):i for i,node in enumerate(left)};b={int(node):i for i,node in enumerate(right)}
        costs=np.full((len(left),len(right)),1e4);edge=np.full_like(costs,-1,dtype=int)
        for k in indices:
            i,j=map(int,pairs[:,k])
            if family[i]!=u:i,j=j,i
            costs[a[i],b[j]]=-np.log(max(float(p[k]),1e-6));edge[a[i],b[j]]=int(k)
        lanes=min(lanes,len(left),len(right));chosen=[]
        while lanes:
            size=len(left)+len(right)-lanes;matrix=np.zeros((size,size));matrix[:len(left),:len(right)]=costs
            matrix[len(left):,len(right):]=1e4
            rr,cc=linear_sum_assignment(matrix)
            chosen=[int(edge[i,j]) for i,j in zip(rr,cc) if i<len(left) and j<len(right) and edge[i,j]>=0]
            if len(chosen)==lanes:break
            lanes-=1
        weights[chosen]=validity
    keep=weights>0
    labels=connected_components(coo_matrix((np.ones(keep.sum()),pairs[:,keep]),shape=(len(f['raw']),len(f['raw']))),directed=False)[1]
    groups=[np.flatnonzero(labels==k).tolist() for k in np.unique(labels) if (labels==k).sum()>=5]
    return groups,weights


def T_groups(f,t,q,threshold=.9):
    keep=q>=threshold;links=t[keep,3:5].T;n=f['pairs'].shape[1]
    labels=connected_components(coo_matrix((np.ones(links.shape[1]),links),shape=(n,n)),directed=False)[1]
    confidence=np.zeros(n,np.float32)
    if keep.any():np.maximum.at(confidence,t[keep,3],q[keep]);np.maximum.at(confidence,t[keep,4],q[keep])
    indices=np.flatnonzero(confidence>=threshold);groups=[]
    for label in np.unique(labels[indices]):
        edges=indices[labels[indices]==label];nodes=np.unique(f['pairs'][:,edges]);
        if len(nodes)>=5:groups.append(nodes.tolist())
    return groups,confidence


def geometry_extract(groups,weights,f,geo,drift,baseline):
    stereo=f['stereo'];axial=np.flatnonzero(~stereo);mapping=np.full(len(stereo),-1,dtype=int);mapping[axial]=np.arange(len(axial))
    aa=(mapping[f['pairs'][0]]>=0)&(mapping[f['pairs'][1]]>=0)
    data=Data(wire_x=torch.tensor(geo[axial,0]),wire_y=torch.tensor(geo[axial,1]),drift_d=torch.tensor(drift[axial]),
              edge_index=torch.tensor(mapping[f['pairs'][:,aa]]).long(),num_nodes=len(axial))
    probabilities=weights[aa];candidates=[dict(t,indices=list(t['indices']),origin='baseline_soft_candidate') for t in baseline]
    for group in groups:
        ix=sorted(set(mapping[np.asarray(group,dtype=int)])-{-1})
        if not 5<=len(ix)<=80:continue
        values=hierarchy_partition(data,ix,probabilities,.7) or split_component(data,ix,probabilities,.5,1.5)
        for value in values:value=dict(value,indices=axial[value['indices']].tolist(),origin='new_candidate');candidates.append(value)
    # All candidates compete together. No strict-pass locks before relaxed evidence.
    ss=np.flatnonzero(stereo);matrix=np.zeros((len(stereo),len(stereo)),np.float32)
    matrix[f['pairs'][0],f['pairs'][1]]=f['p'];matrix[f['pairs'][1],f['pairs'][0]]=f['p']
    for c in candidates:
        ix=np.asarray(c['indices']);support=np.zeros(len(ss),bool)
        if len(ss):support=((matrix[np.ix_(ss,ix)]>=.7).sum(1)>=2)&(circle_wire_lower_bound(c['fit'],geo[ss])<=drift[ss]+2.)
        c['selection_utility']=float(score(c)+.05*min(len(np.unique(f['cells'][ss[support]])),15))
    result=[];used=set();seen=set()
    for c in sorted(candidates,key=lambda x:x['selection_utility'],reverse=True):
        signature=tuple(sorted(c['indices']))
        if signature in seen or used.intersection(signature):continue
        seen.add(signature);used.update(signature);result.append(c)
    return result


def seeds(tracks,truth,stereo,cells):
    sizes={int(t):int(((truth==t)&~stereo).sum()) for t in np.unique(truth) if t>=0 and ((truth==t)&~stereo).sum()>=5}
    eligible={t for t in sizes if len(np.unique(cells[(truth==t)&stereo]))>=2};found=set();c=Counter(truth_tracks=len(eligible))
    for track in tracks:
        if not track['coarse_3d']['seed_ready']:continue
        c['output_candidates']+=1;aa=np.array([i for i in track['indices'] if not stereo[i]],int);ss=np.asarray(track['confirmed_stereo_indices'],int)
        ids,ns=np.unique(truth[aa],return_counts=True);t=int(ids[ns.argmax()]);good=ss[truth[ss]==t]
        okay=t in eligible and ns.max()/len(aa)>=.8 and ns.max()/sizes[t]>=.5 and len(np.unique(cells[good]))>=2 and len(good)/max(len(ss),1)>=.8
        if not okay:c['fake_candidates']+=1
        elif t in found:c['clone_candidates']+=1
        else:found.add(t)
    c['matched_tracks']=len(found);return dict(c)


@lru_cache(maxsize=1)
def geometry(path):return np.load(path)


def evaluate_event(task):
    eid,cache,data_root,prediction,out,coarse=task;cache=Path(cache);data_root=Path(data_root)
    torch.set_num_threads(1)
    with np.load(cache/'features'/('event_%05d.npz'%eid)) as source:f={k:source[k] for k in source.files}
    with np.load(cache/'indices'/('event_%05d.npz'%eid)) as source:t=source['t']
    with np.load(data_root/'events'/('event_%05d.npz'%eid)) as source:drift,truth=source['d'],source['truth']
    with np.load(prediction) as source:ep,tq=source['E'],source['T']
    geo=geometry(str(data_root/'wire_geometry.npy'))[f['cells']];stereo=f['stereo'];pairs,p=f['pairs'],f['p']
    baseline,_=axial_two_pass(geo,drift,pairs,p);methods={'raw_baseline':baseline}
    eg,ew=E_edges(f,ep,copy=True);methods['E']=geometry_extract(eg,ew,f,geo,drift,baseline)
    tg,tw=T_groups(f,t,tq);methods['T']=geometry_extract(tg,tw,f,geo,drift,baseline)
    result={}
    for key,tracks in methods.items():
        joint,_=attach_stereo(tracks,geo,drift,f['cells'],pairs,p,DEFAULT_CONFIG)
        counts,found=score_tracks(joint,truth,stereo,DEFAULT_CONFIG);row=dict(counts=counts,found=found)
        if coarse:
            confirmed,_=select(joint,propose(joint,geo,drift,f['cells'],pairs,p,refine=False),geo,f['cells'],'guarded_loose')
            row['seeds']=seeds(confirmed,truth,stereo,f['cells'])
        result[key]=row
    packet=dict(event_id=eid,methods=result,scope='Raw two-pass reference; same AS/SS attachment/coarse geometry, no old hybrid calibrator',
                prototype_limits='E lane count capped3 and new geometric groups capped80 axial hits; baseline candidates remain available')
    save_json(Path(out),packet);return packet
