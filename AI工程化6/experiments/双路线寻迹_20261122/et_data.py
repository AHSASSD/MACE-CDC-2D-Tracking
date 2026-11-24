"""Frozen-backbone features and separate supervised indices for independent E/T heads."""
from pathlib import Path
import hashlib,inspect,json,os,time
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from common import JointInteractionNetwork,sha,save_json
from server_runtime import dataset

WEIGHT_SHA='5f3f02409399cd62efce5f254be02fa004237fb61a5b4717681cad1ad90cdcac'


def atomic_npz(path,compress=False,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        (np.savez_compressed if compress else np.savez)(stream,**values)
    tmp.replace(path)


def triplets(pairs,n,probability):
    adjacency=[[] for _ in range(n)]
    for k,(a,b) in enumerate(pairs.T):adjacency[int(a)].append(k);adjacency[int(b)].append(k)
    pieces=[];templates={}
    for c,incident in enumerate(adjacency):
        degree=len(incident)
        if degree<2:continue
        ids=np.asarray(incident,dtype=np.int32)
        if degree<=64:
            if degree not in templates:templates[degree]=np.triu_indices(degree,1)
            i,j=templates[degree];u,v=ids[i],ids[j]
        else:
            ordered=ids[np.argsort(-probability[ids],kind='stable')];anchors=ordered[:24];rest=ordered[24:]
            i,j=np.triu_indices(len(anchors),1)
            u=np.r_[anchors[i],np.repeat(rest,8)];v=np.r_[anchors[j],np.tile(anchors[:8],len(rest))]
        a=np.where(pairs[0,u]==c,pairs[1,u],pairs[0,u]);b=np.where(pairs[0,v]==c,pairs[1,v],pairs[0,v])
        pieces.append(np.column_stack([a,np.full(len(u),c),b,u,v]).astype(np.int32))
    return np.concatenate(pieces) if pieces else np.empty((0,5),dtype=np.int32)


def bundles(cells,pairs,truth):
    unique,family=np.unique(cells,return_inverse=True);a,b=family[pairs[0]],family[pairs[1]]
    valid=a!=b;key=np.minimum(a[valid],b[valid])*len(unique)+np.maximum(a[valid],b[valid])
    keys,inverse=np.unique(key,return_inverse=True);edge_bundle=np.full(pairs.shape[1],-1,dtype=np.int32);edge_bundle[valid]=inverse
    endpoints=np.vstack([keys//len(unique),keys%len(unique)]).astype(np.int32)
    good=valid&(truth[pairs[0]]==truth[pairs[1]])&(truth[pairs[0]]>=0)
    counts=np.zeros(len(keys),dtype=np.int64)
    if good.any():
        support=np.unique(np.column_stack([edge_bundle[good],truth[pairs[0,good]]]),axis=0)
        counts=np.bincount(support[:,0].astype(int),minlength=len(keys))
    return family.astype(np.int32),edge_bundle,endpoints,np.minimum(counts,3).astype(np.int64)


def pruning_audit(pairs,t,ty,truth,stereo):
    same=(truth[pairs[0]]==truth[pairs[1]])&(truth[pairs[0]]>=0);pp=pairs[:,same]
    full=connected_components(coo_matrix((np.ones(pp.shape[1]),pp),shape=(len(truth),len(truth))),directed=False)[1]
    links=t[ty,3:5].T
    labels=connected_components(coo_matrix((np.ones(links.shape[1]),links),shape=(pairs.shape[1],pairs.shape[1])),directed=False)[1]
    ids=np.flatnonzero(same);edge=np.r_[ids,ids];nodes=np.r_[pairs[0,ids],pairs[1,ids]];keep=~stereo[nodes]
    triples=np.unique(np.column_stack([truth[nodes[keep]],labels[edge[keep]],nodes[keep]]),axis=0)
    maxima={}
    if len(triples):
        keys,counts=np.unique(triples[:,:2],axis=0,return_counts=True)
        for (tid,_),count in zip(keys,counts):maxima[int(tid)]=max(maxima.get(int(tid),0),int(count))
    reference=[];lost=[]
    for tid in np.unique(truth[(truth>=0)&~stereo]):
        ix=np.flatnonzero((truth==tid)&~stereo)
        if len(ix)<5:continue
        need=max(5,int(np.ceil(len(ix)*.5)))
        if np.unique(full[ix],return_counts=True)[1].max()>=need:
            reference.append(int(tid))
            if maxima.get(int(tid),0)<need:lost.append(int(tid))
    return dict(reference_reconstructible=len(reference),lost_by_bounded_triplets=len(lost),lost_truth_ids=lost)


@torch.no_grad()
def encode(model,g,device,threshold=.4):
    model.eval();raw=g.x.cpu().numpy().copy();g=g.to(device)
    with torch.autocast(device.type,enabled=device.type=='cuda'):
        h=g.x
        for _ in range(model.n_passes-1):h=model._message(h,g.edge_index,g.edge_attr)
        scores=[]
        for begin in range(0,g.num_edges,model.chunk_size):
            end=begin+model.chunk_size
            scores.append(model._classify(h,g.edge_index[:,begin:end],g.edge_attr[begin:end]).sigmoid().float().cpu().numpy())
        directed=np.concatenate(scores);half=len(directed)//2
        assert np.array_equal(g.edge_index[:,:half].cpu().numpy(),g.edge_index.flip(0)[:,half:].cpu().numpy())
        probability=(directed[:half]+directed[half:])/2;keep=np.flatnonzero(probability>=threshold)
        selected=np.r_[keep,keep+half];effects=[]
        for begin in range(0,len(selected),model.chunk_size):
            ix=torch.as_tensor(selected[begin:begin+model.chunk_size],device=device)
            ei=g.edge_index[:,ix]
            effects.append(model.relational_model(torch.cat([h[ei[0]],h[ei[1]],g.edge_attr[ix]],-1)).float().cpu().numpy())
    effect=np.concatenate(effects) if effects else np.empty((0,256),np.float32)
    latent=((effect[:len(keep)]+effect[len(keep):])/2).astype(np.float16)
    return dict(raw=raw.astype(np.float32),h=h.float().cpu().numpy(),latent=latent,
                pairs=g.edge_index[:,:half][:,keep].cpu().numpy().astype(np.int32),p=probability[keep].astype(np.float32),
                cells=g.cell_id.cpu().numpy(),hit_id=g.hit_id.cpu().numpy(),stereo=g.stereo.cpu().numpy())


def prepare(data_root,weight,cache,ids,device='cuda',threshold=.4):
    data_root=Path(data_root);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    assert sha(weight)==WEIGHT_SHA
    signature=dict(dataset_sha256=sha(data_root/'dataset_metadata_joint_no_time.json'),weight_sha256=sha(weight),
        feature_source_sha256=hashlib.sha256(''.join(inspect.getsource(f) for f in [encode,bundles,triplets,pruning_audit]).encode()).hexdigest(),common_source_sha256=sha(Path(__file__).with_name('common.py')),
        threshold=threshold,format=2,torch=torch.__version__,latent='Mean bidirectional256D, fp16',labels_in_model_features=False,
        triplet_policy='Degree<=64 all pairs; larger degree top24 anchors plus every other edge paired with top8; oracle audit required')
    path=cache/'contract.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==signature,'Feature cache contract changed'
    else:save_json(path,signature)
    for name in ['features','indices']:(cache/name).mkdir(exist_ok=True)
    metadata=json.loads((data_root/'dataset_metadata_joint_no_time.json').read_text(encoding='utf-8'))
    sources={s:dataset(data_root,s) for s in metadata['event_ids']};lookup={eid:(s,j) for s,ds in sources.items() for j,eid in enumerate(ds.ids)}
    model=JointInteractionNetwork(chunk_size=65536,recompute_policy='all').to(device)
    model.load_state_dict(torch.load(weight,map_location=device,weights_only=True));model.eval();started=time.time()
    for j,eid in enumerate(ids):
        fpath=cache/'features'/('event_%05d.npz'%eid);ipath=cache/'indices'/fpath.name
        if fpath.exists() and ipath.exists() and (cache/'stats'/('event_%05d.json'%eid)).exists():continue
        split,index=lookup[eid];g=sources[split][index];truth=g.truth.cpu().numpy()
        if fpath.exists():
            with np.load(fpath) as saved:features={k:saved[k] for k in saved.files}
        else:features=encode(model,g,torch.device(device),threshold)
        family,mapping,endpoints,ey=bundles(features['cells'],features['pairs'],truth)
        features.update(family=family,edge_bundle=mapping,bundle_endpoints=endpoints)
        t=triplets(features['pairs'],len(truth),features['p']);ty=(truth[t[:,0]]==truth[t[:,1]])&(truth[t[:,1]]==truth[t[:,2]])&(truth[t[:,1]]>=0)
        hardness=np.minimum(features['p'][t[:,3]],features['p'][t[:,4]]).astype(np.float16)
        atomic_npz(fpath,**features);atomic_npz(ipath,compress=True,t=t,ty=ty,hardness=hardness,ey=ey)
        stats=dict(event_id=eid,split=split,nodes=len(truth),edges=len(features['p']),triplets=len(t),bundles=len(ey),
                   E_class_counts=np.bincount(ey,minlength=4).tolist(),T_positive=int(ty.sum()),
                   triplet_pruning=pruning_audit(features['pairs'],t,ty,truth,features['stereo']))
        (cache/'stats').mkdir(exist_ok=True);save_json(cache/'stats'/('event_%05d.json'%eid),stats)
        if (j+1)%10==0 or j+1==len(ids):
            progress=dict(phase='feature_cache',completed=j+1,total=len(ids),event_id=eid,triplets=len(t),bundles=len(ey),seconds=time.time()-started,unix=time.time())
            save_json(cache/'progress.json',progress);print(json.dumps(progress),flush=True)
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()


def choose(indices,n,rng):
    return rng.choice(indices,size=min(n,len(indices)),replace=False) if n and len(indices) else np.empty(0,dtype=np.int64)


def sample_t(y,hard,n,rng):
    if len(y)<=n:return np.arange(len(y))
    positive=np.flatnonzero(y);negative=np.flatnonzero(~y)
    p=choose(positive,n//2,rng);negative_quota=min(n-len(p),len(negative))
    poolsize=min(len(negative),max(negative_quota*4,1))
    hardpool=negative[np.argpartition(hard[negative],len(negative)-poolsize)[-poolsize:]] if poolsize<len(negative) else negative
    h=choose(hardpool,negative_quota//2,rng);other=np.setdiff1d(negative,h,assume_unique=True)
    selected=np.r_[p,h,choose(other,negative_quota-len(h),rng)]
    if len(selected)<n:selected=np.r_[selected,choose(np.setdiff1d(np.arange(len(y)),selected),n-len(selected),rng)]
    rng.shuffle(selected);return selected


def sample_e(y,n,rng):
    if len(y)<=n:return np.arange(len(y))
    selected=np.concatenate([choose(np.flatnonzero(y==k),n//4,rng) for k in range(4)])
    if len(selected)<n:selected=np.r_[selected,choose(np.setdiff1d(np.arange(len(y)),selected),n-len(selected),rng)]
    rng.shuffle(selected);return selected


class Features(torch.utils.data.Dataset):
    def __init__(self,root,ids,epoch=0,training=False,seed=20261122):
        self.root=Path(root);self.ids=list(ids);self.epoch=epoch;self.training=training;self.seed=seed
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        eid=self.ids[index]
        with np.load(self.root/'features'/('event_%05d.npz'%eid)) as f:result={k:f[k] for k in f.files}
        with np.load(self.root/'indices'/('event_%05d.npz'%eid)) as f:
            t,y,ey=f['t'],f['ty'],f['ey'];hard=f['hardness']
        if self.training:
            rng=np.random.default_rng(self.seed+self.epoch*100003+eid)
            selected=sample_t(y,hard,8192,rng);eidx=sample_e(ey,4096,rng)
            t,y=t[selected],y[selected];ey=ey[eidx]
        else:eidx=np.arange(len(ey))
        result.update(t=t,ty=y.astype(np.float32),ey=ey,eidx=eidx,eid=eid)
        return {k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in result.items()}


def to_device(batch,device):
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
