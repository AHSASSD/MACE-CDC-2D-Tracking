"""Versioned exact reductions plus an experimental hard-neighbor instance loss.
Old membership.py and its checkpoints remain immutable.
"""
import numpy as np
import torch
import torch.nn.functional as F
import membership as legacy

def balanced(values,types,truth=None):
    if not values.numel():return values.sum()*0
    if truth is None:
        _,ix=torch.unique(types,return_inverse=True)
        count=torch.bincount(ix);s=values.new_zeros(len(count)).index_add(0,ix,values)
        return (s/count).mean()
    kinds,ki=torch.unique(types,return_inverse=True);tracks,ti=torch.unique(truth,return_inverse=True)
    ix=ki*len(tracks)+ti;size=len(kinds)*len(tracks)
    count=torch.bincount(ix,minlength=size);s=values.new_zeros(size).index_add(0,ix,values)/count.clamp_min(1)
    return (s.reshape(len(kinds),-1).sum(1)/(count.reshape(len(kinds),-1)>0).sum(1).clamp_min(1)).mean()

def centers(output,truth):
    valid=truth>=0;ids=torch.nonzero(valid).flatten()
    _,inv=torch.unique(truth[valid],return_inverse=True);k=int(inv.max())+1 if len(ids) else 0
    beta=output[:,3].sigmoid().clamp(1e-5,1-1e-5)
    maxb=beta.new_full((k,),-1).scatter_reduce(0,inv,beta[ids],reduce='amax')
    candidate=torch.where(beta[ids]==maxb[inv],ids,torch.full_like(ids,len(truth)))
    alpha=torch.full((k,),len(truth),device=truth.device,dtype=torch.long).scatter_reduce(0,inv,candidate,reduce='amin')
    return ids,inv,alpha,beta

def row_type_mean(values,mask,stereo):
    parts=[];available=[]
    for kind in [False,True]:
        m=mask&(stereo[None,:]==kind);n=m.sum(1)
        parts.append((values*m).sum(1)/n.clamp_min(1));available.append(n>0)
    return (torch.stack(parts)*torch.stack(available)).sum(0)/torch.stack(available).sum(0).clamp_min(1)

def oc_v1(output,truth,stereo):
    ids,inv,alpha,beta=centers(output,truth)
    if not len(alpha):return beta.mean() if len(beta) else output.sum()*0
    x=output[:,:3];q=torch.atanh(beta).square()+.1
    d=(x[alpha,None,:]-x[None,:,:]).square().sum(-1)
    own=truth[alpha,None]==truth[None,:];other=(~own)&(truth[None,:]>=0)
    weights=q[alpha,None]*q[None,:]
    attraction=row_type_mean(weights*d,own,stereo)
    repulsion=row_type_mean(weights*(1-d.clamp_min(1e-12).sqrt()).relu().square(),other,stereo)
    result=(attraction+repulsion+1-beta[alpha]).mean()
    noise=truth<0
    return result+(beta[noise].mean() if noise.any() else 0)

def oc_v2(output,truth,stereo,components=False):
    """Experimental metric loss: charge-independent, per-hit nearest foreign centers.
    Radius .25 attraction, unit margin repulsion; nearest3 avoid far-negative dilution.
    Beta uses one selected peak per truth instance, low non-center penalty .1.
    This is an adaptation, not a claim of reproducing standard OC.
    """
    ids,inv,alpha,beta=centers(output,truth)
    if not len(alpha):return output.sum()*0
    x=output[:,:3];dist=(x[ids,None,:]-x[alpha][None,:,:]).square().sum(-1).clamp_min(1e-12).sqrt()
    own=dist.gather(1,inv[:,None]).squeeze(1)
    attraction=balanced((own-.25).relu().square(),stereo[ids],truth[ids])
    mask=F.one_hot(inv,len(alpha)).bool();foreign=dist.masked_fill(mask,float('inf'))
    if len(alpha)>1:
        hard=foreign.topk(min(3,len(alpha)-1),largest=False).values
        repulsion=balanced((1-hard).relu().square().mean(1),stereo[ids],truth[ids])
    else:repulsion=attraction*0
    target=torch.zeros_like(beta);target[alpha]=1
    confidence=F.binary_cross_entropy_with_logits(output[alpha,3],torch.ones_like(beta[alpha]))
    noncenter=(truth>=0)&(target==0)
    low=balanced(F.softplus(output[noncenter,3]),stereo[noncenter],truth[noncenter]) if noncenter.any() else confidence*0
    confidence=confidence+.1*low
    total=attraction+repulsion+.2*confidence
    return (total,dict(attraction=attraction,repulsion=repulsion,confidence=confidence)) if components else total

def sample(g,seed,limit=8192):
    rng=np.random.default_rng(seed);ei=g.edge_index.numpy();truth=g.truth.numpy();half=ei.shape[1]//2
    assert np.array_equal(ei[:,:half],ei[:,half:][::-1])
    order=np.argsort(ei[1],kind='stable');boundary=np.searchsorted(ei[1,order],np.arange(len(truth)+1));pos=[];neg=[]
    distance=g.edge_attr[:,4].numpy()
    for c in range(len(truth)):
        neighbors=order[boundary[c]:boundary[c+1]]
        if len(neighbors)<2:continue
        same=neighbors[(truth[ei[0,neighbors]]==truth[c])&(truth[c]>=0)]
        for pool,target,count in [(same,pos,32),(neighbors,neg,32),(neighbors[np.argsort(distance[neighbors],kind='stable')[:32]],neg,32)]:
            if len(pool)<2:continue
            a=rng.integers(len(pool),size=count);b=rng.integers(len(pool)-1,size=count);b+=b>=a
            u,v=pool[a],pool[b];aa,zz=ei[0,u],ei[0,v]
            good=(truth[aa]==truth[c])&(truth[zz]==truth[c])&(truth[c]>=0)
            keep=np.ones(len(u),bool) if target is pos else ~good
            target.append(np.column_stack([aa[keep],np.full(keep.sum(),c),zz[keep],u[keep],v[keep]]))
    def pick(chunks):
        if not chunks:return np.empty((0,5),np.int64)
        a=np.unique(np.concatenate(chunks),axis=0)
        return a[rng.choice(len(a),min(limit//2,len(a)),replace=False)]
    rows=np.concatenate([pick(pos),pick(neg)]).astype(np.int64)
    g.triplet_edges=torch.from_numpy(rows[:,3:]);g.reverse_attr=torch.cat([g.edge_attr[half:],g.edge_attr[:half]])
    return torch.from_numpy(rows[:,:3])

def install(version='v1'):
    legacy.balanced=balanced;legacy.oc_loss=oc_v1 if version=='v1' else oc_v2
    return legacy.Membership,legacy.objectives

def oc_hard(output,truth,stereo):
    """Single-variable OC-v1 ablation: average nearest8 foreign hits, not all negatives."""
    ids,inv,alpha,beta=centers(output,truth)
    if not len(alpha):return output.sum()*0
    x=output[:,:3];q=torch.atanh(beta).square()+.1
    d=(x[alpha,None,:]-x[None,:,:]).square().sum(-1)
    own=truth[alpha,None]==truth[None,:];other=(~own)&(truth[None,:]>=0)
    weights=q[alpha,None]*q[None,:]
    attraction=row_type_mean(weights*d,own,stereo)
    nearest=d.masked_fill(~other,float('inf')).topk(min(8,len(truth)),largest=False).indices
    hard=torch.zeros_like(other).scatter(1,nearest,True)&other
    repulsion=row_type_mean(weights*(1-d.clamp_min(1e-12).sqrt()).relu().square(),hard,stereo)
    result=(attraction+repulsion+1-beta[alpha]).mean()
    return result+(beta[truth<0].mean() if (truth<0).any() else 0)
