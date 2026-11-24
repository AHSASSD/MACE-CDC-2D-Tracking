"""Observable-only joint encoder; OC-v1 and sampled full-graph triplets."""
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

def mlp(a,b,c):
    return nn.Sequential(nn.Linear(a,b),nn.LayerNorm(b),nn.SiLU(),nn.Linear(b,c))

class Membership(nn.Module):
    def __init__(self,chunk=4096):
        super().__init__(); self.chunk=chunk
        self.encode=mlp(12,128,64); self.message=mlp(140,256,256)
        self.update=mlp(332,256,64); self.norm=nn.LayerNorm(64)
        self.edge=mlp(256,128,3)
        self.node_projection=nn.Linear(64,32); self.edge_projection=nn.Linear(256,64)
        self.triplet=mlp(262,128,1); self.oc=mlp(64,64,4)
    def effects(self,h,ei,ea):
        return self.message(torch.cat([h[ei[0]],h[ei[1]],ea],-1))
    def step(self,h,raw,ei,ea):
        msg=h.new_zeros((len(h),256),dtype=torch.float32)
        for b in range(0,ei.shape[1],self.chunk):
            e=ei[:,b:b+self.chunk]; v=self.effects(h,e,ea[b:b+self.chunk])
            msg=msg.index_add(0,e[1],v.float())
        return self.norm(h+self.update(torch.cat([h,raw,msg],-1)))
    def edge_logits(self,h,ei,ea):
        kind=(ea[:,9]+ea[:,10]).long()
        return self.edge(self.effects(h,ei,ea)).gather(1,kind[:,None]).squeeze(1)
    def forward(self,g,t):
        raw=g.x; h=self.encode(raw)
        for _ in range(7):
            h=checkpoint(self.step,h,raw,g.edge_index,g.edge_attr,use_reentrant=False) if self.training else self.step(h,raw,g.edge_index,g.edge_attr)
        logits=[]
        for b in range(0,g.edge_index.shape[1],self.chunk):
            ei=g.edge_index[:,b:b+self.chunk]; ea=g.edge_attr[b:b+self.chunk]
            logits.append(checkpoint(self.edge_logits,h,ei,ea,use_reentrant=False) if self.training else self.edge_logits(h,ei,ea))
        if len(t):
            a,c,z=t.T; n=self.node_projection(h)
            # Triplets store nodes; gather original directed edge records via lookup supplied on graph.
            ix=g.triplet_edges
            e1=self.edge_projection((self.effects(h,g.edge_index[:,ix[:,0]],g.edge_attr[ix[:,0]])+self.effects(h,g.edge_index[:,ix[:,0]].flip(0),g.reverse_attr[ix[:,0]]))/2)
            e2=self.edge_projection((self.effects(h,g.edge_index[:,ix[:,1]],g.edge_attr[ix[:,1]])+self.effects(h,g.edge_index[:,ix[:,1]].flip(0),g.reverse_attr[ix[:,1]]))/2)
            full=torch.cat(logits); half=len(full)//2
            prob=(full.sigmoid()+torch.roll(full.sigmoid(),half))/2
            p1=prob[ix[:,0]]; p2=prob[ix[:,1]]
            x=torch.cat([n[a],n[c],n[z],e1,e2,raw[a],raw[c],raw[z],p1[:,None],p2[:,None]],-1)
            y=torch.cat([n[z],n[c],n[a],e2,e1,raw[z],raw[c],raw[a],p2[:,None],p1[:,None]],-1)
            tq=(self.triplet(x).squeeze(1),self.triplet(y).squeeze(1))
        else: full=torch.cat(logits); tq=(full[:0],full[:0])
        return full,tq,self.oc(h).float()

def balanced(values,types,truth=None):
    pieces=[]
    for k in torch.unique(types):
        mask=types==k
        if truth is None: pieces.append(values[mask].mean())
        else:
            terms=[]
            for tid in torch.unique(truth[mask]): terms.append(values[mask & (truth==tid)].mean())
            pieces.append(torch.stack(terms).mean())
    return torch.stack(pieces).mean() if pieces else values.sum()*0

def oc_loss(output,truth,stereo):
    """Version1: q=atanh(beta)^2+.1, truth-max-beta centers, hinge repulsion radius1.
    Each signal instance has equal weight; A/S balanced attraction and repulsion.
    Confidence is 1-beta(center); negative labels receive beta background loss.
    """
    x=output[:,:3]; beta=output[:,3].sigmoid().clamp(1e-5,1-1e-5)
    q=torch.atanh(beta).square()+.1; terms=[]
    for tid in torch.unique(truth[truth>=0]):
        own=truth==tid; ids=torch.nonzero(own).flatten(); center=ids[beta[ids].argmax()]
        dist=(x-x[center]).square().sum(1)
        attract=balanced(q[center]*q[own]*dist[own],stereo[own])
        other=(truth!=tid)&(truth>=0)
        repel=balanced(q[center]*q[other]*(1-dist[other].clamp_min(1e-12).sqrt()).relu().square(),stereo[other]) if other.any() else dist.sum()*0
        terms.append(attract+repel+1-beta[center])
    result=torch.stack(terms).mean() if terms else output.sum()*0
    if (truth<0).any(): result=result+beta[truth<0].mean()
    return result

def sample(g,seed,limit=8192):
    """Training only: sample positive and negative neighbor pairs without score cuts."""
    rng=np.random.default_rng(seed); ei=g.edge_index.numpy(); truth=g.truth.numpy()
    half=ei.shape[1]//2
    assert np.array_equal(ei[:,:half],ei[:,half:][::-1])
    adjacency=[[] for _ in truth]
    for j,(a,c) in enumerate(ei.T): adjacency[c].append((a,j))
    pos=[]; neg=[]
    # Bound work independently of dense full-graph combinatorics.
    for c,neighbors in enumerate(adjacency):
        same=[p for p in neighbors if truth[p[0]]==truth[c] and truth[c]>=0]
        for pool,target,count in [(same,pos,32),(neighbors,neg,64)]:
            if len(pool)<2: continue
            for _ in range(count):
                i,j=rng.choice(len(pool),2,replace=False); a,e=pool[i]; b,f=pool[j]
                good=truth[a]==truth[c]==truth[b] and truth[c]>=0
                if target is neg and good: continue
                target.append((a,c,b,e,f))
    def pick(rows,n):
        if not rows:return np.empty((0,5),dtype=np.int64)
        a=np.unique(np.asarray(rows,dtype=np.int64),axis=0)
        return a[rng.choice(len(a),min(n,len(a)),replace=False)]
    rows=np.concatenate([pick(pos,limit//2),pick(neg,limit//2)])
    g.triplet_edges=torch.from_numpy(rows[:,3:]); g.reverse_attr=torch.cat([g.edge_attr[half:],g.edge_attr[:half]])
    return torch.from_numpy(rows[:,:3])

def objectives(g,t,result):
    edge,(a,b),output=result; y=(g.truth[g.edge_index[0]]==g.truth[g.edge_index[1]])&(g.truth[g.edge_index[0]]>=0)
    e=torch.nn.functional.binary_cross_entropy_with_logits(edge.float(),y.float(),reduction='none')
    el=[]
    for value in [True,False]:
        mask=y==value
        if mask.any():el.append(balanced(e[mask],g.edge_kind[mask],g.truth[g.edge_index[0,mask]] if value else None))
    le=torch.stack(el).mean()
    if len(t):
        ty=(g.truth[t[:,0]]==g.truth[t[:,1]])&(g.truth[t[:,1]]==g.truth[t[:,2]])&(g.truth[t[:,1]]>=0)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(a.float(),ty.float(),reduction='none')+torch.nn.functional.binary_cross_entropy_with_logits(b.float(),ty.float(),reduction='none'))/2
        kind=g.stereo[t].long().sum(1); parts=[]
        for v in [True,False]:
            mask=ty==v
            if mask.any():parts.append(balanced(loss[mask],kind[mask],g.truth[t[mask,1]] if v else None))
        lt=torch.stack(parts).mean()
    else:lt=le*0
    return le,lt,oc_loss(output,g.truth,g.stereo)
