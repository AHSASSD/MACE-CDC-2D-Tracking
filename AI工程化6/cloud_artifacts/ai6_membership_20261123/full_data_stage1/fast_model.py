"""Same parameters and mathematics as Membership; configurable recomputation."""
import torch
from torch.utils.checkpoint import checkpoint
from membership import Membership

class FastMembership(Membership):
    def __init__(self,chunk=4096,direct_steps=0):
        super().__init__(chunk);self.direct_steps=direct_steps
    def forward(self,g,t):
        raw=g.x;h=self.encode(raw)
        for step in range(7):
            h=checkpoint(self.step,h,raw,g.edge_index,g.edge_attr,use_reentrant=False) if self.training and step>=self.direct_steps else self.step(h,raw,g.edge_index,g.edge_attr)
        logits=[]
        for begin in range(0,g.edge_index.shape[1],self.chunk):
            ei=g.edge_index[:,begin:begin+self.chunk];ea=g.edge_attr[begin:begin+self.chunk]
            logits.append(checkpoint(self.edge_logits,h,ei,ea,use_reentrant=False) if self.training else self.edge_logits(h,ei,ea))
        full=torch.cat(logits)
        if len(t):
            a,c,z=t.T;n=self.node_projection(h);ix=g.triplet_edges
            latent=[]
            for k in [0,1]:
                indices=ix[:,k];ei=g.edge_index[:,indices]
                latent.append(self.edge_projection((self.effects(h,ei,g.edge_attr[indices])+self.effects(h,ei.flip(0),g.reverse_attr[indices]))/2))
            e1,e2=latent
            reverse=getattr(g,'reverse_edge',None)
            prob=(full.sigmoid()+(full[reverse].sigmoid() if reverse is not None else torch.roll(full.sigmoid(),len(full)//2)))/2
            p1,p2=prob[ix[:,0]],prob[ix[:,1]]
            x=torch.cat([n[a],n[c],n[z],e1,e2,raw[a],raw[c],raw[z],p1[:,None],p2[:,None]],-1)
            y=torch.cat([n[z],n[c],n[a],e2,e1,raw[z],raw[c],raw[a],p2[:,None],p1[:,None]],-1)
            tq=(self.triplet(x).squeeze(1),self.triplet(y).squeeze(1))
        else:tq=(full[:0],full[:0])
        return full,tq,self.oc(h).float()

def pack(graphs,triplets):
    from torch_geometric.data import Data
    node=edge=triplet=0;fields={k:[] for k in ['x','edge_index','edge_attr','edge_kind','truth','stereo','triplet_edges','reverse_attr','reverse_edge']};ts=[];slices=[]
    for g,t in zip(graphs,triplets):
        n,e=g.num_nodes,g.num_edges;half=e//2
        for key in ['x','edge_attr','edge_kind','truth','stereo','reverse_attr']:fields[key].append(getattr(g,key))
        fields['edge_index'].append(g.edge_index+node);fields['triplet_edges'].append(g.triplet_edges+edge)
        fields['reverse_edge'].append(torch.cat([torch.arange(half,e),torch.arange(half)])+edge)
        ts.append(t+node);slices.append((node,node+n,edge,edge+e,triplet,triplet+len(t)))
        node+=n;edge+=e;triplet+=len(t)
    data=Data(**{k:torch.cat(v,dim=1 if k=='edge_index' else 0) for k,v in fields.items()})
    return data,torch.cat(ts),slices

def event_objectives(g,t,result,slices,objective):
    from torch_geometric.data import Data
    edge,(ta,tb),oc=result;terms=[]
    for ns,ne,es,ee,ts,te in slices:
        local=Data(truth=g.truth[ns:ne],stereo=g.stereo[ns:ne],edge_index=g.edge_index[:,es:ee]-ns,edge_kind=g.edge_kind[es:ee])
        terms.append(torch.stack(objective(local,t[ts:te]-ns,(edge[es:ee],(ta[ts:te],tb[ts:te]),oc[ns:ne]))))
    return torch.stack(terms).mean(0).unbind()
