"""Independent edge-bundle(E) and triplet(T) models. No truth fields used here."""
import torch
from torch import nn


def classifier(inputs,h1,h2,outputs):
    return nn.Sequential(nn.Linear(inputs,h1),nn.LayerNorm(h1),nn.SiLU(),
                         nn.Linear(h1,h2),nn.LayerNorm(h2),nn.SiLU(),nn.Linear(h2,outputs))


class THead(nn.Module):
    def __init__(self,wide=False):
        super().__init__();self.node_projection=nn.Linear(12,32);self.edge_projection=nn.Linear(256,64)
        self.classifier=classifier(262,256 if wide else 128,128 if wide else 64,1)
    def context(self,b):
        return self.node_projection(b['h'].float()),self.edge_projection(b['latent'].float())
    def ordered(self,b,t,context):
        h,e=context;a,c,z,i,j=t.long().T
        x=torch.cat([h[a],h[c],h[z],e[i],e[j],b['raw'][a],b['raw'][c],b['raw'][z],b['p'][i,None],b['p'][j,None]],-1)
        y=torch.cat([h[z],h[c],h[a],e[j],e[i],b['raw'][z],b['raw'][c],b['raw'][a],b['p'][j,None],b['p'][i,None]],-1)
        return self.classifier(x).squeeze(-1),self.classifier(y).squeeze(-1)
    def forward(self,b,t=None):return self.ordered(b,b['t'] if t is None else t,self.context(b))


def aggregate(values,index,n):
    index=index.long();count=torch.bincount(index,minlength=n).to(values.dtype)
    mean=values.new_zeros((n,values.shape[1])).index_add(0,index,values)/count.clamp_min(1)[:,None]
    maximum=values.new_full((n,values.shape[1]),-torch.inf)
    maximum.scatter_reduce_(0,index[:,None].expand_as(values),values,reduce='amax',include_self=True)
    maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
    return torch.cat([mean,maximum,torch.log1p(count)[:,None]/4],-1),mean


class EHead(nn.Module):
    def __init__(self,wide=False):
        super().__init__();self.hit_encoder=nn.Sequential(nn.Linear(24,64),nn.LayerNorm(64),nn.SiLU())
        self.hit_edge_encoder=nn.Sequential(nn.Linear(257,64),nn.LayerNorm(64),nn.SiLU())
        self.classifier=classifier(515,256 if wide else 128,128 if wide else 64,4)
    def context(self,b):
        family=b['family'].long();ends=b['bundle_endpoints'].long();nf=int(family.max())+1;nb=ends.shape[1]
        node=self.hit_encoder(torch.cat([b['raw'],b['h'].float()],-1));nodes,_=aggregate(node,family,nf)
        encoded=self.hit_edge_encoder(torch.cat([b['latent'].float(),b['p'][:,None]],-1))
        valid=b['edge_bundle']>=0;edges,mean=aggregate(encoded[valid],b['edge_bundle'][valid],nb)
        neighbors=mean.new_zeros((nf,64)).index_add(0,ends[0],mean).index_add(0,ends[1],mean)
        degree=torch.bincount(ends.flatten(),minlength=nf).to(mean.dtype)
        left=(neighbors[ends[0]]-mean)/(degree[ends[0]]-1).clamp_min(1)[:,None]
        right=(neighbors[ends[1]]-mean)/(degree[ends[1]]-1).clamp_min(1)[:,None]
        return nodes,edges,left,right
    def ordered(self,b,index,context):
        nodes,edges,left,right=context;index=index.long();u,v=b['bundle_endpoints'].long()[:,index]
        a=torch.cat([nodes[u],nodes[v],edges[index],left[index],right[index]],-1)
        z=torch.cat([nodes[v],nodes[u],edges[index],right[index],left[index]],-1)
        return self.classifier(a),self.classifier(z)
    def forward(self,b,index=None):return self.ordered(b,b['eidx'] if index is None else index,self.context(b))


@torch.no_grad()
def predict(model,b,route,chunk=8192):
    result=[];context=model.context(b)
    indices=b['t'] if route=='T' else torch.arange(b['bundle_endpoints'].shape[1],device=b['h'].device)
    for begin in range(0,len(indices),chunk):
        a,c=model.ordered(b,indices[begin:begin+chunk],context)
        p=(a.sigmoid()+c.sigmoid())/2 if route=='T' else (a.softmax(-1)+c.softmax(-1))/2
        result.append(p.float().cpu())
    return torch.cat(result).numpy() if result else torch.empty((0,) if route=='T' else (0,4)).numpy()
