"""Event focal + per-track/per-type positive margin + type-balanced hard negatives."""
import math
import torch
import torch.nn.functional as F

TYPE_WEIGHTS=(.5,.3,.2)


def objective(logits,data,epoch):
    z=logits.float();bce=F.binary_cross_entropy_with_logits(z,data.y,reduction='none')
    focal=.5*(1-torch.exp(-bce)).square()*bce
    event=data.batch[data.edge_index[0]];losses=[];records=[]
    coefficient=.02+.06*min(max(epoch-1,0)/4,1)
    for index in range(data.num_graphs):
        select=event==index;v=focal[select];p=data.y[select]>.5;zz=z[select];kind=data.edge_kind[select]
        tid=data.truth[data.edge_index[0,select]];base=v.mean();positive=[];weights=[];negative=[]
        for k,w in enumerate(TYPE_WEIGHTS):
            good=p&(kind==k)&(tid>=0)
            if good.any():
                _,inverse=torch.unique(tid[good],return_inverse=True)
                counts=torch.bincount(inverse).to(zz.dtype)
                penalty=F.relu(math.log(.7/.3)-zz[good]).square()
                grouped=zz.new_zeros(len(counts)).index_add(0,inverse,penalty)/counts
                positive.append(grouped.mean()*w);weights.append(w)
            bad=(~p)&(kind==k)
            if bad.any():
                values=zz[bad];n=max(1,int(len(values)*.1))
                hard=torch.topk(values,n,sorted=False).values
                negative.append(F.relu(hard-math.log(.3/.7)).square().mean())
        pos=sum(positive)/sum(weights) if positive else base*0
        neg=torch.stack(negative).mean() if negative else base*0
        losses.append(base+coefficient*pos+.02*neg)
        records.append(torch.stack([base.detach(),pos.detach(),neg.detach()]))
    metrics=torch.stack(records).mean(0).cpu().tolist()
    return torch.stack(losses).mean(),dict(focal=metrics[0],true_low_margin=metrics[1],false_high_margin=metrics[2],positive_coefficient=coefficient)
