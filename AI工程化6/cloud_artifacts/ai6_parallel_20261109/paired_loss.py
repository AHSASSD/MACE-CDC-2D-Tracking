"""Same GNN: event-balanced focal control vs track-balanced/hard-negative auxiliaries."""
import torch
import torch.nn.functional as F


def objective(logits,data,arm):
    bce=F.binary_cross_entropy_with_logits(logits.float(),data.y,reduction='none')
    focal=.5*(1-torch.exp(-bce)).square()*bce
    edge_event=data.batch[data.edge_index[0]]
    losses=[];base_terms=[];positive_terms=[];negative_terms=[]
    for event in range(data.num_graphs):
        mask=edge_event==event;values=focal[mask];base=values.mean();positive=base*0;negative=base*0
        if arm=='balanced':
            target=data.y[mask]>.5;score=logits[mask].detach().float().sigmoid()
            src=data.edge_index[0,mask];tid=data.truth[src]
            good=target&(tid>=0)
            if good.any():
                _,inverse=torch.unique(tid[good],return_inverse=True)
                count=torch.bincount(inverse).to(values.dtype)
                track_loss=values.new_zeros(len(count)).index_add(0,inverse,values[good])/count
                confidence=values.new_zeros(len(count)).index_add(0,inverse,score[good])/count
                weight=1+torch.clamp((.7-confidence)/.7,min=0,max=1)
                positive=(track_loss*weight).sum()/weight.sum()
            bad=~target
            if bad.any():
                k=max(1,int(bad.sum().item()*.1))
                chosen=torch.topk(score[bad],k,sorted=False).indices
                negative=values[bad][chosen].mean()
        elif arm!='control':raise ValueError(arm)
        losses.append(base+.1*positive+.05*negative)
        base_terms.append(base.detach());positive_terms.append(positive.detach());negative_terms.append(negative.detach())
    return torch.stack(losses).mean(),dict(base=torch.stack(base_terms).mean().item(),
             positive_aux=torch.stack(positive_terms).mean().item(),hard_negative_aux=torch.stack(negative_terms).mean().item())
