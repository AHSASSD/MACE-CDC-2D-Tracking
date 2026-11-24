"""Software smoke using training events only; weights are not reused by cloud training."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from et_data import Features,to_device
from et_models import EHead,THead,predict
from common import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--splits',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(20261122);device=torch.device('cuda');torch.cuda.reset_peak_memory_stats();start=time.time()
    s=json.loads(args.splits.read_text(encoding='utf-8'));ids=s['pilot_train_event_ids'][:7]+s['stress_train_event_ids'][:1]
    ds=Features(args.cache,ids,epoch=1,training=True);models={'E':EHead().to(device),'T':THead().to(device)}
    assert sum(p.numel() for p in models['E'].parameters())==93316
    assert sum(p.numel() for p in models['T'].parameters())==59233
    optim={k:torch.optim.AdamW(v.parameters(),lr=1e-3) for k,v in models.items()};losses={k:[] for k in models}
    for j in range(len(ds)):
        b=to_device(ds[j],device)
        for route,model in models.items():
            model.train();optim[route].zero_grad()
            with torch.autocast('cuda'):
                a,c=model(b)
                if route=='E':loss=(torch.nn.functional.cross_entropy(a.float(),b['ey'].long())+torch.nn.functional.cross_entropy(c.float(),b['ey'].long()))/2
                else:loss=(torch.nn.functional.binary_cross_entropy_with_logits(a.float(),b['ty'])+torch.nn.functional.binary_cross_entropy_with_logits(c.float(),b['ty']))/2
            assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optim[route].step();losses[route].append(float(loss.detach()))
        print(json.dumps(dict(smoke_event=j+1,event_id=int(b['eid']),loss={r:losses[r][-1] for r in models})),flush=True)
    b=to_device(ds[0],device)
    for m in models.values():m.eval()
    with torch.no_grad():
        a,c=models['T'](b);q=(a.sigmoid()+c.sigmoid())/2
        a,c=models['T'](b,b['t'][:,[2,1,0,4,3]]);qq=(a.sigmoid()+c.sigmoid())/2
        assert torch.allclose(q,qq,atol=1e-6,rtol=1e-5)
        a,c=models['E'](b);q=(a.softmax(-1)+c.softmax(-1))/2
        shuffled=dict(b,bundle_endpoints=b['bundle_endpoints'].flip(0));a,c=models['E'](shuffled);qq=(a.softmax(-1)+c.softmax(-1))/2
        assert torch.allclose(q,qq,atol=1e-5,rtol=1e-4)
    result=dict(status='passed',training_events_only=ids,optimizer_updates_per_route=len(ds),losses=losses,
        endpoint_exchange_invariant=True,gradient_finite=True,peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start,
        note='Smoke model states are discarded; cloud models start from fresh seeds.')
    save_json(args.cache.parent/'local_smoke.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
