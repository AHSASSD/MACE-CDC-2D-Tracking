"""Optimizer/AMP checkpoint roundtrip against uninterrupted next updates."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import torch
from et_data import Features,to_device
from et_models import EHead,THead
from common import save_json


def build():
    m={'E':EHead().cuda(),'T':THead().cuda()}
    o={k:torch.optim.AdamW(x.parameters(),lr=.001,weight_decay=.0001) for k,x in m.items()}
    s={k:torch.amp.GradScaler('cuda',init_scale=1024.) for k in m}
    return m,o,s


def step(m,o,s,b):
    for k in m:
        o[k].zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            a,c=m[k](b)
            if k=='E':loss=(torch.nn.functional.cross_entropy(a.float(),b['ey'].long())+torch.nn.functional.cross_entropy(c.float(),b['ey'].long()))/2
            else:loss=(torch.nn.functional.binary_cross_entropy_with_logits(a.float(),b['ty'])+torch.nn.functional.binary_cross_entropy_with_logits(c.float(),b['ty']))/2
        s[k].scale(loss).backward();s[k].unscale_(o[k]);torch.nn.utils.clip_grad_norm_(m[k].parameters(),5)
        s[k].step(o[k]);s[k].update()


def main():
    torch.set_num_threads(1);torch.manual_seed(20261122)
    b=to_device(Features(ROOT/'local_cache_v3',[24],epoch=1,training=True)[0],'cuda')
    m,o,s=build();step(m,o,s,b)
    path=ROOT/'resume_roundtrip.pth'
    torch.save(dict(models={k:x.state_dict() for k,x in m.items()},optimizers={k:x.state_dict() for k,x in o.items()},
                    scalers={k:x.state_dict() for k,x in s.items()}),path)
    step(m,o,s,b);reference={k:{n:p.detach().clone() for n,p in x.state_dict().items()} for k,x in m.items()}
    restored=torch.load(path,map_location='cuda',weights_only=False);mm,oo,ss=build()
    for k in mm:mm[k].load_state_dict(restored['models'][k]);oo[k].load_state_dict(restored['optimizers'][k]);ss[k].load_state_dict(restored['scalers'][k])
    step(mm,oo,ss,b);differences={k:max(float((v-reference[k][n]).abs().max()) for n,v in x.state_dict().items()) for k,x in mm.items()}
    assert max(differences.values())<1e-4,differences
    result=dict(status='passed',maximum_parameter_difference=differences,optimizer_and_scaler_restored=True,
                note='Training-event software test; not a model-performance claim.')
    save_json(ROOT/'resume_roundtrip.json',result);print(json.dumps(result))


if __name__=='__main__':main()
