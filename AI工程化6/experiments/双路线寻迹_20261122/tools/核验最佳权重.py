from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import torch,numpy as np
from et_data import Features,to_device
from et_models import EHead,THead,predict

def main():
    torch.set_num_threads(1);p=ROOT/'evaluation_download';eid=json.loads((p/'full/contract.json').read_text(encoding='utf-8'))['validation_ids'][0]
    b=to_device(Features(p/'cache',[eid])[0],'cuda');result=dict(event_id=eid,device=torch.cuda.get_device_name(),routes={})
    for k,cls,epoch in [('E',EHead,35),('T',THead,25)]:
        f=p/'full'/('%s_best_members.pth'%k);sha=hashlib.sha256(f.read_bytes()).hexdigest()
        model=cls().cuda();state=torch.load(f,map_location='cuda',weights_only=True);other=torch.load(p/'full'/('%s_epoch_%03d.pth'%(k,epoch)),map_location='cuda',weights_only=True)
        assert all(torch.equal(v,other[key]) for key,v in state.items())
        model.load_state_dict(state);model.eval();torch.cuda.synchronize();started=time.perf_counter()
        with torch.no_grad(),torch.autocast('cuda'):prob=predict(model,b,k)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        reference=np.load(p/'full/validation'/('epoch_%03d'%epoch)/('prediction_%05d.npz'%eid))[k]
        diff=float(np.max(np.abs(prob-reference)));assert diff<.01
        result['routes'][k]=dict(epoch=epoch,sha256=sha,max_probability_difference_vs_cloud=diff,inference_seconds=seconds,best_equals_epoch_state=True)
    (ROOT/'evaluation_20261122/weights_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
