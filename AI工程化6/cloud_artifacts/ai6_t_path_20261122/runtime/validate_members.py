import bootstrap
from bootstrap import HERE,BASE
from pathlib import Path
import argparse,json,os
import torch,numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from common import save_json
from et_data import Features,to_device,atomic_npz
from et_models import THead,predict
from evaluate import event

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--splits',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();torch.set_num_threads(1)
    ids=json.loads(args.splits.read_text(encoding='utf-8'))['periodic_member_validation_event_ids'][:16];args.out.mkdir(exist_ok=True)
    model=THead().cuda();model.load_state_dict(torch.load(args.weights,map_location='cuda',weights_only=True));model.eval();tasks=[]
    for j,eid in enumerate(ids):
        if (args.out/('event_%05d.json'%eid)).exists():continue
        b=to_device(Features(args.cache,[eid])[0],'cuda')
        with torch.no_grad(),torch.autocast('cuda'):q=predict(model,b,'T')
        pred=args.out/('prediction_%05d.npz'%eid);atomic_npz(pred,compress=True,T=q);tasks.append((eid,str(args.cache),str(pred),str(args.out)))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(event,t) for t in tasks]):print('member',f.result(),flush=True)
    from STEP5_两级寻迹 import summarize
    from STEP2_训练双路线 import seed_summary
    packets=[json.loads((args.out/('event_%05d.json'%eid)).read_text(encoding='utf-8')) for eid in ids]
    result=dict(events=len(ids),weights=str(args.weights),scope='Development member check, not independent confirmation; T_old means old decoder with the specified head',methods={k:dict(tracks=summarize([r['methods'][k]['counts'] for r in packets]),seeds=seed_summary([r['methods'][k]['seeds'] for r in packets]),mean_seconds=float(np.mean([r['methods'][k]['seconds'] for r in packets]))) for k in packets[0]['methods']})
    save_json(args.out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
