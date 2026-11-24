"""Development-only matched precision audit of frozen saved heads."""
from pathlib import Path
import sys,os,json,hashlib,time
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
import bootstrap
os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import torch
from et_data import Features,to_device
from et_models import THead,predict
OUT=ROOT/'training_review_20261122';SOURCE=OUT/'download';BINS=10000

def metrics(pos,neg,k):
    a=int(pos[k]);b=int(neg[k]);total=int(pos[0]);return dict(threshold=k/BINS,tp=a,fp=b,precision=a/max(a+b,1),recall=a/max(total,1))

def main():
    torch.set_num_threads(1);contract=json.loads((SOURCE/'training/contract.json').read_text(encoding='utf-8'));ids=contract['validation_ids'];assert len(ids)==200
    models={}
    for k,path in [('T25',SOURCE/'initial_T25.pth'),('focused7',SOURCE/'training/T_epoch_007.pth'),('focused10',SOURCE/'training/T_epoch_010.pth')]:
        m=THead().cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True));m.eval();models[k]=m
    hist={k:{g:np.zeros((2,BINS+1),np.int64) for g in ['all','weak','axial','weak_axial','path_axial']} for k in models};started=time.time()
    for j,eid in enumerate(ids):
        b=Features(SOURCE/'cache',[eid])[0];t=b['t'].numpy();y=b['ty'].numpy().astype(bool);p=b['p'].numpy();stereo=b['stereo'].numpy();raw=b['raw'].numpy()
        weak=np.minimum(p[t[:,3]],p[t[:,4]])<.7;aa=~stereo[t[:,:3]].any(1)
        xy=raw[:,:2];v1=xy[t[:,1]]-xy[t[:,0]];v2=xy[t[:,2]]-xy[t[:,1]];path=aa&((v1*v2).sum(1)>=0)
        masks=dict(all=np.ones(len(t),bool),weak=weak,axial=aa,weak_axial=weak&aa,path_axial=path);gpu=to_device(b,'cuda')
        for k,m in models.items():
            with torch.no_grad(),torch.autocast('cuda'):q=predict(m,gpu,'T')
            bucket=np.minimum((q*BINS).astype(int),BINS)
            for g,mask in masks.items():
                hist[k][g][0]+=np.bincount(bucket[mask&y],minlength=BINS+1);hist[k][g][1]+=np.bincount(bucket[mask&~y],minlength=BINS+1)
        if (j+1)%20==0:
            (OUT/'pr_progress.json').write_text(json.dumps(dict(completed=j+1,total=len(ids),seconds=time.time()-started)),encoding='utf-8');print('scored',j+1,'/200',flush=True)
    cumulative={k:{g:np.cumsum(h[:,::-1],axis=1)[:,::-1] for g,h in groups.items()} for k,groups in hist.items()}
    result=dict(events=200,scope='Development validation PR audit; checkpoints7and10 selected retrospectively, not blind confirmation. Weak means either original edge score<0.7.',models={},seconds=time.time()-started)
    for k,groups in cumulative.items():
        out={}
        for g,(pos,neg) in groups.items():
            bp,bn=cumulative['T25'][g];target=metrics(bp,bn,5000)['precision'];prec=pos/np.maximum(pos+neg,1)
            feasible=np.flatnonzero((prec>=target)&(pos>0));threshold=int(feasible[np.argmax(pos[feasible])]) if len(feasible) else BINS
            out[g]=dict(at05=metrics(pos,neg,5000),at09=metrics(pos,neg,9000),matched_precision=metrics(pos,neg,threshold),precision_target=target)
        result['models'][k]=out
    # Global-precision matched cutoff: report weak recall at the SAME cutoff,
    # rather than choosing a different threshold for each subgroup.
    for k,groups in cumulative.items():
        cutoff=result['models'][k]['all']['matched_precision']['threshold'];i=round(cutoff*BINS)
        result['models'][k]['at_global_precision_match']={g:metrics(*v,i) for g,v in groups.items()}
    (OUT/'matched_precision.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/'pr_histograms.npz',**{k+'_'+g:h for k,groups in hist.items() for g,h in groups.items()})
    print('RESULT',json.dumps(result),flush=True)

if __name__=='__main__':main()
