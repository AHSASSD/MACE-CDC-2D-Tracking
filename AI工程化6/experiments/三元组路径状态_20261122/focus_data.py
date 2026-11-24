"""Label-only hard-positive mining; observable features remain unchanged."""
import bootstrap
from pathlib import Path
import numpy as np
import torch
from et_data import atomic_npz

def strata(f,t,y):
    p=f['p'];lo=np.minimum(p[t[:,3]],p[t[:,4]]);hi=np.maximum(p[t[:,3]],p[t[:,4]])
    _,family,counts=np.unique(f['cells'],return_inverse=True,return_counts=True)
    shared=(counts[family[t[:,:3]]]>1).any(1)
    y=y.astype(bool);category=np.full(len(y),4,np.uint8)
    category[y]=0;category[y&shared]=2;category[y&(lo<.7)]=1
    category[~y&((lo>=.65)|shared)]=3
    weight=np.ones(len(y),np.float32);weight[y&(lo<.7)]=2.5
    weight[y&(lo<.7)&(hi>=.7)]=3.;weight[(category==2)|(category==3)]=1.5
    return category,weight

class FocusDataset(torch.utils.data.Dataset):
    def __init__(self,cache,packs,ids,epoch):self.cache=Path(cache);self.packs=Path(packs);self.ids=list(ids);self.epoch=epoch
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        eid=self.ids[index];name='event_%05d.npz'%eid
        with np.load(self.cache/'features'/name) as s:f={k:s[k] for k in s.files}
        target=self.packs/name
        if not target.exists():
            with np.load(self.cache/'indices'/name) as s:t=s['t'];y=s['ty']
            cat,weight=strata(f,t,y);rng=np.random.default_rng(20261123+eid);selected=[]
            for k in range(5):
                ix=np.flatnonzero(cat==k);selected.extend(rng.choice(ix,min(len(ix),8192),replace=False).tolist())
            ix=np.asarray(selected,int)
            atomic_npz(target,compress=True,t=t[ix],ty=y[ix],category=cat[ix],loss_weight=weight[ix])
        with np.load(target) as s:t=s['t'];y=s['ty'];cat=s['category'];weight=s['loss_weight']
        rng=np.random.default_rng(20261123+eid+100003*self.epoch);selected=[]
        for k,n in enumerate([1536,2048,512,3072,1024]):
            ix=np.flatnonzero(cat==k);selected.extend(rng.choice(ix,min(n,len(ix)),replace=False).tolist())
        selected=np.asarray(selected,int)
        # Fill each sign up to4096; don't let a scarce hard class reduce the batch.
        for sign in [True,False]:
            need=4096-int((y[selected].astype(bool)==sign).sum())
            pool=np.setdiff1d(np.flatnonzero(y.astype(bool)==sign),selected,assume_unique=False)
            if need>0 and len(pool):selected=np.r_[selected,rng.choice(pool,min(need,len(pool)),replace=False)]
        rng.shuffle(selected)
        f.update(t=t[selected],ty=y[selected].astype(np.float32),loss_weight=weight[selected],category=cat[selected],eid=eid)
        return {k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in f.items()}
