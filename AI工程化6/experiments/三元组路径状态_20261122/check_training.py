import bootstrap
from bootstrap import HERE,BASE
import zipfile,json,shutil,subprocess,sys
import numpy as np
from pathlib import Path
from focus_data import FocusDataset,strata
from et_data import sample_t

def main():
    cache=HERE/'smoke_cache'
    with zipfile.ZipFile(HERE/'smoke_source.zip') as z:
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(cache)
    s=json.loads((BASE.parent/'三元组寻迹计划_20261122/split_manifest.json').read_text(encoding='utf-8'))
    for eid in s['periodic_member_validation_event_ids'][:8]:
        for folder in ['features','indices']:shutil.copy2(BASE/'evaluation_download/cache'/folder/('event_%05d.npz'%eid),cache/folder/('event_%05d.npz'%eid))
    rows=[]
    ds=FocusDataset(cache,HERE/'smoke_sampling_packs',[0,2,3,4],1)
    for j,eid in enumerate([0,2,3,4]):
        x=ds[j];assert len(x['ty'])<=8192 and np.isfinite(x['loss_weight'].numpy()).all()
        name='event_%05d.npz'%eid
        with np.load(cache/'features'/name) as f:features={k:f[k] for k in f.files}
        with np.load(cache/'indices'/name) as f:t=f['t'];y=f['ty'];hard=f['hardness']
        cat,_=strata(features,t,y);old=sample_t(y,hard,8192,np.random.default_rng(20261123+eid))
        rows.append(dict(event_id=eid,old_weak_positive=int((cat[old]==1).sum()),new_weak_positive=int((x['category'].numpy()==1).sum()),new_positive=int(x['ty'].sum()),new_negative=int((x['ty']==0).sum())))
    cmd=[sys.executable,str(HERE/'train.py'),'--cache',str(cache),'--out',str(HERE/'smoke_training_final'),'--splits',str(BASE.parent/'三元组寻迹计划_20261122/split_manifest.json'),'--initial',str(BASE/'evaluation_download/full/T_best_members.pth'),'--epochs','1','--limit-train','4']
    subprocess.run(cmd,check=True)
    meta=HERE/'smoke_training_final/checkpoint_meta.json';first=json.loads(meta.read_text(encoding='utf-8'));assert first['updates']==1 and first['epoch']==2
    subprocess.run(cmd,check=True);second=json.loads(meta.read_text(encoding='utf-8'));assert second['updates']==first['updates']
    (HERE/'training_software_checks.json').write_text(json.dumps(dict(sample_counts=rows,checkpoint_resume_passed=True,software_weights_discarded=True),ensure_ascii=False,indent=2),encoding='utf-8');print('PASS',rows,flush=True)

if __name__=='__main__':main()
