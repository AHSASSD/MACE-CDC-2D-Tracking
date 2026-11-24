"""Post-test descriptive overlap stratification; does not select thresholds/models."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'confirmation_20261122';DATA=ROOT.parent.parent/'joint_event_dataset'
import numpy as np
from scipy.spatial import cKDTree

def main():
    s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'));assert s['events']==500
    ids=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))['ids'];geometry=np.load(DATA/'wire_geometry.npy')
    groups={name:dict(truth_tracks=0,baseline_found=0,T_found=0) for name in ['high_overlap','other']}
    events=[]
    for eid in ids:
        with np.load(DATA/'events'/('event_%05d.npz'%eid)) as f:cell=f['cell_id'];truth=f['truth']
        geo=geometry[cell];axial=np.flatnonzero(np.linalg.norm(geo[:,2:4],axis=1)<=1e-9)
        a=truth[axial];pairs=cKDTree(geo[axial,:2]).query_pairs(10.,output_type='ndarray')
        good=(a[pairs[:,0]]!=a[pairs[:,1]])&(a[pairs[:,0]]>=0)&(a[pairs[:,1]]>=0);crowded=np.zeros(len(axial),bool);crowded[pairs[good].ravel()]=True
        row=json.loads((OUT/'events'/('event_%05d.json'%eid)).read_text(encoding='utf-8'));found={k:set(v['found']) for k,v in row['methods'].items()};detail=[]
        for tid in np.unique(a[a>=0]):
            mask=a==tid
            if mask.sum()<5:continue
            frac=float(crowded[mask].mean());name='high_overlap' if frac>=.5 else 'other';g=groups[name]
            g['truth_tracks']+=1;g['baseline_found']+=int(tid in found['raw_baseline']);g['T_found']+=int(tid in found['T'])
            detail.append(dict(truth_id=int(tid),overlap_fraction=frac,baseline_found=tid in found['raw_baseline'],T_found=tid in found['T']))
        events.append(dict(event_id=eid,tracks=detail))
    for g in groups.values():
        g['baseline_efficiency']=g['baseline_found']/g['truth_tracks'];g['T_efficiency']=g['T_found']/g['truth_tracks'];g['net']=g['T_found']-g['baseline_found']
    assert sum(g['truth_tracks'] for g in groups.values())==s['methods']['raw_baseline']['tracks']['truth_tracks']
    result=dict(definition='At least half axial wire-center positions have a different non-noise truth track wire-center within10mm. Diagnostic uses truth; wire centers are not actual hit coordinates.',groups=groups,events=events)
    (OUT/'overlap_strata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    path=OUT/'确认集评估报告.md'
    with path.open('a',encoding='utf-8') as f:
        f.write('\n\n## 空间重叠分层\n\n使用历史诊断约定：至少一半直丝线中心在10mm内有其他非噪声真轨的直丝线中心。真值仅用于评价分层；线中心不是实际hit坐标。\n\n|类别|真轨数|基线效率|T效率|净增正确轨|\n|---|---:|---:|---:|---:|\n')
        for k,g in groups.items():f.write('|%s|%d|%.3f%%|%.3f%%|%+d|\n'%(k,g['truth_tracks'],100*g['baseline_efficiency'],100*g['T_efficiency'],g['net']))
        f.write('\n这是固定模型测试后的描述性分层，未用来调整阈值。\n\n![确认集指标](confirmation_metrics.png)\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(14,4));labels=['Baseline','T25']
    for ax,section,metric,title,scale in zip(axes,['tracks','tracks','seeds',None],['track_efficiency','fake_fraction','efficiency','mean_decode_seconds'],['Axial efficiency (%)','Fake rate (%)','Coarse3D seed efficiency (%)','CPU decode seconds/event'],[100,100,100,1]):
        values=[m[section][metric]*scale if section else m[metric] for m in s['methods'].values()]
        ax.bar(labels,values,color=['#888888','#2a9d8f']);ax.set_title(title);ax.grid(axis='y',alpha=.2)
        for j,v in enumerate(values):ax.text(j,v,'%.3f'%v,ha='center',va='bottom')
        ax.set_ylim(0,max(values)*1.15)
    fig.tight_layout();fig.savefig(OUT/'confirmation_metrics.png',dpi=160);plt.close(fig)
    print(json.dumps(groups),flush=True)

if __name__=='__main__':main()
