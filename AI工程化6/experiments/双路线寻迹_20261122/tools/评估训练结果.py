"""Paired review of all completed member checks; local decoder ablations."""
from pathlib import Path
import sys,json,time,os
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np
DOWNLOAD=ROOT/'evaluation_download'
OUT=ROOT/'evaluation_20261122'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def aggregate(rows):
    from STEP5_两级寻迹 import summarize
    from STEP2_训练双路线 import seed_summary
    return dict(tracks=summarize([x['counts'] for x in rows]),seeds=seed_summary([x['seeds'] for x in rows]))

def summary():
    h=read(DOWNLOAD/'full/history.json');checks=[x for x in h if 'members' in x['validation']]
    best={k:max(checks,key=lambda r:r['validation']['members'][k]['seeds']['f1'])['epoch'] for k in ['E','T']}
    rows={k:{r['event_id']:r['methods'][k] for f in (DOWNLOAD/'full/validation'/('epoch_%03d'%best[k])).glob('member_*.json') for r in [read(f)]} for k in best}
    base={r['event_id']:r['methods']['raw_baseline'] for f in (DOWNLOAD/'full/validation'/('epoch_%03d'%best['T'])).glob('member_*.json') for r in [read(f)]}
    ids=sorted(base);assert len(ids)==200 and all(set(v)==set(ids) for v in rows.values())
    rng=np.random.default_rng(20261122);samples=rng.integers(0,len(ids),(10000,len(ids)))
    result=dict(best_epochs=best,events=200,last_complete_epoch=h[-1]['epoch'],methods={'raw_baseline':aggregate(list(base.values()))},paired={})
    for k in rows:
        result['methods'][k]=aggregate(list(rows[k].values()));gains=[];lost=[];wins=ties=losses=0
        for eid in ids:
            a=set(rows[k][eid]['found']);b=set(base[eid]['found'])
            gains.extend([[eid,int(t)] for t in sorted(a-b)]);lost.extend([[eid,int(t)] for t in sorted(b-a)])
            d=len(a)-len(b);wins+=d>0;losses+=d<0;ties+=d==0
        ci={}
        for label,section,num,den in [('track_efficiency','counts','matched_tracks','truth_tracks'),('fake_fraction','counts','fake_candidates','output_candidates'),('seed_efficiency','seeds','matched_tracks','truth_tracks')]:
            def calc(source):
                a=np.array([source[eid][section].get(num,0) for eid in ids]);b=np.array([source[eid][section].get(den,0) for eid in ids]);return a[samples].sum(1)/np.maximum(b[samples].sum(1),1)
            ci[label]=np.quantile(100*(calc(rows[k])-calc(base)),[.025,.975]).tolist()
        result['paired'][k]=dict(recovered=len(gains),lost=len(lost),net=len(gains)-len(lost),improved_events=int(wins),tied_events=int(ties),worsened_events=int(losses),delta_95ci_percentage_points=ci,recovered_tracks=gains,lost_tracks=lost)
    result['epoch_trend']=[dict(epoch=r['epoch'],E_class_f1=r['validation']['E']['macro_f1'],T_class_f1=r['validation']['T']['f1'],members=r['validation']['members']) for r in checks]
    save(OUT/'paired_summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for k in ['raw_baseline','E','T']:
        for ax,section,metric,title in zip(axes,['tracks','tracks','seeds'],['track_efficiency','fake_fraction','efficiency'],['Axial track efficiency (%)','Fake candidates (%)','Coarse 3D seed efficiency (%)']):
            ax.plot([r['epoch'] for r in checks],[100*r['validation']['members'][k][section][metric] for r in checks],marker='o',label=k);ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.25)
    axes[0].legend();fig.tight_layout();fig.savefig(OUT/'training_member_curves.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ['epoch_trend','paired']},ensure_ascii=False),flush=True)
    print('PAIRED',json.dumps({k:{a:b for a,b in v.items() if not a.endswith('_tracks')} for k,v in result['paired'].items()}),flush=True)

def ablate_one(eid):
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    import torch
    torch.set_num_threads(1)
    from et_decode import E_edges,T_groups,geometry_extract,seeds
    from STEP5_两级寻迹 import axial_two_pass,attach_stereo,score_tracks,DEFAULT_CONFIG
    from coarse_3d_membership import propose,select
    with np.load(DOWNLOAD/'cache/features'/('event_%05d.npz'%eid)) as s:f={k:s[k] for k in s.files}
    with np.load(DOWNLOAD/'cache/indices'/('event_%05d.npz'%eid)) as s:t=s['t'];ey=s['ey']
    with np.load(ROOT.parent.parent/'joint_event_dataset/events'/('event_%05d.npz'%eid)) as s:drift=s['d'];truth=s['truth']
    geometry=np.load(ROOT.parent.parent/'joint_event_dataset/wire_geometry.npy');geo=geometry[f['cells']]
    ep=np.load(DOWNLOAD/'full/validation/epoch_035'/('prediction_%05d.npz'%eid))['E'];tq=np.load(DOWNLOAD/'full/validation/epoch_025'/('prediction_%05d.npz'%eid))['T']
    started=time.perf_counter();baseline,_=axial_two_pass(geo,drift,f['pairs'],f['p']);base_seconds=time.perf_counter()-started
    result={}
    for key in ['raw_baseline','E','E_no_copy','E_oracle_color','T']:
        started=time.perf_counter()
        if key=='raw_baseline':tracks=baseline
        elif key=='T':groups,w=T_groups(f,t,tq);tracks=geometry_extract(groups,w,f,geo,drift,baseline)
        else:
            prob=np.eye(4,dtype=np.float32)[ey] if key=='E_oracle_color' else ep
            groups,w=E_edges(f,prob,copy=key!='E_no_copy');tracks=geometry_extract(groups,w,f,geo,drift,baseline)
        joint,_=attach_stereo(tracks,geo,drift,f['cells'],f['pairs'],f['p'],DEFAULT_CONFIG)
        counts,found=score_tracks(joint,truth,f['stereo'],DEFAULT_CONFIG)
        confirmed,_=select(joint,propose(joint,geo,drift,f['cells'],f['pairs'],f['p'],refine=False),geo,f['cells'],'guarded_loose')
        result[key]=dict(counts=counts,found=found,seeds=seeds(confirmed,truth,f['stereo'],f['cells']),seconds=time.perf_counter()-started+base_seconds)
    save(OUT/'ablations'/('event_%05d.json'%eid),dict(event_id=eid,methods=result));return eid

def ablations():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    ids=read(DOWNLOAD/'full/contract.json')['validation_ids'][:16]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(ablate_one,eid) for eid in ids]):print('ablated',f.result(),flush=True)
    packets=[read(OUT/'ablations'/('event_%05d.json'%eid)) for eid in ids]
    out={k:dict(aggregate([r['methods'][k] for r in packets]),mean_decode_seconds=float(np.mean([r['methods'][k]['seconds'] for r in packets]))) for k in packets[0]['methods']}
    save(OUT/'ablation_summary.json',dict(events=16,scope='Diagnostic validation subset, oracle uses truth only for upper-bound diagnosis; CPU decoding with 2 workers, excludes neural inference and feature preparation',methods=out));print('ABLATIONS',json.dumps(out),flush=True)

def report():
    r=read(OUT/'paired_summary.json');a=read(OUT/'ablation_summary.json');w=read(OUT/'weights_verification.json')
    lines=['# 双路线训练与寻迹评估：2026-11-22','',
      '结论：T三元组路线取得小幅收益；当前E二号方案原型退化，暂不替换稳定流程。两者从第5到45轮的成员曲线均接近平台，不建议仅为凑满60轮继续消耗GPU。', '',
      '## 训练状态','',
      '作业1858533在gpu11运行六小时，于北京时间2026-11-22 10:46:36因TIME LIMIT结束，并非正常收敛结束。完成49轮完整训练与分类验证，第50轮7000个事件的参数更新已完成，成员验证保存45/200例。第50轮不完整评估不进入比较；该轮状态已下载，可续接。', '',
      '## 相同200例验证事例上的对照','',
      '按粗三维种子F1分别选择E第35轮、T第25轮。5969条可评价直丝真轨、5968条满足斜丝条件的真轨。直丝匹配要求至少5个直丝命中，纯度≥80%、完整度≥50%；粗三维种子要求初步几何判据及至少两个正确斜丝CellID和斜丝纯度。它不是GenFit精确拟合成功率。', '',
      '|方法|直丝效率|假轨率|克隆率|粗三维种子效率|种子假轨率|',
      '|---|---:|---:|---:|---:|---:|']
    for k,m in r['methods'].items():
        t=m['tracks'];s=m['seeds'];lines.append('|%s|%.3f%%|%.3f%%|%.3f%%|%.3f%%|%.3f%%|'%(k,100*t['track_efficiency'],100*t['fake_fraction'],100*t['clone_fraction'],100*s['efficiency'],100*s['fake_fraction']))
    lines+=['','T：找回原先漏掉的22条，同时损失原先找到的4条，净增18条；假候选156→136，克隆3→1。粗三维正确种子5610→5630。克隆数量很小，不宜据此外推稳定的三倍改善。', '',
      'E：找回15条，同时损失161条，净少146条；假候选156→286。其共用边分类能力不能直接转化为成员提取质量。', '',
      '以事件为单位做10000次配对bootstrap：T效率增益95%区间为+0.116～+0.518个百分点，假轨率变化为−0.550～−0.151个百分点；E效率变化为−2.982～−1.942个百分点。区间条件于当前验证样本和所选权重，未校正选轮偏差。', '',
      '注意：这200例同时用于训练期间选轮，不能当成独立盲测。基线是原AA两级提取＋相同AS/SS附着和guarded_loose粗三维，不含历史hybrid校准器；不能与旧完整部署指标直接混比。', '',
      '![各轮成员指标](training_member_curves.png)', '',
      '## 本机16例重放与E消融诊断','',
      '选取固定验证序列前16例；数据与200例重叠，仅用于定位机制。oracle-color输入正确颜色标签，属于真值辅助诊断，不能作为实际性能。', '',
      '|方法|直丝效率|假轨率|粗三维种子效率|平均CPU解码秒/事件|',
      '|---|---:|---:|---:|---:|']
    for k,m in a['methods'].items():
        lines.append('|%s|%.3f%%|%.3f%%|%.3f%%|%.3f|'%(k,100*m['tracks']['track_efficiency'],100*m['tracks']['fake_fraction'],100*m['seeds']['efficiency'],m['mean_decode_seconds']))
    replay={}
    for k,epoch in [('raw_baseline',25),('E',35),('T',25)]:
        diffs=[]
        for f in (OUT/'ablations').glob('event_*.json'):
            local=read(f);remote=read(DOWNLOAD/'full/validation'/('epoch_%03d'%epoch)/('member_%05d.json'%local['event_id']))
            for section in ['counts','seeds']:
                for metric in ['matched_tracks','fake_candidates','clone_candidates','output_candidates']:
                    if local['methods'][k][section].get(metric,0)!=remote['methods'][k][section].get(metric,0):diffs.append([local['event_id'],section,metric])
        replay[k]=diffs
    save(OUT/'replay_verification.json',replay)
    lines+=['', '消融发现：E正确颜色标签仅把正确直丝轨数446→447，仍低于基线457，说明颜色分类不是唯一瓶颈，束内配对、组级提取和候选竞争仍需改进。开启复制相对关闭复制442→446，表明这一小样本中复制有帮助，但未弥补整体退化。这不能证明完整的二号思路失败。', '',
      'T在16例中正确直丝457→462；CPU解码3.33→4.28秒/例，约增加28.5%。质量提升当前伴随解码成本增加，尚未实现提速。', '',
      'CPU耗时为两个单线程worker并发时的每事件实测，包含原直丝候选生成、对应解码、斜丝附着和粗三维；不含主干特征准备和神经网络推理，不能作为全流程吞吐。', '',
      '本机重放与服务器保存记录的计数差异：'+json.dumps(replay,ensure_ascii=False), '',
      '最佳权重已在本机RTX4060加载并推理事例232：E/T与V100缓存概率最大差异分别为0.000415/0.000488，符合混合精度跨设备差异；最佳权重内容分别与保存的第35/25轮一致。', '',
      '## 后续优先级','',
      '1. 先保留T第25轮作为候选，在未用于选轮的确认集上检查收益和耗时，再决定接入。',
      '2. 分析T丢失的4条，约束候选竞争时的原直丝损耗。',
      '3. E先定位普通真束误剔除、束内hit匹配及长共用段入口—出口配对；当前实现限制为最多3通道、新几何组最多80个直丝命中。不能把首版退化当成二号思路无价值的证明。',
      '4. 首版E把各束独立匹配后再做连通分组，尚未实现跨整段的全局通道连续性。优先改进该机制，再判断是否需要增加模型容量或训练轮数。', '',
      '原始数据见paired_summary.json、ablation_summary.json、weights_verification.json、replay_verification.json；服务器下载原件保留在相邻evaluation_download目录。']
    (OUT/'模型评估报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT',OUT/'模型评估报告.md')

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if '--ablations' in sys.argv:ablations()
    elif '--report' in sys.argv:report()
    else:summary()
