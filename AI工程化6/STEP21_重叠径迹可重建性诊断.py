"""Survey cached joint tracking; run truth-assisted witnesses on representative events."""
from pathlib import Path
import argparse,csv,json,sys,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent
EXP=ROOT/'experiments/重叠可重建性_20261122'
sys.path.insert(0,str(EXP))
import reconstructibility as r
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False


def check_helpers():
    # Same wire can still carry distinguishable drift measurements.
    cells=np.array([1,2,1,3]);drift=np.array([.2,.4,.8,.1]);truth=np.array([10,10,20,20])
    xy=np.array([[0.,0.],[20.,0.],[0.,0.],[40.,0.]])
    f=r.overlap_features(cells,drift,truth,xy,np.ones(4,dtype=bool))
    assert f[10]['shared_cell_fraction']==.5 and f[10]['exact_input_equivalent_fraction']==0.
    drift[2]=drift[0];f=r.overlap_features(cells,drift,truth,xy,np.ones(4,dtype=bool))
    assert f[10]['exact_input_equivalent_fraction']==.5
    pairs=np.array([[0,1,2],[1,2,3]])
    labels=r.labels_at(4,pairs,np.array([True,False,True]));assert labels[0]==labels[1] and labels[2]==labels[3] and labels[0]!=labels[2]


def survey_summary(rows,event_ids):
    result=dict(events=len(event_ids),eligible_tracks=len(rows),matched=sum(x['found'] for x in rows),missed=sum(not x['found'] for x in rows))
    result['threshold_sensitivity']={}
    for radius in [2,5,10,20]:
        key='foreign_within_%dmm_fraction'%radius;high=[x for x in rows if x[key]>=.5];low=[x for x in rows if x[key]<.5]
        hm=sum(not x['found'] for x in high);lm=sum(not x['found'] for x in low)
        result['threshold_sensitivity'][radius]=dict(high_overlap_tracks=len(high),high_overlap_missed=hm,high_overlap_miss_rate=hm/max(len(high),1),
            low_overlap_tracks=len(low),low_overlap_missed=lm,low_overlap_miss_rate=lm/max(len(low),1),fraction_of_misses_high_overlap=hm/max(hm+lm,1))
    for group,sub in [('missed',[x for x in rows if not x['found']]),('found',[x for x in rows if x['found']])]:
        result[group+'_details']=dict(tracks=len(sub),mean_shared_cell_fraction=float(np.mean([x['shared_cell_fraction'] for x in sub])),
            any_exact_equivalent_tracks=sum(x['exact_input_equivalent_fraction']>0 for x in sub),
            graph_categories=dict(Counter(x['graph_diagnosis'] for x in sub)))
    result['by_split']={p:dict(tracks=len(sub),missed=sum(not x['found'] for x in sub),
        high_overlap_missed=sum(not x['found'] and x['foreign_within_10mm_fraction']>=.5 for x in sub)) for p in ['train','validation','test'] if (sub:=[x for x in rows if x['partition']==p])}
    # Event bootstrap retains within-event dependence. Association is not causality.
    event_table=[]
    for eid in event_ids:
        sub=[x for x in rows if x['event_id']==eid];high=[x for x in sub if x['foreign_within_10mm_fraction']>=.5];low=[x for x in sub if x['foreign_within_10mm_fraction']<.5]
        event_table.append([len(high),sum(not x['found'] for x in high),len(low),sum(not x['found'] for x in low)])
    table=np.asarray(event_table);samples=np.random.default_rng(20261122).integers(len(table),size=(2000,len(table)))
    totals=table[samples].sum(1);delta=totals[:,1]/np.maximum(totals[:,0],1)-totals[:,3]/np.maximum(totals[:,2],1)
    result['high_minus_low_miss_rate_event_bootstrap95']=np.quantile(delta,[.025,.975]).tolist()
    return result


def plot_case(detail):
    eid=detail['event_id'];r.initialize();cells,d,truth,hit,geo,stereo,pairs,p,packet=r.load(eid)
    focus=sorted(detail['focus'],key=lambda x:(not x['found'],x['foreign_within_10mm_fraction'],x['axial_hits']),reverse=True)[0]
    tid=focus['truth_id'];other=focus['primary_competitor'];targets=[tid,other] if other is not None else [tid]
    fig,axes=plt.subplots(2,3,figsize=(19,11));aa=np.flatnonzero(~stereo)
    colors={t:plt.get_cmap('tab20')(j%20) for j,t in enumerate(sorted(set(truth[aa])))}
    for t in sorted(set(truth[aa])):
        ix=aa[truth[aa]==t];axes[0,0].scatter(*geo[ix,:2].T,s=14,c=[colors[t]],marker='x')
    for t in detail['missed_axial']:
        ix=aa[truth[aa]==t];axes[0,0].scatter(*geo[ix,:2].T,s=50,facecolors='none',edgecolors='#c2185b');axes[0,0].annotate('漏轨T%d'%t,geo[ix,:2].mean(0),fontsize=7,color='#c2185b')
    axes[0,0].set_title('全事件直丝投影：紫红圈为漏轨')
    zoom=np.concatenate([np.flatnonzero((truth==t)&~stereo) for t in targets]);xy=geo[zoom,:2]
    lower=xy.min(0)-15;upper=xy.max(0)+15
    for ax in axes[0,1:]:ax.scatter(*geo[aa,:2].T,s=7,c='lightgray');ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]))
    for number,t in enumerate(targets):
        ix=np.flatnonzero((truth==t)&~stereo);color=['#c2185b','#0072b2'][number]
        axes[0,1].scatter(*geo[ix,:2].T,s=28,facecolors='none',edgecolors=color,marker=['o','s'][number],label='T%d'%t)
        for i in ix:axes[0,1].add_patch(Circle(geo[i,:2],d[i],fill=False,color=color,lw=.7,alpha=.8))
        for i in ix:axes[0,1].annotate('%.2f'%d[i],geo[i,:2],fontsize=6,color=color)
    axes[0,1].set_title('重叠轨迹与漂移圆；数字为d/mm');axes[0,1].legend()
    selected=np.isin(truth[pairs[0]],targets)|np.isin(truth[pairs[1]],targets)
    selected&=~stereo[pairs[0]]&~stereo[pairs[1]]&(p>=.575)
    same=truth[pairs[0]]==truth[pairs[1]]
    for mask,color,label in [(selected&same,'green','真边≥0.575'),(selected&~same,'red','跨轨假边≥0.575')]:
        axes[0,2].add_collection(LineCollection(geo[pairs[:,mask].T,:2],colors=color,alpha=.5,lw=.8,label=label))
    axes[0,2].set_title('局部真边和跨轨连接');axes[0,2].legend()
    with np.load(r.OUT/'details'/('event_%05d_residuals.npz'%eid)) as f:
        circle,three,model_ids=f['circle'],f['three_d'],f['model_ids'].tolist()
    for ax,matrix,mask,title in [(axes[1,0],circle,~stereo,'直丝：本轨圆残差与其他轨圆残差'),(axes[1,1],three,stereo,'斜丝：本轨三维残差与其他轨三维残差')]:
        ix=np.flatnonzero((truth==tid)&mask);k=model_ids.index(tid);own=matrix[ix,k];foreign=np.delete(matrix[ix],k,axis=1).min(1);finite=np.isfinite(own)&np.isfinite(foreign)
        ax.scatter(own[finite],foreign[finite],s=23,color='#c2185b');cap=max(2.,min(20.,float(np.max(np.r_[own[finite],foreign[finite]])) if finite.any() else 2.))
        ax.plot([0,cap],[0,cap],'k--',lw=.8);ax.set(xlim=(0,cap),ylim=(0,cap),xlabel='本轨模板 |残差| / mm',ylabel='最佳竞争模板 |残差| / mm',title=title)
        ax.text(.03,.97,'虚线上方：本轨几何更兼容\n模板由真值分组拟合，仅作诊断',va='top',transform=ax.transAxes,fontsize=8)
    axes[1,2].axis('off');lines=['事件%d，重点T%d，邻近T%s'%(eid,tid,other),'原始：%d条漏轨，%d条仅缺三维种子'%(len(detail['missed_axial']),len(detail['seed_only_miss']))]
    for x in detail['focus']:
        m=x['physical'];lines.append('T%d A%d/S%d 圆RMS %.2f mm\n  去假边%s；理想真边%s；三维独立丝支持%d'%(x['truth_id'],x['axial_hits'],m['stereo_hits'],m.get('axial_rms_mm',np.nan),
            '找回' if x['oracle_remove_false_found'] else '未找回','找回' if x['oracle_perfect_graph_found'] else '未找回',m.get('supported_distinct_stereo_cells',0)))
    axes[1,2].text(0,1,'\n'.join(lines),va='top',fontsize=9)
    for ax in axes.ravel()[:3]:ax.set(xlabel='丝中心x / mm',ylabel='丝中心y / mm');ax.set_aspect('equal');ax.grid(alpha=.2)
    for ax in axes[1,:2]:ax.grid(alpha=.2)
    fig.suptitle('EvtID=%d｜直丝+斜丝联合epoch126｜重叠可重建性诊断（真值干预不是实际效率）'%eid,fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96));path=r.OUT/'details'/('event_%05d_diagnosis.png'%eid);fig.savefig(path,dpi=140);plt.close(fig);return path


def report(summary,details,rows):
    out=r.OUT;stat=summary['threshold_sensitivity'][10]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar(['高投影重叠','较低投影重叠'],[stat['high_overlap_miss_rate']*100,stat['low_overlap_miss_rate']*100],color=['#c2185b','#0072b2'])
    axes[0].set(ylabel='直丝漏轨率 / %',title='至少一半直丝hit附近10mm内有其他真轨hit')
    axes[1].bar(['高投影重叠漏轨','其他漏轨'],[stat['high_overlap_missed'],stat['low_overlap_missed']],color=['#c2185b','#777777']);axes[1].set(ylabel='条数',title='501个事例的漏轨组成')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'overlap_summary.png',dpi=150);plt.close(fig)
    focuses=[x for d in details for x in d['focus'] if not x['found']]
    stats=dict(detail_events=[d['event_id'] for d in details],missed_tracks=len(focuses),
        restored_by_removing_false_edges=sum(x['oracle_remove_false_found'] for x in focuses),restored_by_perfect_candidate_scores=sum(x['oracle_perfect_graph_found'] for x in focuses),
        restored_denoised_nonlocking=sum(x['oracle_denoised_nonlocking_found'] for x in focuses),
        restored_oracle_circle_assignment=sum(x['oracle_circle_assignment_found'] for x in focuses),
        restored_oracle_3d_assignment=sum(x['oracle_3d_assignment_found'] for x in focuses),
        restored_oracle_3d_cell_matching=sum(x['oracle_3d_cell_matching_found'] for x in focuses),
        axial_witness=sum(x['has_axial_reconstruction_witness'] for x in focuses),stereo_geometry_witness=sum(x['has_3d_geometry_witness'] for x in focuses))
    requested=[x for event in details if event['event_id'] in [1500,1501,1503,1504] for x in event['focus'] if not x['found']]
    summary['requested_four_events']=dict(missed_tracks=len(requested),removed_false_found=sum(x['oracle_remove_false_found'] for x in requested),
        denoised_nonlocking_found=sum(x['oracle_denoised_nonlocking_found'] for x in requested),perfect_graph_found=sum(x['oracle_perfect_graph_found'] for x in requested),
        geometry_witness=sum(x['has_3d_geometry_witness'] for x in requested),oracle_circle_assignment_found=sum(x['oracle_circle_assignment_found'] for x in requested),
        oracle_3d_assignment_found=sum(x['oracle_3d_assignment_found'] for x in requested),
        oracle_3d_cell_matching_found=sum(x['oracle_3d_cell_matching_found'] for x in requested))
    summary['detailed_witnesses']=stats;r.e.save_json(out/'summary.json',summary)
    lines=['# 重叠径迹可重建性诊断','',
        '固定使用直丝、斜丝联合第126轮权重的已存预测和稳定寻迹结果。这里只做离线诊断，不修改训练清单、正在运行的训练或稳定部署。',
        '',f"扫描{summary['events']}个事例，共{summary['eligible_tracks']}条可评价真轨；已找到{summary['matched']}条，漏掉{summary['missed']}条。",
        '',f"按至少一半直丝hit在10mm内存在其他真轨的直丝丝中心定义高投影重叠：漏轨中{stat['high_overlap_missed']}/{summary['missed']}（{stat['fraction_of_misses_high_overlap']:.1%}）满足。高重叠组漏轨率{stat['high_overlap_miss_rate']:.2%}，其余{stat['low_overlap_miss_rate']:.2%}。",
        '', '这些是丝中心二维投影重叠，不能解释为实际三维hit重合；统计是关联，不单独证明漏轨因果。2/5/10/20mm敏感性、按原train/validation/test分组和事例bootstrap结果见summary.json。',
        '', '## 代表事例逐轨结果','',
        '|事件|真轨|A/S命中|10mm重叠比例|原结果|只去假边|去假边且取消两级锁点|理想真边|真值圆RMS/mm|正确斜丝三维支持丝数|',
        '|---|---|---|---|---|---|---|---|---|---|']
    for event in details:
        for x in event['focus']:
            m=x['physical'];lines.append('|%d|T%d|%d/%d|%.0f%%|%s|%s|%s|%s|%.3f|%d|'%(event['event_id'],x['truth_id'],x['axial_hits'],m['stereo_hits'],100*x['foreign_within_10mm_fraction'],
                '直丝漏轨' if not x['found'] else '仅缺三维种子','找回' if x['oracle_remove_false_found'] else '未找回','找回' if x['oracle_denoised_nonlocking_found'] else '未找回',
                '找回' if x['oracle_perfect_graph_found'] else '未找回',m.get('axial_rms_mm',np.nan),m.get('supported_distinct_stereo_cells',0)))
    lines+=['','## 如何解释','',
        '去假边即可找回：原观测与原真边分数已有一定重建支撑，跨轨竞争或提取流程是主要嫌疑。去假边后还需取消两级锁点才找回：提取过程先锁定短碎片造成损失的证据。当前阈值下纯真边图仍碎裂、理想真边才能找回：评分与阈值连通性是瓶颈。两者都失败时，还应检查当前圆模型、裁点和提取规则，不能立即判成物理不可重建。',
        '',f"你指定四个事例的{len(requested)}条直丝漏轨：去假边找回{summary['requested_four_events']['removed_false_found']}条；去假边并取消两级锁点找回{summary['requested_four_events']['denoised_nonlocking_found']}条；理想真边找回{summary['requested_four_events']['perfect_graph_found']}条；具有正确斜丝三维几何支持{summary['requested_four_events']['geometry_witness']}条。",
        '',f"另外，以真值分组拟合出的全部轨迹作为模板，再仅按测得的漂移距离和有限丝几何重新归属hit：二维圆模板找回上述漏轨{summary['requested_four_events']['oracle_circle_assignment_found']}条，三维模板找回{summary['requested_four_events']['oracle_3d_assignment_found']}条。该实验仍然使用真值种子，不能当成实际算法效果或独立验证，只用于检查测量残差中是否存在分离信号。",
        '',f"在三维模板基础上，再按每根丝做全局一对一分配，找回上述漏轨{summary['requested_four_events']['oracle_3d_cell_matching_found']}条。这仍是依赖真值模板的条件实验，且要求单轨不重复经过同一根丝；各事件是否违反此假设记录在oracle_trials中。与逐点最近模板相比，它检查了重复丝命中之间的竞争约束价值。",
        '', '不能把理想连边找回16条等同于已经证明16条都能被无真值算法唯一恢复。几何模板实验仍会混淆部分轨迹，说明有几何可行性与现有信息能否可靠区分是两个问题；后者还取决于漂移测量误差、候选假设质量及联合竞争。',
        '', '真值分组圆拟合、真值斜丝池和真值模板归属都使用了离线标签，是可行性证据或局限性定位，不是无真值寻迹算法，也不能作为实际重建效率。模板内样本参与拟合，模板归属指标不是泛化指标。',
        '', '三维诊断使用有限丝长、倾角、漂移半径以及粗螺旋约束；独立斜丝支持至少3根是当前程序的验证门槛。两个正确斜丝点可以生成候选，但左右漂移分支或相位杠杆不足可能留下多个解。未提供精确测量误差模型，因此本诊断不宣称证明不可辨识。',
        '', '缺少三维几何证据，可能来自非单圆轨迹、左右分支、粗拟合偏差或搜索预算；不等于该轨迹在物理上不可恢复。',
        '', '## 图片','', '![总体重叠统计](overlap_summary.png)']
    for event in details:lines+=['',f"![事件{event['event_id']}六联图](details/event_{event['event_id']:05d}_diagnosis.png)"]
    (out/'诊断报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import html
    images=['overlap_summary.png']+['details/event_%05d_diagnosis.png'%d['event_id'] for d in details]
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>重叠可重建性诊断</title><style>body{font:16px sans-serif;margin:24px;background:#f7f7f7}img{max-width:100%;background:white}a{display:block;margin:20px 0}</style><h1>重叠径迹可重建性诊断</h1><p>联合epoch126；真值干预仅为诊断，不是实际重建效率。</p><a href="诊断报告.md">逐轨报告</a>'+''.join('<a href="%s"><img loading="lazy" src="%s"></a>'%(html.escape(i),html.escape(i)) for i in images),encoding='utf-8')
    return stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--event-range',nargs=2,type=int,default=[1500,2000]);p.add_argument('--detail-events',nargs='+',type=int,default=[1500,1501,1503,1504]);p.add_argument('--extra-events',type=int,default=2);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    check_helpers();r.e.verify_pins();out=r.OUT
    for name in ['survey','details']:(out/name).mkdir(parents=True,exist_ok=True)
    ids=list(range(args.event_range[0],args.event_range[1]+1));assert set(args.detail_events)<=set(ids)
    contract=dict(event_ids=ids,representatives=args.detail_events,extra_event_selection='largest count of high-overlap missed tracks, then largest missed count',
        weight_sha256=r.e.CONTRACT['weight_sha256'],dataset_sha256=r.e.sha(r.e.DATA_ROOT/'dataset_metadata_joint_no_time.json'),
        source_hashes={str(f):r.e.sha(f) for f in [Path(__file__),EXP/'reconstructibility.py']},truth_assisted=True,deployment_changed=False)
    r.e.save_json(out/'contract.json',contract);survey={};started=time.time()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=r.initialize) as pool:
        futures=[pool.submit(r.light,eid) for eid in ids]
        for future in as_completed(futures):
            row=future.result();survey[row['event_id']]=row
            if len(survey)%50==0:print(json.dumps(dict(phase='overlap_survey',completed=len(survey),total=len(ids),seconds=time.time()-started)),flush=True)
    rows=[x for eid in ids for x in survey[eid]['tracks']];summary=survey_summary(rows,ids)
    if ids==list(range(1500,2001)):
        old=r.read(r.BATCH/'summary.json');assert summary['eligible_tracks']==old['metrics']['truth_tracks'] and summary['matched']==old['metrics']['matched_tracks']
    fields=[k for k in rows[0] if k not in ['largest_true_component','best_overlap_group']]
    with (out/'all_tracks.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    ranks=sorted(ids,key=lambda eid:(sum(not x['found'] and x['foreign_within_10mm_fraction']>=.5 for x in survey[eid]['tracks']),sum(not x['found'] for x in survey[eid]['tracks'])),reverse=True)
    detail_ids=list(dict.fromkeys(args.detail_events+[eid for eid in ranks if eid not in args.detail_events][:args.extra_events]));r.e.save_json(out/'selected_details.json',detail_ids)
    details=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=r.initialize) as pool:
        for future in as_completed([pool.submit(r.detail,eid) for eid in detail_ids]):
            row=future.result();details.append(row);print(json.dumps(dict(phase='oracle_and_geometry',event_id=row['event_id'],focus_tracks=len(row['focus']),seconds=row['seconds'])),flush=True)
    details.sort(key=lambda d:detail_ids.index(d['event_id']))
    for event in details:plot_case(event)
    stats=report(summary,details,rows);r.e.verify_pins();print(json.dumps(dict(status='completed',summary=summary,detailed=stats,seconds=time.time()-started,output=str(out))),flush=True)


if __name__=='__main__':main()
