from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'calibrated_tracking_20261122'

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    r=read(OUT/'confirmation/summary.json');c=read(OUT/'confirmation/contract.json');d=read(OUT/'development/contract.json');assert r['events']==200
    for field in ['alignment_sha256','decoder_sha256','runner_sha256','initial_sha256','focused_sha256','preselection_budget']:assert c[field]==d[field],field
    split=read(ROOT.parent/'三元组寻迹计划_20261122/split_manifest.json');assert not set(c['ids'])&(set(split['original_train_event_ids'])|set(split['periodic_member_validation_event_ids'])|set(split['validation_confirm_event_ids']))
    rows=[read(OUT/'confirmation/events'/('event_%05d.json'%eid)) for eid in c['ids']];sample=np.random.default_rng(20261123).integers(0,200,(5000,200));ci={}
    for key in ['T10_cal_old','T10_cal_path']:
        ci[key]={}
        for label,section,numerator,denominator in [('efficiency','counts','matched_tracks','truth_tracks'),('fake_fraction','counts','fake_candidates','output_candidates'),('seed_efficiency','seeds','matched_tracks','truth_tracks')]:
            values={}
            for method in ['T25_old',key]:
                a=np.array([x['methods'][method][section].get(numerator,0) for x in rows]);b=np.array([x['methods'][method][section].get(denominator,0) for x in rows]);values[method]=a[sample].sum(1)/np.maximum(b[sample].sum(1),1)
            ci[key][label]=np.quantile(100*(values[key]-values['T25_old']),[.025,.975]).tolist()
    (OUT/'confirmation/paired_intervals.json').write_text(json.dumps(ci,indent=2),encoding='utf-8')
    lines=['# 校准新T与路径状态解码：确认测试','',
      '本次先在原200例开发验证上固定分数对齐，在固定16例开发事件上检查提取流程，再冻结规则，使用另行预留的200例确认。确认事件与T微调训练、选轮验证及此前查看的500例均不重叠；原主干历史使用过原验证集，因此不是整个流水线的全新盲测。', '',
      '## 固定设置','',
      '新模型为接口微调第10轮；直丝三元组与其余三元组分别做单调logit分段映射，在旧分数0.2/0.5/0.9处对齐开发集精度，不将其称为真实后验概率。映射后，旧三元组提取器仍用0.9，路径搜索仍用0.2，保持原分数使用约定。', '',
      '路径原型仅把选择前粗三维证据预算固定为64个配对种子、保留3个、池最多32；最终粗三维评价仍为512个配对种子。beam最多4条、128个种子、每路径最多32个命中、最多拟合48个路径候选。确认期间未调参。', '',
      '## 200例确认结果','',
      '|方法|正确直丝轨|直丝效率|假轨率|克隆率|粗三维种子效率|CPU解码秒/例|','|---|---:|---:|---:|---:|---:|---:|']
    for key,m in r['methods'].items():
        t=m['tracks'];s=m['seeds'];lines.append('|%s|%d|%.3f%%|%.3f%%|%.3f%%|%.3f%%|%.3f|'%(key,t['matched_tracks'],100*t['track_efficiency'],100*t['fake_fraction'],100*t['clone_fraction'],100*s['efficiency'],m['mean_seconds']))
    lines+=['','baseline为原两级；T25_old为原T25＋原三元组提取器；T10_cal_old只替换为校准后的新头；T10_cal_path同时替换为路径状态解码。基线均使用相同联合GNN、原AS/SS评分与guarded_loose粗三维，不含旧hybrid校准器。', '',
      '粗三维指标是成员种子质量，非GenFit精确拟合成功率。CPU耗时为6个单线程worker并发下测量，包含候选、斜丝附着和粗三维，排除共享主干特征准备及GPU小头评分。', '',
      '## 相对原T25提取器的逐事件得失','']
    for k in ['T10_cal_old','T10_cal_path']:
        p=r['paired'][k];lines+=['- %s：找回%d条、损失%d条，净增%+d条。'%(k,len(p['gained']),len(p['lost']),p['net'])]
    lines+=['','以事件为单位做5000次配对bootstrap，变化区间单位为百分点：','', '```json',json.dumps(ci,ensure_ascii=False,indent=2),'```','',
      '此确认集现已被查看；后续若据此调整模型、分数或解码规则，不能继续把它当作下一版的独立确认。', '',
      '## 判断规则与证据','',
      '分别评价模型变更和提取器变更；三元组分类召回的改善不能代替上述成员结果。若成员效率、假轨/克隆或耗时退化，保留旧稳定流程，不因模型较新而自动替换。', '',
      '原始逐事件结果、协议、得失径迹清单与统计区间保留在confirmation/。分数映射保留在alignment.json。全部测试在本机完成，没有新增GPU训练配额。', '',
      '![确认集比较](confirmation/comparison.png)']
    (OUT/'校准与寻迹确认报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(15,4));names=list(r['methods']);labels=['Baseline','T25 old','T10 cal old','T10 cal path']
    for ax,field,title in zip(axes,['track_efficiency','fake_fraction','seed','time'],['Axial efficiency (%)','Fake candidates (%)','Coarse3D seed efficiency (%)','CPU seconds/event']):
        vals=[r['methods'][k]['seeds']['efficiency']*100 if field=='seed' else r['methods'][k]['mean_seconds'] if field=='time' else r['methods'][k]['tracks'][field]*100 for k in names]
        ax.bar(range(4),vals,color=['gray','#277da8','#43aa8b','#f8961e']);ax.set_xticks(range(4),labels,rotation=25,ha='right');ax.set_title(title);ax.set_ylim(0,max(vals)*1.17);ax.grid(axis='y',alpha=.2)
        for i,v in enumerate(vals):ax.text(i,v,'%.3f'%v,ha='center',va='bottom',fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'confirmation/comparison.png',dpi=160);plt.close(fig)
    print('PAIRED_CI',json.dumps(ci),flush=True)

if __name__=='__main__':main()
