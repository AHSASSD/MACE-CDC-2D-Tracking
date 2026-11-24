"""Summarize exact case transitions without reassigning the original cohorts."""
from pathlib import Path
import json
import csv

root=Path(__file__).resolve().parent
data=json.loads((root/'summary.json').read_text(encoding='utf-8'))
models=data['models'];keys=['control_012','balanced_012','balanced_015']
labels={'control_012':'原Focal +12轮','balanced_012':'整轨辅助目标 +12轮','balanced_015':'整轨辅助目标 +15轮'}
groups=list(models[keys[0]]['groups'])
lines=['# 原223条漏轨的固定分组复盘','',
       '这里固定原epoch126诊断200事件中漏掉的同一223条径迹，沿用原五类标签。新模型重新分类只用于观察状态变化，不改变原分组分母。重建调用冻结稳定hybrid与粗三维流程；真值仅在重建之后做归因。原诊断数据已被多次查看，本表不是额外测试成绩。','',
       '## 最终实际找回','',
       '| 原类别 | 原漏轨数 | 原Focal +12轮找回 | 整轨辅助 +12轮找回 | 整轨辅助 +15轮找回 |','| --- | ---: | ---: | ---: | ---: |']
for g in groups:
    d=models[keys[0]]['groups'][g]
    lines.append('| %s | %d | %d | %d | %d |'%(d['label'],d['total'],*[models[k]['groups'][g]['recovered'] for k in keys]))
lines+=['','这里的“找回”只统计原来的223条漏轨，不能直接当作总效率增加。三组也会失去部分原正确匹配：','',
        '| 模型 | 找回原漏轨 | 丢失原正确径迹 | 净变化 | 正确直丝总数 | 假轨 | 克隆 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for k in keys:
    m=models[k];a=m['actual_tracks'];assert a['matched_tracks']==5746+m['recovered']-m['lost']
    lines.append('| %s | %d | %d | %+d | %d | %d | %d |'%(labels[k],m['recovered'],m['lost'],m['recovered']-m['lost'],a['matched_tracks'],a.get('fake_candidates',0),a.get('clone_candidates',0)))
lines+=['','同一200事件的粗三维种子对照（可评真径迹5968条）：','',
        '| 模型 | 正确种子 | 假种子 | 克隆种子 |','| --- | ---: | ---: | ---: |']
for k in keys:
    s=models[k]['coarse_seed'];lines.append('| %s | %d | %d | %d |'%(labels[k],s['matched_tracks'],s.get('fake_candidates',0),s.get('clone_candidates',0)))
lines+=['','## 评分与连通性','',
        '下表只看整轨辅助目标第12轮。连通性仍按原定义：只保留属于同一真径迹的AA候选边，在0.575阈值下最大连通块达到max(5, ceil(该轨AA命中数/2))。这是有真值的离线诊断，不是线上去错边。假边接受率是与该真径迹相接的AA假边中，分数达到0.575的比例，按径迹等权汇总。','',
        '| 原类别 | 原本纯图够连通 | 现在纯图够连通 | 真边均分变化 | 假边接受比例变化/百分点 |','| --- | ---: | ---: | ---: | ---: |']
for g in groups:
    d=models['balanced_012']['groups'][g];old=d['total'] if g=='pure_connected_at_existing_relaxed_cut' else 0
    score='-' if d['mean_true_score_change'] is None else '%+.5f'%d['mean_true_score_change']
    false='-' if d['mean_incident_false_acceptance_change'] is None else '%+.4f'%(d['mean_incident_false_acceptance_change']*100)
    lines.append('| %s | %d | %d | %s | %s |'%(d['label'],old,d['now_connected'],score,false))
lines+=['','五类是原模型的现象标签，不是互相独立的因果机制。真边均分上升可能只是校准变化；原已连通病例找回，可能涉及假边、几何拟合或候选竞争，不能仅凭找回数断言改好了某个提取器环节。','',
        '所有新旧候选边索引都逐事件核对一致。因此AA候选图的结构不足没有被训练补齐；新模型可以改变已有边的权重，联合AS/SS证据与几何也可能间接绕过一部分AA问题。','',
        '## 继续训练的证据','',
        '本轮是从epoch126出发、计划最多18轮的微调，实际因配额余量在两组15轮后停止。15/18不是从零模型的训练总轮数。下面是服务器整轨辅助目标的原两级成员验证，不与上表稳定hybrid结果混为同一提取器。','',
        '| 微调轮次 | 验证边Focal | 原两级直丝效率 | 原两级假轨 | 原两级克隆 | 原两级直丝+斜丝成员效率 |','| --- | ---: | ---: | ---: | ---: | ---: |']
history=json.loads((root.parent/'cloud/balanced/history.json').read_text(encoding='utf-8'))
for row in history:
    if row['epoch'] not in (6,12,15):continue
    m=row['membership'];lines.append('| %d | %.8f | %.4f%% | %d | %d | %.4f%% |'%(row['epoch'],row['validation']['loss'],m['track_efficiency']*100,m.get('fake_candidates',0),m.get('clone_candidates',0),m['joint_track_efficiency']*100))
lines+=['','固定测试复测中，整轨辅助第12轮的粗三维正确种子8262→8271，假种子290→310；效率变化95%配对区间[-0.1918,+0.3953]个百分点。验证病例中的局部恢复尚未形成稳定测试收益。','',
        '当前损失将同一径迹的AA/AS/SS正边一起平均，没有按径迹和边类型再分组，也没有直接优化所需AA连接是否存在。这是目标设计上可核实的事实；它是否造成这批弱AA病例未改善，还需要训练集同类病例对照，不能仅凭验证病例确认因果。','']
rows=[]
for key in keys:
    for p in sorted((root/'events'/key).glob('event_*.json')):
        packet=json.loads(p.read_text(encoding='utf-8'))
        rows.extend(dict(model=key,event_id=packet['event_id'],**t) for t in packet['tracks'])
weak=[t for t in rows if t['model']=='balanced_012' and t['old_category']=='broadly_low_scores']
increased=sum(t['new_true_mean'] is not None and t['old_true_mean'] is not None and t['new_true_mean']>t['old_true_mean'] for t in weak)
old_weak={(t['event_id'],t['truth_id']) for t in rows if t['model']=='control_012' and t['old_category']=='broadly_low_scores' and t['recovered']}
new_weak={(t['event_id'],t['truth_id']) for t in weak if t['recovered']}
lines+=['','补充：83条普遍低分病例中，整轨辅助第12轮有%d条真边均分上升，但只有8条达到原纯图连通要求、8条最终找回。两组都找回8条，其中%d条共同找回、辅助目标独有%d条、原Focal独有%d条；相同总数不能当作完全相同的病例。'%(increased,len(old_weak&new_weak),len(new_weak-old_weak),len(old_weak-new_weak)),'',
        '**续训判断：不建议按原配置仅为补齐15→18轮而续用GPU。** 第12→15轮的正确直丝总数没有增加，假轨153→163、克隆1→4；粗三维正确种子5671→5663、假种子166→177。少数弱评分病例仍有波动性恢复，但没有体现为总体收益。这不等于证明第18轮绝不可能反弹，也不等于模型已数学收敛。','',
        '更值得先做的是：在训练集内按“径迹×边类型”平衡AA/AS/SS正边，明确约束所需AA连接；另在验证集校准新分数与候选竞争规则。原223条诊断病例只用于分析和验证，不回流成训练事件。小规模对照出现稳定收益后，再决定较长训练预算。','',
        '本次分组复盘在本机完成，使用已下载并核对的第12/15轮权重。临时SSH状态复查超时，因此没有把云端状态称为本次实时确认；15/18轮信息来自上一轮完整下载的训练记录。']
lines.insert(2,'**主要恢复出现在混合碎裂和原已连通两类；普遍低分这一大类只有有限实际恢复。当前不建议原配置续训以凑满轮数。**\n')
(root/'分组改善与续训判断.md').write_text('\n'.join(lines),encoding='utf-8')
with (root/'逐径迹变化.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(report=str(root/'分组改善与续训判断.md'),diagnostic_rows=len(rows)),ensure_ascii=False))
