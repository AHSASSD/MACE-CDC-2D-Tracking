import json
from pathlib import Path
import numpy as np
root=Path(__file__).parent;r=root/'review_20261123/end_to_end_probe'
probe=json.loads((r/'progress.json').read_text());h=probe['history'];expanded=json.loads((r/'expanded_validation.json').read_text())
rows=[]
for e in h:
    c=e['members']['control_v1']['validation'];d=e['members']['hard_v1']['validation']
    rows.append(f'|{e["epoch"]}|{c["correct"]}/491|{100*c["fake_fraction"]:.2f}%|{d["correct"]}/491|{100*d["fake_fraction"]:.2f}%|')
a=expanded['arms']['D10']['validation'];b=expanded['arms']['control7']['validation']
train=np.mean([x['train_seconds'] for x in h[1:]]);whole=np.mean([x['epoch_seconds_including_validation'] for x in h[1:]])
text=f'''# 对照训练复核与全量扩量决策

## 已完成对照

作业1862499，gpu14，Slurm COMPLETED，ExitCode=0:0，15分40秒。两组从相同D10权重开始，256个训练事件、10轮、各80次更新。原始观测同时含直丝和斜丝；不同事件不连边、不合并同名TrkID。

|轮数|普通OC正确轨|普通OC假候选率|近邻排斥正确轨|近邻排斥假候选率|
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

这些是8例校准、16例开发评价的原始OC聚类成员，未加粗几何/三维。每轮校准参数可能变化，轮间差异不能全部解释为模型更新。普通OC末轮75/491，初始49/491；近邻排斥末轮33/491，不支持继续投入该修改。第7轮是普通OC在该小开发区按成员F1选出的检查点（80/491，假候选198/278）。不存在新独立盲测或已部署效率提升的结论。

## 200例扩展验证

两份权重使用相同扩展网格：β阈值0.3/0.5/0.7，半径0.125/0.25/0.375/0.5/0.75/1.0。只用32个校准事件选择配置，再评估固定200个开发选轮事件；包含已查看的16例，属于开发验证，不称为盲测。

|权重|正确轨/可评价真轨|匹配比例|假候选率|克隆数|
|---|---:|---:|---:|---:|
|初始D10|{a['correct']}/{a['truth']}|{100*a['efficiency']:.2f}%|{100*a['fake_fraction']:.2f}%|{a['clones']}|
|普通OC对照第7轮|{b['correct']}/{b['truth']}|{100*b['efficiency']:.2f}%|{100*b['fake_fraction']:.2f}%|{b['clones']}|

改善方向在更大开发集合仍成立：多匹配{b['correct']-a['correct']}条，匹配比例提高{100*(b['efficiency']-a['efficiency']):.2f}个百分点，假候选率下降{100*(a['fake_fraction']-b['fake_fraction']):.2f}个百分点。但约60%的假候选仍不可用；没有与旧稳定完整流程做同事件公平比较，不能拿20.20%与旧约95%直接作架构优劣判断。

目前D10仅有320次优化更新，第7轮对照又增加56次，总计376次。按7000事件、有效batch32，每个全量epoch约219次更新；当前训练曝光和优化步数远少于原计划的全量长训。这是支持扩量学习曲线的理由，不是保证长训一定能成功。

## 吞吐核查

两组各256例，完整训练轮平均{train:.2f}秒，包含成员验证平均{whole:.2f}秒。1 worker首轮77.54秒，2 worker第二轮78.00秒，差别很小且各只测一次；保守选1 worker，不宣称统计上证明它更优。

新阶段沿用microbatch4、累计到32、edge chunk16384、前两次消息更新不重算、后续保留重算、磁盘几何缓存、prefetch2、固定内存和异步传输。完整7000例的速度还要实测，不把此前3图5.5倍基准直接当作全量轮加速比。

## 已提交服务器阶段

作业1862832，gpu18，2026-11-23北京时间16:17:48开始、22:17:48配额结束，单GPU/3CPU/6小时。

本阶段是7000训练事件的普通OC扩量学习曲线，最多20轮，继承普通OC第7轮权重，新建AdamW优化器、初始LR1e-4，20轮余弦降到1e-5。不是预定C/D从头200轮正式消融，不把小样本轮数与全量轮数相加冒充同样训练预算。

每轮32例校准＋200例开发成员评价，保留最佳成员F1检查点。至少完成10个全量轮后，如连续5轮没有成员改善，则结束本阶段。配额不足在更新边界保存，最多损失约96个事件的未落盘进度；未完成的轮从游标继续。训练损失/耗时在中断后明确标注为本次恢复段统计，成员验证仍使用完整固定集合。

独立监督器检查进度心跳，启动阶段最多等待900秒，已开始更新后600秒无进展会终止自己启动的训练进程组并恢复；连续3次无新进展后退出独立训练作业。不scancel用户交互作业。原稳定基线、历史源码和权重保留。

训练尚在运行；最终采用与否须看后续成员、粗几何和三维评价。当前未评价GenFit、粗三维种子效率，也没有用真值轨迹作为模型输入。
'''
(r/'对照评估与扩量决策.md').write_text(text,encoding='utf-8');print(r/'对照评估与扩量决策.md')
