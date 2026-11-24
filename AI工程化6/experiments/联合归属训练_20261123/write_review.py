"""Render the audited training history and preliminary member results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=Path(__file__).parent/'review_20261123'
history=json.loads((r/'progress.json').read_text())['history']
v=json.loads((r/'validation.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
for k,name in enumerate(['Edge loss','Triplet loss']):
 for arm in ['C64','D64']:axes[k].plot([x['epoch'] for x in history],[x['losses'][arm][k] for x in history],marker='o',label=arm)
 axes[k].set(title=name,xlabel='Pilot epoch');axes[k].legend();axes[k].grid(alpha=.2)
axes[2].plot([x['epoch'] for x in history],[x['losses']['D64'][2] for x in history],marker='o',label='D64 training OC')
axes[2].scatter([10],[v['validation_oc_loss']],label='D64 validation OC (16 events)',color='red')
axes[2].set(title='OC loss (not track efficiency)',xlabel='Pilot epoch');axes[2].legend();axes[2].grid(alpha=.2)
fig.savefig(r/'training_curves.png',dpi=160);plt.close(fig)
rows=[]
for arm in ['C64','D64']:
 a=history[0]['losses'][arm];b=history[-1]['losses'][arm]
 rows.append(f'|{arm}|{a[0]:.4f} → {b[0]:.4f}|{a[1]:.4f} → {b[1]:.4f}|'+(f'{a[2]:.4f} → {b[2]:.4f}|' if arm=='D64' else '未训练归属头，不比较其OC值|'))
m=v['validation_raw_oc'];o=v['truth_center_oracle'];seconds=np.mean([x['seconds'] for x in history])
text=f'''# C/D联合归属模型：10轮试训复核

结论：运行稳定性试训已完成，但第10轮D64原始归属分组尚不可用。不能凭OC loss下降宣布寻迹改善；正式7000例/200轮尚未启动，稳定模型不替换。

## 作业核查

作业1860878，gpu14、V100 32GB，Slurm为COMPLETED，ExitCode=0:0。北京时间2026-11-23 00:43:07至05:56:06，耗时5小时12分59秒；结束后正常释放资源，本次查询无运行中GPU作业。

日志只有一次训练启动，320个共同更新进度记录；未见Traceback、OOM、nonfinite、恢复重试或resource_tracker警告。两组各完成10轮、每轮1000训练事件、各320次优化器更新。每组第1—10轮权重及完整状态保存在服务器，已下载第10轮权重和日志至本目录。

## 训练曲线

|组|边损失：首轮→末轮|三元组损失：首轮→末轮|OC损失|
|---|---:|---:|---|
{chr(10).join(rows)}

D64验证OC均值为{v['validation_oc_loss']:.4f}，在16例开发验证事件上接近训练值。但归属损失与真实成员分组质量仍脱节；不能由此证明没有过拟合或已经学会整轨。

![训练曲线](training_curves.png)

## 新做的小规模成员验证

使用冻结清单中的8例解码校准、另16例成员验证；均不属于训练集，二者不重叠。按原登记β阈值和半径网格，仅在8例选择配置，再冻结用于16例。选择β≥{v['selected_decoder']['beta_threshold']}、半径{v['selected_decoder']['radius']}学习空间单位，中心抑制距离等于半径。

评价为D64学习空间直接分组后的直丝成员诊断：真轨至少5个直丝命中，候选至少5个正确命中、纯度≥80%、完整度≥50%。尚未加入漂移圆/有限斜丝几何、粗三维及GenFit，也未运行同事件稳定基线，故不是完整流水线效率或与旧95%结果的公平对照。

|16例原始OC分组指标|数量/数值|
|---|---:|
|可评价真轨|{m['truth']}|
|匹配真轨|{m['correct']}|
|输出候选|{m['outputs']}|
|假候选|{m['fakes']}|
|克隆候选|{m['clones']}|
|匹配比例|{100*m['efficiency']:.2f}%|
|假候选比例|{100*m['fake_fraction']:.2f}%|

零克隆不代表成功，因为绝大多数真轨尚未被正确分组。这16例本次已查看，今后只能作为开发诊断，不再称为新盲测。

## 中心提取还是表示不足

另做仅用于离线诊断的“真值指定中心”：每条真轨内取最高β命中作为中心，然后在全部中心之间按学习空间最近邻分组。该操作使用真值，不能部署，也不是数学意义的理论效率上限。

这个诊断匹配{o['correct']}/{o['truth']}，约{100*o['efficiency']:.2f}%，仍有{o['fakes']}个假候选。它比实际中心提取好，提示中心选择/抑制有损失；但已提供真中心后仍很低，说明当前嵌入表示及最近中心关联也不足，不能只归罪于阈值。这不能证明200轮训练也无效。

C/D在固定0.5边分数下的详细AA/AS/SS计数保存在validation.json。D边/T训练损失略高于C，暂未显示增加OC监督同时提升局部判别。未做同等精度PR比较，不能仅凭同阈值宣布哪组判别更好。

## 训练成本与下一步

当前两组串行、microbatch1、全重算时，1000事件一轮平均{seconds/60:.2f}分钟。按同分布线性外推，7000事件两组各200轮约{seconds*7*200/3600:.0f} GPU小时，仅为未优化实现的粗估，不是正式耗时承诺。不能未经吞吐优化直接照此投入全部预算。

下一步应先拆解吸引/排斥/β损失、审计近邻异轨与接口监督，检查中心抑制与嵌入尺度，并接入实际成员选轮；同时优化采样、损失归约、分块和重算。保留试训检查点。正式训练按修正后的冻结协议重新初始化C/D，不把试训权重悄悄当作正式从头训练起点。

原始证据：progress.json、slurm-1860878.log、validation.json、weights/。本次仅做下载、检查和本机验证，未新增GPU训练作业。
'''
(r/'试训结果复核.md').write_text(text,encoding='utf-8')
print(r/'试训结果复核.md')
