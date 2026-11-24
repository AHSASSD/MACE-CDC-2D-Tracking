import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
run=root/'memory_safe_gpu'
report=json.loads((run/'fresh_test_report.json').read_text(encoding='utf-8'))
joint_seconds=report.get('joint_processing_seconds',report.get('joint_forward_seconds',0))
baseline_seconds=report.get('baseline_processing_seconds',report.get('baseline_forward_seconds',0))
joint=report['joint'];base=report['axial_baseline']
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for name in ['pilot_v1','mean_ablation','extended_sum','cpu_extension','memory_safe_gpu']:
    history=json.loads((root/name/'history.json').read_text(encoding='utf-8'))
    axes[0].plot([h['epoch'] for h in history],[h['validation_macro_f1'] for h in history],label=name)
axes[0].set(xlabel='Epoch within stage (optimizer resets)',ylabel='Validation AA/AS/SS macro F1',title='Validation-only model selection');axes[0].legend(fontsize=7)
x=np.arange(3);axes[1].bar(x-.18,[joint[k]['f1'] for k in ['AA','AS','SS']],.36,label='Joint')
axes[1].bar([.18],[base['AA']['f1']],.36,label='Frozen axial baseline')
axes[1].set(xticks=x,xticklabels=['AA','AS','SS'],ylim=(0,1),ylabel='Edge F1',title=f'Fresh holdout: {report["events"]} events');axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(run/'comparison.png',dpi=160);plt.close(fig)
rows='\n'.join(f"| {kind} | {joint[kind]['precision']:.4f} | {joint[kind]['recall']:.4f} | {joint[kind]['f1']:.4f} |" for kind in ['AA','AS','SS'])
jc=joint['candidates'];bc=base['candidates']
text=f'''# AI6 联合训练实验结果

已实现并在本机训练同时处理直丝和斜丝的无时间神经网络。以下结果来自模型与阈值冻结后新抽取的 {report['events']} 个完整测试事件，与训练、验证和此前已查看的 pilot test 均无交集。事件 ID 与权重哈希见 fresh_test_ids.json。

## 独立测试边关联

| 边类型 | Precision | Recall | F1 |
|---|---:|---:|---:|
{rows}

原 no-time62 直丝基线在同批事件、相同直丝候选边规则上的 AA F1 为 **{base['AA']['f1']:.4f}**。联合模型与原基线容量、训练量不同，不能把差异全部归因于加入斜丝。

## 候选解码诊断

两模型都使用相同的连通分量解码器，真轨分母为同样的 {jc['truth_tracks']} 条至少有 5 个直丝 hit 的真轨。按直丝纯度和完整度均 >=70% 匹配。

边关联阈值：联合 {joint['threshold']}、基线 {base['threshold']}。候选阈值分别为 {joint['candidate_threshold']}、{base['candidate_threshold']}，均仅按验证集目标 (matched -0.25*fake -0.25*clones)/truth 选择，候选阈值与边 F1 阈值分开。

| 指标 | 联合模型 | 原直丝基线 |
|---|---:|---:|
| 直丝真轨匹配效率 | {jc['axial_efficiency']:.2%} | {bc['axial_efficiency']:.2%} |
| 假候选比例 | {jc['fake_fraction']:.2%} | {bc['fake_fraction']:.2%} |
| 克隆候选比例 | {jc['clone_fraction']:.2%} | {bc['clone_fraction']:.2%} |
| 匹配候选的直丝平均完整度 | {jc['axial_completeness_matched']:.2%} | {bc['axial_completeness_matched']:.2%} |

联合模型匹配候选内斜丝纯度 {jc['stereo_purity']:.2%}、斜丝完整度 {jc['stereo_completeness_matched']:.2%}。这两个条件指标必须结合总匹配效率阅读，不能代表所有斜丝均已重建。连通分量本身不共享 hit，因此共享数为零不是额外的物理判别能力。

本次预测与部分评分处理耗时：联合模型 {joint_seconds:.3f} 秒，基线 {baseline_seconds:.3f} 秒。初版计时范围包含候选评分且两处评分次数不同，因此不作速度优劣比较；后续脚本已经将前向计时位置修正。

## 结论与限制

联合输入、训练和独立评估流程已跑通；当前模型{'仍低于' if joint['AA']['f1']<base['AA']['f1'] else '达到或超过'}原直丝基线的 AA 边 F1。当前不能仅根据接入斜丝就宣布方法有效。

本次训练仅使用 128 个完整事件，hidden=64、3 次消息传递。原直丝基线 hidden=256、8 次消息传递，且已经接受更多训练。本轮属于可复现的小规模方向验证。

没有使用时间、真实位置或动量构建输入；不重复给 d 加噪；保留独立 hit 身份。生产几何哈希、原始 CSV 哈希与既有事件划分均记录在 pilot_v1/manifest.json。验证集 957 条合格真轨保持图内连通，但这不是三维可重建性的充分证明。

本报告的候选解码不是原两级几何寻迹。没有将斜丝参考点当作真实 hit 做圆拟合，没有三维拟合或动量精度结果。后续应先解决 AS/SS 关联质量和候选串联，再接直丝几何拟合及带全局竞争的斜丝归属；全量实验前需改为分片惰性数据读取并释放磁盘空间。

![比较图](comparison.png)
'''
(root/'实验结果.md').write_text(text,encoding='utf-8')
print(root/'实验结果.md')
