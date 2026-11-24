from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'training_review_20261122';SOURCE=OUT/'download'

def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    h=read(SOURCE/'training/history.json');r=read(OUT/'matched_precision.json');hist=np.load(OUT/'pr_histograms.npz')
    old=hashlib.sha256((SOURCE/'initial_T25.pth').read_bytes()).hexdigest();best=hashlib.sha256((SOURCE/'training/T_best_junction.pth').read_bytes()).hexdigest()
    import torch
    initial=torch.load(SOURCE/'initial_T25.pth',map_location='cpu',weights_only=True);selected=torch.load(SOURCE/'training/T_best_junction.pth',map_location='cpu',weights_only=True)
    equal=set(initial)==set(selected) and all(torch.equal(v,selected[k]) for k,v in initial.items());assert equal
    (OUT/'weights_verification.json').write_text(json.dumps(dict(initial_file_sha256=old,selected_file_sha256=best,selected_parameters_equal_initial=equal),indent=2),encoding='utf-8')
    lines=['# 弱连接三元组微调结果复核：2026-11-22','',
      '结论：训练正常早停，固定阈值选模未接受新权重。匹配精度补查发现小幅判别收益，分数整体抬高占主要部分，尚不能宣称寻迹效率改善。', '',
      '## 训练与资源','',
      '作业1859338（gpu10）正常COMPLETED，退出码0:0。北京时间14:06:39开始，14:33:40结束，总计27分01秒；完成7000训练事例的10轮微调、17500次优化更新和每轮200例开发验证。第10轮按最少10轮且连续8轮无合格改善的规则早停，未因配额不足中断。作业已退出，没有继续占用GPU至原17:06上限。', '',
      '选模规则要求阈值0.5下全体三元组精度相对原T25下降不超过0.003，各轮均不合格，T_best_junction.pth实际仍为原T25，已逐张量验证参数完全一致。所有10轮新权重均已保存并下载。', '',
      '## 固定0.5阈值的结果','',
      '|模型|整体精度|整体召回|弱正三元组召回|','|---|---:|---:|---:|']
    for row in [h[0],h[-1]]:
        v=row['validation'];lines.append('|%s|%.3f%%|%.3f%%|%.3f%%|'%('原T25' if row['epoch']==0 else '微调第10轮',100*v['all05']['precision'],100*v['all05']['recall'],100*v['weak05']['recall']))
    lines+=['', '弱例为至少一条原边分数小于0.7的三元组，包含AA/AS/SS，不等于精确的主干—分支接口。弱项验证是正例子集，其日志precision=1由定义造成，不能当成独立精度成绩。', '',
      '## 匹配精度补查','',
      '在同一200例开发验证上，用原T25、第7轮（固定0.9下弱召回最高）及第10轮保存权重重新评分。按0.0001步长扫描阈值，保持原T25在0.5阈值下的直丝三元组精度约98.285%。这是回顾性开发集分析，未经新确认集检验。', '',
      '|模型|阈值|直丝三元组精度|直丝三元组召回|同阈值弱直丝三元组召回|', '|---|---:|---:|---:|---:|']
    for k,m in r['models'].items():
        a=m['axial']['matched_precision'];idx=round(a['threshold']*10000);hh=hist[k+'_weak_axial'];weak=float(np.cumsum(hh[0,::-1])[::-1][idx]/hh[0].sum())
        lines.append('|%s|%.4f|%.3f%%|%.3f%%|%.3f%%|'%(k,a['threshold'],100*a['precision'],100*a['recall'],100*weak))
    lines+=['', '按全AA/AS/SS总体精度约97.51%匹配时，第10轮阈值约0.8497，整体召回77.09%→77.39%，弱三元组召回15.34%→15.53%。这也表明固定0.5时的大幅涨分不能直接解释成同等规模的能力提升。', '',
      '![开发集PR曲线](precision_recall_review.png)','',
      '## 实际寻迹结论与后续','',
      '本次复核尚未给出新权重的直丝寻迹效率、假轨率、克隆率或粗三维种子效率；三元组精度不等于径迹纯度，三元组召回不等于寻迹效率。稳定部署保持原版本。', '',
      '建议保留第10轮作为需校准的实验候选，在开发集固定分数校准/阈值后接入路径状态解码，再使用next_member_manifest.json中预留的200例检验。不要直接把0.854当成整套解码器所有阈值的替代：旧T的0.9兼容阈值和路径搜索的低分转移阈值也需分别校准。', '',
      '暂不建议按原损失与固定0.5选模规则继续加轮数。第1到10轮同一阈值表现已基本进入平台。之后应优先控制难负例代价、分数校准和AA共享接口专门评价，再决定是否进一步微调。', '',
      '证据：download/training/history.json、status.json、best.json，matched_precision.json及pr_histograms.npz。程序复核未使用500例确认集或下一组预留200例选阈值。']
    (OUT/'训练结果复核.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for k in r['models']:
        for ax,g in zip(axes,['all','axial']):
            pos,neg=np.cumsum(hist[k+'_'+g][:,::-1],axis=1)[:,::-1];precision=pos/np.maximum(pos+neg,1);recall=pos/max(pos[0],1)
            mask=(precision>=.95)&(pos>0)&(recall>.5)
            ax.plot(100*recall[mask],100*precision[mask],label=k)
            m=r['models'][k][g]['matched_precision'];ax.scatter([100*m['recall']],[100*m['precision']],s=25)
    for ax,title in zip(axes,['All triplets','Axial-only triplets']):ax.set_title(title);ax.set_xlabel('Recall (%)');ax.set_ylabel('Precision (%)');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'precision_recall_review.png',dpi=160);plt.close(fig);print('report saved',OUT/'训练结果复核.md')

if __name__=='__main__':main()
