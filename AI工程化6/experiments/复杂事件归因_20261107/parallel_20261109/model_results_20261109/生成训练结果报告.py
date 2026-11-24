"""Produce an auditable report and conventional scientific training curves."""
from pathlib import Path
import json
import sys
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT.parent))
import experiment_common as e


def main():
    history={arm:json.loads((ROOT/'cloud'/arm/'history.json').read_text(encoding='utf-8')) for arm in ['control','balanced']}
    status=json.loads((ROOT/'cloud/status.json').read_text(encoding='utf-8'))
    incidents=[json.loads(line) for line in (ROOT/'cloud/incidents.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    reference=json.loads((e.SNAPSHOT/'best_model_selection.json').read_text(encoding='utf-8'))
    result=json.loads((ROOT/'summary.json').read_text(encoding='utf-8')) if (ROOT/'summary.json').exists() else None
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for arm,color,label in [('control','#2364aa','Original focal'),('balanced','#d05b24','Track-balanced auxiliaries')]:
        rows=history[arm];x=[r['epoch'] for r in rows]
        axes[0,0].plot(x,[r['validation']['loss'] for r in rows],marker='.',label=label,color=color)
        members=[r for r in rows if r['membership'] is not None]
        for ax,key in [(axes[0,1],'track_efficiency'),(axes[1,0],'joint_track_efficiency'),(axes[1,1],'fake_fraction')]:
            ax.plot([r['epoch'] for r in members],[r['membership'][key]*100 for r in members],marker='o',color=color,label=label)
    axes[0,0].axhline(reference['validation']['loss'],ls='--',color='gray',label='Initial epoch126')
    if result:
        m=result['initial126_original_two_pass_200']
        for ax,key in [(axes[0,1],'track_efficiency'),(axes[1,0],'joint_track_efficiency'),(axes[1,1],'fake_fraction')]:ax.axhline(m[key]*100,ls='--',color='gray',label='Initial126 local reference')
    for ax,title,ylabel in [(axes[0,0],'Common focal validation: all 1500 events','Loss'),
                            (axes[0,1],'Original two-pass axial membership: fixed200','Efficiency (%)'),
                            (axes[1,0],'2D + stereo attachment: fixed200 (not coarse3D)','Efficiency (%)'),
                            (axes[1,1],'Original two-pass fake fraction: fixed200','Fake fraction (%)')]:
        ax.set_title(title,fontsize=10);ax.set_xlabel('Fine-tuning epoch');ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.savefig(ROOT/'training_curves.png',dpi=180);plt.close(fig)
    ended=dt.datetime.fromtimestamp(status['unix'],tz=dt.timezone(dt.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    lines=['# 双目标微调训练结果与同流程复测','',
           '本次实际运行是从固定epoch126出发的受控微调。网络仍为隐藏维度256、8次消息传递，分别使用原Focal和整轨平衡/高分假边辅助目标；不是一次改变网络层结构的从零训练。','',
           f'两组各完成15轮，分别累计3285次优化器更新，AMP跳步均为0。计划上限18轮，程序在北京时间{ended}为避免剩余配额内无法完成下一对训练/验证而停止；停止状态为`stopped_before_incomplete_pair`，两组最后一轮均有全状态断点。','',
           f'期间出现{len(incidents)}次退出码137，日志显示`srun: task 0: Killed`、Slurm步骤退出信号9；守护程序都成功恢复。证据只能确认被SIGKILL，不能仅凭该日志认定CUDA显存不足或Python semaphore警告是根因。','',
           '## 训练记录中的原两级成员验证','',
           '同一200验证事件、5969条可评直丝径迹，每3轮评价一次。下面的joint指标是斜丝附着的成员匹配，不是下表的粗三维种子，也不是GenFit精拟合。','',
           '| 目标 | 微调轮次 | 正确直丝匹配 | 直丝效率 | 假轨 | 克隆 | 直丝+斜丝成员效率 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ['control','balanced']:
        for row in history[arm]:
            m=row['membership']
            if m is None:continue
            lines.append('| %s | %d | %d | %.4f%% | %d (%.4f%%) | %d (%.4f%%) | %.4f%% |'%('原Focal' if arm=='control' else '整轨辅助目标',row['epoch'],m['matched_tracks'],m['track_efficiency']*100,m.get('fake_candidates',0),m['fake_fraction']*100,m.get('clone_candidates',0),m['clone_fraction']*100,m['joint_track_efficiency']*100))
    lines+=['','第6轮在两组已记录的成员验证中效率较高；第12轮则提供较低假轨或克隆的取舍。因此在另300个固定验证事件中复测这四个权重，不直接使用最后15轮或仅按最小边loss挑模型。控制组最小边loss在第1轮，实验组也在第1轮，后段没有持续改善验证损失。','',
            '![训练曲线](training_curves.png)','',
            '## 固定稳定提取器与粗三维流程','',
            '全部模型使用同一本机推理运行时和同一300验证事件；冻结原候选校准器、hybrid参数.02/.2、guarded_loose粗三维设置，不叠加上轮未通过检查的新提取器。候选权重从原200成员验证选取。本组300事件以前用于提取器研究，属于额外固定验证复测，不能称为从未查看的测试集。']
    if result:
        a=result['stable_tracker_300'];names={'initial126':'原epoch126','control_006':'原Focal +6轮','control_012':'原Focal +12轮','balanced_006':'整轨辅助目标 +6轮','balanced_012':'整轨辅助目标 +12轮'}
        lines+=['','| 模型 | 正确直丝匹配 | 效率 | 假轨率 | 克隆率 | 新增/丢失原正确匹配 |','| --- | ---: | ---: | ---: | ---: | ---: |']
        for name,packet in a.items():
            m=packet['tracks'];lines.append('| %s | %d | %.4f%% | %.4f%% (%d) | %.4f%% (%d) | %s |'%(names[name],m['matched_tracks'],m['track_efficiency']*100,m['fake_fraction']*100,m.get('fake_candidates',0),m['clone_fraction']*100,m.get('clone_candidates',0),'-' if name=='initial126' else '%d/%d'%(packet['gained'],packet['lost'])))
        lines+=['','| 模型 | 粗三维种子正确匹配 | 种子效率 | 种子假轨率 | 种子克隆率 |','| --- | ---: | ---: | ---: | ---: |']
        for name,packet in a.items():
            m=packet['seed'];lines.append('| %s | %d | %.4f%% | %.4f%% (%d) | %.4f%% (%d) |'%(names[name],m['matched_tracks'],m['efficiency']*100,m['fake_fraction']*100,m.get('fake_candidates',0),m['clone_fraction']*100,m.get('clone_candidates',0)))
        lines+=['','同事件配对bootstrap的直丝效率变化95%区间：','']
        for name,packet in a.items():
            if name=='initial126':continue
            lo,hi=packet['paired_efficiency_difference_95_ci'];lines.append('- %s：[%+.4f, %+.4f]个百分点。'%(names[name],lo*100,hi*100))
        diagnostic=ROOT/'coarse_and_axial_diagnostics.json'
        if diagnostic.exists():
            detail=json.loads(diagnostic.read_text(encoding='utf-8'))
            lines+=['','## 原直丝保留与粗三维不确定性','',
                    '以下真值只用于离线诊断。旧直丝hit保留率表示原输出中的hit仍出现在新输出中；“原正确hit仍分配正确”进一步要求它仍属于满足80%纯度/50%完整度口径的正确直丝组。两者不能混同。','',
                    '| 模型 | 旧直丝hit保留率 | 原正确hit仍正确分组 | 粗三维种子效率变化及95%区间/百分点 |','| --- | ---: | ---: | ---: |']
            for name,d in detail.items():
                lo,hi=d['seed_efficiency_difference_95_ci']
                lines.append('| %s | %.4f%% | %.4f%% | %+.4f [%+.4f, %+.4f] |'%(names[name],d['old_axial_hit_retention']*100,d['old_correct_hit_correct_assignment_retention']*100,d['seed_efficiency_difference']*100,lo*100,hi*100))
    else:lines+=['','同流程复测正在运行，结果将保存到`summary.json`。']
    test_paths=[ROOT/'fixed_test/summary.json',ROOT/'fixed_test_balanced_012/summary.json']
    if all(p.exists() for p in test_paths):
        tests=[json.loads(p.read_text(encoding='utf-8')) for p in test_paths]
        test_models={'initial126':tests[0]['models']['initial126'],'control_012':tests[0]['models']['control_012'],'balanced_012':tests[1]['models']['balanced_012']}
        assert tests[0]['models']['initial126']==tests[1]['models']['initial126']
        lines.insert(2,'**结论：本轮微调尚不足以替换原epoch126稳定模型。** 固定300测试、8818条真实径迹中，原模型的正确粗三维种子8262条；原Focal微调12轮为8265条、假种子290→291；整轨辅助目标微调12轮为8271条、假种子290→310。两组粗三维效率增益的配对区间均跨零。斜丝附着覆盖有所改善，但没有稳定地转化为粗三维寻迹收益。\n')
        lines+=['','## 冻结候选的固定300测试复测','',
                '控制组12轮先冻结；四个验证候选全部完成后，在查看候选测试结果前，依据实验组12轮的粗三维验证表现追加冻结第二个候选。两组本轮测试均没有用于继续选择轮次、调整阈值或训练。该固定测试以前已多次查看，仍不能称为全新的盲测样本。','',
                '| 模型 | 正确直丝匹配 | 直丝效率 | 直丝假轨率 | 直丝克隆率 |','| --- | ---: | ---: | ---: | ---: |']
        for name,packet in test_models.items():
            m=packet['tracks'];lines.append('| %s | %d | %.4f%% | %.4f%% (%d) | %.4f%% (%d) |'%(names[name],m['matched_tracks'],m['track_efficiency']*100,m['fake_fraction']*100,m.get('fake_candidates',0),m['clone_fraction']*100,m.get('clone_candidates',0)))
        lines+=['','| 模型 | 正确粗三维种子 | 种子效率 | 种子假轨率 | 种子克隆率 |','| --- | ---: | ---: | ---: | ---: |']
        for name,packet in test_models.items():
            m=packet['seed'];lines.append('| %s | %d | %.4f%% | %.4f%% (%d) | %.4f%% (%d) |'%(names[name],m['matched_tracks'],m['efficiency']*100,m['fake_fraction']*100,m.get('fake_candidates',0),m['clone_fraction']*100,m.get('clone_candidates',0)))
        lines+=['','逐径迹变化与配对bootstrap区间：','']
        for test,name in zip(tests,['control_012','balanced_012']):
            row=test['models'][name];lo,hi=test['paired_efficiency_difference_95_ci']['seed']
            lines.append('- %s：直丝新增正确%d条，同时丢失原正确%d条；粗三维种子效率变化95%%区间[%+.4f, %+.4f]个百分点。'%(names[name],row['gained'],row['lost'],lo*100,hi*100))
        lines+=['','训练目标改动与原Focal续训的作用需要区分：原Focal微调12轮在固定测试中将直丝假轨273→255，但正确匹配仅8404→8409、克隆5→6；辅助目标将正确匹配提高到8416，却将假轨增至287。两者均未同时改善全部指标。','',
                '辅助目标模型在固定测试中的斜丝附着完整度84.5117%→85.7158%，但纯度99.0862%→98.8943%。因此不能仅以附着斜丝点更多或joint成员匹配更多，就判断粗三维种子质量更好。','',
                '上述结果评估的是新权重直接接入冻结旧框架的实际效果，尚未针对新模型重新校准候选评分或提取阈值。分数分布与旧校准器的配合可能影响结果，但本次未做额外实验，不能将其写成已确定的原因。']
    lines+=['','稳定部署尚未替换。所有下载权重均保留在`cloud/control`、`cloud/balanced`，下载归档SHA256已经核对；`evaluation_contract.json`固定模型SHA、事件ID、选择口径和运行时。评价不包括三维位置、方向、动量精拟合，精拟合由GenFit负责。','']
    (ROOT/'训练结果报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(report=str(ROOT/'训练结果报告.md'),complete_evaluation=result is not None),ensure_ascii=False))


if __name__=='__main__':main()
