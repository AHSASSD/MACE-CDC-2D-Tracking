"""Small validation-only geometric gate; does not replace the membership baseline."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from common import ROOT, DATA_ROOT, MODEL_ROOT, save_json, sha
from STEP5_两级寻迹 import score_tracks, summarize, DEFAULT_CONFIG

spec=importlib.util.spec_from_file_location('helix_prototype',ROOT/'三维拟合草稿_暂未启用.py')
helix=importlib.util.module_from_spec(spec);spec.loader.exec_module(helix)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='local_evaluation_epoch051_20261106')
    parser.add_argument('--events',type=int,default=5);args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot;reference=snapshot/'two_stage_validation_v1'
    contract=json.loads((reference/'experiment_contract.json').read_text(encoding='utf-8'))
    if not 1<=args.events<=len(contract['event_ids']):raise ValueError('Invalid validation sample size')
    out=snapshot/f'geometry_3d_gate_{args.events:03d}_v1';out.mkdir(exist_ok=False)
    geometry=np.load(DATA_ROOT/'wire_geometry.npy');all_rows=[];reasons=Counter();started=time.time()
    for eid in contract['event_ids'][:args.events]:
        old=json.loads((reference/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
        with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,drift,truth=f['cell_id'],f['d'],f['truth']
        geo=geometry[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9;accepted=[];failures=[]
        for i,track in enumerate(old['tracks']['stereo_assisted']):
            fitted,reason=helix.fit_helix(track['fit'],track['indices'],geo,drift,1.,0)
            phase='strict_3d'
            if fitted is None:
                fitted,reason=helix.fit_helix(track['fit'],track['indices'],geo,drift,1.5,2);phase='relaxed_3d'
            if fitted is None:reasons[reason]+=1;failures.append(dict(candidate=i,reason=reason))
            else:
                fitted['stage']=phase;accepted.append(fitted);reasons[phase]+=1
            print(json.dumps(dict(event_id=eid,candidate=i+1,total=len(old['tracks']['stereo_assisted']),result=reason,seconds=time.time()-started)),flush=True)
        counts,matched=score_tracks(accepted,truth,stereo,DEFAULT_CONFIG)
        row=dict(event_id=eid,baseline_counts=old['methods']['stereo_assisted'],gated_counts=counts,
                 matched_truth_ids=matched,accepted_tracks=accepted,failures=failures)
        save_json(out/f'event_{eid:05d}.json',row);all_rows.append(row)
    summary=dict(events=args.events,event_ids=contract['event_ids'][:args.events],weight_sha256=contract['weight_sha256'],
                 baseline=summarize([r['baseline_counts'] for r in all_rows]),
                 geometry_gated=summarize([r['gated_counts'] for r in all_rows]),reasons=dict(reasons),seconds=time.time()-started,
                 geometry_source_sha256=sha(ROOT/'三维拟合草稿_暂未启用.py'),
                 scope='fixed transverse circle, single-arc longitudinal helix fit; geometry compatibility only, no truth-z or momentum validation')
    save_json(out/'summary.json',summary)
    b=summary['baseline'];g=summary['geometry_gated']
    report=f'''# 小样本三维几何验收试验

同一第51轮模型、固定验证事件中的前{args.events}个事件。候选先由此前两级成员寻迹生成，再用实际有限丝段距离做严格/宽松两级验收。

| 指标 | 原成员候选 | 三维门槛后 |
|---|---:|---:|
| 直丝真轨成员匹配效率 | {b['track_efficiency']:.2%} | {g['track_efficiency']:.2%} |
| 假候选比例 | {b['fake_fraction']:.2%} | {g['fake_fraction']:.2%} |
| 直丝及斜丝成员同时达标比例 | {b['joint_track_efficiency']:.2%} | {g['joint_track_efficiency']:.2%} |

状态计数：{json.dumps(dict(reasons),ensure_ascii=False)}。总耗时{summary['seconds']:.1f}秒。

横向圆由直丝确定并固定，优化z0和每弧度纵向增量；残差为单段螺旋曲线到有限丝段的三维最短距离减去测量d。粗网格加受保护Newton迭代求最近点，距离已与独立标量最小化器核对，包含端点约束；合成非退化例子验证纵向参数恢复。至少5个直丝、4个斜丝及两种斜丝旋向，检查Jacobian条件和候选种子间多解。严格RMS<=1mm，宽松<=1.5mm并最多去除2个斜丝异常成员。

该原型受单段螺旋、固定横向圆、相位覆盖、有限种子搜索及退化判断限制。拟合失败可能来自约束不足、初值、模型近似或数值问题，不能一律解释为候选成员错误；通过验收也不证明纵向参数真实正确。本次没有用真实z或动量评分。暂不以该门槛替换已有成员基线，保留失败原因供改进。

三维拟合草稿_暂未启用.py已通过有限几何距离和合成纵向恢复的初步检查，本次仅由独立STEP7试验入口加载；未接入STEP5基线或服务器训练。
'''
    (out/'三维几何验收评估.md').write_text(report,encoding='utf-8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
