"""Locked 500-event confirmation of T25 vs raw two-pass; no parameter tuning."""
from pathlib import Path
import sys,os,json,time,hashlib,argparse
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from common import sha,save_json
from et_data import prepare,Features,to_device,atomic_npz
from et_models import THead,predict

DATA=ROOT.parent.parent/'joint_event_dataset'
OUT=ROOT/'confirmation_20261122'
CACHE=OUT/'cache'
WEIGHT=ROOT.parent.parent.parent/'AI工程化6_稳定基线备份/AI6_稳定基线_20261107_230240/AI工程化6/joint_training/local_evaluation_epoch126_20261107/best_interaction_network_joint_no_time.pth'

def evaluate(eid):
    torch.set_num_threads(1)
    from et_decode import T_groups,geometry_extract,seeds
    from STEP5_两级寻迹 import axial_two_pass,attach_stereo,score_tracks,DEFAULT_CONFIG
    from coarse_3d_membership import propose,select
    with np.load(CACHE/'features'/('event_%05d.npz'%eid)) as s:f={k:s[k] for k in s.files}
    with np.load(CACHE/'indices'/('event_%05d.npz'%eid)) as s:t=s['t']
    with np.load(DATA/'events'/('event_%05d.npz'%eid)) as s:drift=s['d'];truth=s['truth']
    tq=np.load(OUT/'predictions'/('event_%05d.npz'%eid))['T']
    geo=np.load(DATA/'wire_geometry.npy')[f['cells']]
    started=time.perf_counter();baseline,_=axial_two_pass(geo,drift,f['pairs'],f['p']);base_seconds=time.perf_counter()-started
    rows={}
    for k in ['raw_baseline','T']:
        started=time.perf_counter()
        if k=='T':groups,weights=T_groups(f,t,tq);tracks=geometry_extract(groups,weights,f,geo,drift,baseline)
        else:tracks=baseline
        joint,_=attach_stereo(tracks,geo,drift,f['cells'],f['pairs'],f['p'],DEFAULT_CONFIG)
        counts,found=score_tracks(joint,truth,f['stereo'],DEFAULT_CONFIG)
        confirmed,_=select(joint,propose(joint,geo,drift,f['cells'],f['pairs'],f['p'],refine=False),geo,f['cells'],'guarded_loose')
        rows[k]=dict(counts=counts,found=found,seeds=seeds(confirmed,truth,f['stereo'],f['cells']),decode_seconds=base_seconds+time.perf_counter()-started)
    save_json(OUT/'events'/('event_%05d.json'%eid),dict(event_id=eid,n_hits=len(truth),methods=rows))
    return eid

def summarize(ids):
    from STEP5_两级寻迹 import summarize as track_summary
    from STEP2_训练双路线 import seed_summary
    rows=[json.loads((OUT/'events'/('event_%05d.json'%eid)).read_text(encoding='utf-8')) for eid in ids]
    result=dict(events=len(ids),scope='500 internal confirmation validation events, disjoint from training and periodic selection; not historically pristine blind test',methods={})
    for k in ['raw_baseline','T']:
        rr=[r['methods'][k] for r in rows]
        result['methods'][k]=dict(tracks=track_summary([r['counts'] for r in rr]),seeds=seed_summary([r['seeds'] for r in rr]),mean_decode_seconds=float(np.mean([r['decode_seconds'] for r in rr])),median_decode_seconds=float(np.median([r['decode_seconds'] for r in rr])))
    gains=[];lost=[]
    for r in rows:
        a=set(r['methods']['raw_baseline']['found']);b=set(r['methods']['T']['found'])
        gains.extend([[r['event_id'],int(t)] for t in b-a]);lost.extend([[r['event_id'],int(t)] for t in a-b])
    result['paired']=dict(recovered=gains,lost=lost,net=len(gains)-len(lost))
    rng=np.random.default_rng(20261122);sample=rng.integers(0,len(rows),(10000,len(rows)));ci={}
    for label,section,num,den in [('track_efficiency','counts','matched_tracks','truth_tracks'),('fake_fraction','counts','fake_candidates','output_candidates'),('seed_efficiency','seeds','matched_tracks','truth_tracks')]:
        ratios={}
        for k in ['raw_baseline','T']:
            a=np.array([r['methods'][k][section].get(num,0) for r in rows]);b=np.array([r['methods'][k][section].get(den,0) for r in rows]);ratios[k]=a[sample].sum(1)/np.maximum(b[sample].sum(1),1)
        ci[label]=np.quantile(100*(ratios['T']-ratios['raw_baseline']),[.025,.975]).tolist()
    result['delta_95ci_percentage_points']=ci
    result['timing_scope']='CPU decode with 2 concurrent single-thread workers; excludes cached graph/GNN feature preparation and GPU T inference. Not end-to-end throughput.'
    timing=json.loads((OUT/'inference_timings.json').read_text(encoding='utf-8'))
    result['mean_T_gpu_seconds']=float(np.mean(list(timing.values())))
    save_json(OUT/'summary.json',result)
    lines=['# T第25轮：500例确认测试','',result['scope'],'',
      '|方法|直丝效率|假轨率|克隆率|粗三维种子效率|CPU解码秒/例|','|---|---:|---:|---:|---:|---:|']
    for k,m in result['methods'].items():
        t=m['tracks'];lines.append('|%s|%.3f%%|%.3f%%|%.3f%%|%.3f%%|%.3f|'%(k,100*t['track_efficiency'],100*t['fake_fraction'],100*t['clone_fraction'],100*m['seeds']['efficiency'],m['mean_decode_seconds']))
    lines+=['','恢复%d条，损失%d条，净增%d条。'%(len(gains),len(lost),len(gains)-len(lost)),'',
      '按事件配对bootstrap的95%变化区间（百分点）：'+json.dumps(ci,ensure_ascii=False),'',
      '本次固定T第25轮和0.9三元组阈值，不在确认集上调参。未修改稳定部署。该集合与训练/选轮集不重叠，但属于历史使用过的数据集，不能宣称全新盲测。','',
      '基线为原两级提取及相同斜丝附着/guarded_loose粗三维，不含旧hybrid校准器。粗三维为成员种子检查，非GenFit精确拟合成功率。','',result['timing_scope']]
    (OUT/'确认集评估报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('SUMMARY',json.dumps(result,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);args=p.parse_args();torch.set_num_threads(1)
    s=json.loads((ROOT.parent/'三元组寻迹计划_20261122/split_manifest.json').read_text(encoding='utf-8'))
    ids=s['validation_confirm_event_ids'];assert len(ids)==500 and not(set(ids)&set(s['periodic_member_validation_event_ids'])) and not(set(ids)&set(s['original_train_event_ids']))
    if args.limit:ids=ids[:args.limit]
    for folder in ['events','predictions']: (OUT/folder).mkdir(parents=True,exist_ok=True)
    tw=ROOT/'evaluation_download/full/T_best_members.pth'
    assert sha(tw)=='46a9d457a19e3e81112534f52e5a76e0437baa26b06c84f98fcd29f0e3b3eed2'
    contract=dict(ids=ids,T_sha256=sha(tw),backbone_sha256=sha(WEIGHT),dataset_sha256=sha(DATA/'dataset_metadata_joint_no_time.json'),triplet_threshold=.9,source_sha256={f.name:sha(f) for f in ROOT.glob('*.py')},runner_sha256=sha(Path(__file__)),no_training=True)
    if (OUT/'contract.json').exists():assert json.loads((OUT/'contract.json').read_text(encoding='utf-8'))==contract
    else:save_json(OUT/'contract.json',contract)
    started=time.time();save_json(OUT/'progress.json',dict(phase='feature_preparation',events=0,total=len(ids),unix=time.time()))
    prepare(DATA,WEIGHT,CACHE,ids)
    model=THead().cuda();model.load_state_dict(torch.load(tw,map_location='cuda',weights_only=True));model.eval()
    ds=Features(CACHE,ids);pending=set();completed=sum((OUT/'events'/('event_%05d.json'%eid)).exists() for eid in ids)
    timings=json.loads((OUT/'inference_timings.json').read_text(encoding='utf-8')) if (OUT/'inference_timings.json').exists() else {}
    def drain(all_done=False):
        nonlocal pending,completed
        if not pending:return
        done,pending=wait(pending,return_when='ALL_COMPLETED' if all_done else FIRST_COMPLETED)
        for f in done:f.result();completed+=1
        save_json(OUT/'progress.json',dict(phase='member_evaluation',events=completed,total=len(ids),seconds=time.time()-started,unix=time.time()));print('completed',completed,'/',len(ids),flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for j,eid in enumerate(ids):
            if (OUT/'events'/('event_%05d.json'%eid)).exists():continue
            path=OUT/'predictions'/('event_%05d.npz'%eid)
            if not path.exists():
                b=to_device(ds[j],'cuda');torch.cuda.synchronize();begin=time.perf_counter()
                with torch.no_grad(),torch.autocast('cuda'):q=predict(model,b,'T')
                torch.cuda.synchronize();timings[str(eid)]=time.perf_counter()-begin;atomic_npz(path,compress=True,T=q);save_json(OUT/'inference_timings.json',timings)
            pending.add(pool.submit(evaluate,eid))
            if len(pending)>=6:drain()
        drain(True)
    summarize(ids);save_json(OUT/'progress.json',dict(phase='completed',events=len(ids),total=len(ids),seconds=time.time()-started,unix=time.time()))

if __name__=='__main__':main()
