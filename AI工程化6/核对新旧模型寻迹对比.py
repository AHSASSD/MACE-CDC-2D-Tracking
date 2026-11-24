"""Audit all saved memberships, score denominators, frozen hashes and deployment."""
import argparse
from collections import Counter
import json
import subprocess
import sys
import numpy as np
from common import ROOT,MODEL_ROOT,DATA_ROOT,sha,save_json
from STEP5_两级寻迹 import score_tracks,summarize,DEFAULT_CONFIG


def same(a,b):
    if set(a)!=set(b):raise AssertionError('Score fields changed')
    if not all(np.isclose(a[k],b[k],rtol=1e-12,atol=1e-10) for k in a):raise AssertionError('Score counts changed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True);args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot;out=snapshot/'model_comparison_v1'
    report=json.loads((out/'paired_comparison.json').read_text(encoding='utf-8'))
    geometry=np.load(DATA_ROOT/'wire_geometry.npy');audits={}
    for summary in [report['reference'],report['latest']]:
        plan=summary['plan'];source=MODEL_ROOT/plan['snapshot'];rows={k:[] for k in summary['methods']};fake_causes={k:Counter() for k in rows}
        if sha(source/'best_interaction_network_joint_no_time.pth')!=plan['weight_sha256']:raise AssertionError('Weight changed')
        confidence=MODEL_ROOT/plan['confidence_snapshot']/'candidate_confidence_v2/candidate_confidence.pkl'
        if sha(confidence)!=plan['confidence_model_sha256']:raise AssertionError('Calibrator changed')
        if any(sha(ROOT/n)!=s for n,s in plan['source_hashes'].items()):raise AssertionError('Source changed')
        for eid in plan['event_ids']:
            row=json.loads((source/'model_comparison_v1/events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
            with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:truth,cells,hits=f['truth'],f['cell_id'],f['hit_id']
            stereo=np.linalg.norm(geometry[cells,2:4],axis=1)>1e-9
            eligible={int(t):int(((truth==t)&~stereo).sum()) for t in np.unique(truth) if t>=0 and ((truth==t)&~stereo).sum()>=5}
            for name,candidates in row['candidates'].items():
                indices=[i for c in candidates for i in c['indices']]
                if len(indices)!=len(set(indices)):raise AssertionError('Repeated hit ownership')
                for c in candidates:
                    ids=np.array(c['indices'],dtype=int)
                    if len(ids)<1 or ids.min()<0 or ids.max()>=len(hits):raise AssertionError('Invalid membership')
                    if hits[ids].tolist()!=c['hit_ids']:raise AssertionError('Independent HitID mismatch')
                    aa=ids[~stereo[ids]]
                    if hits[aa].tolist()!=c['axial_hit_ids']:raise AssertionError('Axial HitID mismatch')
                    tids,n=np.unique(truth[aa],return_counts=True);t=int(tids[n.argmax()]);correct=int(n.max())
                    if t not in eligible:fake_causes[name]['ineligible_dominant_truth']+=1
                    elif correct/len(aa)<.8:fake_causes[name]['mixed_axial_hits']+=1
                    elif correct/eligible[t]<.5:fake_causes[name]['insufficient_truth_coverage']+=1
                counts,matched=score_tracks(candidates,truth,stereo,DEFAULT_CONFIG)
                same(counts,row['methods'][name])
                if matched!=row['matched_truth_ids'][name]:raise AssertionError('Matched truth IDs changed')
                if counts['output_candidates']!=counts['matched_tracks']+counts.get('fake_candidates',0)+counts.get('clone_candidates',0):
                    raise AssertionError('Candidate partition does not add up')
                rows[name].append(counts)
        for name,v in rows.items():
            same(summarize(v),summary['methods'][name])
            if sum(fake_causes[name].values())!=summary['methods'][name].get('fake_candidates',0):raise AssertionError('Fake classification mismatch')
        audits[plan['snapshot']]=dict(events=len(plan['event_ids']),methods=len(rows),all_scores_recomputed=True,
                                    exclusive_hit_ownership=True,all_frozen_hashes_match=True,fake_causes={k:dict(v) for k,v in fake_causes.items()})
    eid=4779;input_path=out/f'observable_only_event_{eid:05d}.npz';inference=out/f'inference_epoch{report["latest"]["plan"]["epoch"]}_{eid:05d}.json'
    if not input_path.exists():
        with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:
            np.savez_compressed(input_path,cell_id=f['cell_id'],d=f['d'],hit_id=f['hit_id'])
    with np.load(input_path) as f:
        if set(f.files)!={'cell_id','d','hit_id'}:raise AssertionError('Unexpected inference input fields')
    if not inference.exists():
        subprocess.run([sys.executable,str(ROOT/'STEP4_联合寻迹推理.py'),'--snapshot',args.snapshot,'--event',str(eid),
                        '--input-npz',str(input_path),'--output',str(inference)],check=True,cwd=ROOT)
    predicted=json.loads(inference.read_text(encoding='utf-8'))
    expected=json.loads((out/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))['candidates']['conservative_joint']
    def members(rows):return sorted(tuple(sorted(c['hit_ids'])) for c in rows)
    if members(predicted['candidates'])!=members(expected):raise AssertionError('Deployment/evaluation membership differs')
    result=dict(audits=audits,truth_free_input_fields=['cell_id','d','hit_id'],inference_event=eid,
                deployment_matches_saved_test_memberships=True,inference_entry_sha256=sha(ROOT/'STEP4_联合寻迹推理.py'))
    save_json(out/'verification.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
