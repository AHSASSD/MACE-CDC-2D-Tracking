"""Recompute saved event scores and audit protection, data separation and hashes."""
import experiment_common as e
import json
import numpy as np
from STEP5_两级寻迹 import score_tracks,summarize,DEFAULT_CONFIG
from 生成粗三维成员报告 import seed_counts
from STEP2_弱边候选校准 import seed_summary


def same(left,right):
    assert set(left)==set(right)
    for key in left:assert np.isclose(left[key],right[key],rtol=1e-12,atol=1e-9),(key,left[key],right[key])


def main():
    calibration=json.loads((e.EXP/'weak_edge_recovery_v1/summary.json').read_text(encoding='utf-8'))
    geometry=np.load(e.DATA_ROOT/'wire_geometry.npy');result={}
    for phase in ['validation','test']:
        folder=e.EXP/'evaluation_v1'/phase
        if not (folder/'summary.json').exists():continue
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'));plan=summary['plan']
        assert plan['scorer_sha256']==calibration['scorer_sha256']==e.sha(e.EXP/'weak_edge_recovery_v1/candidate_recovery_scorer.pkl')
        assert all(e.sha(e.EXP/name)==digest for name,digest in plan['source_sha256'].items())
        assert not set(plan['event_ids']).intersection(e.IDS)
        all_before=[];all_after=[];all_bseed=[];all_aseed=[];protected_a=0
        for eid in plan['event_ids']:
            row=json.loads((folder/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
            baseline=json.loads((folder/'baseline'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
            with np.load(e.DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,truth,hits=f['cell_id'],f['truth'],f['hit_id']
            stereo=np.linalg.norm(geometry[cells,2:4],axis=1)>1e-9
            assert [t['indices'] for t in row['candidates'][:len(baseline['tracks'])]]==[t['indices'] for t in baseline['tracks']]
            assert [t['indices'] for t in row['coarse'][:len(baseline['coarse'])]]==[t['indices'] for t in baseline['coarse']]
            protected_a+=sum(not stereo[i] for t in baseline['tracks'] for i in t['indices'])
            for key in ['candidates','coarse']:
                ids=[i for t in row[key] for i in t['indices']];assert len(ids)==len(set(ids))
                for t in row[key]:assert hits[t['indices']].tolist()==t['hit_ids']
            before,_=score_tracks(baseline['tracks'],truth,stereo,DEFAULT_CONFIG);after,_=score_tracks(row['candidates'],truth,stereo,DEFAULT_CONFIG)
            same(before,row['baseline']);same(after,row['selected'])
            bs=seed_counts(baseline['coarse'],truth,stereo,cells);ms=seed_counts(row['coarse'],truth,stereo,cells)
            same(bs,row['baseline_seed']);same(ms,row['selected_seed'])
            all_before.append(before);all_after.append(after);all_bseed.append(bs);all_aseed.append(ms)
        same(summarize(all_before),summary['baseline']);same(summarize(all_after),summary['selected'])
        same(seed_summary(all_bseed),summary['baseline_seed']);same(seed_summary(all_aseed),summary['selected_seed'])
        result[phase]=dict(events=len(all_before),all_scores_recomputed=True,old_candidate_identities_unchanged=True,
                           exclusive_hit_ownership=True,protected_axial_hits=protected_a,lost_axial_hits=0,
                           no_calibration_event_overlap=True,source_and_scorer_hashes_match=True)
    e.verify_pins();e.save_json(e.EXP/'evaluation_verification.json',dict(phases=result,baseline_pins_unchanged=True))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
