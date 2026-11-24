"""Replay a held validation event without truth fields and audit GenFit IDs."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from common import ROOT,MODEL_ROOT,DATA_ROOT,save_json,sha


def main():
    snapshot=MODEL_ROOT/'local_evaluation_epoch126_20261107';tag='coarse_3d_v4_fast'
    val=json.loads((snapshot/(tag+'_validation')/'summary.json').read_text(encoding='utf-8'))
    assert val['check_passed'];eid=val['contract']['event_ids'][100];profile=val['selected_on_development']
    folder=snapshot/(tag+'_report');folder.mkdir(exist_ok=True)
    source=folder/f'observable_only_event_{eid:05d}.npz';output=folder/f'genfit_event_{eid:05d}.json'
    if not source.exists():
        with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:np.savez_compressed(source,cell_id=f['cell_id'],d=f['d'],hit_id=f['hit_id'])
    with np.load(source) as f:assert set(f.files)=={'cell_id','d','hit_id'}
    if not output.exists():
        subprocess.run([sys.executable,str(ROOT/'STEP18_粗三维寻迹与GenFit导出.py'),'--event',str(eid),
                        '--input-npz',str(source),'--output',str(output)],cwd=ROOT,check=True)
    exported=json.loads(output.read_text(encoding='utf-8'))
    expected=json.loads((snapshot/(tag+'_validation')/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))['candidates'][profile]
    with np.load(source) as f:cells,hits,drift=f['cell_id'],f['hit_id'],f['d']
    geo=np.load(DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    reference={tuple(sorted(int(hits[i]) for i in t['indices'] if not stereo[i])):t for t in expected}
    for group in exported['candidates']:
        key=tuple(sorted(group['axial_hit_ids']));t=reference.pop(key)
        assert set(group['confirmed_stereo_hit_ids'])==set(t['confirmed_stereo_hit_ids'])
        assert set(group['pending_stereo_hit_ids'])==set(t['pending_stereo_hit_ids'])
        assert set(group['genfit_hit_ids'])==set(group['axial_hit_ids'])|set(group['confirmed_stereo_hit_ids'])
        assert not set(group['genfit_hit_ids']).intersection(group['pending_stereo_hit_ids'])
        for measurement in group['measurements']:
            i=int(np.flatnonzero(hits==measurement['hit_id'])[0]);mid=np.r_[geo[i,:2],0.];half=geo[i,2:5]*geo[i,5]/2
            assert np.allclose(measurement['wire_start_mm'],mid-half) and np.allclose(measurement['wire_end_mm'],mid+half)
            assert measurement['drift_distance_mm']==float(drift[i])
    assert not reference
    assert exported['lost_axial_hits']==0 and not exported['truth_used']
    result=dict(event_id=eid,input_fields=['cell_id','d','hit_id'],truth_free_replay_matches_held_event=True,
                exact_axial_identity_preserved=True,genfit_excludes_pending_s=True,finite_wire_measurements_verified=True,
                output=str(output),inference_source_sha256=sha(ROOT/'STEP18_粗三维寻迹与GenFit导出.py'))
    save_json(folder/'genfit_export_verification.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
