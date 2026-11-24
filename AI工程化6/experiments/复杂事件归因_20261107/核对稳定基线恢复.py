"""Run the backed-up deployment entry without writing into the backup."""
import experiment_common as e
import json
import os
import subprocess
import sys


def main():
    out=e.EXP/'baseline_restore_probe';out.mkdir(exist_ok=True)
    copied=e.SNAPSHOT/'coarse_3d_v4_fast_report'
    expected=json.loads((copied/'genfit_event_09408.json').read_text(encoding='utf-8'))
    result=out/'genfit_event_09408.json'
    if not result.exists():
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',LOKY_MAX_CPU_COUNT='4',OMP_NUM_THREADS='1')
        subprocess.run([sys.executable,'-B',str(e.BASELINE/'STEP18_粗三维寻迹与GenFit导出.py'),'--event','9408',
                        '--input-npz',str(copied/'observable_only_event_09408.npz'),'--output',str(result)],
                       cwd=e.BASELINE,env=env,check=True)
    actual=json.loads(result.read_text(encoding='utf-8'))
    def members(payload):return sorted((tuple(sorted(t['axial_hit_ids'])),tuple(sorted(t['confirmed_stereo_hit_ids'])),tuple(sorted(t['pending_stereo_hit_ids']))) for t in payload['candidates'])
    assert members(actual)==members(expected)
    assert actual['lost_axial_hits']==0 and actual['truth_used'] is False
    e.verify_pins()
    verification=dict(backup=e.PIN,event=9408,byte_verified_files=e.manifest['files'],
                      restored_inference_memberships_equal=True,truth_free_input=True,lost_axial_hits=0,
                      note='Executed directly from backup with -B; output outside backup; original deployment source untouched')
    e.save_json(out/'verification.json',verification);print(json.dumps(verification,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
