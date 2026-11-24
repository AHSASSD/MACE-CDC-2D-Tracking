from pathlib import Path
import json,subprocess

ROOT=Path(__file__).resolve().parent.parent
REMOTE=r'''
import json,subprocess,time
from pathlib import Path
p=Path('/seu_nvme/home/zhangyongchao/213230965/ai6_et_20261122')
def read(name,default=None):
 try:return json.loads((p/name).read_text())
 except (OSError,ValueError):return default
job=read('launch.json',{})
rows=[json.loads(f.read_text()) for f in (p/'cache/stats').glob('*.json')]
reference=sum(x['triplet_pruning']['reference_reconstructible'] for x in rows);lost=sum(x['triplet_pruning']['lost_by_bounded_triplets'] for x in rows)
result=dict(unix=time.time(),launch=job,pipeline=read('pipeline_state.json'),pipeline_status=read('pipeline_status.json'),
 cache_progress=read('cache/progress.json'),cache_events=len(rows),pruning_reference=reference,pruning_lost=lost,
 cache_gate_pilot=read('cache_gate_pilot.json'),cache_gate_full=read('cache_gate_full.json'),
 pilot_progress=read('pilot/progress.json'),pilot_checkpoint=read('pilot/checkpoint_meta.json'),pilot_status=read('pilot/status.json'),
 full_progress=read('full/progress.json'),full_checkpoint=read('full/checkpoint_meta.json'),full_status=read('full/status.json'),
 pilot_history=read('pilot/history.json',[])[-2:],full_history=read('full/history.json',[])[-2:])
if job.get('job_id'):result['slurm']=subprocess.run(['scontrol','show','job',job['job_id']],stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.decode()
print(json.dumps(result))
'''


def main():
    proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','seu-gateway','python3','-'],input=REMOTE,
                        encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=40)
    if proc.returncode:raise RuntimeError(proc.stderr)
    result=json.loads(proc.stdout);(ROOT/'cloud_status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    compact={k:v for k,v in result.items() if k not in ['slurm','pilot_history','full_history']}
    print(json.dumps(compact,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
