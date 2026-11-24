"""Read the dedicated job's status and synchronize a compact local snapshot."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
REMOTE=r'''
import json,time,subprocess
from pathlib import Path
p=Path('/seu_nvme/home/zhangyongchao/213230965/ai6_curriculum_20261121')
def read(name,default=None):
    try:return json.loads((p/name).read_text())
    except (ValueError,OSError):return default
job=read('launch.json',{})
status=dict(unix=time.time(),launch=job,progress=read('outputs/progress.json'),
    checkpoint=read('outputs/checkpoint_meta.json'),supervisor=read('supervisor_status.json'),
    initial_members=read('outputs/initial_members.json'),preflight=read('runtime_config.json'),
    history=read('outputs/history.json',[])[-3:])
if job.get('job_id'):
    status['slurm']=subprocess.run(['scontrol','show','job',job['job_id']],stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.decode()
files=sorted((p/'outputs/member_metrics/epoch_000').glob('event_*.json'),key=lambda x:x.stat().st_mtime)
status['initial_member_events']=len(files)
if len(files)>1:
    recent=files[-min(10,len(files)):]
    status['recent_initial_seconds_per_event']=(recent[-1].stat().st_mtime-recent[0].stat().st_mtime)/(len(recent)-1)
logs=sorted(p.glob('attempt_*.log'))
if logs:status['latest_log_tail']=logs[-1].read_text(errors='replace')[-3000:]
print(json.dumps(status))
'''


def main():
    process=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','seu-gateway','python3','-'],
                           input=REMOTE,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=40)
    if process.returncode:raise RuntimeError(process.stderr)
    result=json.loads(process.stdout)
    (ROOT/'cloud_status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
