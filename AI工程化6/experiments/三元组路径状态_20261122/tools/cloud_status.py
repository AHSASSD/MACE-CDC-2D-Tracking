from pathlib import Path
import subprocess,json
ROOT=Path(__file__).resolve().parent.parent
REMOTE="""
from pathlib import Path
import json,subprocess,time
p=Path('/seu_nvme/home/zhangyongchao/213230965/ai6_t_path_20261122')
def read(name,default=None):
 try:return json.loads((p/name).read_text())
 except (OSError,ValueError):return default
launch=read('launch.json',{});h=read('training/history.json',[])
r=dict(unix=time.time(),launch=launch,status=read('status.json'),training_status=read('training/status.json'),progress=read('training/progress.json'),checkpoint=read('training/checkpoint_meta.json'),best=read('training/best.json'),history_tail=h[-2:],incidents=(p/'incidents.jsonl').read_text() if (p/'incidents.jsonl').exists() else '')
if launch:
 job=subprocess.run(['scontrol','show','job',launch['job_id']],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 r['job']=job.stdout.decode()
 if job.returncode:r['accounting']=subprocess.check_output(['sacct','-j',launch['job_id'],'--format=JobID,State,Elapsed,ExitCode','-n']).decode()
print(json.dumps(r))
"""

def main():
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','seu-gateway','python3','-'],input=REMOTE,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=40)
    if p.returncode:raise RuntimeError(p.stderr)
    r=json.loads(p.stdout);(ROOT/'cloud_status.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(r,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
