"""HostPython3.6: restart own step, stop after3 attempts without durable progress."""
from pathlib import Path
import json,subprocess,sys,os,time,signal

def read(p,default=None):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return default

def save(p,value):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(p)

def main():
    root=Path(sys.argv[1]);failed=0;previous=-1
    # GPU allocation is used for training only. Extractor comparisons run locally.
    for stage in ['train']:
        if (root/(stage+'_done.json')).exists():continue
        while True:
            deadline=int(os.environ.get('SLURM_JOB_END_TIME','0') or 0)
            if deadline and time.time()+300>deadline:save(root/'status.json',dict(status='allocation_time_low',stage=stage));return
            save(root/'status.json',dict(status='running',stage=stage,unix=time.time()))
            started=time.time();log=(root/(stage+'.log')).open('ab');child=subprocess.Popen(['bash',str(root/'step.sh'),stage],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                time.sleep(15)
                files=[root/'training/progress.json',root/'training/checkpoint_meta.json']+list((root/stage).glob('event_*.json'))
                heartbeat=max([p.stat().st_mtime for p in files if p.exists()]+[started])
                if time.time()-heartbeat>900:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            log.close()
            status=read(root/'training/status.json',{}) if stage=='train' else {}
            if status.get('status')=='allocation_time_low':save(root/'status.json',dict(status='allocation_time_low',stage=stage));return
            if child.returncode==0:
                save(root/(stage+'_done.json'),dict(unix=time.time()));failed=0;previous=-1;break
            current=read(root/'training/checkpoint_meta.json',{}).get('updates',0)+len(list((root/stage).glob('event_*.json')))
            failed=failed+1 if current<=previous else 1;previous=current
            with (root/'incidents.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,returncode=child.returncode,no_progress_failures=failed,unix=time.time()))+'\n')
            if failed>=3:save(root/'status.json',dict(status='failed_three_times_without_progress',stage=stage));return
    save(root/'status.json',dict(status='completed',unix=time.time()))

if __name__=='__main__':main()
