"""Job-local recovery. Only controls its own srun process; never calls scancel."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError,ValueError):
        return {}


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    root=Path(sys.argv[1]);out=root/'outputs';out.mkdir(exist_ok=True)
    deadline=int(os.environ.get('SLURM_JOB_END_TIME','0') or 0)
    previous=int(read(out/'checkpoint_meta.json').get('updates',0));failures=0;attempt=0
    while failures<3:
        if deadline and time.time()+240>deadline:
            write(root/'supervisor_status.json',dict(status='allocation_time_low',unix=time.time()))
            return 0
        attempt+=1;start=time.time();reason='process_exit'
        log=(root/('attempt_%02d.log'%attempt)).open('ab')
        child=subprocess.Popen(['bash',str(root/'run_step.sh'),'train'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            time.sleep(15)
            heartbeat=max([p.stat().st_mtime for p in [out/'progress.json',out/'checkpoint_meta.json'] if p.exists()]+[start])
            if time.time()-heartbeat>900:
                reason='no_heartbeat_900s';os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        log.close();status=read(out/'status.json');meta=read(out/'checkpoint_meta.json')
        if child.returncode==0 and status.get('status') in ['completed','early_stopped','allocation_time_low']:
            write(root/'supervisor_status.json',dict(status=status['status'],attempts=attempt,unix=time.time()))
            return 0
        updates=int(meta.get('updates',0));failures=failures+1 if updates<=previous else 0
        incident=dict(attempt=attempt,returncode=child.returncode,reason=reason,
                      checkpoint_updates=updates,consecutive_no_progress=failures,unix=time.time())
        with (root/'incidents.jsonl').open('a') as stream:stream.write(json.dumps(incident)+'\n')
        print(json.dumps(incident),flush=True);previous=updates
    write(root/'supervisor_status.json',dict(status='failed_three_times_without_checkpoint_progress',unix=time.time()))
    return 1


if __name__=='__main__':sys.exit(main())
