"""Login-node guard for one existing allocation; Python 3.6 compatible."""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from recovery_budget import update as recovery_update

ROOT=Path.home()/'network_datasets/ai6'

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,ValueError):return default

def epoch():
    rows=read(ROOT/'joint_training/joint_no_time_scratch_v1/training_history.json',[])
    return rows[-1]['epoch'] if rows else 0

def alive(pid,job):
    try:
        p=Path('/proc')/str(pid);cmd=(p/'cmdline').read_bytes().decode()
        return (p/'stat').read_text().split(')')[-1].strip()[0]!='Z' and '/usr/bin/srun' in cmd and '--jobid='+job in cmd
    except (FileNotFoundError,ProcessLookupError):return False

def job_state(job):
    text=subprocess.check_output(['/usr/bin/squeue','-h','-j',job,'-o','%u|%T|%N|%e'],universal_newlines=True).strip()
    parts=text.split('|')
    if len(parts)!=4:return None
    if parts[0]!=getpass.getuser():raise RuntimeError('Allocation owner mismatch')
    return parts

def next_failure(previous_epoch,current_epoch,streak):
    return 0 if current_epoch>previous_epoch else streak+1

def attention(status,job,incident):
    policy=read(ROOT/'supervisor_policy.json',{})
    authorized=bool(policy.get('auto_cancel_no_progress')) and job in policy.get('authorized_job_ids',[])
    write(status,{'status':'needs_attention','details':incident,'auto_cancel_authorized':authorized})
    if authorized and job_state(job):
        subprocess.check_call(['/usr/bin/scancel',job]);write(status,{'status':'allocation_released_after_repeated_failure','details':incident})

def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--job-id',required=True);args=parser.parse_args();job=args.job_id
    if not job.isdigit():raise ValueError('Numeric job ID required')
    lock=(ROOT/f'joint_training/watchdog_{job}.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('Watchdog already active',flush=True);return
    status=ROOT/f'joint_training/watchdog_{job}.json';stop=ROOT/f'joint_training/watchdog_stop_{job}'
    launch=read(ROOT/'slurm_training_launch.json',{});pid=launch.get('srun_pid',0)
    if str(launch.get('job_id'))!=job:pid=0
    baseline=epoch();streak=0;started=time.time();restarts=0;deadline=time.time()+6*3600
    while True:
        if stop.exists():write(status,{'status':'stopped_by_user','job':job,'unix':time.time()});return
        if time.time()>=deadline:write(status,{'status':'allocation_deadline_reached','job':job,'unix':time.time()});return
        try:allocation=job_state(job)
        except subprocess.CalledProcessError:
            write(status,{'status':'scheduler_query_failed','job':job,'unix':time.time()});time.sleep(30);continue
        if not allocation or allocation[1]!='RUNNING':write(status,{'status':'allocation_ended','job':job,'epoch':epoch(),'unix':time.time()});return
        try:deadline=time.mktime(time.strptime(allocation[3],'%Y-%m-%dT%H:%M:%S'))
        except ValueError:pass
        budget=read(ROOT/('joint_training/recovery_budget_'+job+'.json'),{})
        if epoch()>budget.get('last_recovery_epoch',epoch()):budget=recovery_update(ROOT,job,epoch())
        streak=budget.get('no_progress_failures',0)
        if budget.get('needs_attention'):
            attention(status,job,{'job':job,'budget':budget,'unix':time.time()});return
        pipeline=read(ROOT/'joint_training/pipeline_status.json',{})
        if pipeline.get('status')=='complete' and epoch()>=200:write(status,{'status':'complete','epoch':epoch(),'unix':time.time()});return
        if pid and alive(pid,job):
            progress=read(ROOT/'joint_training/joint_no_time_scratch_v1/training_progress.json',{})
            fresh=progress.get('updated_unix',0)>=started
            updated=max(started,progress.get('updated_unix',0));timeout=180 if fresh and pipeline.get('status')!='evaluating' else 600
            stalled=time.time()-updated>timeout
            phase='step_running' if fresh else 'step_starting'
            write(status,{'status':phase if not stalled else 'stalled_step','job':job,'srun_pid':pid,'completed_epoch':epoch(),'progress':progress,'progress_from_current_step':fresh,'restarts':restarts,'no_progress_failures':streak,'unix':time.time()})
            if not stalled:time.sleep(15);continue
            # Cancel only this guard's srun client/step, never the SSH batch service.
            os.kill(pid,signal.SIGTERM)
            for _ in range(30):
                if not alive(pid,job):break
                time.sleep(1)
            if alive(pid,job):os.kill(pid,signal.SIGKILL)
        if pid:
            current=epoch();budget=recovery_update(ROOT,job,current);streak=budget['no_progress_failures']
            incident={'job':job,'srun_pid':pid,'previous_completed_epoch':baseline,'current_completed_epoch':current,'no_progress_failures':streak,'pipeline':pipeline,'unix':time.time()}
            with (ROOT/'joint_training/watchdog_events.jsonl').open('a') as log:log.write(json.dumps(incident)+'\n')
            if streak>=3:
                attention(status,job,incident)
                return
        baseline=epoch();time.sleep(15 if pid else 0)
        recovery_update(ROOT,job,baseline,begin=True)
        text=subprocess.check_output(['python3',str(ROOT/'launch_slurm_step.py'),'--job-id',job],universal_newlines=True)
        launch=json.loads(text.strip().splitlines()[-1]);pid=launch['srun_pid'];started=time.time();restarts+=1
        write(status,{'status':'restarting_step','job':job,'srun_pid':pid,'checkpoint_epoch':baseline,'restarts':restarts,'unix':time.time()})

if __name__=='__main__':main()
