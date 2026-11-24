"""Login-host supervisor; restarts only its own step, never cancels the allocation."""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    assert a.job.isdigit()
    root=a.root;out=root/'outputs';out.mkdir(exist_ok=True)
    lock=(out/'guard.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    home=Path.home();failed=0;attempt=0
    while True:
        info=subprocess.check_output(['squeue','-h','-j',a.job,'-o','%u|%T|%e'],universal_newlines=True).strip().split('|')
        if len(info)!=3 or info[1]!='RUNNING':write(out/'guard_status.json',dict(status='allocation_ended'));return
        assert info[0]==getpass.getuser(), 'Allocation owner mismatch'
        if (out/'STOP_GUARD').exists():write(out/'guard_status.json',dict(status='stopped_by_user'));return
        completed=read(out/'status.json',{}).get('completed',{})
        if read(out/'status.json',{}).get('status') in ('complete','stopped_before_incomplete_pair'):
            write(out/'guard_status.json',dict(status='training_finished',allocation_cancelled=False));return
        prior=sum(completed.values());attempt+=1;started=time.time()
        env=os.environ.copy();env.update(XDG_RUNTIME_DIR='/tmp',MACE_WORKERS='2',MACE_PIN_MEMORY='1',MACE_ASYNC_TRANSFER='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',MACE_GRAPH_CACHE='/seu_nvme/home/zhangyongchao/213230965/ai6_graph_cache_v1')
        end=time.mktime(time.strptime(info[2],'%Y-%m-%dT%H:%M:%S'))
        env['SLURM_JOB_END_TIME']=str(int(end))
        if time.time()+900>end:
            write(out/'guard_status.json',dict(status='insufficient_allocation_time',allocation_cancelled=False));return
        # Reuse the most recent retained AI6 environment. This avoids unpacking another ~60GB copy.
        env['ENROOT_DATA_PATH']=str(home/'.local/share/enroot')
        cmd=['/usr/bin/srun','--jobid='+a.job,'--overlap','--export=ALL','--nodes=1','--ntasks=1','--cpus-per-task=3','--gres=gpu:1','--mpi=none','--job-name=ai6-paired',
             '--container-name=ai6_training_1530118','--container-image='+str(home/'pytorch-24.08-ssh.sqsh'),
             '--container-mounts=/seu_share:/seu_share,/seu_share2:/seu_share2,/seu_nvme:/seu_nvme',
             '--container-workdir='+str(root),'--container-writable','--no-container-remap-root',
             'python','-u',str(root/'paired_train.py'),'--data',str(home/'network_datasets/ai6/joint_event_dataset'),
             '--weight',str(home/'network_datasets/ai6/joint_training/evaluation_snapshot_epoch126_1794052868/best_interaction_network_joint_no_time.pth'),
             '--member-ids',str(root/'member_ids.json'),'--out',str(out),'--epochs','18']
        with (out/('attempt_%02d.log'%attempt)).open('ab') as log:
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        while proc.poll() is None:
            progress=read(out/'progress.json',{});updated=max(started,progress.get('unix',0))
            timeout=900 if progress.get('status')!='training' else 300
            state=dict(status='running',job=a.job,pid=proc.pid,attempt=attempt,consecutive_failures=failed,progress=progress,unix=time.time(),allocation_cancelled=False)
            write(out/'guard_status.json',state)
            if (out/'STOP_GUARD').exists() or time.time()-updated>timeout:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            time.sleep(15)
        now=read(out/'status.json',{})
        if now.get('status') in ('complete','stopped_before_incomplete_pair'):
            write(out/'guard_status.json',dict(status='training_finished',completed=now,allocation_cancelled=False));return
        failed=0 if sum(now.get('completed',{}).values())>prior else failed+1
        with (out/'incidents.jsonl').open('a') as f:
            f.write(json.dumps(dict(attempt=attempt,returncode=proc.returncode,consecutive_failures=failed,unix=time.time()))+'\n')
        if failed>=3:
            write(out/'guard_status.json',dict(status='needs_attention',reason='three attempts without a new completed arm epoch',allocation_cancelled=False));return
        time.sleep(20)


if __name__=='__main__':main()
