"""Run on login node. Launch only inside an existing user-owned allocation."""
import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job-id',required=True);args=parser.parse_args()
    if not args.job_id.isdigit():raise ValueError('Numeric allocation ID required')
    state=subprocess.check_output(['/usr/bin/squeue','-h','-j',args.job_id,'-o','%u|%T|%N'],universal_newlines=True).strip().split('|')
    if len(state)!=3 or state[0]!=getpass.getuser() or state[1]!='RUNNING':raise RuntimeError('Not a running allocation owned by this user')
    home=Path.home();root=home/'network_datasets/ai6'
    cmd=['/usr/bin/srun','--jobid='+args.job_id,'--overlap','--export=ALL,XDG_RUNTIME_DIR=/tmp','--nodes=1','--ntasks=1','--cpus-per-task=3','--gres=gpu:1','--mpi=none','--job-name=ai6-train',
         '--container-image='+str(home/'pytorch-24.08-ssh.sqsh'),'--container-name=ai6_training_'+args.job_id,'--container-mounts=/seu_share:/seu_share,/seu_share2:/seu_share2,/seu_nvme:/seu_nvme',
         '--container-workdir='+str(root),'--container-writable','--no-container-remap-root','/bin/bash',str(root/'start_ai6_training.sh')]
    env=os.environ.copy();env['XDG_RUNTIME_DIR']='/tmp'
    with (root/f'joint_training/slurm_step_{args.job_id}.log').open('ab') as log:
        proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'job_id':args.job_id,'node':state[2],'srun_pid':proc.pid,'command':cmd,'unix':time.time(),'scope':'existing allocation, no new allocation submitted'}
    (root/'slurm_training_launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

if __name__=='__main__':main()
