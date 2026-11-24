"""Run on login node after obtaining an allocation; no new GPU job submitted."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);args=p.parse_args()
if not args.job_id.isdigit():raise ValueError('Numeric allocation ID required')
root=Path.home()/'network_datasets/ai6'
with (root/('joint_training/watchdog_'+args.job_id+'.log')).open('ab') as log:
    child=subprocess.Popen([sys.executable,str(root/'slurm_training_watchdog.py'),'--job-id',args.job_id],cwd=str(root),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'watchdog_pid':child.pid,'job_id':args.job_id}))
