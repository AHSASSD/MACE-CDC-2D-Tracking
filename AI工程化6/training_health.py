"""Low-overhead process/mapping/cgroup diagnostics for external SIGKILL."""
import json
import os
from pathlib import Path
import subprocess
import time

def process_tree(pid):
    result=[];todo=[pid];seen=set()
    while todo:
        current=todo.pop()
        if current in seen:continue
        seen.add(current);base=Path('/proc')/str(current)
        try:
            text=(base/'status').read_text();status=dict(line.split(':',1) for line in text.splitlines() if ':' in line)
            keys=['VmRSS','RssAnon','RssFile','RssShmem','VmSize','VmLck','VmPin']
            row={'pid':current,**{k:status.get(k,'').strip() for k in keys},'fds':len(list((base/'fd').iterdir())),
                 'graph_cache_mappings':sum('ai6_graph_cache_v1' in line for line in (base/'maps').read_text().splitlines())}
            result.append(row)
            todo.extend(int(v) for v in (base/'task'/str(current)/'children').read_text().split())
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return result

def snapshot(pid):
    info={'unix':time.time(),'processes':process_tree(pid),'slurm_job':os.getenv('SLURM_JOB_ID'),'slurm_step':os.getenv('SLURM_STEP_ID')}
    try:info['trainer_cgroup']=(Path('/proc')/str(pid)/'cgroup').read_text()
    except FileNotFoundError:pass
    for name in ['memory.current','memory.peak','memory.events','memory.max']:
        path=Path('/sys/fs/cgroup')/name
        if path.exists():info[name]=path.read_text().strip()
    return info

def run_monitored(cmd,log,root,interval=5):
    health=Path(root)/'joint_training/training_health.jsonl'
    with health.open('a',encoding='utf-8') as output:
        with subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=root) as proc:
            while True:
                row=snapshot(proc.pid);row['trainer_pid']=proc.pid
                output.write(json.dumps(row)+'\n');output.flush()
                try:code=proc.wait(timeout=interval);break
                except subprocess.TimeoutExpired:pass
            output.write(json.dumps({'unix':time.time(),'trainer_pid':proc.pid,'returncode':code,'final':snapshot(proc.pid)})+'\n');output.flush()
            if code:raise subprocess.CalledProcessError(code,cmd)
