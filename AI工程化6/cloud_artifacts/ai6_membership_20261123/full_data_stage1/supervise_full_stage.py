"""Own-process-group supervisor: bounded retries and no-progress termination.
Does not cancel Slurm allocations or touch interactive jobs.
"""
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).parent;run=root/'run';run.mkdir(exist_ok=True)
command=[sys.executable,'-u',str(root/'train_full_stage.py'),'--data','/seu_share/home/zhangyongchao/213230965/network_datasets/ai6/joint_event_dataset','--cache','/seu_nvme/home/zhangyongchao/213230965/ai6_graph_cache_v1','--out',str(run),'--epochs','20','--workers','1']
failures=0;high_water=-1;attempt=0
while failures<3:
    attempt+=1;before=high_water;began=last=time.time();stamp=0
    child=subprocess.Popen(command,start_new_session=True)
    while child.poll() is None:
        time.sleep(10)
        try:
            path=run/'progress.json';modified=path.stat().st_mtime
            if modified>stamp:
                row=json.loads(path.read_text());stamp=modified;last=time.time()
                high_water=max(high_water,int(row.get('updates',-1)))
        except (FileNotFoundError,ValueError,OSError):pass
        timeout=900 if high_water<0 else 600
        if time.time()-last>timeout:
            print(json.dumps(dict(supervisor='no_progress_timeout',attempt=attempt,seconds=timeout)),flush=True)
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            break
    code=child.wait()
    record=dict(attempt=attempt,exit_code=code,updates_high_water=high_water,elapsed=time.time()-began)
    print(json.dumps(dict(supervisor=record)),flush=True)
    if code==0:sys.exit(0)
    failures=0 if high_water>before else failures+1
    record['consecutive_failures_without_progress']=failures
    (run/'supervisor.json').write_text(json.dumps(record))
sys.exit('Three failures without observed forward progress; standalone job exits.')
