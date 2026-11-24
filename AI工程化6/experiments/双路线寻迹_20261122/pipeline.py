"""Job-local staged supervision; own steps only, no scancel and no duplicate trainers."""
import json,os,signal,subprocess,sys,time
from pathlib import Path


def read(p,default=None):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return default


def save(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)


def progress(root):
    cached=len(list((root/'cache/stats').glob('*.json')))
    pilot=read(root/'pilot/checkpoint_meta.json',{}).get('updates',0)
    full=read(root/'full/checkpoint_meta.json',{}).get('updates',0)
    return cached+pilot*10000+full*1000000


def main():
    root=Path(sys.argv[1]);deadline=int(os.environ.get('SLURM_JOB_END_TIME','0') or 0)
    state=read(root/'pipeline_state.json',dict(step=0,attempts=0));stages=['prepare_pilot','check_pilot','pilot','prepare_full','check_full','full']
    failed=0;previous=progress(root)
    for index in range(state['step'],len(stages)):
        mode=stages[index]
        while True:
            if deadline and time.time()+300>deadline:save(root/'pipeline_status.json',dict(status='allocation_time_low',stage=mode,unix=time.time()));return
            state.update(step=index,stage=mode,attempts=state.get('attempts',0)+1);save(root/'pipeline_state.json',state)
            if mode.startswith('check_'):
                command=['python3',str(root/'runtime/check_cache.py'),'--root',str(root),'--mode',mode.split('_')[1]]
            else:command=['bash',str(root/'run_et_step.sh'),mode]
            started=time.time();log=(root/('stage_%s_attempt_%03d.log'%(mode,state['attempts']))).open('ab')
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                time.sleep(15)
                files=[root/'cache/progress.json',root/'pilot/progress.json',root/'pilot/checkpoint_meta.json',root/'full/progress.json',root/'full/checkpoint_meta.json']
                heartbeat=max([p.stat().st_mtime for p in files if p.exists()]+[started])
                if time.time()-heartbeat>900:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=45)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            log.close()
            if mode in ['pilot','full'] and child.returncode==0:
                status=read(root/mode/'status.json',{})
                if status.get('status')=='allocation_time_low':save(root/'pipeline_status.json',dict(status='allocation_time_low',stage=mode));return
                success=status.get('status') in ['completed','early_stopped']
            else:success=child.returncode==0
            if success:
                failed=0;previous=progress(root);state['step']=index+1;save(root/'pipeline_state.json',state);break
            current=progress(root);failed=failed+1 if current<=previous else 0;previous=current
            incident=dict(stage=mode,returncode=child.returncode,no_progress_failures=failed,progress=current,unix=time.time())
            with (root/'incidents.jsonl').open('a') as stream:stream.write(json.dumps(incident)+'\n')
            print(json.dumps(incident),flush=True)
            if failed>=3:save(root/'pipeline_status.json',dict(status='failed_three_times_without_progress',stage=mode));return
    save(root/'pipeline_status.json',dict(status='completed',unix=time.time()))


if __name__=='__main__':main()
