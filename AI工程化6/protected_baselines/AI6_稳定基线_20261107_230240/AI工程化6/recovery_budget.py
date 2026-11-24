"""One shared no-progress recovery budget across process and Slurm-step guards."""
import json
from pathlib import Path

def update(root,job,epoch,begin=False):
    import fcntl
    folder=Path(root)/'joint_training';path=folder/('recovery_budget_'+str(job)+'.json')
    with (folder/('recovery_budget_'+str(job)+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(path.read_text()) if path.exists() else {'pending':False,'no_progress_failures':0,'last_recovery_epoch':epoch}
        if epoch>state['last_recovery_epoch']:state['no_progress_failures']=0
        if begin:
            if state['no_progress_failures']>=3:return state
            state.update(pending=True,last_recovery_epoch=epoch)
        else:
            if state['pending'] and epoch<=state['last_recovery_epoch']:state['no_progress_failures']+=1
            state['pending']=False
        state['needs_attention']=state['no_progress_failures']>=3
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(path)
        return state
