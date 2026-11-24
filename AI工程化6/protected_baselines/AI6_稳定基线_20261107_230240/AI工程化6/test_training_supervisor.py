"""Recovery policy check without using GPU or terminating actual processes."""
import importlib.util
import io
import os
import sys
import subprocess
import tempfile
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pipeline',root/'运行完整流程.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as directory:
    base=Path(directory);dest=base/'joint_training/joint_no_time_scratch_v1';dest.mkdir(parents=True)
    torch.save({'epoch':29},dest/'training_state_joint_no_time.pth')
    calls=[]
    def transient(cmd,log,cwd):
        calls.append(cmd)
        if len(calls)==1:raise subprocess.CalledProcessError(-9,cmd)
    module.supervised_training(['training'],io.StringIO(),base,runner=transient,delay=0)
    assert len(calls)==2
    assert '"checkpoint_epoch": 29' in (base/'joint_training/recovery_events.jsonl').read_text()
    calls.clear()
    def permanent(cmd,log,cwd):calls.append(cmd);raise subprocess.CalledProcessError(-9,cmd)
    try:module.supervised_training(['training'],io.StringIO(),base,runner=permanent,delay=0)
    except subprocess.CalledProcessError:pass
    else:raise AssertionError('Must stop after retry limit')
    assert len(calls)==3
    calls.clear()
    def programming_error(cmd,log,cwd):calls.append(cmd);raise subprocess.CalledProcessError(1,cmd)
    try:module.supervised_training(['training'],io.StringIO(),base,runner=programming_error,delay=0)
    except subprocess.CalledProcessError:pass
    else:raise AssertionError('Do not blindly retry program errors')
    assert len(calls)==1
    if os.name=='posix':
        calls.clear()
        def real_signal(cmd,log,cwd):
            calls.append(cmd)
            code='import os,signal;os.kill(os.getpid(),signal.SIGKILL)' if len(calls)==1 else 'print("recovered")'
            module.run_monitored([sys.executable,'-c',code],log,cwd,interval=.05)
        with (base/'fault_test.log').open('w') as log:
            module.supervised_training(['isolated_fault_test'],log,base,runner=real_signal,delay=0)
        assert len(calls)==2
print('PASS: checkpoint recovery on SIGKILL only, bounded retries, other failures stop')
