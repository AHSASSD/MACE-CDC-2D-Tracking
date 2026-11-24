"""Wait for the fixed fresh validation, then evaluate the frozen test split."""
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent;snapshot='local_evaluation_epoch095_20261107'
target=root/'joint_training'/snapshot/'hybrid_fresh_validation_v1/summary.json'
deadline=time.time()+1200
while not target.exists():
    if time.time()>deadline:raise TimeoutError('Fresh validation not finished')
    time.sleep(5)
subprocess.run([sys.executable,'-u',str(root/'STEP15_保守联合寻迹复核.py'),'--snapshot',snapshot,'--split','test','--workers','12'],cwd=root,check=True)
