"""Run development then a fresh validation gate. No test-set parameter selection."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for phase in ['dev','validation']:
    (root/'local_runner_status.json').write_text(json.dumps(dict(status='running',phase=phase,unix=time.time())))
    result=subprocess.run([sys.executable,'-B',str(root/'local_extractor.py'),'--phase',phase,'--workers','8'],env=env,cwd=str(root))
    if result.returncode:
        (root/'local_runner_status.json').write_text(json.dumps(dict(status='failed',phase=phase,returncode=result.returncode,unix=time.time())));sys.exit(result.returncode)
(root/'local_runner_status.json').write_text(json.dumps(dict(status='completed',unix=time.time(),stable_deployment_changed=False)))
