"""Apply benchmark-selected execution settings and keep the Slurm flock."""
import json
import os
import sys
from pathlib import Path

root=Path(__file__).resolve().parent
selection=json.loads((root/'server_runtime_selection.json').read_text());c=selection['config']
env=os.environ.copy()
env.update({'MACE_GRAPH_CACHE':selection['cache_root'],'MACE_WORKERS':str(c['workers']),
            'MACE_BATCH_SIZE':str(c['batch']),'MACE_VAL_BATCH_SIZE':str(selection.get('validation_batch_size',c['batch'])),
            'MACE_PIN_MEMORY':str(int(c['workers']>0)),'MACE_ASYNC_TRANSFER':str(int(c['workers']>0)),
            'MACE_RECOMPUTE':c['policy'],'OMP_NUM_THREADS':str(max(1,3-c['workers'])),
            'MKL_NUM_THREADS':str(max(1,3-c['workers'])),'OPENBLAS_NUM_THREADS':str(max(1,3-c['workers']))})
os.execve(sys.executable,[sys.executable,'-u',str(root/'运行完整流程.py')],env)
