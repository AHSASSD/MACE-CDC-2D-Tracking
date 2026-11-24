import json
import os
from pathlib import Path
import torch
from graph_cache import CachedGraphDataset
from common import DATA_ROOT

r=Path(__file__).resolve().parent
selection=json.loads((r/'server_runtime_selection.json').read_text())
data=CachedGraphDataset(selection['cache_root'],'train',DATA_ROOT)
print(json.dumps({'job_id':os.getenv('SLURM_JOB_ID'),'step_id':os.getenv('SLURM_STEP_ID'),'gpu':torch.cuda.get_device_name(0),'train_events':len(data),'cgroup':Path('/proc/self/cgroup').read_text(),'torch':torch.__version__}),flush=True)
