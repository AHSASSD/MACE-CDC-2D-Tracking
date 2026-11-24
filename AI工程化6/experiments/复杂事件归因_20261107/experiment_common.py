"""Import a pinned backup baseline; keep every experimental output separate."""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'1')
EXP=Path(__file__).resolve().parent
CONFIG=EXP/'experiment_config.json'
if not CONFIG.exists():
    latest=Path(r'D:\pycharm\root操作\AI工程化6_稳定基线备份\LATEST.json')
    pin=json.loads(latest.read_text(encoding='utf-8'))
    CONFIG.write_text(json.dumps(pin,ensure_ascii=False,indent=2),encoding='utf-8')
PIN=json.loads(CONFIG.read_text(encoding='utf-8'))
BASELINE=Path(PIN['project']);MANIFEST=Path(PIN['backup'])/'manifest.json'
if hashlib.sha256(MANIFEST.read_bytes()).hexdigest()!=PIN['manifest_sha256']:raise RuntimeError('Backup manifest changed')
manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
for name,expected in manifest['deployment_pins'].items():
    if hashlib.sha256((BASELINE/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Backup deployment pin changed: '+name)
sys.path.insert(0,str(BASELINE))
from common import DATA_ROOT,MODEL_ROOT,sha,save_json
SNAPSHOT=MODEL_ROOT/'local_evaluation_epoch126_20261107'
VALIDATION=SNAPSHOT/'coarse_3d_v4_fast_validation'
CONTRACT=json.loads((VALIDATION/'contract.json').read_text(encoding='utf-8'))
IDS=CONTRACT['event_ids']


def initialize_worker():
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1);threadpool_limits(limits=1)


def verify_pins():
    for name,expected in manifest['deployment_pins'].items():
        if sha(BASELINE/name)!=expected:raise RuntimeError('Backup deployment pin changed: '+name)

