"""Create a portable snapshot; never modifies the active local checkpoint."""
import json
import tarfile
import time
import torch
import numpy as np
from common import *

def main():
    stamp=time.strftime('%Y%m%d_%H%M%S');out=ROOT/'remote_transfer'/stamp;out.mkdir(parents=True)
    local=MODEL_ROOT/'joint_no_time_scratch_v1/training_state_joint_no_time.pth'
    snapshot=out/'training_state_joint_no_time.pth';snapshot.write_bytes(local.read_bytes())
    original_hash=sha(snapshot);state=torch.load(snapshot,map_location='cpu',weights_only=False)
    # NumPy 2 ndarray pickles reference numpy._core, absent from remote NumPy 1.24.
    # The MT19937 keys are stored as plain integers; generator state is unchanged.
    rng=state['numpy_rng'];state['numpy_rng']=(rng[0],rng[1].tolist(),rng[2],rng[3],rng[4])
    np.random.set_state(state['numpy_rng'])
    torch.save(state,snapshot)
    record={'source_checkpoint':str(local),'source_sha256':original_hash,'portable_sha256':sha(snapshot),'epoch':state['epoch'],
        'config':state['config'],'numpy_conversion':'MT19937 uint32 ndarray -> list of identical integers; no change to model/optimizer/scheduler/scaler/RNG values',
        'dataset_metadata_sha256':sha(DATA_ROOT/'dataset_metadata_joint_no_time.json'),'created_unix':time.time()}
    save_json(out/'migration_manifest.json',record)
    files=['common.py','STEP2_训练模型.py','STEP3_推理评估.py','test_joint_contract.py','运行完整流程.py','模型与数据约定.md']
    package=out/'ai6_training.tar.gz'
    with tarfile.open(package,'w:gz',compresslevel=3) as tar:
        for name in files:tar.add(ROOT/name,arcname=name)
        tar.add(DATA_ROOT,arcname='joint_event_dataset')
        tar.add(snapshot,arcname='joint_training/joint_no_time_scratch_v1/training_state_joint_no_time.pth')
        for name in ['model_metadata_joint_no_time.json','training_history.json','best_model_selection.json','best_interaction_network_joint_no_time.pth']:
            path=MODEL_ROOT/'joint_no_time_scratch_v1'/name
            if path.exists():tar.add(path,arcname=f'joint_training/joint_no_time_scratch_v1/{name}')
        tar.add(out/'migration_manifest.json',arcname='migration_manifest.json')
        for path,arc in [(ROOT/'pilot_v1/manifest.json','pilot_v1/manifest.json'),(ROOT/'memory_safe_gpu/fresh_test_ids.json','memory_safe_gpu/fresh_test_ids.json')]:
            if path.exists():tar.add(path,arcname=arc)
    save_json(out/'package_manifest.json',{'package':str(package),'sha256':sha(package),'bytes':package.stat().st_size,'checkpoint_epoch':state['epoch']})
    save_json(ROOT/'remote_transfer/latest.json',{'directory':str(out)})
    print(json.dumps({'directory':str(out),'epoch':state['epoch'],'bytes':package.stat().st_size,'sha256':sha(package)}),flush=True)

if __name__=='__main__':main()
