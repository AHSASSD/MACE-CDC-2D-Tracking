"""Copy a consistent, hash-verified validation-selected snapshot over SSH."""
import json
from pathlib import Path
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from common import MODEL_ROOT, DATA_ROOT, sha, save_json


def main():
    base = '/seu_share/home/zhangyongchao/213230965/network_datasets/ai6/joint_training'
    remote = r'''
import hashlib,json,os,shutil,time
from pathlib import Path
base=Path('/seu_share/home/zhangyongchao/213230965/network_datasets/ai6/joint_training')
source=base/'joint_no_time_scratch_v1'
names=['best_interaction_network_joint_no_time.pth','best_model_selection.json','model_metadata_joint_no_time.json']
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
for attempt in range(5):
    before={n:digest(source/n) for n in names}
    selection=json.loads((source/names[1]).read_text())
    dest=base/('evaluation_snapshot_epoch%03d_%d'%(selection['epoch'],int(time.time())))
    dest.mkdir()
    for n in names:shutil.copy2(str(source/n),str(dest/n))
    after={n:digest(source/n) for n in names}
    copied={n:digest(dest/n) for n in names}
    if before==after==copied:
        history=json.loads((source/'training_history.json').read_text())
        row=[r for r in history if r['epoch']==selection['epoch']]
        if len(row)!=1 or row[0]['validation']['loss']!=selection['validation']['loss']:
            raise RuntimeError('Selection/history mismatch')
        manifest=dict(epoch=selection['epoch'],remote_snapshot=str(dest),sha256=copied,
                      selected_history=row[0],capture_unix=time.time(),training_history_last=history[-1])
        (dest/'snapshot_manifest.json').write_text(json.dumps(manifest,indent=2))
        print(json.dumps(manifest));break
    shutil.rmtree(str(dest));time.sleep(1)
else:raise RuntimeError('Source kept changing')
'''
    flags = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '-o', 'StrictHostKeyChecking=yes']
    result = subprocess.run(['ssh', *flags, 'seu-gateway', 'python3 -'], input=remote,
                            encoding='utf-8', capture_output=True, check=True, timeout=90)
    manifest = json.loads(result.stdout)
    date = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    dest = MODEL_ROOT / ('local_evaluation_epoch%03d_%s' % (manifest['epoch'], date))
    dest.mkdir(exist_ok=True)
    for name, digest in manifest['sha256'].items():
        if (dest/name).exists():
            if sha(dest/name) != digest:raise RuntimeError('Existing snapshot differs')
        else:
            subprocess.run(['scp', *flags, 'seu-gateway:'+manifest['remote_snapshot']+'/'+name,
                            str(dest/name)], check=True, timeout=90)
        if sha(dest/name) != digest:raise RuntimeError('Download hash mismatch')
    metadata = json.loads((dest/'model_metadata_joint_no_time.json').read_text(encoding='utf-8'))
    if metadata['dataset_metadata_sha256'] != sha(DATA_ROOT/'dataset_metadata_joint_no_time.json'):
        raise RuntimeError('Dataset differs')
    save_json(dest/'snapshot_manifest.json', manifest)
    print(json.dumps(dict(snapshot=str(dest), epoch=manifest['epoch'], sha256=manifest['sha256']),ensure_ascii=False))


if __name__ == '__main__':main()
