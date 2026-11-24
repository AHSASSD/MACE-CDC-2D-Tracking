"""Run on the login host: archive user files, then remove three expired AI6 containers."""
import hashlib
import argparse
import json
import os
import subprocess
import tarfile
import time


NAMES = ['pyxis_ai6_training_1527110', 'pyxis_ai6_training_1528054',
         'pyxis_ai6_training_1529367']
parser=argparse.ArgumentParser()
parser.add_argument('--old-ssh',action='store_true')
args=parser.parse_args()
if args.old_ssh:
    NAMES=['pyxis_1527110.4294967291','pyxis_1528054.4294967291','pyxis_1529367.4294967291']
base = os.path.realpath(os.path.expanduser('~/.local/share/enroot'))
archive_root = '/seu_nvme/home/zhangyongchao/213230965/cloud_cleanup_' + time.strftime('%Y%m%d_%H%M%S')
os.makedirs(archive_root, mode=0o700)
log_path = os.path.join(archive_root, 'manifest.json')
records = []


def save():
    with open(log_path, 'w') as f:
        json.dump({'container_root': base, 'records': records}, f, indent=2)


def quota():
    result = subprocess.run(['/usr/lpp/mmfs/bin/mmlsquota', '-g', 'zhangyongchao',
                             '--block-size', 'G', 'ddnfs2'], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    print(result.stdout, flush=True)
    return result.stdout


assert os.getuid() == 4977
states=subprocess.check_output(['sacct','-j','1527110,1528054,1529367','-n','-X','-o','JobID,State','-P'],universal_newlines=True)
for job in ['1527110','1528054','1529367']:
    matches=[line.split('|')[1].strip() for line in states.splitlines() if line.split('|')[0].strip()==job]
    assert len(matches)==1 and matches[0] in ('TIMEOUT','COMPLETED','FAILED','CANCELLED','CANCELLED+'), 'Target job is not terminal'
print(json.dumps({'archive_root': archive_root, 'quota_before': quota()}), flush=True)
for name in NAMES:
    path = os.path.join(base, name)
    assert os.path.realpath(path) == path and os.path.dirname(path) == base
    assert not os.path.islink(path)
    if not os.path.isdir(path):
        continue
    mountpoints = [line.split()[4].replace('\\040', ' ') for line in open('/proc/self/mountinfo')]
    assert not any(m == path or m.startswith(path + '/') for m in mountpoints), 'Nested mount: stop'
    record = {'name': name, 'started_at': time.time(), 'preserved_files': 0, 'preserved_bytes': 0}
    archive = os.path.join(archive_root, name + '_user_files.tar.gz')

    def filter_entry(info):
        parts = info.name.split('/')
        if any(p in ('.cache', '__pycache__', '.nv') for p in parts):
            return None
        if info.isfile():
            record['preserved_files'] += 1
            record['preserved_bytes'] += info.size
        return info

    with tarfile.open(archive, 'w:gz', dereference=False) as tf:
        for sub in ('root', 'home', 'workspace', 'tmp'):
            source = os.path.join(path, sub)
            if os.path.lexists(source):
                tf.add(source, arcname=sub, filter=filter_entry)
    with tarfile.open(archive, 'r:gz') as tf:
        assert sum(1 for member in tf if member.isfile()) == record['preserved_files']
    h = hashlib.sha256()
    with open(archive, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    record.update(archive=archive, archive_sha256=h.hexdigest(), archive_bytes=os.path.getsize(archive), status='archived')
    records.append(record)
    save()
    print(json.dumps(record), flush=True)
    result = subprocess.run(['enroot', 'remove', '--force', name])
    record.update(remove_exit_code=result.returncode, removed=not os.path.lexists(path), ended_at=time.time())
    save()
    print(json.dumps(record), flush=True)
    assert record['removed'] and result.returncode == 0, 'Removal incomplete: stop'
print(json.dumps({'quota_after': quota(), 'manifest': log_path}), flush=True)
