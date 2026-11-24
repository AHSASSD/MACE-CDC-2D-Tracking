"""Login-hostPython3.6, one idempotent bounded GPU submission."""
from pathlib import Path
import json,zipfile,subprocess,hashlib,shutil
ROOT=Path('/seu_nvme/home/zhangyongchao/213230965/ai6_t_path_20261122')
ARCHIVE=Path('/tmp/ai6_t_path_20261122.zip')

def main():
    if (ROOT/'launch.json').exists():print((ROOT/'launch.json').read_text());return
    own=subprocess.check_output(['squeue','-h','-u','213230965','-o','%i|%j|%T|%N']).decode();print('OWN_JOBS',own,flush=True)
    assert not any('ai6-triplet-path-junction' in row for row in own.splitlines()),'Matching job already exists; inspect before retry'
    assert shutil.disk_usage(str(ROOT.parent)).free>5*1024**3,'Need5GiB disk for focused training packs'
    initial=ROOT.parent/'ai6_et_20261122/full/T_best_members.pth'
    assert hashlib.sha256(initial.read_bytes()).hexdigest()=='46a9d457a19e3e81112534f52e5a76e0437baa26b06c84f98fcd29f0e3b3eed2'
    ROOT.mkdir(exist_ok=True)
    with zipfile.ZipFile(str(ARCHIVE)) as z:
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist());z.extractall(str(ROOT))
    subprocess.check_call(['bash','-n',str(ROOT/'step.sh')]);subprocess.check_call(['bash','-n',str(ROOT/'train_focus.sbatch')])
    check=subprocess.run(['/usr/bin/sbatch','--test-only',str(ROOT/'train_focus.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);print(check.stdout.decode(),flush=True);assert check.returncode==0
    result=subprocess.check_output(['/usr/bin/sbatch','--parsable',str(ROOT/'train_focus.sbatch')],stderr=subprocess.STDOUT).decode().strip();job=result.splitlines()[-1].split(';')[0];assert job.isdigit(),result
    record=dict(job_id=job,root=str(ROOT),source_archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),time_limit='03:00:00',max_finetune_epochs=30,initial='T25',frozen_backbone=True,auto_deploy=False)
    (ROOT/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

if __name__=='__main__':main()
