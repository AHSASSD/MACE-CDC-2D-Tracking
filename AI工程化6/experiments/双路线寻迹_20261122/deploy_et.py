"""Login-host deployment. Test resources then submit one dedicated E/T job."""
import hashlib,json,subprocess,zipfile
from pathlib import Path

ROOT=Path('/seu_nvme/home/zhangyongchao/213230965/ai6_et_20261122')
ZIP=Path('/tmp/ai6_et_runtime_20261122.zip')


def main():
    own=subprocess.check_output(['squeue','-a','-h','-o','%i|%u|%T|%j|%N|%L']).decode()
    print(json.dumps(dict(existing_own_jobs=[x for x in own.splitlines() if '|213230965|' in x])),flush=True)
    if (ROOT/'launch.json').exists():
        print((ROOT/'launch.json').read_text(),flush=True);return
    assert Path('/seu_share/home/zhangyongchao/213230965/.local/share/enroot/pyxis_ai6_training_1530118').exists()
    ROOT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(str(ZIP)) as archive:
        assert all(not x.startswith('/') and '..' not in Path(x).parts for x in archive.namelist())
        archive.extractall(str(ROOT))
    subprocess.check_call(['bash','-n',str(ROOT/'run_et_step.sh')]);subprocess.check_call(['bash','-n',str(ROOT/'train_et.sbatch')])
    preflight=subprocess.run(['/usr/bin/sbatch','--test-only',str(ROOT/'train_et.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(preflight.stdout.decode(),flush=True)
    if preflight.returncode:raise RuntimeError('Slurm resource preflight failed')
    result=subprocess.check_output(['/usr/bin/sbatch','--parsable',str(ROOT/'train_et.sbatch')],stderr=subprocess.STDOUT).decode().strip()
    job=result.splitlines()[-1].split(';')[0];assert job.isdigit(),result
    record=dict(job_id=job,root=str(ROOT),source_archive_sha256=hashlib.sha256(ZIP.read_bytes()).hexdigest(),
                time_limit='06:00:00',routes=['E-small','T-small'],formal_max_epochs=60)
    (ROOT/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
