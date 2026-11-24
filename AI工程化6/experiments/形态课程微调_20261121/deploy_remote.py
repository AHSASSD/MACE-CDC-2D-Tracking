"""Run on the login host. Inspect existing work, validate submission, then submit once."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

RUN=Path('/seu_nvme/home/zhangyongchao/213230965/ai6_curriculum_20261121')
ZIP=Path('/tmp/ai6_curriculum_20261121_upload.zip')


def main():
    inventory=subprocess.check_output(['squeue','-a','-h','-o','%i|%u|%T|%j|%N|%L']).decode()
    own=[line for line in inventory.splitlines() if '|213230965|' in line]
    print(json.dumps(dict(existing_jobs=own)),flush=True)
    existing=[line for line in own if '|ai6-curriculum-recall|' in line]
    if existing:
        print('Existing matching job; no duplicate submission',flush=True)
        return
    if (RUN/'launch.json').exists():
        raise RuntimeError('A launch record already exists; inspect it before resubmitting')
    assert Path('/seu_share/home/zhangyongchao/213230965/.local/share/enroot/pyxis_ai6_training_1530118').exists()
    RUN.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(str(ZIP)) as archive:
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in archive.namelist())
        archive.extractall(str(RUN))
    subprocess.check_call(['bash','-n',str(RUN/'run_step.sh')])
    subprocess.check_call(['bash','-n',str(RUN/'train.sbatch')])
    test=subprocess.run(['/usr/bin/sbatch','--test-only',str(RUN/'train.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(test.stdout.decode(),flush=True)
    if test.returncode:raise RuntimeError('Submission preflight failed')
    if '--submit' not in sys.argv:
        return
    answer=subprocess.check_output(['/usr/bin/sbatch','--parsable',str(RUN/'train.sbatch')],stderr=subprocess.STDOUT).decode().strip()
    job=answer.splitlines()[-1].split(';')[0]
    assert job.isdigit(),answer
    record=dict(job_id=job,submission=answer,archive_sha256=hashlib.sha256(ZIP.read_bytes()).hexdigest(),root=str(RUN),time_limit='06:00:00')
    (RUN/'launch.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
