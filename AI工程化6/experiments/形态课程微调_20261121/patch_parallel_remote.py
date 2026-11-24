"""Replace serial evaluation before any optimizer update; restart only this training step."""
import hashlib,json,subprocess,time,zipfile
from pathlib import Path

ROOT=Path('/seu_nvme/home/zhangyongchao/213230965/ai6_curriculum_20261121')
ZIP=Path('/tmp/ai6_curriculum_20261121_parallel.zip')


def main():
    meta=json.loads((ROOT/'outputs/checkpoint_meta.json').read_text())
    assert meta['updates']==0 and meta['cursor']==0 and meta['epoch']==1
    step=subprocess.check_output(['scontrol','show','step','1857763.1']).decode()
    assert 'UserId=4977' in step and 'State=RUNNING' in step
    with zipfile.ZipFile(str(ROOT/'runtime_before_parallel.zip'),'w',zipfile.ZIP_DEFLATED) as backup:
        for path in (ROOT/'runtime').iterdir():
            if path.is_file():backup.write(str(path),path.name)
    with zipfile.ZipFile(str(ZIP)) as archive:
        for name in archive.namelist():
            if not name.startswith('runtime/'):continue
            assert '..' not in Path(name).parts and not name.startswith('/')
            content=archive.read(name);path=ROOT/name;temp=path.with_suffix('.patch')
            # Login Python is3.6; actual container runtime is3.10. Syntax was checked locally.
            if path.suffix=='.py':content.decode('utf-8')
            temp.write_bytes(content);temp.replace(path)
    contract=ROOT/'outputs/contract.json'
    contract.rename(ROOT/'outputs/contract_before_parallel.json')
    result=dict(unix=time.time(),reason='parallel CPU reconstruction and overlap initial baseline with GPU training',
        before_checkpoint=meta,archive_sha256=hashlib.sha256(ZIP.read_bytes()).hexdigest(),
        original_job_retained=True,training_step_to_restart='1857763.1')
    (ROOT/'parallel_patch.json').write_text(json.dumps(result,indent=2))
    # Step ID is explicit. The GPU allocation and its supervisor remain running.
    subprocess.check_call(['scancel','--signal=TERM','1857763.1'])
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
