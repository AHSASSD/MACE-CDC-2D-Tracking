"""Package source only. Local smoke caches and trained smoke states are excluded."""
from pathlib import Path
import ast,hashlib,json,zipfile

ROOT=Path(__file__).resolve().parent


def main():
    for path in ROOT.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,10))
    with zipfile.ZipFile(ROOT/'runtime_upload.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in ROOT.glob('*.py'):archive.write(path,'runtime/'+path.name)
        archive.write(ROOT/'split_manifest.json','runtime/split_manifest.json')
        for name in ['run_et_step.sh','train_et.sbatch','pipeline.py']:archive.write(ROOT/name,name)
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    (ROOT/'packaged_sources.json').write_text(json.dumps(source,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(package_bytes=(ROOT/'runtime_upload.zip').stat().st_size,python_sources=len(source),smoke_weights_included=False)),flush=True)


if __name__=='__main__':main()
