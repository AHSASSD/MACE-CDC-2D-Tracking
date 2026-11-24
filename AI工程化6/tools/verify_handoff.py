"""Offline verification of archived model bytes. No training or network access."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
catalog=json.loads((root/'docs/MODEL_CATALOG.json').read_text(encoding='utf-8'))
errors=[]
for entry in catalog['models']:
    p=root/entry['path']
    if not p.exists():errors.append(dict(path=entry['path'],reason='missing (run git lfs pull)'));continue
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    if h!=entry['sha256']:errors.append(dict(path=entry['path'],reason='checksum mismatch or LFS pointer'))
if errors:raise SystemExit(json.dumps(errors,ensure_ascii=False,indent=2))
print(json.dumps(dict(verified_models=len(catalog['models']),all_bytes_match=True)))
