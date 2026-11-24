"""Read-only cloud export of AI6 code, weights and run metadata."""
from pathlib import Path
import tarfile,json,hashlib
base=Path('/seu_nvme/home/zhangyongchao/213230965')
roots=[p for p in base.glob('ai6_*') if p.is_dir()]
roots.append(Path('/seu_share/home/zhangyongchao/213230965/network_datasets/ai6/joint_training'))
selected=[];weights=[]
skip={'cache','features','indices','__pycache__','.git','validation','events'}
for root in roots:
 for p in root.rglob('*'):
  if not p.is_file() or any(x in skip for x in p.relative_to(root).parts):continue
  suffix=p.suffix.lower()
  if suffix in {'.pth','.pt','.ckpt','.safetensors','.onnx'}:
   if p.name in {'train.pt','validation.pt','test.pt'}:continue
   weights.append(p)
  elif suffix not in {'.py','.md','.json','.sbatch','.sh','.yaml','.yml','.toml','.txt','.log'}:continue
  if suffix not in {'.pth','.pt','.ckpt','.safetensors','.onnx'} and p.stat().st_size>2*1024**2:continue
  if p.name.startswith('.env') or 'private_key' in p.name.lower():continue
  selected.append((p,root.name+'/'+str(p.relative_to(root))))
inventory=[]
with tarfile.open(base/'ai6_handoff_20261124.tar.gz','w:gz') as tar:
 for p,name in sorted(selected):
  tar.add(p,arcname=name,recursive=False)
  inventory.append(dict(path=name,source=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),is_weight=p in weights))
(base/'ai6_handoff_inventory_20261124.json').write_text(json.dumps(inventory,indent=2))
print(json.dumps(dict(files=len(selected),weights=len(weights),bytes=sum(x['bytes'] for x in inventory),archive=str(base/'ai6_handoff_20261124.tar.gz'))))
