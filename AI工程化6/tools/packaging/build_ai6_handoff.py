"""Publish snapshot: all AI6 source/model weights; explicit cache exclusions."""
from pathlib import Path
import shutil,json,hashlib
source=Path(r'D:\pycharm\root操作\AI工程化6')
repo=Path(r'D:\pycharm\root操作\MACE-CDC-Tracking-Publish')
target=repo/'AI工程化6';target.mkdir(exist_ok=True)
code={'.py','.sh','.sbatch','.ps1','.bat','.cpp','.cc','.c','.h','.hpp','.ipynb'}
weights={'.pth','.pt','.ckpt','.onnx','.safetensors'}
meta={'.md','.json','.yaml','.yml','.toml','.txt','.ini'}
images={'.png','.jpg','.jpeg','.gif','.svg','.html'}
skip={'__pycache__','.git','.idea','.venv','cache','features','indices','graph_cache','predictions'}
inventory=[];excluded=[]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
for p in source.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(source);suffix=p.suffix.lower();reason=None
 if any(x in skip for x in rel.parts):reason='generated_cache_or_tool_state'
 elif p.name.startswith('.env') or p.name in {'train.pt','validation.pt','test.pt'}:reason='environment_or_regenerable_graph_dataset'
 elif suffix in code|weights|images:pass
 elif suffix in meta and p.stat().st_size<=1024**2:pass
 elif rel.parts[0]=='joint_event_dataset' and suffix in {'.npz','.npy'}:pass
 else:reason='regenerable_array_or_large_raw_report_or_duplicate_archive'
 # Model/source files take precedence over cache-location exclusions.
 if suffix in code|weights and '__pycache__' not in rel.parts and p.name not in {'train.pt','validation.pt','test.pt'}:reason=None
 if reason:
  excluded.append(dict(path=str(rel.as_posix()),bytes=p.stat().st_size,reason=reason));continue
 dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 inventory.append(dict(path='AI工程化6/'+rel.as_posix(),source=str(p),bytes=p.stat().st_size,sha256=digest(p),kind='code' if suffix in code else ('weight' if suffix in weights else 'artifact')))
docs=target/'docs';docs.mkdir(exist_ok=True)
(docs/'LOCAL_FILE_INVENTORY.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
(docs/'EXCLUDED_REGENERABLE_FILES.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(inventory),weights=sum(x['kind']=='weight' for x in inventory),bytes=sum(x['bytes'] for x in inventory),excluded=len(excluded))))
