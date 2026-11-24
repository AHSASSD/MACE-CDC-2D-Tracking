from pathlib import Path
import shutil,json,hashlib,sys,platform,importlib.metadata as md,re
repo=Path(r'D:\pycharm\root操作\MACE-CDC-Tracking-Publish');root=repo/'AI工程化6';source=Path(r'D:\pycharm\root操作\AI工程化6')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
extra={'.pkl','.joblib','.pickle','.cmd','.pyw','.m','.cu','.cxx','.csv'}
added=[]
for p in source.rglob('*'):
 if p.is_file() and p.suffix.lower() in extra:
  dest=root/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);added.append(dict(path=dest.relative_to(root).as_posix(),source=str(p),sha256=sha(p),bytes=p.stat().st_size))
backup=Path(r'D:\pycharm\root操作\AI工程化6_稳定基线备份')
for p in backup.rglob('*') if backup.exists() else []:
 if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower() in {'.py','.md','.json','.pkl','.pth','.pt','.sh','.sbatch','.cmd'} and p.name not in {'train.pt','test.pt','validation.pt'} and p.stat().st_size<10*1024**2:
  dest=root/'protected_baselines'/p.relative_to(backup);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
models=[]
known={'5f3f02409399cd62efce5f254be02fa004237fb61a5b4717681cad1ad90cdcac':'stable_joint_epoch126','46a9d457a19e3e81112534f52e5a76e0437baa26b06c84f98fcd29f0e3b3eed2':'experimental_T25','be28c5472f612a5000a7a134363cef9769cd13d42fb27fc40535137a489de9f7':'experimental_OC_control7'}
for p in root.rglob('*'):
 if p.is_file() and p.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors','.onnx','.pkl','.joblib','.pickle'}:
  h=sha(p);models.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=h,role=known.get(h,'read_associated_contract_before_use'),format='calibrator_or_estimator' if p.suffix.lower() in {'.pkl','.joblib','.pickle'} else ('resume_state' if 'state' in p.name.lower() else 'model_or_checkpoint')))
(root/'docs/MODEL_CATALOG.json').write_text(json.dumps(dict(date='2026-11-24',count=len(models),models=models),ensure_ascii=False,indent=2),encoding='utf-8')
(root/'docs/ADDITIONAL_FILES.json').write_text(json.dumps(added,ensure_ascii=False,indent=2),encoding='utf-8')
env=dict(python=sys.version,platform=platform.platform(),packages={})
for name in ['torch','torch-geometric','numpy','scipy','pandas','matplotlib','scikit-learn','openpyxl']:
 try:env['packages'][name]=md.version(name)
 except md.PackageNotFoundError:env['packages'][name]='not installed'
(root/'docs/ENVIRONMENT_LOCAL.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
doc=root/'docs/HANDOFF.md';doc.write_text(doc.read_text(encoding='utf-8').replace('STEP19_六面板可视化诊断.py','STEP19_六联图可视化诊断.py'),encoding='utf-8')
for name,title in [('README.md','## AI6 joint axial/stereo research handoff (2026-11-24)'),('README.zh-CN.md','## AI6直丝—斜丝联合研究交接（2026-11-24）')]:
 p=repo/name;text=p.read_text(encoding='utf-8')
 if title not in text:text+='\n'+title+'\n\n[交接总览 / Handoff](AI工程化6/docs/HANDOFF.md) · [效果与口径 / Results](AI工程化6/docs/RESULTS.md) · [模型清单 / Model catalog](AI工程化6/docs/MODEL_CATALOG.json) · [代理与同学入口 / Agent entry](AGENT.md)\n\nIncludes mature and experimental AI6 code, all located model/resume checkpoints, figures, metadata, measurement NPZ and geometry. Experimental OC is not promoted. Raw OC cluster metrics are not directly comparable with the complete two-stage tracker. Large regenerable graph/latent caches and duplicate archives are listed separately. Existing versions remain intact.\n'
 p.write_text(text,encoding='utf-8')
gallery=['# 可视化索引','现有图片随原相对路径保存。所有图片路径和SHA可由本次快照清单追踪。']
for rel in ['visual_diagnosis_20261121/joint_events_1500_2000/event_01500_T1317_baseline.png','visual_diagnosis_20261121/joint_events_1500_2000/event_01501_T2102_baseline.png','experiments/联合归属训练_20261123/review_20261124/stage_review.png']:
 if (root/rel).exists():gallery+=['',f'## {Path(rel).name}',f'![{Path(rel).name}](../{rel})']
(root/'docs/VISUALIZATIONS.md').write_text('\n'.join(gallery)+'\n',encoding='utf-8')
bad=[];pattern=re.compile(r'gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|(?i:password|api_key|access_token)\s*[:=]\s*[\"\x27][^\"\x27\n]{6,}[\"\x27]')
for p in root.rglob('*'):
 if p.is_file() and p.suffix.lower() in {'.py','.md','.json','.sh','.sbatch','.txt','.toml','.yml','.yaml','.log'}:
  content=p.read_text(encoding='utf-8',errors='replace')
  if pattern.search(content):bad.append(p.relative_to(root).as_posix())
print(json.dumps(dict(models=len(models),model_bytes=sum(x['bytes'] for x in models),additional=len(added),secret_scan_candidates=bad),ensure_ascii=False))
if bad:raise SystemExit('Review candidate files before publishing')
