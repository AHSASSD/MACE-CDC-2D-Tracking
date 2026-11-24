"""Full local project snapshot, byte verification and frozen deployment pins."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

SOURCE=Path(r'D:\pycharm\root操作\AI工程化6')
BACKUPS=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def main():
    started=time.time();stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    target=BACKUPS/('AI6_稳定基线_'+stamp);target.mkdir(exist_ok=False);project=target/SOURCE.name;project.mkdir()
    paths=[p for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.tmp']
    size=sum(p.stat().st_size for p in paths)
    if shutil.disk_usage(BACKUPS).free<size+2*1024**3:raise RuntimeError('Insufficient disk margin')
    contract=json.loads((SOURCE/'joint_training/local_evaluation_epoch126_20261107/coarse_3d_v4_fast_validation/contract.json').read_text(encoding='utf-8'))
    pins={**contract['source_sha256'],
          'joint_training/local_evaluation_epoch126_20261107/best_interaction_network_joint_no_time.pth':contract['weight_sha256'],
          'joint_training/local_evaluation_epoch095_20261107/candidate_confidence_v2/candidate_confidence.pkl':contract['confidence_sha256'],
          'joint_event_dataset/dataset_metadata_joint_no_time.json':contract['dataset_metadata_sha256']}
    for name,expected in pins.items():
        if digest(SOURCE/name)!=expected:raise RuntimeError('Verified baseline changed: '+name)
    def copy(path):
        rel=path.relative_to(SOURCE);dest=project/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        for _ in range(3):
            before=digest(path);shutil.copy2(path,dest);after=digest(path);copied=digest(dest)
            if before==after==copied:return rel.as_posix(),dict(bytes=dest.stat().st_size,sha256=copied)
        raise RuntimeError('File kept changing: '+str(rel))
    files={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(copy,p) for p in paths]):
            name,row=f.result();files[name]=row
            if len(files)%1000==0:
                progress=dict(files=len(files),total=len(paths),bytes=size,seconds=time.time()-started,target=str(target));save(BACKUPS/'backup_progress.json',progress);print(json.dumps(progress,ensure_ascii=False),flush=True)
    for name,expected in pins.items():
        if digest(project/name)!=expected or digest(SOURCE/name)!=expected:raise RuntimeError('Deployment pin changed: '+name)
    manifest=dict(created_beijing=stamp,source=str(SOURCE),backup_project=str(project),files=len(files),bytes=size,
                  sha256={k:v for k,v in sorted(files.items())},deployment_pins=pins,
                  excluded=['__pycache__ directories','*.tmp atomic/transient files'],
                  stable_models=dict(gnn_epoch=126,candidate_calibrator_epoch=95,coarse_profile='guarded_loose',coarse_run='coarse_3d_v4_fast'),
                  external_dependencies=dict(raw_csv='AI工程化4 - 副本/whole data (only needed to regenerate the dataset)',
                                             source_geometry_xlsx=r'D:\SRTP\CDC_cell_map\cdc_cell_map.xlsx'),
                  portable_inference_inputs_included=True,all_copied_files_verified=True,seconds=time.time()-started)
    save(target/'manifest.json',manifest)
    (target/'备份说明.md').write_text(f'''# AI6 稳定基线备份

时间（北京时间）：{stamp}。复制原AI工程化6完整目录（仅排除Python字节码缓存和临时文件），包含代码、10,000事件数据、冻结权重、候选校准器、概率缓存、逐事件评估及历史实验。

共{len(files):,}个文件，{size/1024**3:.3f}GiB。每个文件在复制前、复制后和备份副本中核对SHA256；清单见manifest.json。

稳定部署固定为第126轮GNN、第95轮候选校准器及coarse_3d_v4_fast的guarded_loose粗三维成员方案。备份用于恢复与只读基线对照；后续实验输出请放在独立目录。

推理与已有数据评估所需输入已包含。原CSV和原始几何XLSX是重新制集时的外部依赖，原文件继续保留在原位置。此备份不包含远端服务器磁盘或远端最新训练状态。

恢复时将本目录内AI工程化6复制到独立目标位置；不要直接覆盖存在未备份修改的目录。运行入口为STEP4_联合寻迹推理.py和STEP18_粗三维寻迹与GenFit导出.py。
''',encoding='utf-8')
    save(BACKUPS/'LATEST.json',dict(backup=str(target),project=str(project),manifest_sha256=digest(target/'manifest.json')))
    print(json.dumps(dict(backup=str(target),files=len(files),gib=size/1024**3,verified=True,seconds=time.time()-started),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
