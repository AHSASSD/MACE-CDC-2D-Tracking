"""Finalize saved batch artifacts; bound to the one known batch process."""
from pathlib import Path
import ctypes,json,time,hashlib
from collections import Counter
r=Path(__file__).resolve().parent/'joint_events_1500_2000'
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.OpenProcess.restype=ctypes.c_void_p
kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
kernel.CloseHandle.argtypes=[ctypes.c_void_p]
handle=kernel.OpenProcess(0x100000,False,13152)
deadline=time.time()+4*3600
while not (r/'summary.json').exists() and time.time()<deadline:
    if not handle or kernel.WaitForSingleObject(handle,0)==0:break
    time.sleep(10)
if handle:kernel.CloseHandle(handle)
rows=[json.loads(p.read_text(encoding='utf-8')) for p in r.glob('completed_*.json')]
by_split={}
missing=[]
for x in rows:
    by_split.setdefault(x['partition'],Counter()).update(x['metrics'])
    stem=Path(x['image']).stem
    for suffix in ['.png','_geometry.png','.json','_hits.csv','_edges.csv']:
        if not (r/(stem+suffix)).exists():missing.append(stem+suffix)
page=r/'index.html'
if page.exists():
    text=page.read_text(encoding='utf-8').replace('<img src=','<img loading="lazy" src=')
    text=text.replace('<h1>AI6 六联图诊断</h1>','<h1>直丝+斜丝联合诊断 · EvtID 1500–2000</h1><p>501事件闭区间；跨训练/验证/测试划分，仅作逐事件诊断，不能将混合汇总当独立测试成绩。图像按需加载。</p>')
    page.write_text(text,encoding='utf-8')
expected=set(range(1500,2001));actual={x['event_id'] for x in rows}
audit=dict(status='completed' if expected==actual and not missing else 'incomplete',completed=len(rows),missing_events=sorted(expected-actual),missing_files=missing,has_stereo=all(x['stereo_hits']>0 and x['edge_counts']['AS']>0 and x['edge_counts']['SS']>0 for x in rows),per_split_counts={k:dict(v) for k,v in by_split.items()},unix=time.time())
(r/'保存完整性核对.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit),flush=True)
