"""Close the two user-requested applications at the original training deadline."""
import json
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parent
info=json.loads((root/'pilot_v1/training_started.json').read_text(encoding='utf-8'))
deadline=info['app_close']['deadline_unix']
time.sleep(max(0,deadline-time.time()))
result=subprocess.run(['powershell','-NoProfile','-Command',
    "$ErrorActionPreference='Stop'; Get-Process -Name Endfield,MaaEnd -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep -Seconds 2; $remaining=@(Get-Process -Name Endfield,MaaEnd -ErrorAction SilentlyContinue); if ($remaining.Count -gt 0) { exit 1 }"],capture_output=True,text=True)
(root/'app_close_result.json').write_text(json.dumps({'deadline_unix':deadline,'executed_unix':time.time(),
    'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr},ensure_ascii=False,indent=2),encoding='utf-8')
