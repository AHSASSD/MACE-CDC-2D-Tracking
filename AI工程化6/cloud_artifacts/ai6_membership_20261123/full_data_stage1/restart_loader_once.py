"""One-time termination of our own training child; allocation/supervisor stay live."""
import json,os,signal
from pathlib import Path
root=Path('/seu_nvme/home/zhangyongchao/213230965/ai6_membership_20261123/full_data_stage1')
record=root/'loader_restart_request.json'
if record.exists():raise SystemExit('Loader restart already requested; refusing duplicate')
pid=1834915
proc=Path('/proc')/str(pid)
assert proc.stat().st_uid==os.getuid()
assert b'/full_data_stage1/train_full_stage.py' in (proc/'cmdline').read_bytes()
assert os.getpgid(pid)==pid
assert (root/'run/state.pth').exists()
record.write_text(json.dumps(dict(reason='loader-only transition to2workers',pid=pid,allocation=1862832,checkpoint_preserved=True)))
os.killpg(pid,signal.SIGTERM)
print('Owned training process group stopped; existing supervisor will resume checkpoint.')
