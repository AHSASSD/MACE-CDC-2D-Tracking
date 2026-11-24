"""Prepare immutable runtime copies and check the paired optimization contract."""
from pathlib import Path
import json
import shutil
import sys
import zipfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import experiment_common as e
e.verify_pins()
for name in ['common.py','server_runtime.py','graph_cache.py','two_stage_core.py','STEP5_两级寻迹.py']:
    shutil.copyfile(e.BASELINE/name,ROOT/name)
(ROOT/'member_ids.json').write_text(json.dumps(e.IDS),encoding='utf-8')
import torch
from types import SimpleNamespace
from paired_loss import objective
z=torch.tensor([-.8,1.2,-2.,.2,2.,-.3,1.],requires_grad=True)
d=SimpleNamespace(y=torch.tensor([1.,1.,0.,0.,1.,0.,1.]),batch=torch.tensor([0,0,0,1,1]),
                  edge_index=torch.tensor([[0,0,1,1,3,3,4],[1,2,0,2,4,4,3]]),
                  truth=torch.tensor([5,5,8,6,6]),num_graphs=2)
bce=torch.nn.functional.binary_cross_entropy_with_logits(z,d.y,reduction='none')
f=.5*(1-torch.exp(-bce)).square()*bce
expected=(f[:4].mean()+f[4:].mean())/2
control,_=objective(z,d,'control')
torch.testing.assert_close(control,expected,rtol=0,atol=0)
torch.testing.assert_close(torch.autograd.grad(control,z,retain_graph=True)[0],torch.autograd.grad(expected,z,retain_graph=True)[0])
balanced,terms=objective(z,d,'balanced');balanced.backward()
assert torch.isfinite(z.grad).all() and balanced>=control and terms['positive_aux']>0 and terms['hard_negative_aux']>0
for name in ['paired_train.py','paired_guard.py']:
    compile((ROOT/name).read_text(encoding='utf-8'),str(ROOT/name),'exec')
report=dict(control_matches_baseline_loss_and_gradient=True,balanced_backward_finite=True,train_and_guard_syntax='passed',initial_weight_sha256=e.CONTRACT['weight_sha256'],members=len(e.IDS))
(ROOT/'preflight_local.json').write_text(json.dumps(report,indent=2))
with zipfile.ZipFile(ROOT/'paired_runtime.zip','w',compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(ROOT.glob('*.py')):zf.write(p,arcname=p.name)
    zf.write(ROOT/'member_ids.json',arcname='member_ids.json')
print(json.dumps(report))
