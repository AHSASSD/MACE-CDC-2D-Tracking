import json,time
import torch
import membership as old
from repair_v2 import balanced,oc_v1,oc_v2
torch.set_num_threads(1);torch.manual_seed(123)
for n in [1,32,123]:
    z=torch.randn(n,4,requires_grad=True);truth=torch.randint(-1,5,(n,));stereo=torch.rand(n)>.5
    a=old.oc_loss(z,truth,stereo);b=oc_v1(z,truth,stereo)
    assert torch.allclose(a,b,atol=2e-5,rtol=2e-5),(a,b)
    ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z)[0]
    assert torch.allclose(ga,gb,atol=2e-5,rtol=2e-5)
    value=torch.rand(n,requires_grad=True)
    assert torch.allclose(old.balanced(value,stereo,truth),balanced(value,stereo,truth),atol=1e-6)
# Geometric loss must penalize mixed centers more than separated compact groups.
truth=torch.tensor([0,0,1,1]);stereo=torch.tensor([False,True,False,True])
good=torch.tensor([[0.,0,0,3],[.1,0,0,-3],[2.,0,0,3],[2.1,0,0,-3]],requires_grad=True)
bad=good.detach().clone();bad[2:,0]-=2;bad.requires_grad_()
assert oc_v2(good,truth,stereo)<oc_v2(bad,truth,stereo)
loss=oc_v2(bad,truth,stereo);loss.backward();assert torch.isfinite(bad.grad).all()
perm=torch.tensor([3,0,2,1]);assert torch.allclose(oc_v2(good,truth,stereo),oc_v2(good[perm],truth[perm],stereo[perm]))
print(json.dumps(dict(passed=True,vectorized_v1_value_and_gradient_equivalence=True,v2_geometry_and_permutation=True)))
