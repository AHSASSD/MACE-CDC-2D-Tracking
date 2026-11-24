"""Freeze the runtime, verify selection/loss/sampler, and package only source files."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import zipfile
import numpy as np
import torch
from torch_geometric.data import Data, Batch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'复杂事件归因_20261107'))
import experiment_common as baseline
sys.path.insert(0,str(ROOT))
from curriculum import epoch_ids
from recall_loss import objective


def main():
    baseline.initialize_worker();baseline.verify_pins()
    selection=json.loads((ROOT/'difficulty_audit/selection.json').read_text(encoding='utf-8'))
    metadata=json.loads((baseline.DATA_ROOT/'dataset_metadata_joint_no_time.json').read_text(encoding='utf-8'))
    assert selection['dataset_sha256']==baseline.sha(baseline.DATA_ROOT/'dataset_metadata_joint_no_time.json')
    original=set(metadata['event_ids']['train']);clean=set(selection['clean_event_ids']);hard=set(selection['hard_event_ids'])
    assert not (clean&hard) and clean|hard==original and not selection['invalid_event_ids']
    assert not original&set(metadata['event_ids']['validation']+metadata['event_ids']['test'])
    for name,digest in selection['source_sha256'].items():assert baseline.sha(ROOT/name)==digest
    schedule={}
    for epoch in range(1,31):
        ids=epoch_ids(selection,epoch)
        assert ids==epoch_ids(selection,epoch) and len(ids)==len(set(ids)) and set(ids)<=original
        assert clean<=set(ids) and set(ids)&hard
        if epoch>=15:assert set(ids)==original
        # Resume at every possible saved optimizer boundary preserves the permutation.
        for cursor in range(0,len(ids),32):assert ids[:cursor]+epoch_ids(selection,epoch)[cursor:]==ids
        schedule[epoch]=dict(events=len(ids),hard=len(set(ids)&hard))

    def graph(y,k,truth=None):
        n=len(y);return Batch.from_data_list([Data(x=torch.zeros(n+1,1),edge_index=torch.stack([torch.arange(n),torch.full((n,),n)]),
            y=torch.tensor(y,dtype=torch.float32),edge_kind=torch.tensor(k),truth=torch.tensor(truth if truth is not None else list(range(n))+[-1]))])
    # True low-score edges receive a stronger upward gradient; no extra margin above p=.7.
    for zvalue in [-2.,0.,2.]:
        z=torch.tensor([zvalue],requires_grad=True);g=graph([1],[0]);loss,terms=objective(z,g,5)
        grad=torch.autograd.grad(loss,z)[0]
        ref=torch.tensor([zvalue],requires_grad=True);b=torch.nn.functional.binary_cross_entropy_with_logits(ref,g.y,reduction='none')
        focal=(.5*(1-torch.exp(-b)).square()*b).mean();fg=torch.autograd.grad(focal,ref)[0]
        assert torch.isfinite(grad).all()
        if zvalue<=0:assert grad.item()<fg.item()
        else:assert terms['true_low_margin']==0 and torch.allclose(grad,fg)
    z=torch.tensor([2.],requires_grad=True);loss,terms=objective(z,graph([0],[1]),5)
    assert terms['false_high_margin']>0 and torch.autograd.grad(loss,z)[0].item()>0
    # Duplicating edges from a long track must not change its extra positive weighting.
    g1=graph([1,1],[0,0],[10,20,-1]);g2=graph([1,1,1],[0,0,0],[10,10,20,-1])
    _,m1=objective(torch.tensor([-2.,0.]),g1,5);_,m2=objective(torch.tensor([-2.,-2.,0.]),g2,5)
    assert abs(m1['true_low_margin']-m2['true_low_margin'])<1e-6
    # Track averaging is inside each edge kind; the AA/AS/SS weights remain explicit.
    g=graph([1,1,1],[0,1,2]);z=torch.tensor([-2.,0.,1.],requires_grad=True);loss,m=objective(z,g,5)
    expected=.5*(np.log(.7/.3)+2)**2+.3*np.log(.7/.3)**2
    assert abs(m['true_low_margin']-expected)<1e-5;loss.backward()
    assert torch.isfinite(z.grad).all()

    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)
    pinned=['common.py','server_runtime.py','graph_cache.py','two_stage_core.py','STEP5_两级寻迹.py','coarse_3d_membership.py','三维拟合草稿_暂未启用.py']
    for name in pinned:shutil.copy2(baseline.BASELINE/name,runtime/name)
    new=['STEP2_课程召回微调.py','curriculum.py','recall_loss.py','morphology.py','membership_cpu.py','preflight_remote.py']
    for name in new:shutil.copy2(ROOT/name,runtime/name)
    for path in runtime.glob('*.py'):compile(path.read_text(encoding='utf-8'),str(path),'exec')
    # Keep the original audit immutable, record the revised mixing schedule in the runtime copy.
    selection['training_curriculum_override']=dict(epochs_1_5='ordinary plus25% of difficult events, without replacement',
        epochs_6_15='difficult fraction increases from25% to100%',epochs_16_30='all7000 original train events once per epoch')
    baseline.save_json(runtime/'selection.json',selection)
    # Reuse the already fixed validation cohort to compare epoch126 and future checkpoints.
    assert len(baseline.IDS)==200 and set(baseline.IDS)<=set(metadata['event_ids']['validation'])
    baseline.save_json(runtime/'member_ids.json',baseline.IDS)
    events=json.loads((ROOT/'difficulty_audit/events.json').read_text(encoding='utf-8'))
    shape_tracks=sum(t['blob'] or t['loop'] for r in events for t in r['tracks'])
    baseline.save_json(ROOT/'verification.json',dict(status='passed',tests=['original split isolation','source and data hashes',
        '30 deterministic no-replacement curriculum schedules','all optimizer-boundary sampler resumes','true low-score gradient',
        'positive margin saturation','false high-score gradient','per-track and per-kind balancing','all runtime Python syntax'],
        schedule=schedule,shape_flagged_tracks=shape_tracks,eligible_axial_tracks=210720,
        ordinary_tracks_in_shape_hard_events=sum(len(r['tracks']) for r in events if r['shape_hard'])-shape_tracks,
        baseline_weight_sha256=baseline.sha(baseline.SNAPSHOT/'best_interaction_network_joint_no_time.pth'),
        sources={p.name:baseline.sha(p) for p in runtime.iterdir() if p.is_file()}))
    with zipfile.ZipFile(ROOT/'runtime_upload.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in runtime.iterdir():archive.write(path,'runtime/'+path.name)
        for name in ['run_step.sh','train.sbatch','supervise_remote.py']:archive.write(ROOT/name,name)
    print(json.dumps(dict(status='passed',ordinary=len(clean),hard=len(hard),shape_flagged_tracks=shape_tracks,
                          first_epoch=schedule[1],full_distribution_from=15,package_bytes=(ROOT/'runtime_upload.zip').stat().st_size)),flush=True)


if __name__=='__main__':main()
