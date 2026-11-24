"""Observable-only coarse membership export; GenFit owns the precision fit."""
import argparse
import json
from pathlib import Path
import pickle
import time
import numpy as np
import torch
from common import ROOT,MODEL_ROOT,DATA_ROOT,sha,save_json,build_graph,JointInteractionNetwork
from STEP5_两级寻迹 import undirected_predictions,axial_two_pass,attach_stereo,DEFAULT_CONFIG
from hybrid_joint_tracker import track
from coarse_3d_membership import propose,select,finite_cylinder_intersections


def seed_state(circle,hypothesis):
    origin,span=hypothesis['arc'];phi=origin+span/2;radius=circle[2]
    position=[circle[0]+radius*np.cos(phi),circle[1]+radius*np.sin(phi),hypothesis['z0']+hypothesis['pitch']*(phi-origin)]
    tangent=np.array([-radius*np.sin(phi),radius*np.cos(phi),hypothesis['pitch']]);tangent/=np.linalg.norm(tangent)
    return dict(position_mm=list(map(float,position)),direction_candidates=[tangent.tolist(),(-tangent).tolist()],
                reference_phi_rad=float(phi),tan_lambda=float(hypothesis['pitch']/radius),
                charge=None,momentum_magnitude=None,covariance=None)


def export_result(eid,tracks,before,geo,drift,cells,hit_ids,plan,profile,seconds):
    stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9;groups=[]
    for candidate,(t,old) in enumerate(zip(tracks,before)):
        aa=[i for i in t['indices'] if not stereo[i]]
        assert aa==[i for i in old['indices'] if not stereo[i]]
        verified=t['confirmed_stereo_indices'];pending=t['pending_stereo_indices'];hypotheses=t['coarse_3d']['hypotheses']
        observations=[]
        for i in aa+verified+pending:
            g=geo[i];midpoint=np.array([g[0],g[1],0.]);half=g[2:5]*g[5]/2
            role='axial_protected' if not stereo[i] else 'stereo_geometry_confirmed' if i in verified else 'stereo_pending'
            measurement=dict(hit_id=int(hit_ids[i]),cell_id=int(cells[i]),role=role,drift_distance_mm=float(drift[i]),
                             wire_start_mm=(midpoint-half).tolist(),wire_end_mm=(midpoint+half).tolist())
            if stereo[i]:
                measurement['cylinder_intersection_candidates']=[point for offset in [0.,-drift[i],drift[i]]
                    for point in finite_cylinder_intersections(t['fit'],g,float(offset),0.)]
            observations.append(measurement)
        groups.append(dict(candidate_id=candidate,axial_hit_ids=hit_ids[aa].tolist(),
                           confirmed_stereo_hit_ids=hit_ids[verified].tolist(),pending_stereo_hit_ids=hit_ids[pending].tolist(),
                           rejected_stereo_hit_ids=hit_ids[t['rejected_stereo_indices']].tolist(),
                           genfit_hit_ids=hit_ids[aa+verified].tolist(),genfit_seed_ready=t['coarse_3d']['seed_ready'],
                           transverse_circle_mm=list(map(float,t['fit'])),transverse_rms_mm=float(t['rms']),
                           longitudinal_status=t['coarse_3d']['status'],longitudinal_hypotheses=hypotheses,
                           seed_states=[seed_state(t['fit'],h) for h in hypotheses],measurements=observations))
    original_a=[int(hit_ids[i]) for t in before for i in t['indices'] if not stereo[i]]
    final_a=[h for t in groups for h in t['axial_hit_ids']]
    assert original_a==final_a
    all_selected=[h for t in groups for h in t['axial_hit_ids']+t['confirmed_stereo_hit_ids']+t['pending_stereo_hit_ids']]
    assert len(all_selected)==len(set(all_selected))
    return dict(schema='AI6-coarse-3d-genfit-input-v1',event_id=eid,units=dict(position='mm',angle='rad',drift='mm'),
                inputs_read=['cell_id','d','hit_id'],truth_used=False,profile=profile,weight_sha256=plan['weight_sha256'],
                candidate_calibrator_sha256=plan['confidence_sha256'],axial_hits_before=len(original_a),axial_hits_after=len(final_a),
                lost_axial_hits=0,candidates=groups,seconds=seconds,
                scope='Rough seeds and protected hit membership. GenFit supplies field/charge/momentum/covariance and precision fit.',
                pending_policy='Pending S is retained separately and excluded from genfit_hit_ids; unresolved A candidates remain present.')


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='local_evaluation_epoch126_20261107')
    parser.add_argument('--run-tag',default='coarse_3d_v4_fast');parser.add_argument('--event',type=int,required=True)
    parser.add_argument('--input-npz',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    snapshot=MODEL_ROOT/args.snapshot;validation=json.loads((snapshot/(args.run_tag+'_validation')/'summary.json').read_text(encoding='utf-8'))
    if not validation['check_passed']:raise ValueError('Stereo profile has not passed held validation')
    profile=validation['selected_on_development'];plan=validation['contract']
    weight=snapshot/'best_interaction_network_joint_no_time.pth'
    confidence=MODEL_ROOT/'local_evaluation_epoch095_20261107/candidate_confidence_v2/candidate_confidence.pkl'
    if sha(weight)!=plan['weight_sha256'] or sha(confidence)!=plan['confidence_sha256']:raise ValueError('Frozen weights changed')
    if sha(DATA_ROOT/'dataset_metadata_joint_no_time.json')!=plan['dataset_metadata_sha256']:raise ValueError('Dataset changed')
    if any(sha(ROOT/n)!=h for n,h in plan['source_sha256'].items()):raise ValueError('Frozen tracking code changed')
    path=args.input_npz or DATA_ROOT/'events'/f'event_{args.event:05d}.npz'
    with np.load(path) as f:cells,drift,hit_ids=f['cell_id'].astype(np.int64),f['d'].astype(float),f['hit_id'].astype(np.int64)
    geometry=np.load(DATA_ROOT/'wire_geometry.npy')
    if len(cells)<1 or cells.min()<0 or cells.max()>=len(geometry):raise ValueError('Invalid CellID')
    if len(np.unique(hit_ids))!=len(hit_ids):raise ValueError('Independent HitID must be unique')
    geo=geometry[cells]
    started=time.perf_counter();g=build_graph(cells,drift,hit_ids,geometry,truth=None,event_id=args.event)
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=JointInteractionNetwork(chunk_size=65536).to(device);model.load_state_dict(torch.load(weight,weights_only=True,map_location=device));model.eval();g=g.to(device)
    with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):p=model(g.x,g.edge_index,g.edge_attr).sigmoid().float().cpu().numpy()
    pairs,p=undirected_predictions(g.edge_index.cpu().numpy(),p)
    axial,_=axial_two_pass(geo,drift,pairs,p);joint,_=attach_stereo(axial,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
    with confidence.open('rb') as f:calibrator=pickle.load(f)
    before=track(geo,drift,cells,pairs,p,axial,joint,calibrator,.02,.2)
    proposals=propose(before,geo,drift,cells,pairs,p,refine=plan['refine']);result,_=select(before,proposals,geo,cells,profile)
    output=export_result(args.event,result,before,geo,drift,cells,hit_ids,plan,profile,time.perf_counter()-started)
    args.output.parent.mkdir(parents=True,exist_ok=True);save_json(args.output,output)
    print(json.dumps(dict(candidates=len(result),seed_ready=sum(t['genfit_seed_ready'] for t in output['candidates']),
                          confirmed_stereo_hits=sum(len(t['confirmed_stereo_hit_ids']) for t in output['candidates']),
                          lost_axial_hits=0,seconds=output['seconds'])),flush=True)


if __name__=='__main__':main()
