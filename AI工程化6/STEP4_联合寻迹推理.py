"""Frozen AI6 inference: observations in, exclusive independent HitID groups out."""
import argparse
import json
from pathlib import Path
import pickle
import time
import numpy as np
import torch
from common import ROOT,DATA_ROOT,MODEL_ROOT,build_graph,JointInteractionNetwork,sha,save_json
from STEP5_两级寻迹 import undirected_predictions,axial_two_pass,attach_stereo,DEFAULT_CONFIG
from hybrid_joint_tracker import track


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--event',type=int,required=True)
    parser.add_argument('--input-npz',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--snapshot',default='local_evaluation_epoch095_20261107');args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    snapshot=MODEL_ROOT/args.snapshot
    comparison=snapshot/'model_comparison_v1/paired_comparison.json'
    if comparison.exists():
        # A new GNN may reuse the frozen epoch-95 calibrator after paired testing.
        contract=json.loads(comparison.read_text(encoding='utf-8'))['latest']['plan']
        if contract['snapshot']!=args.snapshot:raise ValueError('Comparison belongs to another snapshot')
        confidence=MODEL_ROOT/contract['confidence_snapshot']/'candidate_confidence_v2/candidate_confidence.pkl'
    else:
        contract=json.loads((snapshot/'hybrid_frozen_test_v1/contract.json').read_text(encoding='utf-8'))
        confidence=snapshot/'candidate_confidence_v2/candidate_confidence.pkl'
    if 'dataset_metadata_sha256' in contract and sha(DATA_ROOT/'dataset_metadata_joint_no_time.json')!=contract['dataset_metadata_sha256']:
        raise ValueError('Frozen dataset changed')
    if 'geometry_sha256' in contract and sha(DATA_ROOT/'wire_geometry.npy')!=contract['geometry_sha256']:
        raise ValueError('Frozen geometry changed')
    weight=snapshot/'best_interaction_network_joint_no_time.pth'
    if sha(weight)!=contract['weight_sha256'] or sha(confidence)!=contract['confidence_model_sha256']:raise ValueError('Frozen model changed')
    if any(sha(ROOT/name)!=digest for name,digest in contract['source_hashes'].items()):raise ValueError('Frozen tracking source changed')
    path=args.input_npz or DATA_ROOT/'events'/f'event_{args.event:05d}.npz'
    with np.load(path) as f:
        # Only observable wire identity, independent hit identity, and measured d.
        cells=f['cell_id'].astype(np.int64);drift=f['d'].astype(float);hit_ids=f['hit_id'].astype(np.int64)
    geometry=np.load(DATA_ROOT/'wire_geometry.npy')
    if len(cells)<1 or cells.min()<0 or cells.max()>=len(geometry):raise ValueError('Invalid cell identity')
    if len(np.unique(hit_ids))!=len(hit_ids):raise ValueError('Duplicate independent HitID')
    started=time.time();g=build_graph(cells,drift,hit_ids,geometry,truth=None,event_id=args.event)
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=JointInteractionNetwork(chunk_size=65536).to(device)
    model.load_state_dict(torch.load(weight,weights_only=True,map_location=device));model.eval();g=g.to(device)
    with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
        probability=model(g.x,g.edge_index,g.edge_attr).sigmoid().float().cpu().numpy()
    pairs,p=undirected_predictions(g.edge_index.cpu().numpy(),probability);geo=geometry[cells]
    base,_=axial_two_pass(geo,drift,pairs,p);base_joint,_=attach_stereo(base,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
    with confidence.open('rb') as f:calibrator=pickle.load(f)
    tracks=track(geo,drift,cells,pairs,p,base,base_joint,calibrator,contract['base_cut'],contract['new_cut'])
    stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9;output=[];used=set()
    for t in tracks:
        ids=np.array(t['indices'],dtype=int);assert not used.intersection(ids);used.update(ids.tolist())
        output.append(dict(hit_ids=hit_ids[ids].tolist(),cell_ids=cells[ids].tolist(),
                           axial_hit_ids=hit_ids[ids[~stereo[ids]]].tolist(),stereo_hit_ids=hit_ids[ids[stereo[ids]]].tolist(),
                           axial_circle=t['fit'],axial_rms_mm=t['rms'],stage=t['stage']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_json(args.output,dict(event_id=args.event,input_hits=len(cells),candidates=output,
              unassigned_hit_ids=hit_ids[[i for i in range(len(cells)) if i not in used]].tolist(),
              snapshot=args.snapshot,selected_epoch=json.loads((snapshot/'best_model_selection.json').read_text(encoding='utf-8'))['epoch'],
              weight_sha256=sha(weight),confidence_model_sha256=sha(confidence),seconds=time.time()-started,
              inputs_read=['cell_id','hit_id','d'],truth_used=False,scope='hit memberships; axial circle geometry; no momentum or fitted 3D positions'))
    print(json.dumps(dict(candidates=len(output),assigned_hits=len(used),input_hits=len(cells),seconds=time.time()-started)),flush=True)


if __name__=='__main__':main()
