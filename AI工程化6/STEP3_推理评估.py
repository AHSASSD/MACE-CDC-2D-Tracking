"""Streaming independent edge evaluation after training/selection are frozen."""
import argparse
import json
import torch
from torch_geometric.loader import DataLoader
from common import *
from STEP2_训练模型 import evaluate
from server_runtime import dataset as runtime_dataset,loader as runtime_loader,DevicePrefetch

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-tag',default='joint_no_time_scratch_v1');p.add_argument('--split',choices=['validation','test'],default='test');args=p.parse_args()
    torch.set_num_threads(2);out=MODEL_ROOT/args.run_tag;target=out/f'edge_evaluation_{args.split}.json'
    if target.exists():raise FileExistsError('Evaluation already exists')
    config=json.loads((out/'model_metadata_joint_no_time.json').read_text(encoding='utf-8'))
    if config['smoke']:raise ValueError('Smoke model cannot be a production model')
    dataset=runtime_dataset(DATA_ROOT,args.split)
    if sha(DATA_ROOT/'dataset_metadata_joint_no_time.json')!=config['dataset_metadata_sha256']:raise ValueError('Dataset changed')
    device=torch.device('cuda');model=JointInteractionNetwork(chunk_size=config['chunk_size']).to(device)
    model.load_state_dict(torch.load(out/'best_interaction_network_joint_no_time.pth',weights_only=True,map_location=device))
    import os
    batch_size=int(os.getenv('MACE_VAL_BATCH_SIZE','1'))
    result=evaluate(model,DevicePrefetch(runtime_loader(dataset,batch_size),device),device,True)
    report={'split':args.split,'events':len(dataset),'edge_threshold':.5,'result':result,'weight_sha256':sha(out/'best_interaction_network_joint_no_time.pth'),'scope':'edge association only, not track efficiency or 3D momentum'}
    if args.split=='test':
        exposed=set()
        pilot=ROOT/'pilot_v1/manifest.json';fresh=ROOT/'memory_safe_gpu/fresh_test_ids.json'
        if pilot.exists():exposed.update(json.loads(pilot.read_text(encoding='utf-8'))['ids']['test'])
        if fresh.exists():exposed.update(json.loads(fresh.read_text(encoding='utf-8'))['ids'])
        report['previously_inspected_pilot_test_ids']=sorted(exposed&set(dataset.ids))
        indices=[i for i,eid in enumerate(dataset.ids) if eid not in exposed]
        report['unexposed_subset']={'events':len(indices),'result':evaluate(model,DevicePrefetch(runtime_loader(torch.utils.data.Subset(dataset,indices),batch_size),device),device,True)}
    save_json(target,report)
    print(result,flush=True)

if __name__=='__main__':main()
