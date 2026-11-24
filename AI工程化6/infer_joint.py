"""Export joint associations using only wire identity and measured d from CSV."""
import argparse
import json
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
from joint_experiment import HERE,MAP,graph,legacy,MeanInteractionNetwork,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--event',type=int,required=True);p.add_argument('--run',default='memory_safe_gpu');p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2)
    # No truth, timing, particle kinematics, Good, or creation-process columns read.
    frame=pd.read_csv(args.csv,usecols=['EvtID','CellID','HitID','d'])
    frame=frame.loc[frame.EvtID.eq(args.event)].copy()
    if frame.empty: raise ValueError('Requested event not present')
    assert not frame.HitID.duplicated().any()
    frame=frame.merge(pd.read_excel(MAP),on='CellID',how='left',validate='many_to_one')
    frame['TrkID']=-1  # Dummy labels never passed to the network.
    g=graph(frame)
    run=HERE/args.run
    config=json.loads((run/'configuration.json').read_text(encoding='utf-8'))
    checkpoint=torch.load(run/'best.pt',weights_only=True,map_location='cpu')
    cls=MeanInteractionNetwork if config['architecture']['aggregation']=='mean' else legacy.InteractionNetwork
    model=cls(12,11,hidden=64,n_passes=3);model.load_state_dict(checkpoint['model']);model.eval()
    with torch.no_grad(): probabilities=model(g.x,g.edge_index,g.edge_attr).sigmoid().numpy()
    calibration=run/'decoding_validation.json'
    threshold=json.loads(calibration.read_text(encoding='utf-8'))['joint']['selected'] if calibration.exists() else checkpoint['threshold']
    a,b=g.edge_index.numpy(); keep=probabilities>=threshold
    _,labels=connected_components(coo_matrix((np.ones(keep.sum()),(a[keep],b[keep])),shape=(g.num_nodes,g.num_nodes)),directed=False)
    candidates=[np.where(labels==k)[0] for k in np.unique(labels)]
    candidates=[c for c in candidates if np.sum(~g.stereo.numpy()[c])>=5]
    save(__import__('pathlib').Path(args.output),{'event':args.event,'weight_sha256':sha(run/'best.pt'),
         'threshold':threshold,'association_threshold':checkpoint['threshold'],'input_hits':g.num_nodes,'stereo_hits':int(g.stereo.sum()),
         'candidates':[{'hit_ids':g.hit_id[c].tolist(),'cell_ids':g.cell_id[c].tolist(),'axial_hits':int((~g.stereo[c]).sum()),'stereo_hits':int(g.stereo[c].sum())} for c in candidates],
         'note':'Exploratory association components, without geometric track fit or 3D momentum. Inference reads all CSV hits for event, whereas training/evaluation use the inherited Good/CreatProc sample filter; input populations may differ.'})
    print('Saved',args.output,flush=True)

if __name__=='__main__': main()
