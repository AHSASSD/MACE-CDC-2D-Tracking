from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from common import JointInteractionNetwork,JointEventDataset,save_json
from et_data import encode


def main():
    torch.set_num_threads(1);data=ROOT.parent.parent/'joint_event_dataset'
    weight=ROOT.parent.parent.parent/'AI工程化6_稳定基线备份/AI6_稳定基线_20261107_230240/AI工程化6/joint_training/local_evaluation_epoch126_20261107/best_interaction_network_joint_no_time.pth'
    ds=JointEventDataset(data,'train');g=ds[ds.ids.index(24)];model=JointInteractionNetwork(chunk_size=65536).cuda().eval()
    model.load_state_dict(torch.load(weight,map_location='cuda',weights_only=True))
    feature=encode(model,g.clone(),torch.device('cuda'),.4);gpu=g.to('cuda')
    with torch.no_grad(),torch.autocast('cuda'):
        directed=model(gpu.x,gpu.edge_index,gpu.edge_attr).sigmoid().float().cpu().numpy()
    half=len(directed)//2;p=(directed[:half]+directed[half:])/2;ix=np.flatnonzero(p>=.4)
    assert np.array_equal(feature['pairs'],gpu.edge_index[:,:half][:,ix].cpu().numpy())
    difference=float(np.max(np.abs(p[ix]-feature['p'])))
    assert np.allclose(p[ix],feature['p'],atol=.001,rtol=.001)
    result=dict(status='passed',event_id=24,maximum_probability_difference=difference,retained_edges=len(ix),
                original_model_logits_preserved=True,actual_node_state_width=feature['h'].shape[1],edge_latent_width=feature['latent'].shape[1])
    save_json(ROOT/'backbone_interface_check.json',result);print(json.dumps(result))


if __name__=='__main__':main()
