import feasibility as f
import json,numpy as np,torch
from et_data import Features,to_device
from et_models import THead,predict

def main():
    torch.set_num_threads(1);eid=1500;b=Features(f.BASE/'junction_audit_20261122/cache',[eid])[0]
    original={k:v.numpy() if torch.is_tensor(v) else v for k,v in b.items()}
    aa=~original['stereo'][original['pairs'][0]]&~original['stereo'][original['pairs'][1]]
    x=dict(original,pairs=original['pairs'][:,aa],p=original['p'][aa],latent=original['latent'][aa]);known={tuple(e) for e in x['pairs'].T}
    records=json.loads((f.OUT/'event_01500.json').read_text(encoding='utf-8'))['methods']['geometry_interior']['proposal_records'];records=[r for r in records if (r['u'],r['v']) in known]
    proposals=[(r['u'],r['v']) for r in records];xy=x['raw'][:,:2]*800
    t,owners=f.aux_triplets(proposals,x['pairs'],x['p'],xy);x['t']=t
    gpu={k:torch.from_numpy(v).cuda() if isinstance(v,np.ndarray) else v for k,v in x.items()}
    model=THead().cuda();model.load_state_dict(torch.load(f.BASE/'evaluation_download/full/T_best_members.pth',map_location='cuda',weights_only=True));model.eval()
    with torch.no_grad(),torch.autocast('cuda'):q=predict(model,gpu,'T')
    maxima=np.zeros(len(proposals));np.maximum.at(maxima,owners,q)
    reference=np.array([r['max_T'] for r in records]);diff=float(np.max(abs(maxima-reference))) if len(records) else 0.
    assert len(records)>0 and diff<.005,(len(records),diff)
    result=dict(event_id=eid,edges_checked=len(records),max_score_difference_vs_existing_cached_feature_pipeline=diff,passed=True)
    (f.OUT/'score_parity_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)

if __name__=='__main__':main()
