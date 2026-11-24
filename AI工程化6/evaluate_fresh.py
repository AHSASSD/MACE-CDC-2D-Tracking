"""One-shot fresh holdout after validation-only checkpoint selection."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from joint_experiment import HERE,SOURCE,MAP,BASELINE,legacy,graph,save,sha,metrics,predict_all,MeanInteractionNetwork

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='extended_sum');p.add_argument('--events',type=int,default=32);args=p.parse_args()
    torch.set_num_threads(2)
    out=HERE/args.run
    if (out/'fresh_test_report.json').exists(): raise FileExistsError('Fresh test already scored; do not tune against it')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    config=json.loads((out/'configuration.json').read_text(encoding='utf-8'))
    cls=MeanInteractionNetwork if config['architecture']['aggregation']=='mean' else legacy.InteractionNetwork
    checkpoint=torch.load(out/'best.pt',weights_only=True,map_location=device)
    model=cls(12,11,hidden=64,n_passes=3).to(device);model.load_state_dict(checkpoint['model'])
    baseline=legacy.InteractionNetwork(6,4).to(device);baseline.load_state_dict(torch.load(BASELINE,weights_only=True,map_location=device))
    validation=torch.load(HERE/'pilot_v1/validation.pt',weights_only=False)
    calibration={}; candidate_thresholds={}
    for name,network,is_baseline in [('joint',model,False),('axial_baseline',baseline,True)]:
        pred=predict_all(network,validation,device,is_baseline)
        trials=[metrics(validation,pred,t,is_baseline) for t in [.3,.5,.7,.85,.95,.98]]
        # Candidate thresholds are separate from F1-optimal association thresholds.
        def utility(row):
            c=row['candidates']; return (c['matched_tracks']-.25*c['fake_candidates']-.25*c['clone_candidates'])/max(c['truth_tracks'],1)
        chosen=max(trials,key=utility)
        candidate_thresholds[name]=chosen['threshold'];calibration[name]={'trials':trials,'selected':chosen['threshold'],'objective':'(matched - .25*fake - .25*clones)/truth'}
    save(out/'decoding_validation.json',calibration)
    del validation,pred
    old=json.loads((HERE/'pilot_v1/manifest.json').read_text(encoding='utf-8'))
    split=json.loads((SOURCE/'whole_event_retraining/event_split.json').read_text(encoding='utf-8'))
    eligible=sorted(set(split['test_evt_ids'])-set(old['ids']['test']))
    ids=sorted(np.random.default_rng(620261105).choice(eligible,args.events,replace=False).tolist())
    assert not set(ids)&set(old['ids']['train']+old['ids']['validation']+old['ids']['test'])
    save(out/'fresh_test_ids.json',{'ids':ids,'excluded_pilot_test':old['ids']['test'],'checkpoint_sha256':sha(out/'best.pt')})
    geometry=pd.read_excel(MAP); graphs=[]
    for path in sorted((SOURCE/'whole data').glob('CDCSimHit_*.csv')):
        frame=pd.read_csv(path,usecols=['EvtID','TrkID','HitID','CellID','Good','CreatProc','WireType','d'])
        frame=frame.loc[frame.EvtID.isin(ids)&frame.Good.eq(True)&frame.CreatProc.eq('ExtendedDecayWithSpin')]
        frame=frame.merge(geometry,on='CellID',how='left',validate='many_to_one')
        graphs.extend(graph(f) for _,f in frame.groupby('EvtID',sort=True))
    assert len(graphs)==len(ids)
    start=time.perf_counter();pred=predict_all(model,graphs,device)
    elapsed=time.perf_counter()-start
    report={'joint':metrics(graphs,pred,checkpoint['threshold']), 'joint_forward_seconds':elapsed}
    report['joint']['candidates']=metrics(graphs,pred,candidate_thresholds['joint'])['candidates']
    report['joint']['candidate_threshold']=candidate_thresholds['joint']
    del model
    threshold=json.loads((HERE/'pilot_v1/test_report.json').read_text(encoding='utf-8'))['axial_baseline']['threshold']
    start=time.perf_counter();pred=predict_all(baseline,graphs,device,True)
    elapsed=time.perf_counter()-start
    report['axial_baseline']=metrics(graphs,pred,threshold,True)
    report['axial_baseline']['candidates']=metrics(graphs,pred,candidate_thresholds['axial_baseline'],True)['candidates']
    report['axial_baseline']['candidate_threshold']=candidate_thresholds['axial_baseline']
    report['baseline_forward_seconds']=elapsed
    report['events']=len(ids);report['weight_sha256']=sha(out/'best.pt');report['baseline_sha256']=sha(BASELINE)
    report['limitations']=['Training uses only 128 complete events.','Candidate results use the same connected-component decoder on both models, not the official two-pass geometry fitter.','Candidate matching uses >=5 axial hits, >=70% axial purity and completeness.','No 3D reconstruction or momentum evaluated.','Forward time includes CPU-to-GPU transfer and copying probabilities; each model has different architecture size.']
    save(out/'fresh_test_report.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
