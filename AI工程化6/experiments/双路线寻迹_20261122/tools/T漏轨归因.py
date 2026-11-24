from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch,numpy as np
from common import save_json
from et_decode import T_groups,geometry_extract
from STEP5_两级寻迹 import axial_two_pass

def main():
    torch.set_num_threads(1);p=ROOT/'evaluation_download';data=ROOT.parent.parent/'joint_event_dataset'
    lost=json.loads((ROOT/'evaluation_20261122/paired_summary.json').read_text(encoding='utf-8'))['paired']['T']['lost_tracks'];result=[]
    for eid in sorted({x[0] for x in lost}):
        with np.load(p/'cache/features'/('event_%05d.npz'%eid)) as s:f={k:s[k] for k in s.files}
        with np.load(p/'cache/indices'/('event_%05d.npz'%eid)) as s:t=s['t']
        with np.load(data/'events'/('event_%05d.npz'%eid)) as s:d=s['d'];truth=s['truth']
        geo=np.load(data/'wire_geometry.npy')[f['cells']];q=np.load(p/'full/validation/epoch_025'/('prediction_%05d.npz'%eid))['T']
        base,_=axial_two_pass(geo,d,f['pairs'],f['p']);groups,w=T_groups(f,t,q);tracks=geometry_extract(groups,w,f,geo,d,base)
        def info(c,tid):
            ix=np.asarray(c['indices']);values,counts=np.unique(truth[ix],return_counts=True)
            return dict(indices=ix.tolist(),size=len(ix),target_hits=int((truth[ix]==tid).sum()),target_purity=float((truth[ix]==tid).mean()),target_completeness=float((truth[ix]==tid).sum()/((truth==tid)&~f['stereo']).sum()),dominant_truth=int(values[counts.argmax()]),origin=c.get('origin','baseline'),fit_rms=float(c.get('rms',c.get('fit_rms',-1))),selection_utility=c.get('selection_utility'))
        for _,tid in [x for x in lost if x[0]==eid]:
            original=[c for c in base if info(c,tid)['target_purity']>=.8 and info(c,tid)['target_completeness']>=.5]
            overlaps=[]
            for c in tracks:
                common=set(c['indices']).intersection({i for b in original for i in b['indices']})
                if common:overlaps.append(dict(info(c,tid),overlap_original=len(common)))
            exact_survives=any(set(b['indices'])==set(c['indices']) for b in original for c in tracks)
            result.append(dict(event_id=eid,truth_id=tid,baseline_candidates=[info(c,tid) for c in original],selected_overlaps=overlaps,original_candidate_survives=exact_survives,cause='Original baseline candidate replaced in greedy overlap competition' if not exact_survives else 'Original candidate survived; inspect evaluation/attachment'))
        print('attributed',eid,flush=True)
    save_json(ROOT/'decoder_diagnosis_20261122/T_lost_tracks.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
