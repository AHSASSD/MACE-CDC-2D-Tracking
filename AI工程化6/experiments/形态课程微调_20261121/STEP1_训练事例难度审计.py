"""Audit only original7000 train events. Nothing is removed from source data."""
from pathlib import Path
import json,hashlib,csv,time,argparse
from collections import Counter
import numpy as np
from morphology import track_shape,RULES

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent.parent/'joint_event_dataset'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def event(eid,geometry):
    with np.load(DATA/'events'/f'event_{eid:05d}.npz') as f:cells,d,truth,hit=f['cell_id'],f['d'],f['truth'],f['hit_id']
    g=geometry[cells];a=np.linalg.norm(g[:,2:4],axis=1)<=1e-9
    invalid=not np.isfinite(d).all() or not np.isfinite(g).all() or len(set(map(int,hit)))!=len(hit) or (d<0).any()
    tracks=[]
    for tid in np.unique(truth[a&(truth>=0)]):
        mask=a&(truth==tid)
        if mask.sum()<5:continue
        tracks.append(dict(truth_id=int(tid),axial_hits=int(mask.sum()),**track_shape(cells[mask],g[mask,:2])))
    return dict(event_id=eid,hits=len(cells),axial_hits=int(a.sum()),stereo_hits=int((~a).sum()),invalid=bool(invalid),tracks=tracks)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe',nargs=2,type=int);args=parser.parse_args()
    metadata=json.loads((DATA/'dataset_metadata_joint_no_time.json').read_text(encoding='utf-8'));geometry=np.load(DATA/'wire_geometry.npy')
    if args.probe:
        eid,tid=args.probe;assert eid in metadata['event_ids']['train']
        print(json.dumps(next(t for t in event(eid,geometry)['tracks'] if t['truth_id']==tid)));return
    out=ROOT/'difficulty_audit';out.mkdir(exist_ok=True);ids=metadata['event_ids']['train'];rows=[];started=time.time()
    for j,eid in enumerate(ids):
        rows.append(event(eid,geometry))
        if (j+1)%500==0:print(json.dumps(dict(events=j+1,total=len(ids),seconds=time.time()-started)),flush=True)
    cutoff=float(np.quantile([r['hits'] for r in rows],RULES['high_occupancy_quantile']));clean=[];hard=[];invalid=[]
    for r in rows:
        r['shape_hard']=any(t['blob'] or t['loop'] for t in r['tracks']);r['high_occupancy']=r['hits']>cutoff
        r['difficulty']='invalid' if r['invalid'] else 'hard' if r['shape_hard'] or r['high_occupancy'] else 'clean'
        (invalid if r['invalid'] else hard if r['difficulty']=='hard' else clean).append(r['event_id'])
    packet=dict(dataset_sha256=sha(DATA/'dataset_metadata_joint_no_time.json'),train_event_ids=ids,clean_event_ids=clean,hard_event_ids=hard,invalid_event_ids=invalid,
                rules=RULES,hit_cutoff=cutoff,source_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'morphology.py']},
                curriculum=dict(epochs_1_5='clean only',epochs_6_15='anneal hard sampling probability to its original frequency',epochs_16_30='original train distribution, each valid event once'),
                validation_test_policy='Full original validation/test untouched; no validation IDs, predictions or missed-track lists used to select train data')
    (out/'selection.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'events.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    summary=dict(train_events=len(ids),clean_events=len(clean),hard_events=len(hard),invalid_events=len(invalid),hit_percentiles={str(q):float(np.quantile([r['hits'] for r in rows],q)) for q in [.5,.9,.95,.99]},
                 blob_tracks=sum(t['blob'] for r in rows for t in r['tracks']),loop_tracks=sum(t['loop'] for r in rows for t in r['tracks']),
                 total_eligible_axial_tracks=sum(len(r['tracks']) for r in rows),tracks_in_events_with_shape_hard=sum(len(r['tracks']) for r in rows if r['shape_hard']),
                 shape_hard_events=sum(r['shape_hard'] for r in rows),high_occupancy_events=sum(r['high_occupancy'] for r in rows),rules=RULES)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'tracks.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['event_id','truth_id','axial_hits','unique_cells','extent_mm','minor_major','coverage_deg','radius_mm','radial_cv','blob','loop'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerows(dict(event_id=r['event_id'],**t) for t in r['tracks'])
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
