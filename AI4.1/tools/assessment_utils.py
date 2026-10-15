from collections import Counter
import time
import torch
from STEP4_两级寻迹 import run_two_pass, evaluate, truth_sets

def init_worker():
    torch.set_num_threads(1)

def track(task):
    evt,graph,probability=task
    start=time.perf_counter()
    tracks,_=run_two_pass(graph,None,None,'time',.70,.575,probability)
    metrics=evaluate(graph,tracks)
    sizes=Counter(k for values in truth_sets(graph) for k in values)
    missed=set(map(tuple,metrics['missed_tracks']))
    bins={name:{'truth':0,'matched':0} for name in ['5_6','7_10','11_20','21_plus']}
    for key,size in sizes.items():
        if size<5:continue
        name='5_6' if size<7 else '7_10' if size<11 else '11_20' if size<21 else '21_plus'
        bins[name]['truth']+=1;bins[name]['matched']+=key not in missed
    return {'EvtID':evt,**metrics,'output_tracks':len(tracks),'by_hit_count':bins,
            'mixed_candidates':sum(t['purity']<.8 for t in tracks),
            'incomplete_candidates':sum(t['purity']>=.8 and t['completeness']<.5 for t in tracks),
            'elapsed_seconds':time.perf_counter()-start}

def summary(rows):
    keys=['truth_tracks','matched_tracks','fake_tracks','clone_tracks','output_tracks']
    total={key:sum(r[key] for r in rows) for key in keys}
    assert total['output_tracks']==total['matched_tracks']+total['fake_tracks']+total['clone_tracks']
    total.update(events=len(rows),efficiency=total['matched_tracks']/total['truth_tracks'],
                 fake_rate=total['fake_tracks']/max(total['output_tracks'],1),
                 clone_rate=total['clone_tracks']/max(total['output_tracks'],1))
    return total
