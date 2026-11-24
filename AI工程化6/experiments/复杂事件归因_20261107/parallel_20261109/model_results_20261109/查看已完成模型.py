"""Inspect only complete model cohorts while the remaining cohorts finish."""
from collections import Counter
from pathlib import Path
import json

root=Path(__file__).resolve().parent
plan=json.loads((root/'evaluation_contract.json').read_text())
ids=plan['event_ids']
reference=json.loads((root.parent/'extractor_results/validation/summary.json').read_text())
base={i:json.loads((root.parent/'extractor_results/validation/events'/('event_%05d.json'%i)).read_text())['matched']['baseline'] for i in ids}
result={'initial126':dict(tracks=reference['methods']['baseline'],seed=reference['seeds']['baseline'])}
for model in plan['models']:
    rows={r['event_id']:r for r in (json.loads(p.read_text()) for p in (root/'events'/model).glob('event_*.json'))}
    if set(rows)!=set(ids):continue
    tracks=Counter();seed=Counter()
    for r in rows.values():tracks.update(r['counts']);seed.update(r['seed'])
    t=dict(tracks);s=dict(seed)
    t.update(track_efficiency=t['matched_tracks']/t['truth_tracks'],fake_fraction=t.get('fake_candidates',0)/t['output_candidates'],clone_fraction=t.get('clone_candidates',0)/t['output_candidates'])
    s.update(efficiency=s['matched_tracks']/s['truth_tracks'],fake_fraction=s.get('fake_candidates',0)/s['output_candidates'],clone_fraction=s.get('clone_candidates',0)/s['output_candidates'])
    result[model]=dict(tracks=t,seed=s,gained=sum(len(set(r['matched'])-set(base[i])) for i,r in rows.items()),lost=sum(len(set(base[i])-set(r['matched'])) for i,r in rows.items()))
(root/'completed_cohorts_snapshot.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:dict(matched=v['tracks']['matched_tracks'],efficiency=v['tracks']['track_efficiency'],fake=v['tracks'].get('fake_candidates',0),clone=v['tracks'].get('clone_candidates',0),seed_matched=v['seed']['matched_tracks'],seed_fake=v['seed'].get('fake_candidates',0),seed_clone=v['seed'].get('clone_candidates',0),gained=v.get('gained'),lost=v.get('lost')) for k,v in result.items()}))
