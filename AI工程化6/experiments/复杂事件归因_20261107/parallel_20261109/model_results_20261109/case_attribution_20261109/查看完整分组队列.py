from pathlib import Path
from collections import Counter
import json

root=Path(__file__).resolve().parent
result={}
for key in ['control_012','balanced_012','balanced_015']:
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in (root/'events'/key).glob('event_*.json')]
    if len(rows)!=200:continue
    counts=Counter();connected=Counter();positive=Counter();false=Counter()
    for row in rows:
        for t in row['tracks']:
            if t['recovered']:counts[t['old_category']]+=1
            if t['new_connected']:connected[t['old_category']]+=1
            if t['new_true_mean'] is not None and t['old_true_mean'] is not None and t['new_true_mean']>t['old_true_mean']:positive[t['old_category']]+=1
    result[key]=dict(recovered_by_group=dict(counts),pure_connected_by_group=dict(connected),increased_mean_score=dict(positive),
                     recovered=sum(len(r['recovered_old']) for r in rows),lost=sum(len(r['lost_old']) for r in rows))
(root/'completed_cohorts_snapshot.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
