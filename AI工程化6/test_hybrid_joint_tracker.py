"""Deployment replay on a held calibration-check event, plus baseline protection."""
import json
from common import MODEL_ROOT
from STEP15_保守联合寻迹复核 import evaluate_event
from hybrid_candidate_tracking import hybrid_select

snapshot=MODEL_ROOT/'local_evaluation_epoch095_20261107'
summary=json.loads((snapshot/'hybrid_confidence_v1/summary.json').read_text(encoding='utf-8'))
eid=summary['check_events'][0]
expected=json.loads((snapshot/'hybrid_confidence_v1'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))['methods']['base_0.02_new_0.2']
row=evaluate_event(eid,str(snapshot),'validation')
assert row['selected']==expected
pool=[dict(indices=[0,1,2,3,4]),dict(indices=[3,4,5,6,7]),dict(indices=[8,9,10,11,12])]
chosen=hybrid_select(pool,[.5,.99,.8],[0],13,.02,.2)
assert pool[0] in chosen and pool[1] not in chosen and pool[2] in chosen
members=[i for t in chosen for i in t['indices']];assert len(members)==len(set(members))
print('deployment matches held-event calibration replay; protected baseline and exclusive additions verified')
