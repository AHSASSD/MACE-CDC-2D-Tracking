"""Combine local initial-reference counts with the recorded server member checks."""
from collections import Counter
from pathlib import Path
import json

root=Path(__file__).resolve().parent
rows=[json.loads(p.read_text()) for p in (root/'events/initial126_raw').glob('event_*.json')]
assert len(rows)==200
c=Counter()
for row in rows:c.update(row['counts'])
initial=dict(c)
initial.update(track_efficiency=c['matched_tracks']/c['truth_tracks'],fake_fraction=c['fake_candidates']/c['output_candidates'],clone_fraction=c['clone_candidates']/c['output_candidates'],joint_track_efficiency=c['joint_matched_tracks']/c['truth_tracks'])
result={'initial126_local_reference':initial,'runtime_note':'Initial reference uses cached local126 probabilities; recorded arm members use the server runtime. Stable_tracker300 re-evaluates all models under the same local runtime.'}
for arm in ['control','balanced']:
    history=json.loads((root/'cloud'/arm/'history.json').read_text())
    result[arm]=[dict(epoch=r['epoch'],validation=r['validation'],membership=r['membership']) for r in history if r['membership']]
(root/'raw_member_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'initial126_original_two_pass_200':initial}))
