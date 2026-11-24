import numpy as np
from common import MODEL_ROOT
from STEP12_候选可靠性校准 import records,features,FEATURES

row=records(232,MODEL_ROOT/'local_evaluation_epoch095_20261107')
assert row['pairs'].ndim==2 and row['pairs'].shape[0]==2 and np.issubdtype(row['pairs'].dtype,np.integer)
assert len(row['p'])==row['pairs'].shape[1]
assert row['x'].shape==(len(row['candidates']),len(FEATURES)) and np.isfinite(row['x']).all()
assert not any('truth' in name or 'event_id' in name or 'track_id' in name for name in FEATURES)
print('candidate features finite; edge indices distinct from probabilities; truth identifiers excluded')
