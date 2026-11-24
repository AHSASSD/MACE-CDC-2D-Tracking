"""Truth-based diagnostics only; never called by graph construction or inference."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

root=Path(__file__).resolve().parent/'pilot_v1'
graphs=torch.load(root/'validation.pt',weights_only=False)
counts=np.zeros((3,3),dtype=np.int64)
degrees=[]; track_count=connected=0
for g in graphs:
    a,b=g.edge_index.numpy(); st=g.stereo.numpy(); truth=g.truth.numpy()
    kinds=st[a].astype(int)+st[b].astype(int)
    degrees.extend(np.bincount(b,minlength=g.num_nodes).tolist())
    for k in range(3):
        counts[k,0]+=np.sum(kinds==k)
        counts[k,1]+=np.sum((kinds==k)&(truth[a]==truth[b]))
    for t in np.unique(truth):
        members=np.where(truth==t)[0]; n=len(members)
        aa=np.sum(~st[members]); ss=np.sum(st[members])
        counts[:,2]+=[aa*(aa-1),2*aa*ss,ss*(ss-1)]
        if aa<5: continue
        track_count+=1
        keep=(truth[a]==t)&(truth[b]==t)
        remap=np.full(g.num_nodes,-1); remap[members]=np.arange(n)
        matrix=coo_matrix((np.ones(keep.sum()),(remap[a[keep]],remap[b[keep]])),shape=(n,n))
        connected+=connected_components(matrix,directed=False,return_labels=False)==1
report={'validation_events':len(graphs),'degree_percentiles':dict(zip(['min','median','p95','max'],np.percentile(degrees,[0,50,95,100]).tolist())),
        'eligible_truth_tracks':track_count,'fully_connected_truth_tracks':int(connected),
        'fully_connected_fraction':float(connected/max(track_count,1)),
        'edge_types':{k:{'candidates':int(n),'positive_candidates':int(p),'all_same_track_pairs':int(total),'same_track_all_pair_coverage':float(p/max(total,1))} for k,(n,p,total) in zip(['AA','AS','SS'],counts)},
        'note':'All-pair coverage is not local-successor recall; distant hits on one track need not have a direct edge. Truth connectivity is a necessary diagnostic, not proof of reconstructability.'}
(root/'graph_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
