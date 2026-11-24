"""Regression: three locked 5-hit fragments can be merged without dropping an old hit."""
import json
import numpy as np
from local_extractor import proposals,resolve
from STEP5_两级寻迹 import score_tracks,DEFAULT_CONFIG

angle=np.linspace(.1,2.,15);geo=np.zeros((15,6));geo[:,0]=100*np.cos(angle);geo[:,1]=100*np.sin(angle);geo[:,4]=1;geo[:,5]=1600
pairs=np.array([(i,i+1) for i in range(14)]).T
p=np.array([.6 if i in (4,9) else .95 for i in range(14)])
pool=[dict(indices=list(range(j,j+5)),fit=[0.,0.,100.],rms=0.) for j in (0,5,10)]
proposal=next(t for t in proposals(geo,np.zeros(15),pairs,p) if len(t['indices'])==15)
pool.append(proposal)
chosen,_=resolve(pool,np.array([.1,.1,.1,.99]),3,set(range(15)),15,dict(cut=.8,protect=.8))
assert chosen==[3]
before,_=score_tracks(pool[:3],np.zeros(15,dtype=int),np.zeros(15,dtype=bool),DEFAULT_CONFIG)
after,_=score_tracks([pool[i] for i in chosen],np.zeros(15,dtype=int),np.zeros(15,dtype=bool),DEFAULT_CONFIG)
assert before['matched_tracks']==0 and after['matched_tracks']==1
protected,_=resolve(pool,np.array([.99,.99,.99,.99]),3,set(range(15)),15,dict(cut=.8,protect=.8))
assert protected==[0,1,2]
partial=[dict(proposal,indices=list(range(1,15)))]
blocked,_=resolve(pool[:3]+partial,np.array([.1,.1,.1,.99]),3,set(range(15)),15,dict(cut=.8,protect=.8))
assert blocked==[0,1,2]
print(json.dumps(dict(locked_fragment_merge='passed',high_confidence_protection='passed',partial_old_group_replacement_rejected='passed')))
