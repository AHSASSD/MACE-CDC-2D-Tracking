"""AI6: observable-only joint axial/stereo graph and independent network."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'AI工程化4 - 副本'
DATA_ROOT=ROOT/'joint_event_dataset'
MODEL_ROOT=ROOT/'joint_training'
MAP_FILE=Path(r'D:\SRTP\CDC_cell_map\cdc_cell_map.xlsx')
SEED=20260823
NODE_FEATURES=['wire_x/800','wire_y/800','wire_r/800','d/10','sin_phi','cos_phi','u','v','w','length/1600','is_stereo','signed_tilt_rad']
EDGE_FEATURES=['dr/800','reference_xy_distance/800','dphi','dphi/(distance+1e-6)','finite_common_z_min_xy/800','min_xy_z/800','common_half_z/800','delta_d/10','direction_dot','source_stereo','target_stereo','delta_signed_tilt']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def save_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def signed_tilt(geometry):
    x,y,u,v,w=geometry[:,:5].T
    r=np.hypot(x,y)
    # Tangential component fixes stereo handedness, unlike acos(w).
    return np.arctan2((-y*u+x*v)/r,w)

def build_graph(cell,drift,hit_id,geometry,truth=None,event_id=-1):
    cell=np.asarray(cell,dtype=np.int64);drift=np.asarray(drift,dtype=float)
    hit_id=np.asarray(hit_id,dtype=np.int64);geo=np.asarray(geometry,dtype=float)[cell]
    xy=geo[:,:2];direction=geo[:,2:5];length=geo[:,5]
    if len(np.unique(hit_id))!=len(hit_id): raise ValueError('Duplicate independent hit identity')
    if not np.isfinite(geo).all() or not np.isfinite(drift).all(): raise ValueError('Nonfinite geometry/d')
    if not np.allclose(np.linalg.norm(direction,axis=1),1,atol=1e-6): raise ValueError('Non-unit wire direction')
    slope=direction[:,:2]/direction[:,2:3];halfz=length*np.abs(direction[:,2])/2
    stereo=np.linalg.norm(direction[:,:2],axis=1)>1e-9
    radius=np.linalg.norm(xy,axis=1);phi=np.arctan2(xy[:,1],xy[:,0]);tilt=signed_tilt(geo)
    excursion=np.linalg.norm(direction[:,:2],axis=1)*length/2
    pairs=cKDTree(xy).query_pairs(120+2*excursion.max(),output_type='ndarray')
    a,b=pairs.T;delta=xy[b]-xy[a];ds=slope[b]-slope[a];hz=np.minimum(halfz[a],halfz[b])
    z=np.clip(-(delta*ds).sum(1)/np.maximum((ds*ds).sum(1),1e-20),-hz,hz)
    pairs=pairs[np.linalg.norm(delta+ds*z[:,None],axis=1)<=120]
    edges=np.vstack([pairs,pairs[:,::-1]]);a,b=edges.T
    delta=xy[b]-xy[a];ds=slope[b]-slope[a];hz=np.minimum(halfz[a],halfz[b])
    z=np.clip(-(delta*ds).sum(1)/np.maximum((ds*ds).sum(1),1e-20),-hz,hz)
    minimum=np.linalg.norm(delta+ds*z[:,None],axis=1);dist=np.linalg.norm(delta,axis=1)/800
    dp=(phi[b]-phi[a]+np.pi)%(2*np.pi)-np.pi
    x=np.column_stack([xy/800,radius/800,drift/10,np.sin(phi),np.cos(phi),direction,length/1600,stereo,tilt])
    e=np.column_stack([(radius[b]-radius[a])/800,dist,dp,dp/(dist+1e-6),minimum/800,z/800,hz/800,(drift[b]-drift[a])/10,(direction[a]*direction[b]).sum(1),stereo[a],stereo[b],tilt[b]-tilt[a]])
    if not np.isfinite(x).all() or not np.isfinite(e).all(): raise ValueError('Nonfinite feature')
    labels=np.zeros(len(a),np.float32) if truth is None else (np.asarray(truth)[a]==np.asarray(truth)[b]).astype(np.float32)
    return Data(x=torch.tensor(x,dtype=torch.float32),edge_index=torch.tensor(edges.T.copy()),edge_attr=torch.tensor(e,dtype=torch.float32),y=torch.from_numpy(labels),
                edge_kind=torch.tensor(stereo[a].astype(np.int64)+stereo[b].astype(np.int64)),stereo=torch.tensor(stereo),
                hit_id=torch.tensor(hit_id),cell_id=torch.tensor(cell),event_id=event_id,
                truth=torch.tensor(np.full(len(cell),-1) if truth is None else np.asarray(truth)))

class JointEventDataset(torch.utils.data.Dataset):
    def __init__(self,root,split):
        self.root=Path(root);self.metadata=json.loads((self.root/'dataset_metadata_joint_no_time.json').read_text(encoding='utf-8'))
        if self.metadata['status']!='complete': raise ValueError('Dataset incomplete')
        if self.metadata['node_features']!=NODE_FEATURES or self.metadata['edge_features']!=EDGE_FEATURES: raise ValueError('Feature contract mismatch')
        self.ids=self.metadata['event_ids'][split]
        self.geometry=np.load(self.root/'wire_geometry.npy')
    def __len__(self): return len(self.ids)
    def __getitem__(self,index):
        eid=self.ids[index]
        with np.load(self.root/'events'/f'event_{eid:05d}.npz') as f:
            return build_graph(f['cell_id'],f['d'],f['hit_id'],self.geometry,f['truth'],eid)

class JointInteractionNetwork(nn.Module):
    """Shared 256-wide message trunk, eight passes, type-conditioned edge heads.

    Chunking and activation recomputation bound memory without dropping edges.
    Geometry state stays 12-dimensional, as the former physical-state residual
    network; output heads distinguish AA/AS/SS under one jointly trained model.
    """
    def __init__(self,hidden=256,n_passes=8,chunk_size=4096,recompute_policy='all'):
        super().__init__();self.hidden=hidden;self.n_passes=n_passes;self.chunk_size=chunk_size
        self.recompute_policy=recompute_policy
        self.relational_model=nn.Sequential(nn.Linear(36,hidden),nn.LayerNorm(hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.object_model=nn.Sequential(nn.Linear(12+hidden,hidden),nn.LayerNorm(hidden),nn.ReLU(),nn.Linear(hidden,12))
        self.edge_classifier=nn.Sequential(nn.Linear(hidden,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,3))
    def _message(self,x,edge_index,edge_attr):
        row,col=edge_index
        messages=x.new_zeros((len(x),self.hidden),dtype=torch.float32)
        for begin in range(0,len(row),self.chunk_size):
            sl=slice(begin,begin+self.chunk_size)
            effect=self.relational_model(torch.cat([x[row[sl]],x[col[sl]],edge_attr[sl]],-1))
            messages=messages.index_add(0,col[sl],effect.float())
        return x+self.object_model(torch.cat([x,messages],-1))
    def _classify(self,x,edge_index,edge_attr):
        row,col=edge_index
        effect=self.relational_model(torch.cat([x[row],x[col],edge_attr],-1))
        kind=(edge_attr[:,9]+edge_attr[:,10]).long()
        return self.edge_classifier(effect).gather(1,kind[:,None]).squeeze(1)
    def forward(self,x,edge_index,edge_attr):
        # Last pass classifies its effects, matching the original eight-pass semantics.
        normal=0 if self.recompute_policy=='all' else (1 if self.recompute_policy=='one_direct' else min(3,1500000//max(edge_index.size(1),1)))
        for step in range(self.n_passes-1):
            x=checkpoint(self._message,x,edge_index,edge_attr,use_reentrant=False) if self.training and step>=normal else self._message(x,edge_index,edge_attr)
        outputs=[]
        for begin in range(0,edge_index.size(1),self.chunk_size):
            ei=edge_index[:,begin:begin+self.chunk_size];ea=edge_attr[begin:begin+self.chunk_size]
            outputs.append(checkpoint(self._classify,x,ei,ea,use_reentrant=False) if self.training else self._classify(x,ei,ea))
        return torch.cat(outputs) if outputs else x.new_empty(0)

class FocalLoss(nn.Module):
    def forward(self,logits,target):
        bce=nn.functional.binary_cross_entropy_with_logits(logits.float(),target,reduction='none')
        return (.5*(1-torch.exp(-bce)).square()*bce).mean()
