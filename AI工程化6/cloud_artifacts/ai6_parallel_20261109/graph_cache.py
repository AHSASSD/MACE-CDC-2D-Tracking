"""Lossless, memory-mapped graph shards; bounded per-worker mapping LRU."""
from collections import OrderedDict
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from torch_geometric.data import Data
from common import DATA_ROOT,JointEventDataset,build_graph,signed_tilt,sha,save_json

NODE_KEYS=['x','stereo','hit_id','cell_id','truth']
EDGE_KEYS=['edge_index','edge_attr','y','edge_kind']

def contract(root):
    return {'dataset_sha256':sha(Path(root)/'dataset_metadata_joint_no_time.json'),
            'builder_sha256':hashlib.sha256((inspect.getsource(build_graph)+inspect.getsource(signed_tilt)).encode()).hexdigest(),'format':1}

class CachedGraphDataset(torch.utils.data.Dataset):
    def __init__(self,root,split,source=DATA_ROOT,mapped_shards=2):
        self.root=Path(root);m=json.loads((self.root/'manifest.json').read_text())
        if m['contract']!=contract(source) or m['status']!='complete':raise ValueError('Invalid/stale graph cache')
        original=json.loads((Path(source)/'dataset_metadata_joint_no_time.json').read_text())
        if m['event_ids']!=original['event_ids'] or set(m['index'])!=set(map(str,sum(m['event_ids'].values(),[]))):raise ValueError('Incomplete/mismatched cache event inventory')
        self.ids=m['event_ids'][split];self.index=m['index'];self.limit=mapped_shards;self._maps=OrderedDict()
    def __len__(self):return len(self.ids)
    def __getstate__(self):
        state=self.__dict__.copy();state['_maps']=OrderedDict();return state
    def _release(self,part,arrays):
        # Returned graphs own their buffers, so evicted mappings can close safely.
        for array in arrays.values():array._mmap.close()
        # Return the dominant feature/index pages to the OS, not just the VMA.
        # Advisory only: immutable on-disk graph contents are unchanged.
        if hasattr(os,'posix_fadvise'):
            for key in ['edge_attr','edge_index']:
                fd=os.open(self.root/part/(key+'.npy'),os.O_RDONLY)
                try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
                except OSError:pass
                finally:os.close(fd)
    def __getitem__(self,i):
        eid=self.ids[i];entry=self.index[str(eid)];part=entry['shard']
        if part not in self._maps:
            self._maps[part]={k:np.load(self.root/part/(k+'.npy'),mmap_mode='c') for k in NODE_KEYS+EDGE_KEYS}
            while len(self._maps)>self.limit:
                old_part,old_arrays=self._maps.popitem(last=False);self._release(old_part,old_arrays)
        self._maps.move_to_end(part);arrays=self._maps[part]
        ns=slice(entry['node_start'],entry['node_end']);es=slice(entry['edge_start'],entry['edge_end'])
        fields={k:torch.from_numpy(np.array(arrays[k][ns],copy=True)) for k in NODE_KEYS}
        for k in EDGE_KEYS:
            values=arrays[k][:,es] if k=='edge_index' else arrays[k][es]
            fields[k]=torch.from_numpy(np.array(values,copy=True))
        fields['edge_index']=fields['edge_index'].long();fields['edge_kind']=fields['edge_kind'].long();fields['y']=fields['y'].float()
        return Data(**fields,event_id=eid)

def generate(output,source=DATA_ROOT,shard_size=32):
    torch.set_num_threads(1);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    signature=contract(source)
    if (output/'manifest.json').exists():
        m=json.loads((output/'manifest.json').read_text());assert m['contract']==signature and m['status']=='complete';print('Cache already complete',flush=True);return
    if shutil.disk_usage(output).free<80*1024**3:raise OSError('Need 80 GiB free for full lossless cache')
    metadata=json.loads((Path(source)/'dataset_metadata_joint_no_time.json').read_text());index={};total_bytes=0;verified=0
    for split in ['train','validation','test']:
        dataset=JointEventDataset(source,split)
        for start in range(0,len(dataset),shard_size):
            name=f'{split}_{start//shard_size:04d}';target=output/name
            if (target/'shard.json').exists():
                shard=json.loads((target/'shard.json').read_text());assert shard['contract']==signature
                index.update(shard['index']);total_bytes+=shard['bytes'];continue
            temporary=output/(name+'.partial')
            if temporary.exists():raise FileExistsError(f'Inspect incomplete shard before retry: {temporary}')
            temporary.mkdir();graphs=[dataset[i] for i in range(start,min(start+shard_size,len(dataset)))];local={};ns=es=0
            for g in graphs:
                local[str(g.event_id)]={'shard':name,'node_start':ns,'node_end':ns+g.num_nodes,'edge_start':es,'edge_end':es+g.num_edges};ns+=g.num_nodes;es+=g.num_edges
            for key in NODE_KEYS+EDGE_KEYS:
                axis=1 if key=='edge_index' else 0
                values=np.concatenate([getattr(g,key).numpy() for g in graphs],axis=axis)
                if key=='edge_index':values=values.astype(np.int32)
                if key in ['y','edge_kind']:values=values.astype(np.uint8)
                np.save(temporary/(key+'.npy'),values)
            # First/last graph of every shard: byte-exact feature/index/label readback.
            for g in [graphs[0],graphs[-1]]:
                row=local[str(g.event_id)]
                for key in NODE_KEYS+EDGE_KEYS:
                    a=np.load(temporary/(key+'.npy'),mmap_mode='r');sl=slice(row['node_start'],row['node_end']) if key in NODE_KEYS else slice(row['edge_start'],row['edge_end'])
                    actual=a[:,sl] if key=='edge_index' else a[sl]
                    assert np.array_equal(actual,getattr(g,key).numpy()),(name,key)
                verified+=1
            size=sum(p.stat().st_size for p in temporary.glob('*.npy'))
            save_json(temporary/'shard.json',{'contract':signature,'index':local,'bytes':size});temporary.rename(target)
            index.update(local);total_bytes+=size
            save_json(output/'build_progress.json',{'events':len(index),'total':10000,'bytes':total_bytes,'verified_readbacks':verified})
            if start%320==0:print(split,len(index),total_bytes,flush=True)
    assert len(index)==10000 and set(map(int,index))==set(sum(metadata['event_ids'].values(),[]))
    save_json(output/'manifest.json',{'status':'complete','contract':signature,'event_ids':metadata['event_ids'],'index':index,'bytes':total_bytes,'verified_readbacks_this_run':verified,'shard_size':shard_size})
    print('COMPLETE',total_bytes,'bytes',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();generate(a.output)
