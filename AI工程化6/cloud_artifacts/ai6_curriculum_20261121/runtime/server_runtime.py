"""Runtime-only loader/transfer choices; mathematical model state is unchanged."""
import os
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from common import JointEventDataset

def dataset(root,split):
    cache=os.getenv('MACE_GRAPH_CACHE')
    if cache:
        from graph_cache import CachedGraphDataset
        return CachedGraphDataset(cache,split,root,mapped_shards=2)
    return JointEventDataset(root,split)

def loader(data,batch_size,shuffle=False,generator=None,workers=None):
    workers=int(os.getenv('MACE_WORKERS','0')) if workers is None else workers
    kwargs={'num_workers':workers,'pin_memory':torch.cuda.is_available() and os.getenv('MACE_PIN_MEMORY','0')=='1'}
    if workers:kwargs.update(prefetch_factor=2,persistent_workers=True,multiprocessing_context='spawn')
    return DataLoader(data,batch_size=batch_size,shuffle=shuffle,generator=generator,**kwargs)

class DevicePrefetch:
    def __init__(self,source,device):
        self.source=source;self.device=device
    def __iter__(self):
        if self.device.type!='cuda' or os.getenv('MACE_ASYNC_TRANSFER','0')!='1':
            for batch in self.source:yield batch.to(self.device,non_blocking=True)
            return
        stream=torch.cuda.Stream(device=self.device);iterator=iter(self.source)
        def preload():
            try:batch=next(iterator)
            except StopIteration:return None
            with torch.cuda.stream(stream):return batch.to(self.device,non_blocking=True)
        upcoming=preload()
        while upcoming is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            current=upcoming;current.record_stream(torch.cuda.current_stream(self.device))
            upcoming=preload();yield current
    def __len__(self):return len(self.source)
