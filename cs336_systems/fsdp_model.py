from collections import defaultdict
from typing import List, Set, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel

__all__ = [
    "FSDP",
]


class FSDP(nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        
        self.module = module
        
        self.compute_dtype = compute_dtype
        
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        self.rank = dist.get_rank() if dist.is_initialized() else 1
        
        
        
    def _mixed_precision():
        ...
        
    def _get_world_size(self):
        return dist.get_world_size() if dist.is_initialized() else 1
    
    def _get_rank():
        return dist.get_rank() if dist.is_initialized() else 1
        
    def forward(self, *inputs, **kwargs):
        ...
        
    def finish_gradient_synchronization(self):
        ...
