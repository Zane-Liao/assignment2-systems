from collections import defaultdict
from typing import List, Set, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

__all__ = [
    "DDPIndividualParameters",
    "BucketDDPIndividualParameters",
]


class DDPIndividualParameters(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Not Initial DDPIndividualParameters !!!")
        
        self.module = module
        
        self.rank = dist.get_rank()
        
        self.world_size = dist.get_world_size()
        
        self._params: List[nn.Parameter] = [p for _, p in module.named_parameters()]
        
        self._buffers: Dict[str, torch.Tensor | None] = [b for _, b in module.named_buffers()]

        self._hook_handles: List[Any] = []
                
        self._broadcast_parameters()
        
        self._broadcast_buffers()

        self._register_hooks()
        
    def _broadcast_parameters(self):
        seen_params: Set[int] = set()
        with torch.no_grad():
            for p in self._params:
                if id(p) in seen_params:
                    continue
                dist.broadcast(p.data, src=0)
                seen_params.add(id(p))
    
    def _broadcast_buffers(self):
        with torch.no_grad():
            for b in self._buffers:
                try:
                    dist.broadcast(b.data, src=0)
                except Exception:
                    pass
    
    def _register_hooks(self):
        seen_params: Set[int] = set()
        for p in self._params:
            if id(p) in seen_params or not p.requires_grad:
                continue
            seen_params.add(id(p))
            
            def make_hook(param):
                def hook(grad):
                    if grad is None:
                        return
                    
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=False)
                    
                    grad.div_(self.world_size)
                    
                    return grad
                return hook
            
            handle = p.register_hook(make_hook(p))
            self._hook_handles.append(handle)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        pass


class BucketDDPIndividualParameters(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Not Initial BucketDDPIndividualParameters !!!")
        
        self.module = module
        
        self.bucket_bytes_cap = int(bucket_size_mb * 1024 * 1024)
        
        self.rank = dist.get_rank()
        
        self.world_size = dist.get_world_size()
        
        self._params: List[nn.Parameter] = [p for _, p in module.named_parameters()]
        
        self._buffers = [b for _, b in module.named_buffers()]
        
        self._param_info: Dict[int, Tuple[int, int, int, torch.Size, int]] = {}
        
        self.buckets: List[Dict[str, Any]] = []
        
        self._hook_handles: List[Any] = []
        
        self._broadcast_parameters()
        
        self._broadcast_buffers()
        
        self._build_buckets()
        
        self._register_hooks()
        
    def _broadcast_parameters(self):
        seen_params: Set[int] = set()
        with torch.no_grad():
            for p in self._params:
                if id(p) in seen_params:
                    continue
                dist.broadcast(p.data, src=0)
                seen_params.add(id(p))
                
    def _broadcast_buffers(self):
        with torch.no_grad():
            for b in self._buffers:
                try:
                    dist.broadcast(b.data, src=0)
                except Exception:
                    pass
    
    def _create_buckets(self, params: List[nn.Parameter], device: torch.device, dtype: torch.dtype):
        # Cal the total numel and allocate a flat buffer
        numels = [p.numel() for p in params]
        buffer = torch.zeros(sum(numels), dtype=dtype, device=device)
        
        bucket_idx = len(self.buckets)
        offsets = []
        offset = 0
        
        for idx_in_bucket, p in enumerate(params):
            offsets.append(offset)
            self._param_info[id(p)] = (bucket_idx, offset, p.numel(), p.shape, idx_in_bucket)
            offset += p.numel()
            
        bucket = {
            'params': params,
            'buffer': buffer,
            'expected_count': len(params),
            'ready_count': 0,
            'offsets': offsets,
            'ready_flags': [False] * len(params),
            'work': None,
            'pending': False 
        }
        
        self.buckets.append(bucket)
            
    def _build_buckets(self):
        # 1.Grouped by: device + dtype
        groups = defaultdict(list)
        seed_ids: Set[int] = set()
        
        for p in self._params:
            if not p.requires_grad:
                continue
            if id(p) in seed_ids:
                continue
        
            seed_ids.add(id(p))
            key = (p.device, p.dtype)
            groups[key].append(p)
        
        # 2.Split buckets for each group by bucket_bytes_cap
        for (device, dtype), params in groups.items():
            cur_bucket_params: List[nn.Parameter] = []
            cur_size = 0
            
            for p in params:
                # Number of Cal p
                numel = p.numel()
                bytes_needed = numel * p.element_size()
                
                if bytes_needed > self.bucket_bytes_cap and len(cur_bucket_params) == 0:
                    self._create_buckets([p], device=device, dtype=dtype)
                    cur_bucket_params = []
                    cur_size = 0
                    continue
                
                if cur_size + bytes_needed > self.bucket_bytes_cap and len(cur_bucket_params) > 0:
                    self._create_buckets(cur_bucket_params, device=device, dtype=dtype)
                    cur_bucket_params = []
                    cur_size = 0
                
                cur_bucket_params.append(p)
                cur_size += bytes_needed
            
            if len(cur_bucket_params) > 0:
                self._create_buckets(cur_bucket_params, device=device, dtype=dtype)
                
    def _unpack_bucket_into_params(self, bucket_idx: int):
        bucket = self.buckets[bucket_idx]
        buf = bucket['buffer']
        params = bucket['params']
        offsets = bucket['offsets']
        
        with torch.no_grad():
            for p, off in zip(params, offsets):
                numel = p.numel()
                flat = buf[off: off + numel]
                view = flat.view_as(p)
                
                if p.grad is None:
                    p.grad = view.clone()
                else:
                    p.grad.copy_(view)
    
    def _register_hooks(self):
        seen: Set[int] = set()
        for p in self._params:
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            
            def make_hook(param):
                pid = id(param)
                def hook(grad):
                    if grad is None:
                        return
                    
                    if grad.is_sparse:
                        raise RuntimeError("Sparse gradients are not supported in this simple bucket DDP !!!")
                    
                    info = self._param_info.get(pid, None)
                    
                    if info is None:
                        return grad
                    
                    bucket_idx, offset, numel, shape, idx_in_bucket = info
                    bucket = self.buckets[bucket_idx]
                    
                    # We flatten the gradient while ensuring memory contiguousness.
                    # The bucket buffer is a 1D contiguous memory, which we unify into a 1D flat layout.
                    flat = grad.contiguous().view(-1)
                    
                    # Copy to the flat buffer of the bucket
                    with torch.no_grad():
                        bucket['buffer'][offset: offset + numel].copy_(flat)
                        
                    # ready_flags: Avoid duplicate counting
                    if not bucket['ready_flags'][idx_in_bucket]:
                        bucket['ready_flags'][idx_in_bucket] = True
                        bucket['ready_count'] += 1
                        
                    # Determine if the bucket is "fully ready"
                    if bucket['ready_count'] == bucket['expected_count'] and not bucket['pending']:
                        try:
                            work = dist.all_reduce(bucket['buffer'], op=dist.ReduceOp.SUM, async_op=True)
                            bucket['work'] = work
                            bucket['pending'] = True
                            
                        except TypeError:
                            # fallback to synchronized version
                            dist.all_reduce(bucket['buffer'], op=dist.ReduceOp.SUM)
                            bucket['buffer'].div_(self.world_size)
                            
                            self._unpack_bucket_into_params(bucket_idx)
                    
                    return grad
                return hook
            
            handle = p.register_hook(make_hook(p))
            self._hook_handles.append(handle)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for idx, bucket in enumerate(self.buckets):
            if bucket.get('pending', False):
                work = bucket.get('work', None)
                
                if work is not None:
                    work.wait()
                    bucket['work'] = None
                    
                with torch.no_grad():
                    bucket['buffer'].div_(self.world_size)
                    
                self._unpack_bucket_into_params(idx)
                
                bucket['pending'] = False
                bucket['ready_count'] = 0
                bucket['ready_flags'] = [False] * len(bucket['params'])
                
            else:
                bucket['ready_count'] = 0
                bucket['ready_flags'] = [False] * len(bucket['params'])
        
    def __del__(self):
        for h in getattr(self, "_hook_handles", []):
            try:
                h.remove()
            except Exception:
                pass