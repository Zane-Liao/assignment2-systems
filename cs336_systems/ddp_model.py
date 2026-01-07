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
    """A minimal, 🤓Edu implementation of Distributed Data Parallel (DDP)
    that synchronizes gradients **per-parameter**, without gradient bucketing.

    This class wraps a given ``torch.nn.Module`` and performs:
    
    1. Parameter and buffer broadcasting from rank 0 at initialization.
    2. Gradient synchronization by registering autograd hooks on each parameter.
    3. A synchronous ``all_reduce`` for every individual parameter gradient
       during backward propagation.

    Compared to PyTorch's official ``DistributedDataParallel``, this
    implementation is intentionally simple and explicit:
    
    - Each parameter triggers its own ``all_reduce`` call.
    - No gradient bucketing, fusion, or overlap of communication and computation.
    - Intended for learning, debugging, or correctness verification rather
      than performance-critical training.

    This design makes the gradient flow and communication behavior easy to
    reason about, at the cost of significantly higher communication overhead.

    Parameters
    ----------
    module : torch.nn.Module
        The neural network module to be wrapped. All parameters returned by
        ``module.named_parameters()`` will participate in distributed gradient
        synchronization. Buffers returned by ``module.named_buffers()`` will be
        broadcasted at initialization time.

    Notes
    -----
    - ``torch.distributed`` must be available and a process group must already
      be initialized before constructing this class.
    - Gradients are synchronized synchronously (``async_op=False``).
    - Tied/shared parameters are handled correctly by deduplicating based on
      parameter identity.
    - Sparse gradients are not specially handled.
    - This class does **not** perform any gradient bucketing or scheduling
      optimizations.

    Methods
    -------
    forward(*inputs, **kwargs)
        Forwards inputs to the wrapped module.

    finish_gradient_synchronization()
        A placeholder method provided for API symmetry with more advanced DDP
        implementations (e.g., bucket-based or asynchronous variants).
        In this implementation, all gradient synchronization is already
        completed during backward hooks, so this method is a no-op.
    """
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
    """A bucket-based🤩 implementation of Distributed Data Parallel (DDP) that
    synchronizes gradients by **fusing multiple parameters into buckets**
    and performing asynchronous collective communication.

    This class improves upon a naive per-parameter DDP approach by grouping
    parameters into buckets of bounded size and issuing one ``all_reduce``
    per bucket instead of per parameter. This significantly reduces
    communication overhead and better matches the design of production-grade
    DDP systems.

    The core ideas implemented in this class are:

    1. Parameters are grouped by (device, dtype) to ensure communication
       compatibility.
    2. Parameters are packed into contiguous flat buffers ("buckets") with
       a configurable size limit.
    3. During backward propagation, each parameter's gradient is copied into
       its bucket buffer via autograd hooks.
    4. When all gradients in a bucket become ready, an asynchronous
       ``all_reduce`` is launched on the bucket buffer.
    5. After communication completes, the reduced gradients are unpacked
       back into individual parameter gradients.

    Compared to ``DDPIndividualParameters``, this implementation:
    
    - Uses gradient bucketing to amortize communication cost.
    - Supports asynchronous ``all_reduce`` to enable partial overlap of
      communication and computation.
    - More closely resembles the internal behavior of PyTorch's native
      ``DistributedDataParallel``.

    This class is intended for educational, experimental, or systems-research
    purposes, where explicit control and visibility into bucket construction
    and synchronization behavior are desired.

    Parameters
    ----------
    module : torch.nn.Module
        The neural network module to be wrapped. All trainable parameters
        returned by ``module.named_parameters()`` will participate in
        bucketed gradient synchronization.

    bucket_size_mb : float, optional
        Maximum size (in megabytes) of each gradient bucket. Parameters are
        packed into buckets such that the total size of each bucket does not
        exceed this limit, except when a single parameter itself exceeds the
        bucket size, in which case it forms its own bucket. Default is 25.0 MB.

    Notes
    -----
    - ``torch.distributed`` must be available and a process group must already
      be initialized before constructing this class.
    - Parameters are deduplicated by identity to correctly handle tied/shared
      weights.
    - Sparse gradients are explicitly not supported.
    - Gradient averaging is performed by dividing the reduced bucket buffer
      by ``world_size`` after communication.
    - Bucket states are reset after each training iteration via
      ``finish_gradient_synchronization``.

    Methods
    -------
    forward(*inputs, **kwargs)
        Forwards inputs to the wrapped module.

    finish_gradient_synchronization()
        Waits for all pending asynchronous ``all_reduce`` operations to
        complete, averages bucket buffers by ``world_size``, unpacks reduced
        gradients back into individual parameters, and resets bucket state
        in preparation for the next iteration.

    Implementation Details
    ----------------------
    Each bucket internally maintains:
    
    - A flat, contiguous gradient buffer.
    - A list of parameters and their offsets into the buffer.
    - Per-parameter readiness flags to ensure correct synchronization.
    - A pending communication handle for asynchronous ``all_reduce``.

    This explicit state management mirrors the internal logic of
    production DDP systems, while remaining concise and readable.
    """
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