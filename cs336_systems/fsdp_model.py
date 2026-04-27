from typing import List, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel
import contextlib
from torch.distributed import ReduceOp

__all__ = [
    "FSDP",
]


class _AllGatherWithReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_shard: torch.Tensor, all_shard_sizes: torch.Tensor):
        world_size = dist.get_world_size()
        group = dist.group.WORLD
        shard_sizes = all_shard_sizes.tolist()

        gather_list = [
            torch.empty(s, dtype=local_shard.dtype, device=local_shard.device)
            for s in shard_sizes
        ]
        dist.all_gather(gather_list, local_shard, group=group)
        full_param = torch.cat(gather_list)

        ctx.all_shard_sizes = all_shard_sizes
        ctx.group = group
        ctx.world_size = world_size
        ctx.original_dtype = local_shard.dtype
        return full_param

    @staticmethod
    def backward(ctx, grad_full: torch.Tensor):
        all_shard_sizes = ctx.all_shard_sizes
        group = ctx.group
        world_size = ctx.world_size
        original_dtype = ctx.original_dtype

        grad_full = grad_full.clone()
        dist.all_reduce(grad_full, op=ReduceOp.SUM, group=group)
        grad_full.div_(world_size)

        if grad_full.dtype != original_dtype:
            grad_full = grad_full.to(original_dtype)

        rank = dist.get_rank()

        offsets = [0]
        shard_list = all_shard_sizes.tolist()
        for s in shard_list[:-1]:
            offsets.append(offsets[-1] + s)
            
        local_start = offsets[rank]
        local_end = local_start + shard_list[rank]
        
        return grad_full[local_start:local_end].contiguous(), None


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype

        self.sharded_infos: List[Tuple[str, nn.Module, str, torch.Size, torch.dtype, int, int, bool]] = []
        self.replicated_params: Dict[str, torch.nn.Parameter] = {}
        flat_params = []
        offset = 0

        for full_name, param in list(module.named_parameters()):
            *parent_names, attr_name = full_name.split(".")
            parent = module
            for n in parent_names:
                parent = getattr(parent, n)

            if parent.__class__.__name__ in ("Linear", "Embedding"):
                delattr(parent, attr_name) 
                numel = param.numel()
                self.sharded_infos.append(
                    (full_name, parent, attr_name, param.shape, param.dtype, offset, numel, param.requires_grad)
                )
                flat_params.append(param.data.view(-1))
                offset += numel
            else:
                self.replicated_params[full_name] = param

        if len(flat_params) == 0:
            self.local_shard = nn.Parameter(torch.empty(0))
            self.all_shard_sizes = torch.tensor([0], dtype=torch.long)
            self.world_size = 1
        else:
            flat_tensor = torch.cat(flat_params)
            total_numel = flat_tensor.numel()
            if not dist.is_initialized():
                self.world_size = 1
                self.local_shard = nn.Parameter(flat_tensor)
                self.all_shard_sizes = torch.tensor([total_numel], dtype=torch.long)
            else:
                world_size = dist.get_world_size()
                self.world_size = world_size
                rank = dist.get_rank()

                base = total_numel // world_size
                rem = total_numel % world_size
                shard_sizes = [base + (1 if r < rem else 0) for r in range(world_size)]
                offsets = [0]
                for r in range(world_size - 1):
                    offsets.append(offsets[-1] + shard_sizes[r])

                local_start = offsets[rank]
                local_end = local_start + shard_sizes[rank]
                self.local_shard = nn.Parameter(flat_tensor[local_start:local_end].clone())

                size_t = torch.tensor([shard_sizes[rank]], dtype=torch.long, device=flat_tensor.device)
                gathered = [torch.empty(1, dtype=torch.long, device=flat_tensor.device) for _ in range(world_size)]
                dist.all_gather(gathered, size_t)
                self.all_shard_sizes = torch.tensor([g.item() for g in gathered], dtype=torch.long)

    def forward(self, *inputs, **kwargs):
        if self.world_size == 1 and self.local_shard.numel() > 0:
            full_param = self.local_shard
        elif self.local_shard.numel() > 0:
            full_param = _AllGatherWithReduceScatter.apply(self.local_shard, self.all_shard_sizes)
            if self.compute_dtype is not None and full_param.dtype != self.compute_dtype:
                full_param = full_param.to(self.compute_dtype)
        else:
            full_param = None

        if full_param is not None:
            for full_name, mod, attr_name, shape, dtype, start, numel, req_grad in self.sharded_infos:
                view = full_param[start: start + numel].view(shape)
                view.requires_grad_(req_grad)
                setattr(mod, attr_name, view)

        if self.compute_dtype is not None:
            inputs = tuple(x.to(self.compute_dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in inputs)
            kwargs = {k: (v.to(self.compute_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in kwargs.items()}

        autocast_device = "cuda" if torch.cuda.is_available() else "cpu" 
        
        with torch.autocast(device_type=autocast_device, dtype=self.compute_dtype) if self.compute_dtype is not None else contextlib.nullcontext():
            return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        if not dist.is_initialized() or self.world_size <= 1:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return

        world_size = self.world_size
        for param in self.replicated_params.values():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def gather_full_params(self) -> Dict[str, torch.Tensor]:
        result = {}
        for name, param in self.replicated_params.items():
            result[name] = param.data

        if self.local_shard.numel() > 0:
            if self.world_size == 1:
                flat = self.local_shard.data
            else:
                gather_list = [torch.empty(s.item(), dtype=self.local_shard.dtype, device=self.local_shard.device)
                               for s in self.all_shard_sizes]
                dist.all_gather(gather_list, self.local_shard.data)
                flat = torch.cat(gather_list)

            for full_name, _, _, shape, dtype, start, numel, _ in self.sharded_infos:
                result[full_name] = flat[start: start + numel].view(shape).to(dtype)

        return result