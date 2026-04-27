import torch
from typing import Iterable, List, Type, Any, Optional, Dict
from torch.optim import Optimizer
import torch.distributed as dist

__all__ = [
    "OptimizerStateShare",
]


class OptimizerStateShare(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ):
        if not dist.is_initialized():
            self.local_optimizer = optimizer_cls(params, **kwargs)
            self.all_params = list(params)
            self.param_to_rank = {id(p): 0 for p in self.all_params}
            self.world_size = 1
            self._next_rank_idx = len(self.all_params)
            return

        self.world_size = dist.get_world_size()
        rank = dist.get_rank()

        self.all_params = list(params)
        self.param_to_rank = {
            id(p): i % self.world_size for i, p in enumerate(self.all_params)
        }
        self._next_rank_idx = len(self.all_params)

        local_params = [p for p in self.all_params if self.param_to_rank[id(p)] == rank]
        self.local_optimizer = optimizer_cls(local_params, **kwargs)

    def zero_grad(self, set_to_none: bool = False):
        for p in self.all_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None, **kwargs: Any):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if dist.is_initialized() and self.world_size > 1:
            for p in self.all_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(self.world_size)

        self.local_optimizer.step(**kwargs)

        if dist.is_initialized() and self.world_size > 1:
            for p in self.all_params:
                owner_rank = self.param_to_rank[id(p)]
                dist.broadcast(p.data, src=owner_rank)

        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        new_params = list(param_group["params"])
        if not new_params:
            return

        group_kwargs = {k: v for k, v in param_group.items() if k != "params"}

        if not dist.is_initialized() or self.world_size == 1:
            self.local_optimizer.add_param_group(param_group)
            self.all_params.extend(new_params)
            for p in new_params:
                self.param_to_rank[id(p)] = 0
            self._next_rank_idx += len(new_params)
            return

        rank = dist.get_rank()
        start_idx = self._next_rank_idx
        local_new = []
        for i, p in enumerate(new_params):
            owner = (start_idx + i) % self.world_size
            self.param_to_rank[id(p)] = owner
            if owner == rank:
                local_new.append(p)

        if local_new:
            local_group = {"params": local_new, **group_kwargs}
            self.local_optimizer.add_param_group(local_group)

        self.all_params.extend(new_params)
        self._next_rank_idx += len(new_params)

    def state_dict(self):
        return {
            "local_optimizer": self.local_optimizer.state_dict(),
            "param_to_rank": self.param_to_rank,
            "_next_rank_idx": self._next_rank_idx,
        }

    def load_state_dict(self, state_dict):
        self.local_optimizer.load_state_dict(state_dict["local_optimizer"])
        self.param_to_rank = state_dict["param_to_rank"]
        self._next_rank_idx = state_dict.get("_next_rank_idx", len(self.all_params))

    def __getattr__(self, name):
        return getattr(self.local_optimizer, name)