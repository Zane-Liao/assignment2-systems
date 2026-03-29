import math
from typing import Optional
from collections.abc import Callable, Iterable
import torch
from torch.optim import Optimizer
from torch.nn import Parameter

__all__ = [
    "compute_lr",
    "gradient_cliping",
    "SGD",
    "AdamW",
]


class SGD(Optimizer):
    """
    PDF version of SGD would be implemented as a PyTorch Optimizer:
    """
    def __init__(self, params, lr=1e-3):
        """
        should initialize your optimizer. Here, params will be a collection of
        parameters to be optimized (or parameter groups, in case the user 
        wants to use different hyperpa-rameters, such as learning rates, 
        for different parts of the model). Make sure to pass params to the
        __init__ method of the base class, which will store these parameters 
        for use in step. You can take additional arguments depending on the
        optimizer (e.g., the learning rate is a common one), and pass
        them to the base class constructor as a dictionary, where keys are
        the names (strings) you choose for these parameters.
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """
        should make one update of the parameters. During the training loop,
        this will be called after the backward pass, so you have access to
        the gradients on the last batch. This method should iterate through
        each parameter tensor p and modify them in place, i.e. setting p.data,
        which holds the tensor associated with that parameter based on the 
        gradient p.grad (if it exists), the tensor representing the 
        gradient of the loss with respect to that parameter.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss


class AdamW(Optimizer):
    """
    Implement the AdamW optimizer as a subclass of torch.optim.Optimizer.
    Your class should take the learning rate α in __init__, as well as the β,
    ϵ and λ hyperparameters. To help you keep state, the base Optimizer class
    gives you a dictionary self.state, which maps nn.Parameter objects to
    a dictionary that stores any information you need for that parameter
    (for AdamW, this would be the moment estimates).
    """
    def __init__(
        self,
        params: Iterable[Optimizer],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid Learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] <= 1:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] <= 1:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        default = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, default)
    
    def step(self, closure: Callable | None = None):
        loss = None
        if loss is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Check grad is or not sparse
                grad = p.grad.data
                if grad.is_sparse:
                    raise ValueError("Adam does not support sparse gradients")
                
                # Initial Parameters
                state = self.state[p]
                alpha = group["lr"]
                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                t = state.get("t", 1) # Update Times
                prev_m_t = state.get("m", torch.zeros_like(grad))
                prev_v_t = state.get("v", torch.zeros_like(grad))
                
                # Compute Parameters
                m_t = beta_1 * prev_m_t + ((1 - beta_1) * grad)
                v_t = beta_2 * prev_v_t + ((1 - beta_2) * torch.square(grad))
                
                alpha_t = alpha * (math.sqrt(1 - (beta_2**t)) / (1 - (beta_1**t)))
                p.data -= alpha_t * m_t / (torch.sqrt(v_t) + eps)
                p.data -= alpha * group["weight_decay"] * p.data
                
                # Update Parameters
                state["m"] = m_t
                state["v"] = v_t
                state["t"] = t + 1

        return loss

def compute_lr(t, alpha_max, alpha_min, t_w, t_c):
    if t < t_w:
        return t / t_w * alpha_max
    if t_w <= t and t <= t_c:
        cosine = math.cos((t - t_w) / (t_c - t_w) * math.pi)
        return alpha_min + 0.5 * (1 + cosine) * (alpha_max - alpha_min)
    if t > t_c:
        return alpha_min

def gradient_cliping(parameters: Iterable[Parameter], max_l2_norm: float, epsilon = 1e-6):
    total_norm_ = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total_norm_ += p.grad.data.norm(2).item() ** 2
    
    total_norm = math.sqrt(total_norm_)
    
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + epsilon)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data.mul_(scale)

    return total_norm