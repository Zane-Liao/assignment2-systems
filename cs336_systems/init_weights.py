import math
import torch
import torch.nn as nn

def init_weights(module):
    """
    kaiming init
    """
    if hasattr(module, "weight") and module.weight is not None:
        if module.weight.dim() >= 2:
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        else:
            nn.init.ones_(module.weight)

    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)

    if hasattr(module, "E"):
        nn.init.normal_(module.E, mean=0.0, std=0.02)
        
    if hasattr(module, "w1") and hasattr(module, "w2") and hasattr(module, "w3"):
        if module.w1 is not None:
            nn.init.kaiming_uniform_(module.w1, a=math.sqrt(5))
        if module.w2 is not None:
            nn.init.kaiming_uniform_(module.w2, a=math.sqrt(5))
        if module.w3 is not None:
            nn.init.kaiming_uniform_(module.w2, a=math.sqrt(5))