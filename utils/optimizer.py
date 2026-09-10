import torch.nn as nn
from typing import List


def get_param_groups(model: nn.Module, weight_decay: float = 0) -> List:
    """Split trainable parameters into decay and no-decay optimizer groups."""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.},
        {"params": decay, "weight_decay": weight_decay}
    ]
