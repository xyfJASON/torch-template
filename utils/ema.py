import copy
from typing import Any, Optional
from collections import OrderedDict

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay

        self.ema_model = copy.deepcopy(unwrap(model))
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def update(self, model: nn.Module, decay: Optional[float] = None) -> None:
        decay = self.decay if decay is None else decay
        # update parameters
        model_params = OrderedDict(unwrap(model).named_parameters())
        ema_params = OrderedDict(self.ema_model.named_parameters())
        for name, param in model_params.items():
            requires_grad = param.requires_grad
            param = param.detach().to(ema_params[name])
            if requires_grad:
                ema_params[name].lerp_(param, 1.0 - decay)
            else:
                ema_params[name].copy_(param)
        # copy buffers
        model_buffers = OrderedDict(unwrap(model).named_buffers())
        ema_buffers = OrderedDict(self.ema_model.named_buffers())
        for name, buffer in model_buffers.items():
            buffer = buffer.detach().to(ema_buffers[name])
            ema_buffers[name].copy_(buffer)

    def to(self, *args, **kwargs) -> "EMA":
        self.ema_model.to(*args, **kwargs)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "ema_model": self.ema_model.state_dict(),
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.ema_model.load_state_dict(state_dict["ema_model"])
        self.decay = state_dict["decay"]
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
