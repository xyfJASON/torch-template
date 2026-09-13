from typing import Any
from collections import OrderedDict

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class EMA:
    """Exponential moving average of model parameters.

    @crowsonkb's notes on EMA Warmup:
        If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
        to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
        gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
        at 215.4k steps).
    """
    def __init__(
        self,
        model: nn.Module,
        ema_model: nn.Module,
        decay: float = 0.9999,
        warmup: str = "none",
        inv_gamma: float = 1.0,
        power: float = 2/3,
    ) -> None:
        self.model = model
        self.ema_model = ema_model.requires_grad_(False).eval()

        self.decay = decay
        self.warmup = warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.num_updates = 0

    def get_decay(self) -> float:
        if self.warmup == "none":
            return self.decay
        elif self.warmup == "tensorflow":
            return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        elif self.warmup == "crowsonkb":
            return min(self.decay, 1 - (1 + self.num_updates / self.inv_gamma) ** (-self.power))
        else:
            raise ValueError(f"Unsupported warmup {self.warmup}")

    @torch.no_grad()
    def copy_from_model(self) -> None:
        # copy parameters
        model_params = OrderedDict(unwrap(self.model).named_parameters())
        ema_params = OrderedDict(unwrap(self.ema_model).named_parameters())
        for name, param in model_params.items():
            param = param.detach().to(ema_params[name])
            ema_params[name].copy_(param)
        # copy buffers
        model_buffers = OrderedDict(unwrap(self.model).named_buffers())
        ema_buffers = OrderedDict(unwrap(self.ema_model).named_buffers())
        for name, buffer in model_buffers.items():
            buffer = buffer.detach().to(ema_buffers[name])
            ema_buffers[name].copy_(buffer)

    @torch.no_grad()
    def update(self) -> None:
        decay = self.get_decay()
        # update parameters
        model_params = OrderedDict(unwrap(self.model).named_parameters())
        ema_params = OrderedDict(unwrap(self.ema_model).named_parameters())
        for name, param in model_params.items():
            requires_grad = param.requires_grad
            param = param.detach().to(ema_params[name])
            if requires_grad:
                ema_params[name].lerp_(param, 1.0 - decay)
            else:
                ema_params[name].copy_(param)
        # copy buffers
        model_buffers = OrderedDict(unwrap(self.model).named_buffers())
        ema_buffers = OrderedDict(unwrap(self.ema_model).named_buffers())
        for name, buffer in model_buffers.items():
            buffer = buffer.detach().to(ema_buffers[name])
            ema_buffers[name].copy_(buffer)
        self.num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.warmup = state_dict["warmup"]
        self.inv_gamma = state_dict["inv_gamma"]
        self.power = state_dict["power"]
        self.num_updates = state_dict["num_updates"]
