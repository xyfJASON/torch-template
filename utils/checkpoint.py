from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

from .rng import get_rng_state, set_rng_state
from .distributed import gather_object, get_rank, is_main_process, wait_for_everyone


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class Stateful(Protocol):
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> Any: ...


class Checkpointer:
    def __init__(
        self,
        models: Mapping[str, torch.nn.Module],
        optimizers: Optional[Mapping[str, torch.optim.Optimizer]] = None,
        schedulers: Optional[Mapping[str, Stateful]] = None,
        stateful: Optional[Mapping[str, Stateful]] = None,
    ) -> None:
        self.models = dict(models)
        self.optimizers = dict(optimizers or {})
        self.schedulers = dict(schedulers or {})
        self.stateful = dict(stateful or {})

        for name in self.optimizers:
            if name not in self.models:
                raise ValueError(f"Optimizer {name} does not exist in models")

    def save(self, path: Union[str, Path], state: Optional[Mapping[str, Any]] = None) -> None:
        rng_states = gather_object(get_rng_state())
        if is_main_process():
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "models": {name: unwrap(model).state_dict() for name, model in self.models.items()},
                "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
                "schedulers": {name: sch.state_dict() for name, sch in self.schedulers.items()},
                "stateful": {name: obj.state_dict() for name, obj in self.stateful.items()},
                "rng": rng_states,
                "state": dict(state or {}),
            }, path)
        wait_for_everyone()

    def load(self, path: Union[str, Path]) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for name, model in self.models.items():
            unwrap(model).load_state_dict(checkpoint["models"][name])
        for name, opt in self.optimizers.items():
            opt.load_state_dict(checkpoint["optimizers"][name])
        for name, sch in self.schedulers.items():
            sch.load_state_dict(checkpoint["schedulers"][name])
        for name, obj in self.stateful.items():
            obj.load_state_dict(checkpoint["stateful"][name])
        set_rng_state(checkpoint["rng"][get_rank()])
        return checkpoint["state"]


class FSDP2Checkpointer(Checkpointer):
    def save(self, path: Union[str, Path], state: Optional[Mapping[str, Any]] = None) -> None:
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state_dicts = {
            name: get_model_state_dict(model=model, options=options)
            for name, model in self.models.items()
        }
        optimizer_state_dicts = {
            name: get_optimizer_state_dict(model=self.models[name], optimizers=opt, options=options)
            for name, opt in self.optimizers.items()
        }
        rng_states = gather_object(get_rng_state())
        if is_main_process():
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "models": model_state_dicts,
                "optimizers": optimizer_state_dicts,
                "schedulers": {name: sch.state_dict() for name, sch in self.schedulers.items()},
                "stateful": {name: obj.state_dict() for name, obj in self.stateful.items()},
                "rng": rng_states,
                "state": dict(state or {}),
            }, path)
        wait_for_everyone()

    def load(self, path: Union[str, Path]) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for name, model in self.models.items():
            set_model_state_dict(
                model=model,
                model_state_dict=checkpoint["models"][name],
                options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
            )
        for name, opt in self.optimizers.items():
            set_optimizer_state_dict(
                model=self.models[name],
                optimizers=opt,
                optim_state_dict=checkpoint["optimizers"][name],
                options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False),
            )
        for name, sch in self.schedulers.items():
            sch.load_state_dict(checkpoint["schedulers"][name])
        for name, obj in self.stateful.items():
            obj.load_state_dict(checkpoint["stateful"][name])
        set_rng_state(checkpoint["rng"][get_rank()])
        return checkpoint["state"]
