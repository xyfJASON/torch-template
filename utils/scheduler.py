import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_factor(step: int, warmup_steps: int) -> float:
    return 1.0 if warmup_steps == 0 else min(float(step) / float(warmup_steps), 1.0)


class ConstantWarmupLR(LambdaLR):
    """Linearly warm up to the optimizer LR, then keep it constant."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, lr_lambda=self.lr_factor, last_epoch=last_epoch)

    def lr_factor(self, step: int) -> float:
        return warmup_factor(step, self.warmup_steps)


class LinearWarmupLR(LambdaLR):
    """Linearly warm up to the optimizer LR, then linearly anneal to a minimum LR."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, lr_lambda=self.lr_factor, last_epoch=last_epoch)

    def lr_factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return warmup_factor(step, self.warmup_steps)
        progress = min(float(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps), 1.0)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * (1.0 - progress)


class CosineWarmupLR(LambdaLR):
    """Linearly warm up the optimizer LR, then cosine anneal to a minimum LR."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, lr_lambda=self.lr_factor, last_epoch=last_epoch)

    def lr_factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return warmup_factor(step, self.warmup_steps)
        progress = min(float(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
