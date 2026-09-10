import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from torch.utils.tensorboard import SummaryWriter

from .distributed import get_rank


class Tracker:
    def __init__(
        self,
        exp_dir: Union[str, Path],
        rank: Optional[int] = None,
    ) -> None:
        """Track metrics on rank zero; other ranks do nothing.

        Args:
            exp_dir: Experiment directory with no existing tracking output.
            rank: Global rank; defaults to RANK, or zero outside torchrun.
        """
        if rank is None:
            rank = get_rank()

        self.tensorboard = None
        if rank == 0:
            exp_dir = Path(exp_dir)
            exp_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard = SummaryWriter(log_dir=str(exp_dir / "tensorboard"))

    def log(self, prefix: str, metrics: Mapping[str, Any], step: int) -> None:
        """Log scalar metrics at a completed optimizer step.

        Args:
            prefix: Prefix to add to metric names.
            metrics: Flat mapping of names to numbers or single-element tensors.
            step: Non-negative optimizer step; repeated steps are allowed.
        """
        if "global_step" in metrics:
            raise ValueError("global_step is reserved for the tracker.")
        scalars = {
            name: float(value.item() if hasattr(value, "item") else value)
            for name, value in metrics.items()
        }
        if self.tensorboard is not None:
            for name, value in scalars.items():
                self.tensorboard.add_scalar(f"{prefix}/{name}", value, global_step=step)

    def log_image(self, prefix: str, images: Mapping[str, Image.Image], step: int) -> None:
        """Log a PIL image at a completed optimizer step.

        Args:
            prefix: Prefix to add to image names.
            images: Flat mapping of names to PIL images.
            step: Non-negative optimizer step; repeated steps are allowed.
        """
        if "global_step" in images:
            raise ValueError("global_step is reserved for the tracker.")
        if self.tensorboard is not None:
            for name, image in images.items():
                self.tensorboard.add_image(
                    tag=f"{prefix}/{name}",
                    img_tensor=np.asarray(image.convert("RGB")),
                    global_step=step,
                    dataformats="HWC",
                )

    def close(self) -> None:
        """Flush pending data and close the enabled tracking backends."""
        if self.tensorboard is not None:
            self.tensorboard.flush()
            self.tensorboard.close()
