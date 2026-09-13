import json
import math
import time
import argparse
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from models.dit import DiT
from utils.tracker import Tracker
from utils.logger import setup_logger
from utils.data import get_data_iterator
from utils.ema import EMA
from utils.optimizer import get_param_groups
from utils.checkpoint import FSDP2Checkpointer
from utils.rng import seed_everything, seed_worker
from utils.scheduler import (
    ConstantWarmupLR,
    LinearWarmupLR,
    CosineWarmupLR,
)
from utils.distributed import (
    broadcast_object,
    cleanup,
    gather_tensor,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    reduce_tensor,
    setup_distributed,
    wait_for_everyone,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=Path, required=True,
        help="Configuration yaml file",
    )
    parser.add_argument(
        "-e", "--exp-dir",
        type=Path, default=None,
        help="Experiment directory",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=Path, default=None,
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--set",
        dest="overrides", action="append", default=[],
        help="Override or add config entries",
    )
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args = get_parser().parse_args()
    conf_base = OmegaConf.load(args.config)
    conf_overrides = OmegaConf.from_dotlist(args.overrides)
    conf = OmegaConf.merge(conf_base, conf_overrides)

    # SETUP DISTRIBUTED
    device = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = None
    if is_main_process():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_dir = args.exp_dir or Path("runs") / timestamp
        exp_dir = str(exp_dir.expanduser().resolve())
    exp_dir = Path(broadcast_object(exp_dir))
    if is_main_process():
        exp_dir.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(config=conf, f=exp_dir / "config.yaml")
        with (exp_dir / "args.json").open("w", encoding="utf-8") as f:
            json.dump({
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }, f, indent=4)
    wait_for_everyone()

    # INITIALIZE LOGGER AND TRACKER
    logger = setup_logger(log_file=exp_dir / "output.log", rank=rank)
    tracker = Tracker(exp_dir=exp_dir, rank=rank)

    # PREPARE MIXED PRECISION
    precision = str(conf.training.precision).lower()
    if precision not in ["fp32", "bf16"]:
        raise ValueError(f"Unsupported precision: {precision}")
    mp_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    mp_enabled = precision == "bf16"

    # PRINT SETUP INFO
    logger.info("=" * 19 + " System Info " + "=" * 18)
    logger.info(f"Using device: {device}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Number of processes: {world_size}")
    logger.info(f"Distributed mode: {is_dist_avail_and_initialized()}")
    logger.info(f"Mixed precision (bf16): {mp_enabled}")
    if len(args.overrides) > 0:
        logger.info("=" * 19 + " Config Info " + "=" * 18)
        for item in args.overrides:
            key, _ = item.split("=", 1)
            old_value = OmegaConf.select(conf_base, key)
            new_value = OmegaConf.select(conf, key)
            if old_value is None:
                logger.info(f"Override: {key}: <NEW> {new_value}")
            else:
                logger.info(f"Override: {key}: {old_value} -> {new_value}")
    wait_for_everyone()

    # BUILD DATASET
    dataset = MNIST(
        root=str(Path(str(conf.data.root)).expanduser()),
        train=True,
        transform=T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)]),
    )

    # BUILD DATALOADER
    global_batch_size = int(conf.training.global_batch_size)
    micro_batch_size = int(conf.training.micro_batch_size)
    if micro_batch_size == -1:
        assert global_batch_size % world_size == 0
        micro_batch_size = global_batch_size // world_size
        grad_acc_steps = 1
    else:
        batch_per_forward = micro_batch_size * world_size
        assert global_batch_size % batch_per_forward == 0
        grad_acc_steps = global_batch_size // batch_per_forward
    datasampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=micro_batch_size,
        sampler=datasampler,
        num_workers=int(conf.data.num_workers),
        pin_memory=bool(conf.data.get("pin_memory", True)),
        prefetch_factor=int(conf.data.get("prefetch_factor", 2)),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + rank),
    )
    if len(dataloader) == 0:
        raise ValueError("DataLoader is empty.")
    logger.info("=" * 19 + " Data Info " + "=" * 20)
    logger.info(f"Size of dataset: {len(dataset)}")
    logger.info(f"Global batch size: {global_batch_size}")
    logger.info(f"Micro batch size: {micro_batch_size}")
    logger.info(f"Gradient accumulation: {grad_acc_steps}")

    # SEED EVERYTHING (SAME ON EVERY RANK FOR MODEL INITIALIZATION)
    seed_everything(args.seed)

    # BUILD MODEL AND EMA MODEL
    model_kwargs: dict = OmegaConf.to_container(conf.model)
    model_kwargs.pop("type", None)
    if conf.model.type == "dit":
        model = DiT(**model_kwargs)
        ema_model = DiT(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model: {conf.model.type}")
    logger.info("=" * 19 + " Model Info " + "=" * 19)
    logger.info(f"Built model: {model.__class__.__name__}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # FULLY SHARD MODEL
    fsdp_kwargs = {}
    if mp_enabled:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    for block in model.blocks:
        fully_shard(block, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    for block in ema_model.blocks:
        fully_shard(block, **fsdp_kwargs)
    fully_shard(ema_model, **fsdp_kwargs)

    # BUILD EMA MANAGER AND COPY WEIGHTS
    ema = EMA(model, ema_model, **conf.ema)
    ema.copy_from_model()

    # SEED EVERYTHING (DIFFERENT SEED PER RANK)
    seed_everything(args.seed, rank=rank)

    # BUILD OPTIMIZER
    param_groups = get_param_groups(
        model=model,
        weight_decay=float(conf.optimizer.weight_decay),
    )
    if str(conf.optimizer.type).lower() == "adam":
        optimizer = torch.optim.Adam(
            params=param_groups,
            lr=float(conf.optimizer.lr),
            betas=tuple(conf.optimizer.betas),
            weight_decay=float(conf.optimizer.weight_decay),
        )
    elif str(conf.optimizer.type).lower() == "adamw":
        optimizer = torch.optim.AdamW(
            params=param_groups,
            lr=float(conf.optimizer.lr),
            betas=tuple(conf.optimizer.betas),
            weight_decay=float(conf.optimizer.weight_decay),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {conf.optimizer.type}")

    # BUILD SCHEDULER
    if str(conf.scheduler.type).lower() == "constant-warmup":
        scheduler = ConstantWarmupLR(
            optimizer=optimizer,
            warmup_steps=int(conf.scheduler.warmup_steps),
        )
    elif str(conf.scheduler.type).lower() == "linear-warmup":
        scheduler = LinearWarmupLR(
            optimizer=optimizer,
            warmup_steps=int(conf.scheduler.warmup_steps),
            total_steps=int(conf.training.max_steps),
            min_lr_ratio=float(conf.scheduler.get("min_lr_ratio", 0.0)),
        )
    elif str(conf.scheduler.type).lower() == "cosine-warmup":
        scheduler = CosineWarmupLR(
            optimizer=optimizer,
            warmup_steps=int(conf.scheduler.warmup_steps),
            total_steps=int(conf.training.max_steps),
            min_lr_ratio=float(conf.scheduler.get("min_lr_ratio", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported scheduler: {conf.scheduler.type}")
    logger.info("=" * 15 + " Optimization Info " + "=" * 16)
    logger.info(f"Learning rate: {conf.optimizer.lr}")
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Scheduler: {scheduler.__class__.__name__}")

    # PREPARE CHECKPOINTER
    checkpointer = FSDP2Checkpointer(
        models={"unet": model, "unet_ema": ema_model},
        optimizers={"unet": optimizer},
        schedulers={"unet": scheduler},
        stateful={"ema": ema},
    )

    # RESUME
    global_step = 0
    consumed_batches = 0
    logger.info("=" * 17 + " Training Info " + "=" * 18)
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        state = checkpointer.load(resume_path)
        global_step = state["global_step"]
        consumed_batches = state["consumed_batches"]
        logger.info(f"Successfully resumed from {resume_path}")
        logger.info(f"Restart training at step {global_step}")

    # START TRAINING
    logger.info("Start training...")
    last_log_time = time.monotonic()
    last_log_step = global_step
    dataiter = get_data_iterator(
        dataloader=dataloader,
        datasampler=datasampler,
        consumed_batches=consumed_batches,
    )

    while global_step < conf.training.max_steps:
        model.train()

        # zero gradients
        optimizer.zero_grad()

        # accumulate gradients
        accumulated_loss = torch.tensor(0., device=device)
        for grad_acc_index in range(grad_acc_steps):
            # fetch a micro batch
            images, _ = next(dataiter)
            images = images.to(device)
            consumed_batches += 1

            # sync/no-sync gradients
            should_sync = (grad_acc_index == grad_acc_steps - 1)
            model.set_requires_gradient_sync(should_sync)
            # forward (within autocast context)
            with torch.autocast(device.type, dtype=mp_dtype, enabled=mp_enabled):
                t = torch.rand(images.shape[0], device=device)
                t_broadcast = t.reshape(images.shape[0], *([1] * (images.ndim - 1)))
                noise = torch.randn_like(images)
                xt = (1. - t_broadcast) * images + t_broadcast * noise
                v_target = noise - images
                v_pred = model(xt, t)
                loss = F.mse_loss(v_pred, v_target)
            # backward (loss divided by accumulation steps)
            loss = loss / grad_acc_steps
            accumulated_loss += loss.detach()
            loss.backward()

        # update gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=conf.training.gradient_clip,
        )
        optimizer.step()
        ema.update()
        scheduler.step()
        global_step += 1

        # logging
        if global_step % conf.logging.every_steps == 0 or global_step == 1:
            mean_loss = reduce_tensor(accumulated_loss)
            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.full_tensor()
            lr = optimizer.param_groups[0]["lr"]
            elapsed = max(time.monotonic() - last_log_time, 1e-6)
            steps_per_second = (global_step - last_log_step) / elapsed
            vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            vram = reduce_tensor(torch.tensor(vram, device=device), op="max")
            torch.cuda.reset_peak_memory_stats(device)
            logger.info(
                f"step: {global_step}, "
                f"loss: {mean_loss.item():.6f}, "
                f"grad_norm: {grad_norm:.6f}, "
                f"lr: {lr:.2e} | "
                f"vram: {vram.item():.2f}GiB, "
                f"training_speed: {steps_per_second:.2f} steps/s"
            )
            tracker.log(
                prefix="Train",
                metrics={
                    "loss": mean_loss,
                    "grad_norm": grad_norm,
                    "lr": lr,
                    "vram": vram,
                    "training_speed": steps_per_second,
                },
                step=global_step,
            )
            last_log_time = time.monotonic()
            last_log_step = global_step
        model.eval()

        # sampling
        if global_step % conf.sampling.every_steps == 0 or global_step == 1:
            num_samples = math.ceil(conf.sampling.num_samples / world_size)
            with torch.no_grad(), torch.autocast(device.type, dtype=mp_dtype, enabled=mp_enabled):
                timesteps = torch.linspace(1, 0, conf.sampling.nfe + 1, device=device)
                generator = torch.Generator(device=device).manual_seed(args.seed + rank)
                samples_shape = (num_samples, *conf.data.image_shape)
                samples = torch.randn(samples_shape, device=device, generator=generator)
                for t, t_prev in zip(timesteps[:-1], timesteps[1:]):
                    v_pred = ema_model(samples, t.repeat((num_samples, )))
                    samples = samples - v_pred * (t - t_prev)
                ema_model.reshard()  # restore root ema parameters to DTensors
                samples = torch.cat(gather_tensor(samples), dim=0)[:conf.sampling.num_samples]
                samples = (samples.float().clamp(-1, 1).cpu() + 1) / 2
            columns = max(1, math.ceil(math.sqrt(conf.sampling.num_samples)))
            grid = to_pil_image(make_grid(samples, nrow=columns))
            tracker.log_image("Train", images={"generated": grid}, step=global_step)
            wait_for_everyone()

        # save checkpoint
        if global_step % conf.checkpoint.every_steps == 0 or global_step == 1:
            checkpointer.save(
                path=exp_dir / "checkpoints" / f"step_{global_step:08d}.pt",
                state={"global_step": global_step, "consumed_batches": consumed_batches},
            )
            wait_for_everyone()

    # save the last checkpoint
    if global_step % conf.checkpoint.every_steps != 0:
        checkpointer.save(
            path=exp_dir / "checkpoints" / f"step_{global_step:08d}.pt",
            state={"global_step": global_step, "consumed_batches": consumed_batches},
        )
    wait_for_everyone()

    # END OF TRAINING
    logger.info("End of training")
    tracker.close()
    cleanup()


if __name__ == "__main__":
    main()
