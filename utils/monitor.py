import time
import torch


class Monitor:
    """Monitor training speed and accelerator memory."""

    def __init__(self, device: torch.device, global_step: int = 0) -> None:
        self.device = torch.device(device)
        self.reset(global_step)

    def reset(self, global_step: int) -> None:
        self.last_log_time = time.monotonic()
        self.last_log_step = global_step
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def collect(self, global_step: int) -> dict[str, float]:
        elapsed = max(time.monotonic() - self.last_log_time, 1e-6)
        metrics = {"speed": (global_step - self.last_log_step) / elapsed}
        if self.device.type == "cuda":
            metrics.update({
                "vram_allocated": torch.cuda.max_memory_allocated(self.device) / (1024 ** 3),
                "vram_reserved": torch.cuda.max_memory_reserved(self.device) / (1024 ** 3),
                "vram_nvml": torch.cuda.device_memory_used(self.device) / (1024 ** 3),
            })
        elif self.device.type == "mps":
            # MPS has no peak memory stats, so report current usage
            metrics.update({
                "mps_allocated": torch.mps.current_allocated_memory() / (1024 ** 3),
                "mps_driver": torch.mps.driver_allocated_memory() / (1024 ** 3),
            })
        return metrics

    def format(self, metrics: dict[str, float]) -> str:
        items = [f"{name}: {value:.2f}GiB" for name, value in metrics.items() if name.startswith(("vram_", "mps_"))]
        items.append(f"speed: {metrics['speed']:.2f} steps/s")
        return ", ".join(items)
