import time
import torch


class Monitor:
    """Monitor training speed and GPU memory."""

    def __init__(self, device: torch.device, global_step: int = 0) -> None:
        self.device = torch.device(device)

        self.last_log_time = time.monotonic()
        self.last_log_step = global_step
        torch.cuda.reset_peak_memory_stats(self.device)

    def reset(self, global_step: int) -> None:
        self.last_log_time = time.monotonic()
        self.last_log_step = global_step
        torch.cuda.reset_peak_memory_stats(self.device)

    def collect(self, global_step: int) -> dict[str, float]:
        elapsed = max(time.monotonic() - self.last_log_time, 1e-6)
        return {
            "speed": (global_step - self.last_log_step) / elapsed,
            "vram_allocated": torch.cuda.max_memory_allocated(self.device) / (1024 ** 3),
            "vram_reserved": torch.cuda.max_memory_reserved(self.device) / (1024 ** 3),
            "vram_nvml": torch.cuda.device_memory_used(self.device) / (1024 ** 3),
        }
