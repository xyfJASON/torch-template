import logging
from pathlib import Path
from typing import Optional, Union

from .distributed import get_rank


def setup_logger(
    name: str = "torch-template",
    *,
    log_file: Optional[Union[str, Path]] = None,
    level: Union[int, str] = logging.INFO,
    rank: Optional[int] = None,
) -> logging.Logger:
    """Set up console and optional file logging, respecting distributed rank.

    Args:
        name: Non-root logger name. Repeated setup replaces its handlers.
        log_file: Optional file path; messages are appended in UTF-8.
        level: Logging level number or name, such as logging.INFO or "DEBUG".
        rank: Global rank.

    Returns:
        Configured standard Python logger.
    """
    if not name or name == "root":
        raise ValueError("Use a non-root logger name.")

    if rank is None:
        rank = get_rank()

    logger = logging.getLogger(name)
    logger.setLevel(level.upper() if isinstance(level, str) else level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.NOTSET if rank == 0 else logging.WARNING)
    handlers = [console]

    if log_file is not None and rank == 0:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="a", encoding="utf-8"))

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    logger.disabled = False
    return logger
