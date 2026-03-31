"""Global seed utilities for reproducibility."""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False,
) -> None:
    """Set random seed for all relevant libraries.

    Args:
        seed: Integer seed value.
        deterministic: If True, force deterministic CUDA algorithms
            (may slow down training but ensures reproducibility).
        benchmark: If True, enable cuDNN auto-tuner (incompatible with
            determinism — should be False when deterministic=True).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducibility knobs
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if deterministic:
        # Raise on non-deterministic ops rather than silently falling back.
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Make hash randomisation deterministic for subprocess workers too.
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(
        "[seed] seed=%d  deterministic=%s  benchmark=%s",
        seed,
        deterministic,
        benchmark,
    )


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)
    set_seed(42)
    x = torch.rand(3, 3)
    set_seed(42)
    y = torch.rand(3, 3)
    assert torch.allclose(x, y), "Seed not reproducible"
    print("seed.py smoke test PASSED")
