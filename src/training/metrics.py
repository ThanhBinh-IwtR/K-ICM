"""Evaluation metrics for kcat regression.

All metrics operate on numpy arrays. DDP-safe aggregation (all_gather before
calling these functions) is the caller's responsibility.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric result container
# ---------------------------------------------------------------------------


class MetricResult(NamedTuple):
    """Container for a complete set of regression metrics."""

    r2: float
    rmse: float
    mae: float
    spearman: float
    pearson: float


# ---------------------------------------------------------------------------
# Core compute function
# ---------------------------------------------------------------------------


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Compute all regression metrics on numpy arrays.

    Args:
        preds: Predicted values, shape [N].
        targets: Ground truth values, shape [N].

    Returns:
        Dict with keys: ``r2``, ``rmse``, ``mae``, ``spearman``, ``pearson``.

    Raises:
        ValueError: If preds and targets have different lengths or contain
            fewer than 2 samples (required for correlation coefficients).
    """
    preds = np.asarray(preds, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()

    if len(preds) != len(targets):
        raise ValueError(
            f"preds and targets must have the same length "
            f"(got {len(preds)} vs {len(targets)})"
        )
    if len(preds) < 2:
        raise ValueError(
            "Need at least 2 samples to compute metrics. "
            f"Got {len(preds)}."
        )

    r2 = float(r2_score(targets, preds))
    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    mae = float(mean_absolute_error(targets, preds))
    spearman = float(stats.spearmanr(targets, preds).statistic)
    pearson = float(stats.pearsonr(targets, preds).statistic)

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "spearman": spearman,
        "pearson": pearson,
        "pcc": pearson,    # alias for pearson
        "scc": spearman,   # alias for spearman
    }


# ---------------------------------------------------------------------------
# DDP-safe aggregation helper
# ---------------------------------------------------------------------------


def all_gather_tensors(
    local_tensor: torch.Tensor,
) -> torch.Tensor:
    """All-gather a 1-D tensor across DDP processes.

    On single-process runs (no distributed context), returns the input as-is.

    Args:
        local_tensor: 1-D FloatTensor on the local rank's device.

    Returns:
        Concatenated 1-D FloatTensor on CPU containing all ranks' values.
    """
    if not dist.is_available() or not dist.is_initialized():
        return local_tensor.detach().cpu()

    world_size = dist.get_world_size()
    if world_size == 1:
        return local_tensor.detach().cpu()

    # All ranks must share their tensor sizes before gathering.
    local_size = torch.tensor(
        [local_tensor.numel()], dtype=torch.long, device=local_tensor.device
    )
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_size = max(s.item() for s in all_sizes)
    # Pad local tensor to max_size so all_gather can use a fixed-length buffer
    padded = torch.zeros(max_size, dtype=local_tensor.dtype, device=local_tensor.device)
    padded[: local_tensor.numel()] = local_tensor.detach()

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    # Trim padding and concatenate
    result = torch.cat(
        [g[: all_sizes[i].item()] for i, g in enumerate(gathered)]
    )
    return result.cpu()


# ---------------------------------------------------------------------------
# Convenience: run metrics on torch tensors
# ---------------------------------------------------------------------------


def compute_metrics_from_tensors(
    preds: torch.Tensor,
    targets: torch.Tensor,
    gather_ddp: bool = False,
) -> dict[str, float]:
    """Compute metrics from PyTorch tensors, optionally gathering across ranks.

    Args:
        preds: FloatTensor [N].
        targets: FloatTensor [N].
        gather_ddp: If True, all-gather tensors across distributed processes
            before computing metrics (only rank 0 returns valid results).

    Returns:
        Dict with metric values (valid only on rank 0 if gather_ddp=True).
    """
    if gather_ddp:
        preds = all_gather_tensors(preds)
        targets = all_gather_tensors(targets)

    return compute_metrics(
        preds.float().numpy(),
        targets.float().numpy(),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)

    rng = np.random.default_rng(42)
    n = 100
    targets = rng.standard_normal(n)
    preds = targets + 0.3 * rng.standard_normal(n)

    metrics = compute_metrics(preds, targets)
    print("Metrics:", metrics)
    assert "r2" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "spearman" in metrics
    assert "pearson" in metrics
    assert metrics["r2"] > 0.8, f"Expected R2>0.8 for near-perfect preds, got {metrics['r2']:.4f}"

    print("metrics.py smoke test PASSED")
