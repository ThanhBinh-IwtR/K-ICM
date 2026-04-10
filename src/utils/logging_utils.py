"""Logging and W&B integration utilities."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def setup_logger(
    log_dir: str | Path,
    rank: int = 0,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure root logger to write to file and stdout (rank 0 only).

    Args:
        log_dir: Directory to write log files to.
        rank: Distributed rank — only rank 0 logs to file/stdout to avoid
              duplicated output.
        level: Logging level.

    Returns:
        Configured root logger.
    """
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rank == 0:
        # Stdout handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

        # File handler
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return root


def get_wandb_logger(
    config: dict[str, Any],
    rank: int = 0,
) -> Any | None:
    """Initialise a W&B run on rank 0; return None on other ranks or if W&B
    is unavailable.

    Falls back to None gracefully — callers must check for None before logging.

    Args:
        config: Full training config dict (will be logged as W&B config).
        rank: Distributed rank.

    Returns:
        wandb run object or None.
    """
    if rank != 0:
        return None

    backend = config.get("logging", {}).get("backend", "none")
    if backend != "wandb":
        logger.info("[wandb] backend=%s — W&B disabled.", backend)
        return None

    try:
        import wandb  # noqa: PLC0415

        logging_cfg = config.get("logging", {})
        project_cfg = config.get("project", {})

        run = wandb.init(
            project=logging_cfg.get("wandb_project", "kcat-prediction"),
            entity=logging_cfg.get("wandb_entity") or None,
            name=project_cfg.get("name", "esm3-molformer-kcat"),
            config=config,
            dir=project_cfg.get("log_dir", "outputs/logs"),
            resume="allow",
        )
        logger.info("[wandb] Run initialised: %s", run.url)
        return run
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[wandb] Failed to initialise W&B (%s). Falling back to no W&B logging.",
            exc,
        )
        return None


def log_metrics(
    wandb_run: Any | None,
    metrics: dict[str, float],
    step: int,
) -> None:
    """Log a metrics dict to W&B (if available).

    Args:
        wandb_run: Active W&B run or None.
        metrics: Dict of metric_name → float.
        step: Global training step.
    """
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = setup_logger("/tmp/test_logs", rank=0)
    log.info("logging_utils.py smoke test PASSED")
    print("logging_utils.py smoke test PASSED")
