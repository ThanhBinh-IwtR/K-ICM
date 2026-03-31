"""Entry point for KICMBaseModel training using PyTorch Lightning.

Launch (multi-GPU DDP):
    python train.py --config configs/train_config.yaml

Launch (resume from checkpoint):
    python train.py --config configs/train_config.yaml --resume outputs/checkpoints/last.ckpt

Launch (fast dev run — smoke test, 1 batch):
    python train.py --config configs/train_config.yaml --fast_dev_run
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Set process title early — before any heavyweight imports
try:
    from setproctitle import setproctitle

    setproctitle("kicm training")
    _pt_check = subprocess.run(
        ["pgrep", "-a", "kicm"], capture_output=True, text=True
    )
    _proctitle_result = _pt_check.stdout.strip()
except ImportError:
    _proctitle_result = "setproctitle not installed — skipping process title."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KICMBaseModel with PyTorch Lightning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to YAML training config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a Lightning .ckpt checkpoint to resume training from.",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run a single train/val/test batch as a smoke test.",
    )
    return parser.parse_args()


def _load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)

    # Deferred imports — keep ESM-3 weight loading away from arg parsing
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        TQDMProgressBar,
    )
    from pytorch_lightning.loggers import WandbLogger, CSVLogger

    from src.data import KICMDataModule
    from src.models import KICMLightningModule
    from src.utils import set_seed, validate_config

    # Basic logging setup before Lightning takes over
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _log = logging.getLogger(__name__)
    _log.info("[PROCTITLE CHECK] %s", _proctitle_result)

    # Validate config structure before building anything
    validate_config(config)

    # Tensor Core precision — trade precision for speed on L40S / A100
    torch.set_float32_matmul_precision("medium")

    # Reproducibility
    seed = config["project"]["seed"]
    set_seed(
        seed,
        deterministic=config["reproducibility"]["deterministic"],
        benchmark=config["reproducibility"]["benchmark"],
    )
    pl.seed_everything(seed, workers=True)

    # Create output directories
    Path(config["project"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["project"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["project"]["log_dir"]).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DataModule
    # ------------------------------------------------------------------
    datamodule = KICMDataModule(config)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = KICMLightningModule(config)

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    log_backend = config["logging"].get("backend", "none")
    loggers = []

    if log_backend == "wandb" and not args.fast_dev_run:
        try:
            wandb_logger = WandbLogger(
                project=config["logging"]["wandb_project"],
                entity=config["logging"].get("wandb_entity") or None,
                name=config["project"]["name"],
                config=config,
                save_dir=config["project"]["log_dir"],
            )
            loggers.append(wandb_logger)
            _log.info("[train] W&B logger initialised.")
        except Exception as exc:  # noqa: BLE001
            _log.warning("[train] W&B unavailable (%s) — falling back to CSVLogger.", exc)
            loggers.append(CSVLogger(save_dir=config["project"]["log_dir"]))
    else:
        loggers.append(CSVLogger(save_dir=config["project"]["log_dir"]))

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    train_cfg = config["training"]
    es_cfg = train_cfg["early_stopping"]

    ckpt_cfg = train_cfg["checkpoint"]
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["project"]["checkpoint_dir"],
        filename=ckpt_cfg.get("filename", "epoch{epoch:03d}-val_r2{val/r2:.4f}"),
        monitor="val/r2",
        mode=ckpt_cfg.get("mode", "max"),
        save_top_k=ckpt_cfg.get("save_top_k", 3),
        save_last=True,
        save_weights_only=True,      # prevents pickling LightningAdamW
        auto_insert_metric_name=False,
    )
    early_stop = EarlyStopping(
        monitor="val/r2",
        mode="max",
        patience=es_cfg["patience"],
        min_delta=es_cfg["min_delta"],
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(refresh_rate=10)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    num_gpus = min(train_cfg["num_gpus"], torch.cuda.device_count())
    strategy = "ddp" if num_gpus > 1 else "auto"

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="gpu",
        devices=num_gpus,
        strategy=strategy,
        precision=train_cfg["precision"],
        callbacks=[checkpoint_callback, early_stop, lr_monitor, progress_bar],
        logger=loggers,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        deterministic=config["reproducibility"]["deterministic"],
        check_val_every_n_epoch=train_cfg.get("val_every_n_epochs", 1),
        gradient_clip_val=train_cfg["gradient_clip_norm"],
        accumulate_grad_batches=train_cfg.get("gradient_accumulation_steps", 1),
        fast_dev_run=args.fast_dev_run,
    )

    # Startup banner — uses trainer.device_ids / world_size after init
    _log.info("=" * 60)
    _log.info(" KICMBaseModel — Lightning Training Session")
    _log.info(" GPUs visible: %d", torch.cuda.device_count())
    _log.info(" Trainer devices: %s", trainer.device_ids)
    _log.info(" World size: %d", trainer.world_size)
    _log.info("=" * 60)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    ckpt_path = args.resume if args.resume else None
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # ------------------------------------------------------------------
    # Test (skipped during fast_dev_run to keep the smoke test minimal)
    # ------------------------------------------------------------------
    if not args.fast_dev_run and config["data"].get("test_csv"):
        # Only rank 0 runs test — other DDP ranks would OOM trying to load
        # a second model on the same GPU.
        rank = int(os.environ.get("LOCAL_RANK", 0))
        if rank == 0:
            _log.info("[train] Building single-GPU test trainer (rank 0 only).")

            # Free DDP training memory before loading test model
            del trainer
            torch.cuda.empty_cache()

            best_ckpt = checkpoint_callback.best_model_path
            if not best_ckpt:
                _log.warning("[train] No best checkpoint — using last model weights.")
                best_ckpt = None

            _log.info("[train] Best checkpoint for test: %s", best_ckpt)

            test_trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                precision=train_cfg["precision"],
                logger=loggers,
            )
            test_trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt)


if __name__ == "__main__":
    main()
