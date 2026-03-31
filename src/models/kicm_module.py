"""KICMLightningModule — PyTorch Lightning wrapper for the KICM D2 model.

D2 LoRA: ESM-3 frozen with LoRA adapters, MoLFormer fully trainable,
B1 unidirectional cross-attention. No phase-based unfreezing, no augmentation.

Optimizer param groups (5):
    task_decay, task_no_decay           — lr.task
    molformer_decay, molformer_no_decay — lr.molformer
    esm3_lora                           — lr.esm3_lora (single group, no bias)
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl

from .kicm_model import KICMModel
from ..training.metrics import compute_metrics

logger = logging.getLogger(__name__)


class KICMLightningModule(pl.LightningModule):
    """Lightning module wrapping KICMModel with LoRA-based ESM-3 adaptation.

    Args:
        config: Full training config dict (same structure as train_config.yaml).
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters({"config": config})

        # Build the underlying model (freeze + LoRA applied inside KICMModel.__init__)
        self.model = KICMModel(config)

        # Loss
        train_cfg = config["training"]
        self.criterion = nn.SmoothL1Loss(beta=train_cfg["huber_beta"])

        # Epoch-level collection buffers
        self.val_preds: list[Tensor] = []
        self.val_targets: list[Tensor] = []
        self.test_preds: list[Tensor] = []
        self.test_targets: list[Tensor] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sequence_tokens: Tensor,
        padding_mask: Tensor,
        smiles_list: list[str],
    ) -> Tensor:
        return self.model(sequence_tokens, padding_mask, smiles_list)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        sequence_tokens = batch["sequence_tokens"]
        padding_mask = batch["padding_mask"]
        smiles_list = batch["smiles"]
        targets = batch["targets"].squeeze(-1)  # [B,1] -> [B]
        batch_size = targets.size(0)

        preds = self(sequence_tokens, padding_mask, smiles_list)
        loss = self.criterion(preds, targets)

        self.log("train/loss", loss,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=batch_size)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        preds = self(
            batch["sequence_tokens"],
            batch["padding_mask"],
            batch["smiles"],
        )
        targets = batch["targets"].squeeze(-1)
        self.val_preds.append(preds.detach().cpu().reshape(-1))
        self.val_targets.append(targets.detach().cpu().reshape(-1))

    def on_validation_epoch_end(self) -> None:
        if not self.val_preds:
            return

        local_preds = torch.cat(self.val_preds).float()
        local_targets = torch.cat(self.val_targets).float()

        gathered_preds = self.all_gather(local_preds)
        gathered_targets = self.all_gather(local_targets)

        all_preds = gathered_preds.reshape(-1).cpu().numpy()
        all_targets = gathered_targets.reshape(-1).cpu().numpy()

        try:
            metrics = compute_metrics(all_preds, all_targets)
        except ValueError as exc:
            logger.warning("[KICMLightningModule] Could not compute val metrics: %s", exc)
            self.val_preds.clear()
            self.val_targets.clear()
            return

        self.log("val/r2", metrics["r2"], sync_dist=False, prog_bar=True)
        self.log("val/rmse", metrics["rmse"], sync_dist=False, prog_bar=True)
        self.log("val/mae", metrics["mae"], sync_dist=False)
        self.log("val/pcc", metrics["pearson"], sync_dist=False)
        self.log("val/scc", metrics["spearman"], sync_dist=False)

        logger.info(
            "[KICMLightningModule] val  r2=%.4f  rmse=%.4f  pcc=%.4f  scc=%.4f",
            metrics["r2"], metrics["rmse"], metrics["pearson"], metrics["spearman"],
        )

        self.val_preds.clear()
        self.val_targets.clear()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch: dict, batch_idx: int) -> None:
        preds = self(
            batch["sequence_tokens"],
            batch["padding_mask"],
            batch["smiles"],
        )
        targets = batch["targets"].squeeze(-1)
        self.test_preds.append(preds.detach().cpu().reshape(-1))
        self.test_targets.append(targets.detach().cpu().reshape(-1))

        if self.global_rank == 0 and batch_idx == 0 and self.logger is not None:
            try:
                import wandb
                table = wandb.Table(
                    columns=["true_log10_value", "pred_log10_value", "error"],
                    data=[
                        [t, p, round(p - t, 4)]
                        for t, p in zip(
                            targets[:20].tolist(), preds[:20].detach().cpu().tolist()
                        )
                    ],
                )
                self.logger.experiment.log({"test/predictions_sample": table})
            except Exception:
                pass

    def on_test_epoch_end(self) -> None:
        if not self.test_preds:
            return

        local_preds = torch.cat(self.test_preds).float()
        local_targets = torch.cat(self.test_targets).float()

        gathered_preds = self.all_gather(local_preds)
        gathered_targets = self.all_gather(local_targets)
        all_preds = gathered_preds.reshape(-1).cpu().numpy()
        all_targets = gathered_targets.reshape(-1).cpu().numpy()

        try:
            metrics = compute_metrics(all_preds, all_targets)
        except ValueError as exc:
            logger.warning("[KICMLightningModule] Could not compute test metrics: %s", exc)
            self.test_preds.clear()
            self.test_targets.clear()
            return

        self.log("test/r2", metrics["r2"], sync_dist=False)
        self.log("test/rmse", metrics["rmse"], sync_dist=False)
        self.log("test/mae", metrics["mae"], sync_dist=False)
        self.log("test/pcc", metrics["pearson"], sync_dist=False)
        self.log("test/scc", metrics["spearman"], sync_dist=False)

        logger.info(
            "[KICMLightningModule] test  r2=%.4f  rmse=%.4f  pcc=%.4f  scc=%.4f",
            metrics["r2"], metrics["rmse"], metrics["pearson"], metrics["spearman"],
        )

        predictions_csv = self.config.get("data", {}).get("predictions_csv")
        if predictions_csv and self.global_rank == 0:
            self._save_predictions_csv(all_preds, predictions_csv)

        self.test_preds.clear()
        self.test_targets.clear()

    def _save_predictions_csv(self, preds: "np.ndarray", output_path: str) -> None:
        """Save test predictions merged with original test CSV for OOD eval."""
        import numpy as np  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415

        test_csv = self.config["data"]["test_csv"]
        target_col = self.config["data"]["target_col"]
        smiles_col = self.config["data"]["smiles_col"]
        sequence_col = self.config["data"]["sequence_col"]

        df = pd.read_csv(test_csv)

        valid_mask = df[[sequence_col, smiles_col, target_col]].notna().all(axis=1)
        df = df[valid_mask].copy()

        max_seq_len = self.config["data"]["max_seq_len"]
        df = df[df[sequence_col].str.len() <= max_seq_len].copy()

        from rdkit import Chem  # noqa: PLC0415
        valid_smiles = df[smiles_col].apply(
            lambda s: isinstance(s, str) and s.strip() != "" and Chem.MolFromSmiles(s) is not None
        )
        df = df[valid_smiles].reset_index(drop=True)

        if len(df) != len(preds):
            logger.warning(
                "[KICMLightningModule] Prediction count (%d) != filtered rows (%d). "
                "Skipping predictions CSV.",
                len(preds), len(df),
            )
            return

        df["prediction"] = preds
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("[KICMLightningModule] Saved %d predictions to %s", len(df), out)

    # ------------------------------------------------------------------
    # Optimiser & Scheduler
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cosine_lambda(warmup: int, total: int, min_ratio: float):
        """Return a lr_lambda that does linear warmup then cosine decay with floor."""
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total - warmup, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(cosine, min_ratio)
        return lr_lambda

    @staticmethod
    def _is_no_decay_param(name: str) -> bool:
        """Return True if parameter should be excluded from weight decay."""
        if name.endswith(".bias"):
            return True
        parts = name.split(".")
        return any("norm" in part.lower() for part in parts)

    def configure_optimizers(self) -> dict[str, Any]:
        train_cfg = self.config["training"]
        lr_cfg = train_cfg["lr"]
        wd = train_cfg["weight_decay"]

        # LoRA adapters have no bias (bias="none" in LoraConfig)
        # All LoRA params are 2D matrices — all go to decay group
        lora_params = list(self.model.esm3_encoder.lora_parameters())
        lora_ids = {id(p) for p in lora_params}

        # Identify MoLFormer parameters
        molformer_param_ids = set(id(p) for p in self.model.substrate_encoder.parameters())

        task_decay, task_no_decay = [], []
        molformer_decay, molformer_no_decay = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in lora_ids:
                continue  # handled separately as single group
            elif id(param) in molformer_param_ids:
                if self._is_no_decay_param(name):
                    molformer_no_decay.append(param)
                else:
                    molformer_decay.append(param)
            else:
                if self._is_no_decay_param(name):
                    task_no_decay.append(param)
                else:
                    task_decay.append(param)

        lr_task = lr_cfg["task"]
        lr_molformer = lr_cfg["molformer"]
        lr_esm3_lora = lr_cfg["esm3_lora"]

        param_groups = [
            {"params": task_decay,      "lr": lr_task,       "weight_decay": wd,  "name": "task_decay"},
            {"params": task_no_decay,   "lr": lr_task,       "weight_decay": 0.0, "name": "task_no_decay"},
            {"params": molformer_decay, "lr": lr_molformer,  "weight_decay": wd,  "name": "molformer_decay"},
            {"params": molformer_no_decay, "lr": lr_molformer, "weight_decay": 0.0, "name": "molformer_no_decay"},
            {"params": lora_params,     "lr": lr_esm3_lora,  "weight_decay": wd,  "name": "esm3_lora"},
            # Note: single group because LoRA has no bias/norm params (bias="none")
        ]

        logger.info(
            "[KICMLightningModule] Param groups: "
            "task_decay=%d  task_no_decay=%d  "
            "molformer_decay=%d  molformer_no_decay=%d  "
            "esm3_lora=%d",
            len(task_decay), len(task_no_decay),
            len(molformer_decay), len(molformer_no_decay),
            len(lora_params),
        )

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=wd,
            eps=train_cfg["adam_eps"],
            betas=tuple(train_cfg["adam_betas"]),
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = train_cfg["scheduler"]["warmup_steps"]
        min_ratio = train_cfg["scheduler"]["min_lr_ratio"]

        lr_lambda = self._make_cosine_lambda(warmup_steps, total_steps, min_ratio)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=[lr_lambda] * len(optimizer.param_groups),
        )

        logger.info(
            "[KICMLightningModule] Optimizer: AdamW  total_steps=%d  warmup=%d  "
            "lr_task=%.2e  lr_molformer=%.2e  lr_esm3_lora=%.2e",
            total_steps, warmup_steps, lr_task, lr_molformer, lr_esm3_lora,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Gradient monitoring
    # ------------------------------------------------------------------

    def on_after_backward(self) -> None:
        if self.global_step % 100 == 0:
            for name, param in self.model.named_parameters():
                if param.grad is not None and "lora" in name:
                    self.log("grad_norm/esm3_lora", param.grad.norm().item(), on_step=True)
                    break

    # ------------------------------------------------------------------
    # GPU memory & LR logging
    # ------------------------------------------------------------------

    def on_train_epoch_end(self) -> None:
        epoch = self.current_epoch
        optimizer = self.optimizers()

        lr_task = next(
            (pg["lr"] for pg in optimizer.param_groups if pg.get("name", "").startswith("task")),
            0.0,
        )
        lr_esm3 = next(
            (pg["lr"] for pg in optimizer.param_groups if pg.get("name", "").startswith("esm3")),
            0.0,
        )
        lr_molformer = next(
            (pg["lr"] for pg in optimizer.param_groups if pg.get("name", "").startswith("molformer")),
            0.0,
        )

        self.log("lr/task", lr_task, on_step=False, on_epoch=True)
        self.log("lr/esm3_lora", lr_esm3, on_step=False, on_epoch=True)
        self.log("lr/molformer", lr_molformer, on_step=False, on_epoch=True)

        if torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            self.log("gpu/memory_gb", mem_gb, sync_dist=True)
            torch.cuda.reset_peak_memory_stats()
        else:
            mem_gb = 0.0

        print(
            f"=== Epoch {epoch} "
            f"| lr_task={lr_task:.2e} | lr_mol={lr_molformer:.2e} "
            f"| lr_lora={lr_esm3:.2e} | GPU_mem={mem_gb:.1f}GB ==="
        )
        logger.info(
            "=== Epoch %d | lr_task=%.2e | lr_mol=%.2e | lr_lora=%.2e | GPU_mem=%.1fGB ===",
            epoch, lr_task, lr_molformer, lr_esm3, mem_gb,
        )
