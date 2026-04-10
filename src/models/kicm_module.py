"""KICMLightningModule — PyTorch Lightning wrapper for Phase E.

Phase E: ESM-3 (frozen, optional LoRA) + BioT5+ (trainable) + Multi-Query CA + FiLM.

Optimizer param groups (dynamic):
    task_decay, task_no_decay           — lr.task  (incl. Scalar Mix layer_weights)
    biot5_decay, biot5_no_decay         — lr.biot5
    esm3_lora                           — lr.esm3_lora (only when enabled)
"""
from __future__ import annotations

import logging
import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from torchmetrics import MeanMetric

from .kicm_model import KICMModel
from ..training.metrics import compute_metrics, all_gather_tensors

logger = logging.getLogger(__name__)


class KICMLightningModule(pl.LightningModule):
    """Lightning module wrapping KICMModel for Phase E.

    Args:
        config: Full training config dict (same structure as train_config.yaml).
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters({"config": config})

        # Build the underlying model (freeze + LoRA applied inside KICMModel.__init__)
        self.model = KICMModel(config)

        # Loss — Huber(delta=4.0) (S14-01)
        # Quadratic for |error|≤4, linear for |error|>4 — no dead gradient zones.
        # delta=4.0 matches previous clamp threshold; MSE+clamp replaced (S12-01→S14-01).
        # self.criterion removed: loss computed inline in training_step via F.huber_loss

        # Epoch-level collection buffers
        self.val_preds: list[Tensor] = []
        self.val_targets: list[Tensor] = []
        self.test_preds: list[Tensor] = []
        self.test_targets: list[Tensor] = []

        # Epoch-level condition coverage metrics (avoid noisy per-step logging)
        # sync_on_compute=True (default): all-reduces across ranks during .compute()
        # at epoch end — one ALLREDUCE per epoch, not a desync risk.
        # (S10-01 REVERT of Deviation 131: default sync restored;
        #  root cause of S9 DDP issue was S7-01, already fixed.)
        self.ph_coverage_metric = MeanMetric()   # default sync_on_compute=True — all-reduces at epoch end
        self.temp_coverage_metric = MeanMetric() # default sync_on_compute=True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sequence_tokens: Tensor,
        esm_pad_mask: Tensor,
        biot5_input_ids: Tensor,
        biot5_attention_mask: Tensor,
        ph: Tensor,
        temp: Tensor,
        ph_known: Tensor,
        temp_known: Tensor,
        cofactor_fp: Tensor,
    ) -> Tensor:
        return self.model(
            sequence_tokens, esm_pad_mask,
            biot5_input_ids, biot5_attention_mask,
            ph, temp, ph_known, temp_known,
            cofactor_fp,
        )

    def _forward_batch(self, batch: dict) -> Tensor:
        """Extract all 9 fields from batch dict and call forward."""
        _required = {"sequence_tokens", "esm_pad_mask", "biot5_input_ids",
                     "biot5_attention_mask", "ph", "temp", "ph_known", "temp_known",
                     "cofactor_fp"}
        missing = _required - set(batch.keys())
        if missing:
            raise ValueError(f"_forward_batch: missing batch keys: {missing}")
        return self(
            sequence_tokens=batch["sequence_tokens"],
            esm_pad_mask=batch["esm_pad_mask"],
            biot5_input_ids=batch["biot5_input_ids"],
            biot5_attention_mask=batch["biot5_attention_mask"],
            ph=batch["ph"],
            temp=batch["temp"],
            ph_known=batch["ph_known"],
            temp_known=batch["temp_known"],
            cofactor_fp=batch["cofactor_fp"],
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        preds = self._forward_batch(batch)
        targets = batch["targets"].reshape(-1)  # [B] safe for B=1
        batch_size = targets.size(0)

        # Huber(delta=4.0): quadratic for |error|≤4, linear for |error|>4 (S14-01)
        # Preserves gradient signal for outliers — non-zero linear gradient at |error|>4.
        # delta=4.0 matches previous MSE+outlier-cap threshold — same robustness, no dead zones.
        loss = F.huber_loss(preds, targets, delta=4.0, reduction='mean')

        # S10-12: train/loss step-level sync is safe (on_step=True, sync_dist=True).
        # on_epoch=True accumulation must NOT use sync_dist=True — Lightning queues
        # ALLREDUCE to fire at epoch boundary, colliding with MeanMetric ALLGATHER
        # in on_train_epoch_end (→ SeqNum mismatch → 30-min NCCL timeout).
        # Architectural rule: in training_step, sync_dist=True is ONLY safe with
        # on_step=True, on_epoch=False. Epoch-level sync belongs in on_train_epoch_end.
        # train/loss epoch average is computed from rank-0 step values — sufficient
        # for monitoring purposes (loss curves don't need cross-rank epoch sync).
        self.log("train/loss", loss,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=False,
                 batch_size=batch_size)

        # Condition coverage — accumulate over epoch via MeanMetric
        self.ph_coverage_metric.update(batch["ph_known"].float())
        self.temp_coverage_metric.update(batch["temp_known"].float())

        # FiLM gamma/beta statistics (from last forward pass, detached)
        # sync_dist=False: FiLM stats are diagnostic — rank-0 values are sufficient.
        # on_epoch=True + sync_dist=True in training_step causes NCCL desync (S10-12).
        if hasattr(self.model, "_last_gamma"):
            self.log("film/gamma_mean", self.model._last_gamma.mean(),
                     on_step=False, on_epoch=True, sync_dist=False, batch_size=batch_size)
            self.log("film/gamma_std", self.model._last_gamma.std(),
                     on_step=False, on_epoch=True, sync_dist=False, batch_size=batch_size)
            self.log("film/beta_mean", self.model._last_beta.mean(),
                     on_step=False, on_epoch=True, sync_dist=False, batch_size=batch_size)
            self.log("film/beta_std", self.model._last_beta.std(),
                     on_step=False, on_epoch=True, sync_dist=False, batch_size=batch_size)
        if hasattr(self.model, "_last_gamma2"):
            self.log("film/gamma2_mean", self.model._last_gamma2.mean(),
                     on_step=False, on_epoch=True, sync_dist=False, batch_size=batch_size)
            self.log("film/gamma2_std", self.model._last_gamma2.std(),
                     on_step=False, on_epoch=True, sync_dist=False, batch_size=batch_size)
            self.log("film/beta2_mean", self.model._last_beta2.mean(),
                     on_step=False, on_epoch=True, sync_dist=False, batch_size=batch_size)
            self.log("film/beta2_std", self.model._last_beta2.std(),
                     on_step=False, on_epoch=True, sync_dist=False, batch_size=batch_size)

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        preds = self._forward_batch(batch)
        targets = batch["targets"].reshape(-1)  # [B] safe for B=1
        self.val_preds.append(preds.detach().cpu().reshape(-1))
        self.val_targets.append(targets.detach().cpu().reshape(-1))

    def on_validation_epoch_end(self) -> None:
        # S7-01 FIX: Always participate in all_gather collectives even with
        # empty local data to prevent NCCL desync across DDP ranks.
        local_preds = (
            torch.cat(self.val_preds).float()
            if self.val_preds
            else torch.tensor([], dtype=torch.float32)
        )
        local_targets = (
            torch.cat(self.val_targets).float()
            if self.val_targets
            else torch.tensor([], dtype=torch.float32)
        )

        all_preds = all_gather_tensors(local_preds).numpy()
        all_targets = all_gather_tensors(local_targets).numpy()

        self.val_preds.clear()
        self.val_targets.clear()

        if len(all_preds) < 2:
            return

        try:
            metrics = compute_metrics(all_preds, all_targets)
        except ValueError as exc:
            logger.warning("[KICMLightningModule] Could not compute val metrics: %s", exc)
            return

        # Metrics are computed on all-gathered data — identical on all ranks.
        # No DDP sync needed here (would double-sync and risk NCCL desync).
        self.log("val/r2", metrics["r2"], prog_bar=True)
        self.log("val/rmse", metrics["rmse"], prog_bar=True)
        self.log("val/mae", metrics["mae"])
        self.log("val/pcc", metrics["pearson"])
        self.log("val/scc", metrics["spearman"])

        logger.info(
            "[KICMLightningModule] val  r2=%.4f  rmse=%.4f  pcc=%.4f  scc=%.4f",
            metrics["r2"], metrics["rmse"], metrics["pearson"], metrics["spearman"],
        )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch: dict, batch_idx: int) -> None:
        preds = self._forward_batch(batch)
        targets = batch["targets"].reshape(-1)  # [B] safe for B=1
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
        # S7-01 FIX: Always participate in all_gather collectives even with
        # empty local data to prevent NCCL desync across DDP ranks.
        local_preds = (
            torch.cat(self.test_preds).float()
            if self.test_preds
            else torch.tensor([], dtype=torch.float32)
        )
        local_targets = (
            torch.cat(self.test_targets).float()
            if self.test_targets
            else torch.tensor([], dtype=torch.float32)
        )

        all_preds = all_gather_tensors(local_preds).numpy()
        all_targets = all_gather_tensors(local_targets).numpy()

        self.test_preds.clear()
        self.test_targets.clear()

        if len(all_preds) < 2:
            return

        try:
            metrics = compute_metrics(all_preds, all_targets)
        except ValueError as exc:
            logger.warning("[KICMLightningModule] Could not compute test metrics: %s", exc)
            return

        # Metrics are computed on all-gathered data — identical on all ranks
        self.log("test/r2", metrics["r2"])
        self.log("test/rmse", metrics["rmse"])
        self.log("test/mae", metrics["mae"])
        self.log("test/pcc", metrics["pearson"])
        self.log("test/scc", metrics["spearman"])

        logger.info(
            "[KICMLightningModule] test  r2=%.4f  rmse=%.4f  pcc=%.4f  scc=%.4f",
            metrics["r2"], metrics["rmse"], metrics["pearson"], metrics["spearman"],
        )

        predictions_csv = self.config.get("data", {}).get("predictions_csv")
        if predictions_csv and self.global_rank == 0:
            self._save_predictions_csv(all_preds, predictions_csv)

    def _save_predictions_csv(self, preds: "np.ndarray", output_path: str) -> None:
        """Save test predictions merged with original test CSV for OOD eval."""
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
        from src.data.cofactor_registry import separate_reactants  # noqa: PLC0415

        def _is_valid_smiles(raw_smiles: str) -> bool:
            substrate_smiles, _, _ = separate_reactants(str(raw_smiles))
            mol = Chem.MolFromSmiles(substrate_smiles)
            return mol is not None

        valid_smiles = df[smiles_col].apply(
            lambda s: isinstance(s, str) and s.strip() != "" and _is_valid_smiles(s)
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

    _NO_DECAY_NAME_PARTS = frozenset({
        "norm", "norm1", "norm2", "ffn_norm", "input_norm",
        "layernorm", "layer_norm", "batch_norm",
        "protein_proj_norm",   # cross_attn.protein_proj_norm (S15-01 — Sequential decomposed)
        "residual_norm",       # mlp_head.residual_norm (S15-02)
    })

    @staticmethod
    def _is_no_decay_param(name: str) -> bool:
        """Return True if parameter should be excluded from weight decay.

        Matches: bias parameters, and parameters belonging to normalization layers.
        Uses exact part matching (split on '.') to avoid substring false positives.
        """
        if name.endswith(".bias"):
            return True
        parts = name.split(".")
        return any(part in KICMLightningModule._NO_DECAY_NAME_PARTS for part in parts)

    def configure_optimizers(self) -> dict[str, Any]:
        train_cfg = self.config["training"]
        lr_cfg = train_cfg["lr"]
        wd = train_cfg["weight_decay"]
        model_cfg = self.config.get("model", {})

        # ESM3 LoRA params (if enabled)
        esm3_lora_enabled = model_cfg.get("esm3_lora", {}).get("enabled", False)
        lora_params = []
        lora_ids: set[int] = set()
        if esm3_lora_enabled:
            lora_params = list(self.model.esm3_encoder.lora_parameters())
            lora_ids = {id(p) for p in lora_params}

        # BioT5+ params (always trainable in Phase E)
        biot5_param_ids = {id(p) for p in self.model.substrate_encoder.parameters()}

        task_decay, task_no_decay = [], []
        biot5_decay, biot5_no_decay = [], []

        # Scalar Mix layer_weights: task-head-level param, NOT a pretrained weight.
        # Must use lr.task (1.2e-4), not lr.esm3_lora (5e-6). (S13-02)
        # Routed to task_no_decay — Softmax scale weights should not be L2-shrunk. (S15-03)
        if (
            hasattr(self.model.esm3_encoder, "layer_weights")
            and self.model.esm3_encoder.layer_weights is not None
        ):
            task_no_decay.append(self.model.esm3_encoder.layer_weights)  # S15-03: no_decay
            lora_ids.add(id(self.model.esm3_encoder.layer_weights))  # prevent duplicate group

        # Condition embeddings for missing pH/temp: zero-init learned vectors.
        # Exempt from weight decay — L2 would shrink them back to zero. (S15-03)
        _cond = self.model.cond_encoder
        for _attr in ("ph_unknown", "temp_unknown"):
            if hasattr(_cond, _attr):
                _p = getattr(_cond, _attr)
                task_no_decay.append(_p)
                lora_ids.add(id(_p))  # prevent duplicate routing in main loop

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in lora_ids:
                continue  # handled as esm3_lora group
            if pid in biot5_param_ids:
                if self._is_no_decay_param(name):
                    biot5_no_decay.append(param)
                else:
                    biot5_decay.append(param)
            else:
                if self._is_no_decay_param(name):
                    task_no_decay.append(param)
                else:
                    task_decay.append(param)

        lr_task = lr_cfg["task"]
        lr_biot5 = lr_cfg["biot5"]

        # Build param groups — only include non-empty groups
        param_groups = []
        if task_decay:
            param_groups.append(
                {"params": task_decay, "lr": lr_task, "weight_decay": wd, "name": "task_decay"},
            )
        if task_no_decay:
            param_groups.append(
                {"params": task_no_decay, "lr": lr_task, "weight_decay": 0.0, "name": "task_no_decay"},
            )
        if biot5_decay:
            param_groups.append(
                {"params": biot5_decay, "lr": lr_biot5, "weight_decay": wd, "name": "biot5_decay"},
            )
        if biot5_no_decay:
            param_groups.append(
                {"params": biot5_no_decay, "lr": lr_biot5, "weight_decay": 0.0, "name": "biot5_no_decay"},
            )
        if esm3_lora_enabled and lora_params:
            lr_esm3_lora = lr_cfg["esm3_lora"]
            param_groups.append(
                {"params": lora_params,
                 "lr": lr_esm3_lora, "weight_decay": wd, "name": "esm3_lora"},
            )

        # Summary log
        group_summary = "  ".join(
            f"{g['name']}={len(g['params'])}" for g in param_groups
        )
        logger.info("[KICMLightningModule] Param groups: %s", group_summary)

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=wd,
            eps=train_cfg["adam_eps"],
            betas=tuple(train_cfg["adam_betas"]),
        )

        total_steps = self.trainer.estimated_stepping_batches
        if total_steps == 0:
            raise RuntimeError(
                "estimated_stepping_batches=0: configure_optimizers must be called "
                "during trainer.fit(), not standalone."
            )
        sched_cfg = train_cfg["scheduler"]
        min_ratio = sched_cfg["min_lr_ratio"]
        wu_task = sched_cfg["warmup_steps"]
        wu_biot5 = sched_cfg.get("biot5_warmup_steps", wu_task)

        task_lambda = self._make_cosine_lambda(wu_task, total_steps, min_ratio)
        biot5_lambda = self._make_cosine_lambda(wu_biot5, total_steps, min_ratio)

        # Build per-group lr_lambda list matching param_groups order
        lr_lambdas = []
        for pg in param_groups:
            name = pg.get("name", "")
            if name.startswith("biot5"):
                lr_lambdas.append(biot5_lambda)
            else:
                lr_lambdas.append(task_lambda)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)

        # Log BioT5+ trainable param count
        biot5_trainable = sum(
            p.numel() for p in self.model.substrate_encoder.parameters()
            if p.requires_grad
        )
        logger.info(
            "[KICMLightningModule] BioT5+ trainable params: %.2fM", biot5_trainable / 1e6,
        )

        # Log total trainable param count
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "[KICMLightningModule] Total trainable params: %.2fM", total_trainable / 1e6,
        )

        logger.info(
            "[KICMLightningModule] Optimizer: AdamW  total_steps=%d  wu_task=%d  "
            "wu_biot5=%d  groups=%s",
            total_steps, wu_task, wu_biot5, group_summary,
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

    @staticmethod
    def _compute_grad_norm(params) -> float:
        """Compute L2 norm of gradients across all parameters."""
        grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
        if not grads:
            return 0.0
        return torch.cat(grads).norm(2).item()

    def on_after_backward(self) -> None:
        # L-02: Gradient norms are rank-local. Values reflect rank-0 only
        # (W&B logger is rank-0). For cross-rank gradient monitoring,
        # enable DDP gradient hooks separately. Do NOT add sync_dist here.
        if self.global_step % 100 != 0:
            return

        # BioT5+ grad norm
        biot5_norm = self._compute_grad_norm(self.model.substrate_encoder.parameters())
        self.log("grad_norm/biot5", biot5_norm, on_step=True, on_epoch=False)

        # Task head grad norm — covers proj + CA + condition + MLP
        task_params = (
            list(self.model.substrate_proj.parameters())
            + list(self.model.cross_attn.parameters())
            + list(self.model.cond_encoder.parameters())
            + list(self.model.mlp_head.parameters())
        )
        task_norm = self._compute_grad_norm(task_params)
        self.log("grad_norm/task", task_norm, on_step=True, on_epoch=False)

        # ESM3 LoRA grad norm (only when enabled)
        if self.model._esm3_lora_enabled:
            lora_norm = self._compute_grad_norm(self.model.esm3_encoder.lora_parameters())
            self.log("grad_norm/esm3_lora", lora_norm, on_step=True, on_epoch=False)

    # ------------------------------------------------------------------
    # GPU memory & epoch-level logging
    # ------------------------------------------------------------------

    def on_train_epoch_end(self) -> None:
        epoch = self.current_epoch
        optimizer = self.optimizers()

        # Extract current LRs for banner (LearningRateMonitor handles W&B logging)
        lr_task = next(
            (pg["lr"] for pg in optimizer.param_groups if pg.get("name", "").startswith("task")),
            0.0,
        )
        lr_biot5 = next(
            (pg["lr"] for pg in optimizer.param_groups if pg.get("name", "").startswith("biot5")),
            0.0,
        )
        lr_esm3 = next(
            (pg["lr"] for pg in optimizer.param_groups if pg.get("name", "").startswith("esm3")),
            0.0,
        )

        # Epoch-level condition coverage (accumulated via MeanMetric)
        # MeanMetric.compute() with sync_on_compute=True all-reduces internally —
        # do NOT add sync_dist=True here (would double-sync).
        self.log("train/ph_coverage", self.ph_coverage_metric.compute(),
                 on_step=False, on_epoch=True)
        self.log("train/temp_coverage", self.temp_coverage_metric.compute(),
                 on_step=False, on_epoch=True)
        self.ph_coverage_metric.reset()
        self.temp_coverage_metric.reset()

        if torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            # Per-rank memory logging (L-01 / S10-09 / S10-11): sync_dist=False allows per-GPU visibility.
            # Use rank-specific key so all GPUs are visible in W&B simultaneously.
            # S10-11: sync_dist=True mean aggregate removed — ALLREDUCE immediately after
            # MeanMetric's two ALLGATHER collectives (ph + temp coverage) caused SeqNum mismatch
            # at Epoch 1 boundary (Rank 0 SeqNum=485 ALLREDUCE vs Rank 1 SeqNum=484 ALLGATHER).
            self.log(f"gpu/memory_gb/rank{self.global_rank}", mem_gb, sync_dist=False)
            logger.info(
                "[GPU mem] rank=%d  peak=%.2fGB", self.global_rank, mem_gb,
            )
            torch.cuda.reset_peak_memory_stats()
        else:
            mem_gb = 0.0

        print(
            f"=== Epoch {epoch} "
            f"| lr_task={lr_task:.2e} | lr_biot5={lr_biot5:.2e} "
            f"| lr_lora={lr_esm3:.2e} | GPU_mem={mem_gb:.1f}GB ==="
        )
        logger.info(
            "=== Epoch %d | lr_task=%.2e | lr_biot5=%.2e | lr_lora=%.2e | GPU_mem=%.1fGB ===",
            epoch, lr_task, lr_biot5, lr_esm3, mem_gb,
        )

        # Scalar Mix weights diagnostic (S12-02)
        if hasattr(self.model.esm3_encoder, 'layer_weights'):
            w = torch.softmax(self.model.esm3_encoder.layer_weights.detach(), dim=0)
            for i, wi in enumerate(w.tolist()):
                self.log(f"esm3/layer_weight_{i}", wi,
                         on_epoch=True, sync_dist=False)  # diagnostic, rank-0 only
