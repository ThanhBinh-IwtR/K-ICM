"""Integration smoke test for KICM D2 pipeline.

Tests:
  - Full KICMModel forward pass with synthetic B=2 inputs
  - Output shape [2], finite values
  - loss.backward() completes without error
  - Grad norms for each param group are non-zero and finite
"""
from __future__ import annotations

import sys
import os

import pytest
import torch
import torch.nn as nn

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _integration_config() -> dict:
    """Minimal config for integration test (no wandb, cpu-friendly)."""
    return {
        "model": {
            "esm3_variant": "esm3_sm_open_v1",
            "d_esm3": 1536,
            "substrate_encoder": "molformer",
            "d_substrate": 768,
            "d_proj": 512,
            "d_fusion": 512,
            "molformer_model": "ibm/MoLFormer-XL-both-10pct",
            "molformer_max_length": 202,
            "projection_normalize_input": True,
            "esm3_lora": {
                "enabled": True,
                "r": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": ["layernorm_qkv.1", "out_proj", "ffn.1"],
                "lora_start_block": 36,
            },
            "cross_attention": {
                "enabled": True,
                "d_sub": 512,
                "d_prot": 1536,
                "n_heads": 8,
                "dropout": 0.0,
                "n_queries": 4,
            },
            "mlp": {
                "hidden_dims": [512, 256, 128],
                "dropouts": [0.2, 0.1, 0.0],
                "activation": "gelu",
            },
        },
        "conditioning": {
            "d_cond": 64,
            "d_mlp_hidden": 32,
        },
        "data": {
            "train_csv": "data/train.csv",
            "val_csv": "data/val.csv",
            "test_csv": "data/test.csv",
            "predictions_csv": "outputs/test_predictions.csv",
            "sequence_col": "sequence",
            "smiles_col": "reactant_smiles",
            "target_col": "log10_value",
            "max_seq_len": 1024,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {
            "num_gpus": 1,
            "ddp_backend": "nccl",
            "precision": "bf16-mixed",
            "find_unused_parameters": False,
            "batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 2,
            "max_epochs": 1,
            "val_every_n_epochs": 1,
            "loss": "huber",
            "huber_beta": 1.0,
            "gradient_clip_norm": 1.0,
            "lr": {
                "task": 2e-4,
                "molformer": 2e-5,
                "esm3_lora": 2e-5,
            },
            "weight_decay": 1e-2,
            "adam_eps": 1e-8,
            "adam_betas": [0.9, 0.999],
            "scheduler": {
                "type": "cosine_with_warmup",
                "warmup_steps": 5,
                "min_lr_ratio": 0.01,
            },
            "early_stopping": {
                "monitor": "val/r2",
                "mode": "max",
                "patience": 5,
                "min_delta": 0.001,
            },
            "checkpoint": {
                "save_top_k": 1,
                "monitor": "val/r2",
                "mode": "max",
                "filename": "epoch{epoch:03d}-r2{val/r2:.4f}",
            },
        },
        "logging": {
            "backend": "csv",
            "log_every_n_steps": 1,
            "log_gpu_memory": False,
        },
        "evaluation": {
            "metrics": ["r2", "rmse"],
            "primary_metric": "r2",
        },
        "reproducibility": {
            "deterministic": False,
            "benchmark": False,
        },
    }


class TestIntegrationD2:
    """Full pipeline integration: forward + backward + grad norms."""

    @pytest.fixture(scope="class")
    def model_and_batch(self):
        from src.models.kicm_model import KICMModel
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        cfg = _integration_config()
        model = KICMModel(cfg)
        model.train()

        # Build synthetic batch with B=2
        tok = EsmSequenceTokenizer()
        encoded = tok(["MKTLLLTLVVVTIVCLDLGAVAGNLGCDLHMKIILVIFYVLFSILLIILGNDGRPKFEAAKYVRASKIDAGMKPEQPAAAPPQQPQPAP",
                       "ACDEFGHIKLMNPQRSTVWY"], return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]
        padding_mask = encoded["attention_mask"].eq(0)

        B = tokens.size(0)
        batch = {
            "sequence_tokens": tokens,
            "esm_pad_mask":    padding_mask,
            "smiles":          ["CC(=O)O", "c1ccccc1"],
            "ph":              torch.tensor([7.0, 6.5]),
            "temp":            torch.tensor([37.0, 25.0]),
            "ph_known":        torch.tensor([True, True]),
            "temp_known":      torch.tensor([True, False]),
            "cofactor_fp":     torch.zeros(B, 512),
            "targets":         torch.tensor([[1.5], [2.0]]),
        }
        return model, batch, cfg

    def test_output_shape(self, model_and_batch):
        """Forward pass should return [B] tensor."""
        model, batch, _ = model_and_batch
        with torch.no_grad():
            preds = model(
                batch["sequence_tokens"],
                batch["esm_pad_mask"],
                batch["smiles"],
                batch["ph"],
                batch["temp"],
                batch["ph_known"],
                batch["temp_known"],
                batch["cofactor_fp"],
            )
        assert preds.shape == (2,), f"Expected [2], got {preds.shape}"

    def test_output_is_finite(self, model_and_batch):
        """Forward pass output must be finite (no NaN/Inf)."""
        model, batch, _ = model_and_batch
        with torch.no_grad():
            preds = model(
                batch["sequence_tokens"],
                batch["esm_pad_mask"],
                batch["smiles"],
                batch["ph"],
                batch["temp"],
                batch["ph_known"],
                batch["temp_known"],
                batch["cofactor_fp"],
            )
        assert torch.isfinite(preds).all(), f"Non-finite predictions: {preds}"

    def test_backward_completes(self, model_and_batch):
        """loss.backward() must complete without error."""
        model, batch, cfg = model_and_batch
        criterion = nn.SmoothL1Loss(beta=1.0)

        preds = model(
            batch["sequence_tokens"],
            batch["esm_pad_mask"],
            batch["smiles"],
            batch["ph"],
            batch["temp"],
            batch["ph_known"],
            batch["temp_known"],
            batch["cofactor_fp"],
        )
        targets = batch["targets"].squeeze(-1)
        loss = criterion(preds, targets)

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        loss.backward()  # must not raise

    def test_grad_norms_nonzero_and_finite(self, model_and_batch):
        """After backward, trainable params must have non-zero, finite gradients."""
        model, batch, _ = model_and_batch
        criterion = nn.SmoothL1Loss(beta=1.0)

        # Zero out any previous gradients
        model.zero_grad()

        preds = model(
            batch["sequence_tokens"],
            batch["esm_pad_mask"],
            batch["smiles"],
            batch["ph"],
            batch["temp"],
            batch["ph_known"],
            batch["temp_known"],
            batch["cofactor_fp"],
        )
        targets = batch["targets"].squeeze(-1)
        loss = criterion(preds, targets)
        loss.backward()

        grad_norms = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norm = param.grad.norm().item()
                grad_norms[name] = norm

        assert len(grad_norms) > 0, "No gradients found on any trainable parameter"

        for name, norm in grad_norms.items():
            assert norm == norm, f"NaN gradient for {name}"    # NaN != NaN
            assert norm != float("inf"), f"Inf gradient for {name}"
