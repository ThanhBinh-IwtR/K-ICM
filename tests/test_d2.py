"""D2 LoRA pipeline tests.

Tests:
  1. ESM3Encoder: LoRA application, hook survival, lora_parameters()
  2. CrossAttentionFusion: B1-only (no bidirectional args)
  3. KICMModel: end-to-end forward pass with LoRA
  4. KICMLightningModule: 5 param groups (LoRA has no bias), configure_optimizers
  5. Config validation: D2 LoRA config passes, D1 config rejected
"""
from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock

import pytest
import torch

# Add kcat_predictor root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _d2_config() -> dict:
    """Return a minimal D2 LoRA config dict."""
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
            "molformer_use_pooler": True,
            "projection_normalize_input": True,
            "esm3_lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["layernorm_qkv.1"],
            },
            "cross_attention": {
                "enabled": True,
                "d_sub": 512,
                "d_prot": 1536,
                "n_heads": 8,
                "dropout": 0.1,
            },
            "mlp": {
                "hidden_dims": [512, 256, 128],
                "dropouts": [0.2, 0.1, 0.0],
                "activation": "gelu",
            },
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
            "max_epochs": 10,
            "val_every_n_epochs": 1,
            "loss": "huber",
            "huber_beta": 1.0,
            "gradient_clip_norm": 1.0,
            "lr": {
                "task": 2e-4,
                "molformer": 2e-5,
                "esm3_lora": 1e-5,
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
            "backend": "wandb",
            "wandb_project": "test",
            "wandb_entity": None,
            "log_every_n_steps": 1,
            "log_gpu_memory": False,
        },
        "evaluation": {
            "metrics": ["r2", "rmse", "mae", "spearman", "pearson"],
            "primary_metric": "r2",
        },
        "reproducibility": {
            "deterministic": False,
            "benchmark": False,
        },
    }


# ---------------------------------------------------------------------------
# Test 1: ESM3Encoder LoRA
# ---------------------------------------------------------------------------

class TestESM3EncoderLoRA:
    @pytest.fixture(scope="class")
    def encoder(self):
        from src.models.esm3_encoder import ESM3Encoder
        enc = ESM3Encoder()
        enc.freeze()
        enc.apply_lora(r=16, alpha=32, dropout=0.05, target_modules=["layernorm_qkv.1"])
        return enc

    def test_lora_params_exist(self, encoder):
        """LoRA parameters should be trainable."""
        lora_params = list(encoder.lora_parameters())
        assert len(lora_params) > 0, "No LoRA parameters found"

    def test_frozen_backbone(self, encoder):
        """Non-LoRA ESM-3 params should be frozen."""
        lora_ids = {id(p) for p in encoder.lora_parameters()}
        for name, param in encoder.esm3.named_parameters():
            if id(param) not in lora_ids:
                assert not param.requires_grad, f"Non-LoRA param {name} is not frozen"

    def test_hook_survives_peft(self, encoder):
        """Forward hook should survive PEFT wrapping."""
        device = next(encoder.esm3.parameters()).device
        tokens = torch.randint(0, 32, (1, 10), device=device)
        mask = torch.zeros(1, 10, dtype=torch.bool, device=device)

        encoder._penultimate_hidden = None
        with torch.no_grad():
            hidden, _ = encoder(tokens, mask)

        assert hidden is not None, "Hook did not fire after PEFT wrapping"
        assert hidden.shape == (1, 10, 1536)

    def test_forward_shape(self, encoder):
        """Forward should produce [B, L, 1536]."""
        device = next(encoder.esm3.parameters()).device
        B, L = 2, 20
        tokens = torch.randint(0, 32, (B, L), device=device)
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        with torch.no_grad():
            hidden, out_mask = encoder(tokens, mask)

        assert hidden.shape == (B, L, 1536)
        assert out_mask.shape == (B, L)


# ---------------------------------------------------------------------------
# Test 2: CrossAttentionFusion B1-only
# ---------------------------------------------------------------------------

class TestCrossAttentionB1:
    def test_no_bidirectional_arg(self):
        """CrossAttentionFusion should NOT accept bidirectional arg."""
        from src.models.cross_attention import CrossAttentionFusion
        import inspect
        sig = inspect.signature(CrossAttentionFusion.__init__)
        assert "bidirectional" not in sig.parameters, "bidirectional param should be removed"
        assert "d_atom" not in sig.parameters, "d_atom param should be removed"

    def test_forward_shape(self):
        """B1 forward: [B, d_sub] output."""
        from src.models.cross_attention import CrossAttentionFusion
        ca = CrossAttentionFusion(d_sub=512, d_prot=1536, n_heads=8, dropout=0.0)
        h_sub = torch.randn(2, 512)
        prot = torch.randn(2, 50, 1536)
        mask = torch.zeros(2, 50, dtype=torch.bool)

        out = ca(h_sub, prot, mask)
        assert out.shape == (2, 512)

    def test_no_substrate_atoms_arg(self):
        """forward() should NOT accept substrate_atoms."""
        from src.models.cross_attention import CrossAttentionFusion
        import inspect
        sig = inspect.signature(CrossAttentionFusion.forward)
        assert "substrate_atoms" not in sig.parameters


# ---------------------------------------------------------------------------
# Test 3: KICMModel D2 forward
# ---------------------------------------------------------------------------

class TestKICMModelD2:
    @pytest.fixture(scope="class")
    def model(self):
        from src.models.kicm_model import KICMModel
        cfg = _d2_config()
        return KICMModel(cfg)

    def test_forward_shape(self, model):
        """End-to-end forward should produce [B] predictions."""
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        tok = EsmSequenceTokenizer()
        encoded = tok(["MKTLL", "ACDE"], return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]
        mask = encoded["attention_mask"].eq(0)

        model.eval()
        with torch.no_grad():
            preds = model(tokens, mask, ["CC(=O)O", "c1ccccc1"])

        assert preds.shape == (2,), f"Expected [2], got {preds.shape}"
        assert torch.isfinite(preds).all(), "Non-finite predictions"

    def test_no_phase_methods(self, model):
        """D2 model should NOT have phase-based unfreezing methods."""
        assert not hasattr(model, "freeze_esm3"), "freeze_esm3 should be removed"
        assert not hasattr(model, "unfreeze_esm3_partial"), "unfreeze_esm3_partial should be removed"
        assert not hasattr(model, "unfreeze_esm3_full"), "unfreeze_esm3_full should be removed"

    def test_trainable_params(self, model):
        """LoRA + MoLFormer + task params should be trainable; ESM-3 backbone frozen."""
        trainable = model.count_trainable_params()
        total = model.count_total_params()
        assert trainable > 0, "No trainable params"
        assert trainable < total, "All params are trainable — ESM-3 not frozen?"
        # LoRA adds ~4.7M, MoLFormer ~47M, task ~2M — should be well under 1.4B total
        assert trainable < 100_000_000, f"Too many trainable params: {trainable}"


# ---------------------------------------------------------------------------
# Test 4: KICMLightningModule
# ---------------------------------------------------------------------------

class TestKICMLightningModule:
    @pytest.fixture(scope="class")
    def module(self):
        from src.models.kicm_module import KICMLightningModule
        cfg = _d2_config()
        return KICMLightningModule(cfg)

    def test_no_phase_methods(self, module):
        """D2 module should NOT have phase-based methods."""
        assert not hasattr(module, "_current_phase"), "_current_phase should be removed"
        assert not hasattr(module, "_update_scheduler_base_lrs"), "_update_scheduler_base_lrs should be removed"

    def test_no_augmentation(self, module):
        """D2 module should NOT have augmentation."""
        assert not hasattr(module, "aug_enabled"), "aug_enabled should be removed"
        assert not hasattr(module, "augmentor"), "augmentor should be removed"

    def test_configure_optimizers_5_groups(self, module):
        """configure_optimizers should produce 5 param groups (LoRA has no bias)."""
        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 1000
        module._trainer = mock_trainer

        result = module.configure_optimizers()
        optimizer = result["optimizer"]
        groups = optimizer.param_groups

        assert len(groups) == 5, f"Expected 5 param groups, got {len(groups)}"

        names = [g["name"] for g in groups]
        expected_names = [
            "task_decay", "task_no_decay",
            "molformer_decay", "molformer_no_decay",
            "esm3_lora",
        ]
        assert names == expected_names, f"Expected {expected_names}, got {names}"

    def test_lr_values(self, module):
        """LR values should match config."""
        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 1000
        module._trainer = mock_trainer

        result = module.configure_optimizers()
        optimizer = result["optimizer"]

        for g in optimizer.param_groups:
            if g["name"].startswith("task"):
                assert g["initial_lr"] == 2e-4, f"task lr={g['initial_lr']}"
            elif g["name"].startswith("molformer"):
                assert g["initial_lr"] == 2e-5, f"molformer lr={g['initial_lr']}"
            elif g["name"].startswith("esm3_lora"):
                assert g["initial_lr"] == 1e-5, f"esm3_lora lr={g['initial_lr']}"


# ---------------------------------------------------------------------------
# Test 5: Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_d2_config_passes(self):
        """D2 LoRA config should pass validation."""
        from src.utils.config_validation import validate_config
        cfg = _d2_config()
        validate_config(cfg)  # should not raise

    def test_missing_lora_lr_fails(self):
        """Config without lr.esm3_lora should fail."""
        from src.utils.config_validation import validate_config
        cfg = _d2_config()
        del cfg["training"]["lr"]["esm3_lora"]
        with pytest.raises((AssertionError, KeyError)):
            validate_config(cfg)

    def test_dmpnn_encoder_rejected(self):
        """substrate_encoder='dmpnn' should be rejected in D2."""
        from src.utils.config_validation import validate_config
        cfg = _d2_config()
        cfg["model"]["substrate_encoder"] = "dmpnn"
        with pytest.raises(AssertionError, match="molformer"):
            validate_config(cfg)

    def test_empty_target_modules_fails(self):
        """Empty target_modules should fail validation."""
        from src.utils.config_validation import validate_config
        cfg = _d2_config()
        cfg["model"]["esm3_lora"]["target_modules"] = []
        with pytest.raises(AssertionError, match="target_modules"):
            validate_config(cfg)
