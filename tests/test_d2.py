"""Phase E pipeline tests (legacy test_d2.py — audited S9 M-04).

Tests:
  1. ESM3Encoder: LoRA application, hook survival, lora_parameters()
  2. CrossAttentionFusion: Multi-query CA with FFN block
  3. KICMModel: end-to-end forward pass (Phase E)
  4. KICMLightningModule: BioT5+ param groups, configure_optimizers
  5. Config validation: Phase E config passes, invalid config rejected

M-04 Audit (Session 9):
  - TestCrossAttentionB1 / TestKICMModelD2: already removed in prior sessions
  - All remaining tests verified to exercise Phase E code only
  - Deduplicated test_no_bidirectional_arg (covered in test_phase_e.py)
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

def _phase_e_config() -> dict:
    """Return a minimal Phase E config dict."""
    return {
        "model": {
            "esm3_variant": "esm3_sm_open_v1",
            "d_esm3": 1536,
            "substrate_encoder": "biot5",
            "biot5_model": "QizhiPei/biot5-plus-base-mol-instructions-molecule",
            "biot5_max_length": 128,
            "d_substrate": 768,
            "d_proj": 512,
            "d_fusion": 512,
            "d_cond": 256,
            "projection_normalize_input": True,
            "esm3_lora": {
                "enabled": False,
                "r": 12,
                "alpha": 24,
                "dropout": 0.05,
                "target_modules": ["layernorm_qkv.1"],
            },
            "biot5_lora": {
                "enabled": True,
                "r": 4,
                "alpha": 8,
                "dropout": 0.1,
                "target_modules": ["q", "v"],
            },
            "cross_attention": {
                "d_sub": 512,
                "d_prot": 1536,
                "n_heads": 8,
                "dropout": 0.1,
            },
            "mlp": {
                "hidden_dims": [512, 256, 128],
                "dropouts": [0.3, 0.2, 0.1],
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
            "ph_col": "ph",
            "temp_col": "temperature",
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
                "biot5": 2e-5,
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
# Test 1: ESM3Encoder — LoRA
# ---------------------------------------------------------------------------

class TestESM3EncoderLoRA:
    @pytest.fixture(scope="class")
    def encoder(self):
        from src.models.esm3_encoder import ESM3Encoder
        enc = ESM3Encoder("esm3_sm_open_v1", d_model=1536)
        enc.freeze()
        enc.apply_lora(r=16, alpha=32, dropout=0.05, target_modules=["layernorm_qkv.1"])
        return enc

    def test_lora_params_exist(self, encoder):
        """LoRA parameters must exist and be trainable."""
        lora_params = list(encoder.lora_parameters())
        assert len(lora_params) > 0, "No LoRA parameters found"
        for p in lora_params:
            assert p.requires_grad, "LoRA param not trainable"

    def test_backbone_frozen(self, encoder):
        """ESM-3 backbone should remain frozen after LoRA application."""
        base_params = [
            p for n, p in encoder.esm3.named_parameters()
            if "lora" not in n.lower()
        ]
        for p in base_params:
            assert not p.requires_grad, "Backbone param not frozen"

    def test_hooks_survive_peft(self, encoder):
        """All 4 multi-scale hooks must still fire after PEFT wrapping."""
        tokens = torch.randint(0, 32, (1, 10))
        mask = torch.zeros(1, 10, dtype=torch.bool)

        with torch.no_grad():
            hidden, _ = encoder(tokens, mask)

        assert hidden is not None, "Hooks did not fire after PEFT wrapping"
        assert hidden.shape == (1, 10, 1536)
        assert len(encoder._hooks) == 4, f"Expected 4 hooks, got {len(encoder._hooks)}"
        assert len(encoder._hidden_states) == 4, "Not all 4 hooks fired"

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
# Test 2: CrossAttentionFusion — Multi-query with FFN
# ---------------------------------------------------------------------------

class TestCrossAttentionMQCA:
    # M-04 note: test_no_bidirectional_arg moved to test_phase_e.py (deduplicated)

    def test_forward_shape(self):
        """Multi-query forward: [B, d_sub] output."""
        from src.models.cross_attention import CrossAttentionFusion
        ca = CrossAttentionFusion(d_sub=512, d_prot=1536, n_heads=8, dropout=0.0)
        h_sub = torch.randn(2, 10, 512)  # [B, S, d_sub] multi-query
        prot = torch.randn(2, 50, 1536)
        prot_mask = torch.zeros(2, 50, dtype=torch.bool)
        sub_mask = torch.ones(2, 10, dtype=torch.bool)  # biot5_valid_mask convention: True=valid

        out = ca(h_sub, prot, prot_mask, sub_mask)
        assert out.shape == (2, 512)

    def test_no_substrate_atoms_arg(self):
        """forward() should NOT accept substrate_atoms."""
        from src.models.cross_attention import CrossAttentionFusion
        import inspect
        sig = inspect.signature(CrossAttentionFusion.forward)
        assert "substrate_atoms" not in sig.parameters


# ---------------------------------------------------------------------------
# Test 3: KICMModel Phase E forward
# ---------------------------------------------------------------------------

class TestKICMModelPhaseE:
    @pytest.fixture(scope="class")
    def model(self):
        from src.models.kicm_model import KICMModel
        cfg = _phase_e_config()
        return KICMModel(cfg)

    def test_forward_shape(self, model):
        """End-to-end forward should produce [B] predictions."""
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        tok = EsmSequenceTokenizer()
        encoded = tok(["MKTLL", "ACDE"], return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]
        mask = encoded["attention_mask"].eq(0)

        # BioT5+ pre-tokenized inputs (fake token IDs)
        biot5_ids = torch.randint(0, 100, (2, 15))
        biot5_mask = torch.ones(2, 15, dtype=torch.long)

        cofactor_fp = torch.zeros(2, 512)  # S12-06: cofactor fingerprint [B, 512]

        model.eval()
        with torch.no_grad():
            preds = model(tokens, mask, biot5_ids, biot5_mask,
                         ph=torch.tensor([7.0, 6.5]),
                         temp=torch.tensor([25.0, 37.0]),
                         ph_known=torch.tensor([True, True]),
                         temp_known=torch.tensor([True, False]),
                         cofactor_fp=cofactor_fp)

        assert preds.shape == (2,), f"Expected [2], got {preds.shape}"
        assert torch.isfinite(preds).all(), "Non-finite predictions"

    def test_no_phase_methods(self, model):
        """Phase E model should NOT have phase-based unfreezing methods."""
        assert not hasattr(model, "freeze_esm3"), "freeze_esm3 should be removed"
        assert not hasattr(model, "unfreeze_esm3_partial"), "unfreeze_esm3_partial should be removed"
        assert not hasattr(model, "unfreeze_esm3_full"), "unfreeze_esm3_full should be removed"

    def test_trainable_params(self, model):
        """BioT5+ + task params should be trainable; ESM-3 backbone frozen."""
        trainable = model.count_trainable_params()
        total = model.count_total_params()
        assert trainable > 0, "No trainable params"
        assert trainable < total, "All params are trainable — ESM-3 not frozen?"
        # BioT5+ ~112M + task ~4M, well under 1.4B total
        assert trainable < 200_000_000, f"Too many trainable params: {trainable}"


# ---------------------------------------------------------------------------
# Test 4: KICMLightningModule
# ---------------------------------------------------------------------------

class TestKICMLightningModule:
    @pytest.fixture(scope="class")
    def module(self):
        from src.models.kicm_module import KICMLightningModule
        cfg = _phase_e_config()
        return KICMLightningModule(cfg)

    def test_no_phase_methods(self, module):
        """Phase E module should NOT have phase-based methods."""
        assert not hasattr(module, "_current_phase"), "_current_phase should be removed"
        assert not hasattr(module, "_update_scheduler_base_lrs"), "_update_scheduler_base_lrs should be removed"

    def test_no_augmentation(self, module):
        """Phase E module should NOT have augmentation."""
        assert not hasattr(module, "aug_enabled"), "aug_enabled should be removed"
        assert not hasattr(module, "augmentor"), "augmentor should be removed"

    def test_configure_optimizers_groups(self, module):
        """configure_optimizers should produce param groups (no ESM3 LoRA when disabled).

        With BioT5+ LoRA (bias=none), biot5_no_decay may be empty → 3 groups.
        Without LoRA (full trainable), biot5_no_decay exists → 4 groups.
        """
        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 1000
        module._trainer = mock_trainer

        result = module.configure_optimizers()
        optimizer = result["optimizer"]
        groups = optimizer.param_groups

        assert len(groups) in (3, 4), f"Expected 3 or 4 param groups, got {len(groups)}"

        names = [g["name"] for g in groups]
        assert "task_decay" in names, f"task_decay missing from {names}"
        assert "task_no_decay" in names, f"task_no_decay missing from {names}"
        assert any(n.startswith("biot5") for n in names), f"No biot5 group in {names}"

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
            elif g["name"].startswith("biot5"):
                assert g["initial_lr"] == 2e-5, f"biot5 lr={g['initial_lr']}"


# ---------------------------------------------------------------------------
# Test 5: Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_phase_e_config_passes(self):
        """Phase E config should pass validation."""
        from src.utils.config_validation import validate_config
        cfg = _phase_e_config()
        validate_config(cfg)  # should not raise

    def test_missing_biot5_lr_fails(self):
        """Config without lr.biot5 should fail."""
        from src.utils.config_validation import validate_config
        cfg = _phase_e_config()
        del cfg["training"]["lr"]["biot5"]
        with pytest.raises((AssertionError, KeyError)):
            validate_config(cfg)

    def test_dmpnn_encoder_rejected(self):
        """substrate_encoder='dmpnn' should be rejected."""
        from src.utils.config_validation import validate_config
        cfg = _phase_e_config()
        cfg["model"]["substrate_encoder"] = "dmpnn"
        with pytest.raises(AssertionError, match="molformer.*biot5"):
            validate_config(cfg)

    def test_empty_target_modules_fails(self):
        """Empty target_modules should fail validation when LoRA enabled."""
        from src.utils.config_validation import validate_config
        cfg = _phase_e_config()
        cfg["model"]["esm3_lora"]["enabled"] = True
        cfg["model"]["esm3_lora"]["target_modules"] = []
        with pytest.raises(AssertionError, match="target_modules"):
            validate_config(cfg)
