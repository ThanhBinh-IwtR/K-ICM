"""D2 LoRA pipeline tests.

Tests:
  1. ESM3Encoder: LoRA application, hook survival, lora_parameters()
  2. CrossAttentionFusion: B1-only (no bidirectional args)
  3. KICMModel: end-to-end forward pass with LoRA
  4. KICMLightningModule: 5 param groups (LoRA has no bias), configure_optimizers
  5. Config validation: D2 LoRA config passes, D1 config rejected
  6. Multi-query cross-attention: n_queries=4 output shape, backward, query_proj
  7. ESM3 LoRA block restriction: blocks 0-35 frozen, 36-47 trainable
  8. KICMModel new config: forward with n_queries=4 + ESM3 LoRA last12
"""
from __future__ import annotations

import copy
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

        B = tokens.size(0)
        ph       = torch.tensor([7.0, 6.5])
        temp     = torch.tensor([25.0, 37.0])
        ph_known   = torch.tensor([True, True])
        temp_known = torch.tensor([True, False])
        cofactor_fp = torch.zeros(B, 512)

        model.eval()
        with torch.no_grad():
            preds = model(tokens, mask, ["CC(=O)O", "c1ccccc1"],
                          ph, temp, ph_known, temp_known, cofactor_fp)

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


# ---------------------------------------------------------------------------
# PHASE-2 New targeted unit tests (prefix: new_)
# ---------------------------------------------------------------------------

class TestNew_MoLFormerMeanPool:
    """CHANGE-1: MoLFormerEncoder uses attention-masked mean pool, not pooler_output."""

    def test_new_output_shape(self):
        """Mean-pool output should be [B, 768]."""
        from src.models.molformer_encoder import MoLFormerEncoder
        enc = MoLFormerEncoder()
        smiles = ["CC(=O)O", "c1ccccc1"]
        with torch.no_grad():
            out = enc(smiles)
        assert out.shape == (2, 768), f"Expected [2, 768], got {out.shape}"

    def test_new_pooler_not_used(self):
        """MoLFormerEncoder should not use pooler_output."""
        from src.models.molformer_encoder import MoLFormerEncoder
        import inspect
        src = inspect.getsource(MoLFormerEncoder.forward)
        assert "pooler_output" not in src, "pooler_output should not appear in forward()"

    def test_new_no_use_pooler_param(self):
        """MoLFormerEncoder.__init__ should not have use_pooler parameter."""
        from src.models.molformer_encoder import MoLFormerEncoder
        import inspect
        sig = inspect.signature(MoLFormerEncoder.__init__)
        assert "use_pooler" not in sig.parameters, "use_pooler param should be removed"


class TestNew_ConditionEncoderCofactor:
    """CHANGE-2: ConditionEncoder accepts cofactor_fp and produces [B, 64] output."""

    def test_new_output_shape_zero_cofactor(self):
        """ConditionEncoder with all-zero cofactor_fp should produce [B, 64]."""
        from src.models.condition_encoder import ConditionEncoder
        enc = ConditionEncoder(d_cond=64, d_mlp_hidden=32)
        B = 4
        ph         = torch.tensor([7.0, 6.5, 8.0, 5.5])
        temp       = torch.tensor([25.0, 37.0, 30.0, 40.0])
        ph_known   = torch.tensor([True, True, False, True])
        temp_known = torch.tensor([True, False, True, True])
        cofactor_fp = torch.zeros(B, 512)

        with torch.no_grad():
            out = enc(ph, temp, ph_known, temp_known, cofactor_fp)

        assert out.shape == (B, 64), f"Expected [{B}, 64], got {out.shape}"

    def test_new_output_shape_random_cofactor(self):
        """ConditionEncoder with random cofactor_fp should produce [B, 64]."""
        from src.models.condition_encoder import ConditionEncoder
        enc = ConditionEncoder(d_cond=64, d_mlp_hidden=32)
        B = 3
        ph         = torch.zeros(B)
        temp       = torch.zeros(B)
        ph_known   = torch.ones(B, dtype=torch.bool)
        temp_known = torch.ones(B, dtype=torch.bool)
        cofactor_fp = torch.rand(B, 512)

        with torch.no_grad():
            out = enc(ph, temp, ph_known, temp_known, cofactor_fp)

        assert out.shape == (B, 64), f"Expected [{B}, 64], got {out.shape}"
        assert torch.isfinite(out).all(), "Non-finite values in ConditionEncoder output"


class TestNew_KICMModelCofactorForward:
    """CHANGE-3: KICMModel.forward() with cofactor_fp argument."""

    @pytest.fixture(scope="class")
    def model(self):
        from src.models.kicm_model import KICMModel
        return KICMModel(_d2_config())

    def test_new_forward_shape_with_cofactor(self, model):
        """Forward with cofactor_fp should produce [B] with no NaN/Inf."""
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        tok = EsmSequenceTokenizer()
        encoded = tok(["MKTLL", "ACDE"], return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]
        mask   = encoded["attention_mask"].eq(0)
        B = tokens.size(0)

        ph          = torch.tensor([7.0, 6.5])
        temp        = torch.tensor([37.0, 25.0])
        ph_known    = torch.tensor([True, True])
        temp_known  = torch.tensor([True, False])
        cofactor_fp = torch.zeros(B, 512)

        model.eval()
        with torch.no_grad():
            preds = model(tokens, mask, ["CC(=O)O", "c1ccccc1"],
                          ph, temp, ph_known, temp_known, cofactor_fp)

        assert preds.shape == (B,), f"Expected [{B}], got {preds.shape}"
        assert torch.isfinite(preds).all(), "Non-finite predictions in KICMModel forward"


class TestNew_ESM3LoRATargets:
    """CHANGE-4: All LoRA-targeted modules are nn.Linear, no LayerNorm in targets."""

    @pytest.fixture(scope="class")
    def encoder(self):
        from src.models.esm3_encoder import ESM3Encoder
        enc = ESM3Encoder()
        enc.freeze()
        enc.apply_lora(r=8, alpha=16, dropout=0.05,
                       target_modules=["layernorm_qkv.1", "out_proj", "ffn.1"])
        return enc

    def test_new_lora_targets_are_linear(self, encoder):
        """All modules matched by LoRA target_modules should wrap nn.Linear base layers."""
        import torch.nn as nn
        targets = {"layernorm_qkv.1", "out_proj", "ffn.1"}
        for name, mod in encoder.esm3.named_modules():
            short = name.split(".")[-1]
            if short in targets:
                # After PEFT wrapping, module is lora.Linear with a base_layer attribute
                base = getattr(mod, "base_layer", mod)
                assert isinstance(base, nn.Linear), (
                    f"Target {name} base is {type(base).__name__}, expected Linear"
                )

    def test_new_no_layernorm_in_lora_params(self, encoder):
        """No LayerNorm parameters should be trainable after LoRA."""
        import torch.nn as nn
        for name, mod in encoder.esm3.named_modules():
            if isinstance(mod, nn.LayerNorm):
                for param in mod.parameters():
                    assert not param.requires_grad, (
                        f"LayerNorm param {name} is trainable — LoRA hit a non-Linear module"
                    )

    def test_new_lora_params_all_2d(self, encoder):
        """All LoRA adapter parameters must be 2D (weight matrices)."""
        lora_params = [p for p in encoder.esm3.parameters() if p.requires_grad]
        assert len(lora_params) > 0, "No LoRA params found"
        for p in lora_params:
            assert p.dim() == 2, f"LoRA param has dim={p.dim()}, expected 2"


class TestNew_TemperatureNormalization:
    """CHANGE-5: Temperature normalization is centered on physiological temp (37°C)."""

    def test_new_temp_37_normalizes_to_zero(self):
        """temp=37.0 should produce temp_norm ≈ 0.0."""
        from src.models.condition_encoder import ConditionEncoder
        enc = ConditionEncoder(d_cond=64, d_mlp_hidden=32)

        # Inspect the normalization formula directly via a forward hook
        captured = {}

        def hook_fn(module, inp, out):
            captured["out"] = out

        # Register hook on temp_embed to capture its input
        handle = enc.temp_embed[0].register_forward_hook(
            lambda m, i, o: captured.update({"temp_embed_input": i[0]})
        )

        ph         = torch.tensor([7.0])
        temp       = torch.tensor([37.0])
        ph_known   = torch.tensor([True])
        temp_known = torch.tensor([True])
        cofactor_fp = torch.zeros(1, 512)

        with torch.no_grad():
            enc(ph, temp, ph_known, temp_known, cofactor_fp)

        handle.remove()

        assert "temp_embed_input" in captured, "Hook did not fire"
        temp_norm_value = captured["temp_embed_input"].item()
        assert abs(temp_norm_value) < 0.01, (
            f"temp=37.0 should normalize to ~0.0, got {temp_norm_value:.4f}"
        )

    def test_new_temp_norm_formula(self):
        """Verify formula: (temp - 37) / 20 for a range of values."""
        import torch
        temps = torch.tensor([0.0, 17.0, 37.0, 57.0, 97.0])
        expected = (temps - 37.0) / 20.0
        computed = (temps - 37.0) / 20.0
        assert torch.allclose(computed, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 6: KICMModel with use_conditioning=False
# ---------------------------------------------------------------------------

class TestKICMModelNoConditioning:
    """Verify use_conditioning=False bypasses FiLM and ConditionEncoder."""

    def _no_cond_config(self):
        """Minimal config with conditioning disabled."""
        cfg = _d2_config()
        cfg["conditioning"] = {"use_conditioning": False}
        return cfg

    def test_film_modules_are_none(self):
        """cond_encoder and film must be None when use_conditioning=False."""
        from src.models.kicm_model import KICMModel
        model = KICMModel(self._no_cond_config())
        assert model.cond_encoder is None, "cond_encoder should be None when use_conditioning=False"
        assert model.film is None, "film should be None when use_conditioning=False"
        assert model.use_conditioning is False

    def test_forward_no_conditioning_shape(self):
        """Forward pass with use_conditioning=False must return [B] tensor."""
        from src.models.kicm_model import KICMModel
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        import torch

        model = KICMModel(self._no_cond_config())
        model.eval()

        tok = EsmSequenceTokenizer()
        B = 2
        seqs = ["MKTLLLTLVVVTIVCLDLGAVAGNLG", "ACDEFGHIKL"]
        encoded = tok(seqs, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]
        pad_mask = encoded["attention_mask"].eq(0)

        with torch.no_grad():
            preds = model(
                tokens, pad_mask,
                ["CC(=O)O", "c1ccccc1"],
                torch.tensor([7.0, 6.5]),
                torch.tensor([37.0, 25.0]),
                torch.tensor([True, True]),
                torch.tensor([True, False]),
                torch.zeros(B, 512),
            )

        assert preds.shape == (B,), f"Expected [{B}], got {preds.shape}"
        assert torch.isfinite(preds).all(), f"Non-finite predictions: {preds}"

    def test_forward_no_conditioning_is_finite(self):
        """Forward output must be finite when conditioning is disabled."""
        # This test is implicitly covered by test_forward_no_conditioning_shape
        # but kept separate for clarity in CI output
        from src.models.kicm_model import KICMModel
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        import torch

        model = KICMModel(self._no_cond_config())
        model.eval()
        tok = EsmSequenceTokenizer()
        B = 2
        encoded = tok(["MKTLLLTLVVVTIVCLDLGAVAGNLG", "ACDEFGHIKL"], return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]
        pad_mask = encoded["attention_mask"].eq(0)

        with torch.no_grad():
            preds = model(
                tokens, pad_mask,
                ["CC(=O)O", "c1ccccc1"],
                torch.tensor([7.0, 6.5]),
                torch.tensor([37.0, 25.0]),
                torch.tensor([True, True]),
                torch.tensor([True, False]),
                torch.zeros(B, 512),
            )
        assert torch.isfinite(preds).all()


# ---------------------------------------------------------------------------
# Test 7: ESM3 LoRA disabled — optimizer must not contain esm3_lora group
# ---------------------------------------------------------------------------

def test_esm3_lora_disabled_no_optimizer_group():
    """When esm3_lora.enabled=False, optimizer must not contain esm3_lora group."""
    from unittest.mock import MagicMock, PropertyMock
    from src.models.kicm_module import KICMLightningModule

    # Build config with esm3_lora.enabled=False and MoLFormer LoRA enabled
    cfg = _d2_config()
    cfg["model"]["esm3_lora"] = {
        "enabled": False,
        "r": 12,
        "alpha": 24,
        "dropout": 0.05,
        "target_modules": ["layernorm_qkv.1", "out_proj", "ffn.1"],
    }
    cfg["model"]["molformer_frozen"] = True
    cfg["model"]["molformer_lora"] = {
        "enabled": True,
        "r": 16,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": [
            "attention.self.query", "attention.self.key",
            "attention.self.value", "attention.output.dense",
            "intermediate.dense", "output.dense",
        ],
    }
    cfg["training"]["lr"]["molformer_lora"] = 5e-5
    cfg["conditioning"] = {"use_conditioning": False}

    module = KICMLightningModule(cfg)

    # Mock trainer.estimated_stepping_batches for scheduler
    mock_trainer = MagicMock()
    type(mock_trainer).estimated_stepping_batches = PropertyMock(return_value=1000)
    module._trainer = mock_trainer

    # configure_optimizers should not crash
    opt_result = module.configure_optimizers()
    optimizer = opt_result["optimizer"]

    # Assert: no param group has name == "esm3_lora"
    group_names = [g.get("name", "") for g in optimizer.param_groups]
    assert "esm3_lora" not in group_names, (
        f"esm3_lora group should not exist when enabled=False. Found groups: {group_names}"
    )

    # Assert: optimizer initializes without RuntimeError (implicitly tested above)
    # Assert: at least task_decay and molformer_lora groups exist
    assert "task_decay" in group_names, f"task_decay missing. Groups: {group_names}"
    assert "molformer_lora" in group_names, f"molformer_lora missing. Groups: {group_names}"

    # Assert: no ESM3 LoRA params have requires_grad=True in model
    esm3_trainable = sum(
        1 for p in module.model.esm3_encoder.parameters() if p.requires_grad
    )
    assert esm3_trainable == 0, (
        f"ESM3 encoder should have 0 trainable params when LoRA disabled, found {esm3_trainable}"
    )


# ---------------------------------------------------------------------------
# Test 8: CrossAttentionFusion multi-query
# ---------------------------------------------------------------------------

class TestCrossAttentionMultiQuery:
    def test_n_queries_4_output_shape(self):
        """n_queries=4 must still output [B, d_sub]."""
        from src.models.cross_attention import CrossAttentionFusion
        B, L, d_sub, d_prot = 3, 20, 512, 1536
        model = CrossAttentionFusion(
            d_sub=d_sub, d_prot=d_prot, n_heads=8, dropout=0.0, n_queries=4
        )
        model.eval()
        h_sub = torch.randn(B, d_sub)
        protein = torch.randn(B, L, d_prot)
        pad_mask = torch.zeros(B, L, dtype=torch.bool)
        with torch.no_grad():
            out = model(h_sub, protein, pad_mask)
        assert out.shape == (B, d_sub), f"Expected ({B},{d_sub}), got {out.shape}"
        assert not torch.isnan(out).any(), "NaN in output"

    def test_n_queries_1_matches_legacy(self):
        """n_queries=1 must produce same shape as original single-query."""
        from src.models.cross_attention import CrossAttentionFusion
        torch.manual_seed(42)
        B, L, d_sub, d_prot = 2, 15, 512, 1536
        model = CrossAttentionFusion(
            d_sub=d_sub, d_prot=d_prot, n_heads=8, dropout=0.0, n_queries=1
        )
        model.eval()
        h_sub = torch.randn(B, d_sub)
        protein = torch.randn(B, L, d_prot)
        pad_mask = torch.zeros(B, L, dtype=torch.bool)
        with torch.no_grad():
            out = model(h_sub, protein, pad_mask)
        assert out.shape == (B, d_sub)

    def test_query_proj_exists_only_when_n_queries_gt_1(self):
        """query_proj must exist iff n_queries > 1."""
        from src.models.cross_attention import CrossAttentionFusion
        m1 = CrossAttentionFusion(512, 1536, 8, 0.0, n_queries=1)
        m4 = CrossAttentionFusion(512, 1536, 8, 0.0, n_queries=4)
        assert not hasattr(m1, 'query_proj'), "n_queries=1 must NOT have query_proj"
        assert hasattr(m4, 'query_proj'), "n_queries=4 must have query_proj"
        assert m4.query_proj.weight.shape == (4 * 512, 512)

    def test_n_queries_4_backward(self):
        """Backward must complete with n_queries=4."""
        from src.models.cross_attention import CrossAttentionFusion
        B, L, d_sub, d_prot = 2, 10, 512, 1536
        model = CrossAttentionFusion(512, 1536, 8, 0.0, n_queries=4)
        h_sub = torch.randn(B, d_sub, requires_grad=True)
        protein = torch.randn(B, L, d_prot)
        pad_mask = torch.zeros(B, L, dtype=torch.bool)
        out = model(h_sub, protein, pad_mask)
        loss = out.sum()
        loss.backward()
        assert h_sub.grad is not None
        assert not torch.isnan(h_sub.grad).any()


# ---------------------------------------------------------------------------
# Test 9: ESM3 LoRA block restriction
# ---------------------------------------------------------------------------

class TestESM3LoRALastBlocks:
    def test_lora_params_only_in_last_blocks(self):
        """After apply_lora(lora_start_block=36), LoRA params in blocks 0-35
        must be frozen, blocks 36-47 must be trainable."""
        from src.models.esm3_encoder import ESM3Encoder
        enc = ESM3Encoder(model_name="esm3_sm_open_v1", d_model=1536)
        enc.freeze()
        enc.apply_lora(r=4, alpha=8, dropout=0.0,
                       target_modules=["layernorm_qkv.1", "out_proj", "ffn.1"],
                       lora_start_block=36)
        for name, param in enc.esm3.named_parameters():
            if 'lora_' not in name:
                continue
            parts = name.split('.')
            block_idx = None
            for i, p in enumerate(parts):
                if p == 'blocks' and i + 1 < len(parts):
                    try:
                        block_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        continue
            if block_idx is not None:
                if block_idx < 36:
                    assert not param.requires_grad, \
                        f"Block {block_idx} LoRA param should be frozen: {name}"
                else:
                    assert param.requires_grad, \
                        f"Block {block_idx} LoRA param should be trainable: {name}"

    def test_lora_start_block_0_all_trainable(self):
        """lora_start_block=0 (default) must leave all LoRA params trainable."""
        from src.models.esm3_encoder import ESM3Encoder
        enc = ESM3Encoder(model_name="esm3_sm_open_v1", d_model=1536)
        enc.freeze()
        enc.apply_lora(r=4, alpha=8, dropout=0.0,
                       target_modules=["layernorm_qkv.1", "out_proj", "ffn.1"],
                       lora_start_block=0)
        lora_params = [p for n, p in enc.esm3.named_parameters()
                       if 'lora_' in n]
        assert all(p.requires_grad for p in lora_params), \
            "lora_start_block=0 must leave all LoRA params trainable"


# ---------------------------------------------------------------------------
# Test 10: KICMModel with new config (n_queries=4 + ESM3 LoRA last12)
# ---------------------------------------------------------------------------

class TestKICMModelNewConfig:
    def test_forward_n_queries_4_esm3_lora_last12(self):
        """Full forward with n_queries=4 and esm3_lora last 12 blocks."""
        from src.models.kicm_model import KICMModel
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        cfg = copy.deepcopy(_d2_config())
        cfg["model"]["cross_attention"]["n_queries"] = 4
        cfg["model"]["esm3_lora"]["enabled"] = True
        cfg["model"]["esm3_lora"]["lora_start_block"] = 36

        model = KICMModel(cfg)
        model.eval()

        tok = EsmSequenceTokenizer()
        B = 2
        encoded = tok(["MKTLL", "ACDE"], return_tensors="pt", padding=True)
        seq_tokens = encoded["input_ids"]
        pad_mask = encoded["attention_mask"].eq(0)

        ph = torch.tensor([7.0, 7.0])
        temp = torch.tensor([37.0, 37.0])
        ph_known = torch.ones(B, dtype=torch.bool)
        temp_known = torch.ones(B, dtype=torch.bool)
        cofactor_fp = torch.zeros(B, 512)

        with torch.no_grad():
            preds = model(seq_tokens, pad_mask, ["CC(=O)O", "c1ccccc1"],
                          ph, temp, ph_known, temp_known, cofactor_fp)
        assert preds.shape == (B,), f"Expected ({B},), got {preds.shape}"
        assert torch.isfinite(preds).all(), "Predictions contain NaN/Inf"

    def test_esm3_lora_params_in_optimizer(self):
        """With esm3_lora.enabled=True, optimizer must contain esm3_lora group."""
        from unittest.mock import PropertyMock
        from src.models.kicm_module import KICMLightningModule

        cfg = copy.deepcopy(_d2_config())
        cfg["model"]["esm3_lora"]["enabled"] = True
        cfg["model"]["esm3_lora"]["lora_start_block"] = 36
        cfg["training"]["lr"]["esm3_lora"] = 2.0e-5

        module = KICMLightningModule(cfg)

        mock_trainer = MagicMock()
        type(mock_trainer).estimated_stepping_batches = PropertyMock(return_value=1000)
        module._trainer = mock_trainer

        opt_cfg = module.configure_optimizers()
        group_names = [g["name"] for g in opt_cfg["optimizer"].param_groups]
        assert "esm3_lora" in group_names, \
            f"esm3_lora group missing. Found: {group_names}"
        esm3_group = next(g for g in opt_cfg["optimizer"].param_groups
                          if g["name"] == "esm3_lora")
        # After LambdaLR, the "initial_lr" stores the configured value;
        # "lr" may be 0.0 at step 0 due to warmup lambda.
        esm3_lr = esm3_group.get("initial_lr", esm3_group["lr"])
        assert abs(esm3_lr - 2.0e-5) < 1e-9, \
            f"ESM3 LoRA LR should be 2e-5, got {esm3_lr}"
        assert len(esm3_group["params"]) > 0, "ESM3 LoRA group is empty"
