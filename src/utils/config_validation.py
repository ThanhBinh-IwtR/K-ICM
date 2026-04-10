"""Config validation utilities for KICMBaseModel.

Supports both D2 LoRA (MoLFormer) and Phase E (BioT5+) architectures.
Call validate_config(config) immediately after loading the YAML to catch
structural mismatches before any heavyweight model/data initialisation.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def validate_config(config: dict) -> None:
    """Assert structural consistency of the training config.

    Raises:
        AssertionError: If any constraint is violated.
        KeyError: If a required config key is missing.

    Args:
        config: Full training config (top-level keys: project, data, model,
                training, logging, evaluation, reproducibility).
    """
    model_cfg = config["model"]
    train_cfg = config["training"]
    mlp_cfg = model_cfg.get("mlp", {})

    # Model dimension consistency
    assert model_cfg["d_proj"] == model_cfg["d_fusion"], (
        f"d_fusion ({model_cfg['d_fusion']}) must equal d_proj ({model_cfg['d_proj']})"
    )

    # Substrate encoder validation
    encoder_type = model_cfg.get("substrate_encoder", "molformer")
    assert encoder_type in ("molformer", "biot5"), (
        f"substrate_encoder must be 'molformer' or 'biot5', got '{encoder_type}'"
    )

    lr_cfg = train_cfg.get("lr", {})

    if encoder_type == "biot5":
        assert model_cfg.get("d_substrate") == 768, (
            f"d_substrate must be 768 for BioT5+ base variants. "
            f"Got {model_cfg.get('d_substrate')}. If using a different BioT5+ variant, "
            f"update this assertion and ProjectionLayer input dim accordingly."
        )
        assert "biot5" in lr_cfg, "lr.biot5 required when substrate_encoder is 'biot5'"
        assert lr_cfg["biot5"] < lr_cfg.get("task", 1e-4), (
            f"lr.biot5 ({lr_cfg['biot5']}) should be < lr.task ({lr_cfg.get('task')})"
        )
        # ConditionEncoder
        assert model_cfg.get("d_cond", 64) > 0, "d_cond must be > 0"
        # BioT5+ LoRA validation
        biot5_lora = model_cfg.get("biot5_lora", {})
        if biot5_lora.get("enabled", False):
            assert biot5_lora.get("r", 0) > 0, "biot5_lora.r must be > 0"
            assert biot5_lora.get("alpha", 0) > 0, "biot5_lora.alpha must be > 0"
            assert biot5_lora.get("target_modules"), "biot5_lora.target_modules cannot be empty"
            assert 0 <= biot5_lora.get("dropout", 0) < 1, "biot5_lora.dropout must be in [0,1)"
    else:
        # MoLFormer validation (existing D2 logic)
        molformer_frozen = model_cfg.get("molformer_frozen", False)
        mol_lora_enabled = model_cfg.get("molformer_lora", {}).get("enabled", False)
        if not molformer_frozen:
            assert "molformer" in lr_cfg, "lr.molformer must be set when molformer_frozen is false"
            assert lr_cfg["molformer"] < lr_cfg.get("task", 1e-4), (
                f"lr.molformer ({lr_cfg['molformer']}) should be < lr.task ({lr_cfg.get('task')})"
            )
        elif mol_lora_enabled:
            assert "molformer_lora" in lr_cfg, (
                "lr.molformer_lora must be set when molformer_lora.enabled is true"
            )

    # ESM3 LoRA validation (optional via enabled flag)
    lora_cfg = model_cfg.get("esm3_lora", {})
    esm3_lora_enabled = lora_cfg.get("enabled", False)
    if esm3_lora_enabled and lora_cfg:
        assert lora_cfg.get("r", 16) > 0, "esm3_lora.r must be > 0"
        assert lora_cfg.get("alpha", 32) > 0, "esm3_lora.alpha must be > 0"
        assert 0.0 <= lora_cfg.get("dropout", 0.05) < 1.0, (
            "esm3_lora.dropout must be in [0, 1)"
        )
        target_modules = lora_cfg.get("target_modules", [])
        assert len(target_modules) > 0, "esm3_lora.target_modules must not be empty"
        assert "esm3_lora" in lr_cfg, "lr.esm3_lora must be set when ESM3 LoRA is enabled"
        assert lr_cfg["esm3_lora"] < lr_cfg.get("task", 1e-4), (
            f"lr.esm3_lora ({lr_cfg['esm3_lora']}) should be < lr.task"
        )

    # MLP head validation
    hidden_dims = mlp_cfg.get("hidden_dims", [])
    dropouts = mlp_cfg.get("dropouts", [])
    assert len(hidden_dims) == len(dropouts), (
        f"mlp.hidden_dims length ({len(hidden_dims)}) must equal "
        f"mlp.dropouts length ({len(dropouts)})"
    )

    # Effective batch size consistency
    num_gpus = train_cfg["num_gpus"]
    bsz = train_cfg["batch_size_per_gpu"]
    accum = train_cfg["gradient_accumulation_steps"]
    eff_bsz = train_cfg.get("effective_batch_size")
    if eff_bsz is not None:
        expected = num_gpus * bsz * accum
        assert eff_bsz == expected, (
            f"effective_batch_size={eff_bsz} != num_gpus({num_gpus}) × "
            f"batch_size_per_gpu({bsz}) × gradient_accumulation_steps({accum}) = {expected}"
        )

    # Standard training validations
    assert train_cfg["batch_size_per_gpu"] > 0, "batch_size_per_gpu must be > 0"
    warmup = train_cfg["scheduler"]["warmup_steps"]
    assert warmup >= 0, f"warmup_steps must be >= 0, got {warmup}"
    clip = train_cfg["gradient_clip_norm"]
    assert clip > 0, f"gradient_clip_norm must be > 0, got {clip}"

    # Loss function validation (S12-08: accept both "huber" and "mse"; S14-01: also "huber_delta4")
    loss_type = train_cfg.get("loss", "huber")
    assert loss_type in ("huber", "mse", "huber_delta4"), (
        f"training.loss must be 'huber', 'mse', or 'huber_delta4', got '{loss_type}'"
    )
    # BioT5+ max length (S12-08: flexible, not hardcoded to 128)
    biot5_len = model_cfg.get("biot5_max_length", 128)
    assert biot5_len > 0, f"biot5_max_length must be > 0, got {biot5_len}"

    logger.info("[validate_config] Config validation passed (encoder=%s).", encoder_type)
