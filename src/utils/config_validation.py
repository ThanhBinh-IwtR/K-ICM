"""Config validation utilities for KICMBaseModel D2 LoRA.

Call validate_config(config) immediately after loading the YAML to catch
structural mismatches before any heavyweight model/data initialisation.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def validate_config(config: dict) -> None:
    """Assert structural consistency of the D2 LoRA training config.

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

    # Model dimension consistency (cross-attention always active in D2)
    assert model_cfg["d_proj"] == model_cfg["d_fusion"], (
        f"d_fusion ({model_cfg['d_fusion']}) must equal d_proj "
        f"({model_cfg['d_proj']})"
    )

    # Substrate encoder must be molformer in D2
    encoder_type = model_cfg.get("substrate_encoder", "molformer")
    assert encoder_type == "molformer", (
        f"D2 requires substrate_encoder='molformer', got '{encoder_type}'"
    )

    # LoRA config validation
    lora_cfg = model_cfg.get("esm3_lora", {})
    if lora_cfg:
        assert lora_cfg.get("r", 16) > 0, f"esm3_lora.r must be > 0, got {lora_cfg.get('r')}"
        assert lora_cfg.get("alpha", 32) > 0, f"esm3_lora.alpha must be > 0, got {lora_cfg.get('alpha')}"
        assert 0.0 <= lora_cfg.get("dropout", 0.05) < 1.0, (
            f"esm3_lora.dropout must be in [0, 1), got {lora_cfg.get('dropout')}"
        )
        target_modules = lora_cfg.get("target_modules", [])
        assert len(target_modules) > 0, "esm3_lora.target_modules must not be empty"

    # MLP head dimension/dropout list alignment
    hidden_dims = mlp_cfg.get("hidden_dims", [])
    dropouts = mlp_cfg.get("dropouts", [])
    assert len(hidden_dims) == len(dropouts), (
        f"mlp.hidden_dims length ({len(hidden_dims)}) must equal "
        f"mlp.dropouts length ({len(dropouts)})"
    )

    # Batch size sanity
    assert train_cfg["batch_size_per_gpu"] > 0, "batch_size_per_gpu must be > 0"

    # Warmup steps sanity
    warmup = train_cfg["scheduler"]["warmup_steps"]
    assert warmup >= 0, f"warmup_steps must be >= 0, got {warmup}"

    # Gradient clip
    clip = train_cfg["gradient_clip_norm"]
    assert clip > 0, f"gradient_clip_norm must be > 0, got {clip}"

    # LR validation
    lr_cfg = train_cfg.get("lr", {})
    assert "molformer" in lr_cfg, (
        "lr.molformer must be set for D2 config"
    )
    assert "esm3_lora" in lr_cfg, (
        "lr.esm3_lora must be set for D2 LoRA config"
    )
    assert lr_cfg["molformer"] < lr_cfg.get("task", 1e-4), (
        f"lr.molformer ({lr_cfg['molformer']}) should be < lr.task ({lr_cfg.get('task')})"
    )
    assert lr_cfg["esm3_lora"] < lr_cfg.get("task", 1e-4), (
        f"lr.esm3_lora ({lr_cfg['esm3_lora']}) should be < lr.task ({lr_cfg.get('task')})"
    )

    logger.info("[validate_config] D2 LoRA config validation passed.")
