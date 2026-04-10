"""Full KICMModel assembly — Phase E architecture.

Phase E: ESM-3 (frozen, optional LoRA) + BioT5+ (trainable) + Multi-Query CA + FiLM.
"""
from __future__ import annotations

import logging

import torch.nn as nn
from torch import Tensor

from .esm3_encoder import ESM3Encoder
from .biot5_encoder import BioT5PlusEncoder, D_BIOT5
from .mlp_head import MLPHead
from .projection import ProjectionLayer
from .cross_attention import CrossAttentionFusion
from .condition_encoder import ConditionEncoder

logger = logging.getLogger(__name__)


class KICMModel(nn.Module):
    """End-to-end model for kinetic parameter prediction.

    Phase E data flow:
        sequence_tokens [B,L] + esm_pad_mask [B,L] (True=padding)
            -> ESM3Encoder (frozen, optional LoRA) -> hidden [B, L, 1536]

        biot5_input_ids [B,S] + biot5_attention_mask [B,S] (pre-tokenized)
            -> BioT5PlusEncoder (trainable) -> h_sub_raw [B, S, 768], biot5_valid_mask [B, S] (True=valid)
            -> substrate_proj -> h_sub_seq [B, S, 512]

        CrossAttentionFusion multi-query (Q=h_sub_seq, K/V=hidden) -> ctx [B, 512]
        ConditionEncoder double-stage FiLM:
          Stage 1: gamma1*ctx + beta1 [B, 512] after cross-attention
          Stage 2: gamma2*h + beta2 [B, 128] before MLP output layer
        MLPHead -> predictions [B]
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        mcfg = config.get("model", config)

        # --- Protein encoder: ESM3 frozen, optional LoRA ---
        self.esm3_encoder = ESM3Encoder(
            model_name=mcfg.get("esm3_variant", "esm3_sm_open_v1"),
            d_model=mcfg.get("d_esm3", 1536),
        )
        # Note: call esm3_encoder.remove_hooks() on cleanup (M-05)
        self.esm3_encoder.freeze()

        lora_cfg = mcfg.get("esm3_lora", {})
        self._esm3_lora_enabled = lora_cfg.get("enabled", False)
        if self._esm3_lora_enabled:
            self.esm3_encoder.apply_lora(
                r=lora_cfg.get("r", 12),
                alpha=lora_cfg.get("alpha", 24),
                dropout=lora_cfg.get("dropout", 0.05),
                target_modules=lora_cfg.get("target_modules", ["layernorm_qkv.1"]),
            )
            logger.info("[KICMModel] ESM3 LoRA ENABLED r=%d", lora_cfg["r"])
        else:
            logger.info("[KICMModel] ESM3 FROZEN (LoRA disabled)")

        # --- Substrate encoder: BioT5+ (optional LoRA) ---
        d_sub = mcfg.get("d_substrate", D_BIOT5)
        self.substrate_encoder = BioT5PlusEncoder(
            model_name=mcfg.get("biot5_model", "QizhiPei/biot5-plus-base"),
            max_length=mcfg.get("biot5_max_length", 128),
            d_output=d_sub,
        )

        biot5_lora_cfg = mcfg.get("biot5_lora", {})
        self._biot5_lora_enabled = biot5_lora_cfg.get("enabled", False)
        if self._biot5_lora_enabled:
            self.substrate_encoder.apply_lora(
                r=biot5_lora_cfg["r"],
                alpha=biot5_lora_cfg["alpha"],
                dropout=biot5_lora_cfg["dropout"],
                target_modules=biot5_lora_cfg["target_modules"],
            )
            logger.info("[KICMModel] BioT5+ LoRA enabled (base frozen, LoRA trainable)")
        else:
            logger.info("[KICMModel] BioT5+ fully trainable")

        # --- Projection: 768 → 512 (sequence-level, applied per token) ---
        d_proj = mcfg.get("d_proj", 512)
        self.substrate_proj = ProjectionLayer(
            d_in=d_sub, d_out=d_proj,
            normalize_input=mcfg.get("projection_normalize_input", True),
        )

        # --- Multi-query cross-attention ---
        ca_cfg = mcfg.get("cross_attention", {})
        self.cross_attn = CrossAttentionFusion(
            d_sub=ca_cfg.get("d_sub", d_proj),
            d_prot=ca_cfg.get("d_prot", 1536),
            n_heads=ca_cfg.get("n_heads", 8),
            dropout=ca_cfg.get("dropout", 0.1),
        )

        # --- Condition encoder: pH + temperature double-stage FiLM ---
        mlp_cfg = mcfg.get("mlp", {})
        mlp_hidden_dims = tuple(mlp_cfg.get("hidden_dims", [512, 256, 128]))
        d_mlp_last = mlp_hidden_dims[-1]  # 128 — dim before output layer
        self.cond_encoder = ConditionEncoder(
            d_ctx=mcfg.get("d_fusion", 512),
            d_cond=mcfg.get("d_cond", 256),
            d_mlp_hidden=d_mlp_last,
            cofactor_fp_dim=512,  # S12-06: cofactor branch always uses 512-bit Morgan FP
        )

        # --- Regression head ---
        self.mlp_head = MLPHead(
            d_in=mcfg.get("d_fusion", 512),
            hidden_dims=mlp_hidden_dims,
            dropouts=tuple(mlp_cfg.get("dropouts", [0.2, 0.1, 0.0])),
            activation=mlp_cfg.get("activation", "gelu"),
        )

        logger.info(
            "[KICMModel] Phase E: BioT5+(trainable) + ESM3(frozen%s) + "
            "MultiQuery-CA + FiLM",
            "+LoRA" if self._esm3_lora_enabled else "",
        )

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
        cofactor_fp: Tensor,         # [B, 512] binary Morgan FP (S12-06)
    ) -> Tensor:
        """Predict target kinetic parameter for a batch.

        Mask conventions (M-01):
            esm_pad_mask:      [B, L] bool, True = PADDING (ESM-3 convention)
            biot5_valid_mask:   [B, S] bool, True = VALID   (CrossAttention convention)

        Args:
            sequence_tokens: LongTensor [B, L]
            esm_pad_mask: BoolTensor [B, L] — True at padding positions
            biot5_input_ids: LongTensor [B, S] — pre-tokenized substrate
            biot5_attention_mask: LongTensor [B, S] — 1=valid, 0=padding
            ph: FloatTensor [B] — pH values
            temp: FloatTensor [B] — temperature in Celsius
            ph_known: BoolTensor [B] — True = measured value
            temp_known: BoolTensor [B] — True = measured value

        Returns:
            FloatTensor [B] — predicted log10 target values.
        """
        # Protein branch — residue-level hidden states
        hidden, prot_mask = self.esm3_encoder(sequence_tokens, esm_pad_mask)

        # Substrate branch — returns per-token sequence, not pooled
        h_sub_raw, biot5_valid_mask = self.substrate_encoder(biot5_input_ids, biot5_attention_mask)  # [B,S,768], [B,S] True=valid
        h_sub_seq = self.substrate_proj(h_sub_raw)                  # [B,S,512]

        # Multi-query cross-attention
        ctx = self.cross_attn(h_sub_seq, hidden, prot_mask, biot5_valid_mask)  # [B,512]

        # FiLM conditioning (double-stage) with cofactor branch (S12-06)
        gamma1, beta1, gamma2, beta2 = self.cond_encoder(ph, temp, ph_known, temp_known, cofactor_fp)
        self._last_gamma = gamma1.detach()
        self._last_beta = beta1.detach()
        self._last_gamma2 = gamma2.detach()
        self._last_beta2 = beta2.detach()

        # Stage 1: FiLM after cross-attention (d_ctx=512)
        ctx = gamma1 * ctx + beta1  # [B, 512]

        # Stage 2: through MLP hidden layers, then FiLM before output (d_mlp_hidden=128)
        h = self.mlp_head.forward_hidden(ctx)    # [B, 128]
        h = gamma2 * h + beta2                   # FiLM stage 2
        return self.mlp_head.forward_output(h)   # [B]

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
