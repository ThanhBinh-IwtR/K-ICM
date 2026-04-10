"""
BioT5PlusEncoder — wraps BioT5+ T5EncoderModel as a trainable substrate encoder.

BioT5+ (Pei et al., 2024) is a T5-based model pretrained on:
  - ZINC20 molecules (SELFIES format)
  - UniRef50 protein sequences
  - Biomedical literature

Unlike MoLFormer (pooled output), BioT5PlusEncoder returns per-token
hidden states [B, S, 768], enabling multi-query cross-attention where
each SMILES token can independently attend to protein residues.

Input:  Pre-tokenized input_ids and attention_mask (from KICMCollator)
Output: (last_hidden_state [B, S, 768], mask [B, S] bool True=valid)
"""
from __future__ import annotations
import logging
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

D_BIOT5 = 768   # BioT5+ base hidden dim
# BioT5+ mol-instructions (molecule): fine-tuned on Mol-Instructions molecule tasks.
# Produces richer semantic molecular representations vs base pretrained model.
# d_model = 768 — identical to base architecture, no downstream changes needed.
BIOT5_MODEL = "QizhiPei/biot5-plus-base"
BIOT5_MAX_LENGTH = 128   # typical SELFIES molecule < 100 tokens


class BioT5PlusEncoder(nn.Module):
    """
    Trainable BioT5+ encoder returning per-token molecular representations.

    Returns [B, S, d_output] — NOT pooled.
    Caller (CrossAttentionFusion) handles pooling after attending to protein.

    Forward accepts pre-tokenized tensors (tokenization is in KICMCollator).
    """

    def __init__(
        self,
        model_name:  str = BIOT5_MODEL,
        max_length:  int = BIOT5_MAX_LENGTH,
        d_output:    int = D_BIOT5,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.d_output   = d_output

        from transformers import T5EncoderModel
        self._model = T5EncoderModel.from_pretrained(model_name)

        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "[BioT5PlusEncoder] Loaded %s  d_output=%d  params=%.1fM  trainable=True",
            model_name, d_output, n_params / 1e6,
        )

        self._frozen = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            input_ids:      [B, S] int64 — pre-tokenized by KICMCollator
            attention_mask: [B, S] int64 — 1=valid, 0=padding (T5 convention)

        Returns:
            hidden: [B, S, 768]  float32
            mask:   [B, S] bool — True=valid, False=padding
                    Convention: True=valid (matches CrossAttentionFusion pool_proj expectation)
        """
        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden = outputs.last_hidden_state.float()   # [B, S, 768]
        # Convention: True=valid, False=padding
        # CrossAttentionFusion pool_proj uses masked_fill(~mask, -inf)
        # → ~mask is True where mask is False (padding) → padding gets -inf ✓
        mask = attention_mask.bool()
        assert mask.dtype == torch.bool, "biot5_valid_mask must be bool"  # S13-05

        return hidden, mask

    def freeze(self) -> None:
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._frozen = True
        logger.info("[BioT5PlusEncoder] All parameters frozen.")

    def apply_lora(self, r: int, alpha: int, dropout: float, target_modules: list[str]) -> None:
        """Apply LoRA adapters to BioT5+ T5EncoderModel."""
        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=target_modules,
        )
        self._model = get_peft_model(self._model, lora_config)
        n_lora = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logger.info(
            "[BioT5PlusEncoder] LoRA applied: r=%d, alpha=%d, trainable=%.2fM params",
            r, alpha, n_lora / 1e6,
        )
        self._lora_enabled = True

    def lora_parameters(self):
        """Yield only trainable parameters (LoRA adapters when LoRA is applied)."""
        for p in self._model.parameters():
            if p.requires_grad:
                yield p

    def extra_repr(self) -> str:
        return f"model={self.model_name}  d_output={self.d_output}"
