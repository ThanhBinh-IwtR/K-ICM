"""
MoLFormerEncoder — wraps IBM MoLFormer-XL as a trainable PyTorch nn.Module.

MoLFormer (Ross et al., 2022) is a transformer LM trained on 1.1B SMILES strings.
Unlike Uni-Mol, MoLFormer:
  - Runs entirely on GPU (no CPU roundtrip)
  - Requires no pre-computed cache
  - Is trainable — representations adapt to kcat prediction task
  - Has no atom-level outputs (pooled sequence only)

Output: cls_repr [B, d_molformer] — molecular embedding
        where d_molformer = 768 (MoLFormer-XL hidden dim)

Used in Phase D2 as substrate encoder replacing Uni-Mol.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

D_MOLFORMER = 768   # MoLFormer-XL hidden dim


class MoLFormerEncoder(nn.Module):
    """
    Trainable MoLFormer-XL encoder for SMILES → molecular embedding.

    Tokenization is handled internally. Accepts raw SMILES strings,
    returns float32 embeddings on the caller's device.

    Args:
        model_name:  HuggingFace model identifier
        max_length:  Maximum SMILES token length (MoLFormer max = 202)
        use_pooler:  If True and pooler_output available, use it.
                     Else: attention-masked mean pool of last_hidden_state.
        d_output:    Expected output dim. Must equal D_MOLFORMER.
    """

    MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

    def __init__(
        self,
        model_name: str  = MODEL_NAME,
        max_length: int  = 202,
        use_pooler: bool = True,
        d_output:   int  = D_MOLFORMER,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.use_pooler = use_pooler
        self.d_output   = d_output

        # Load tokenizer and model
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            model_name,
            deterministic_eval=True,
            trust_remote_code=True,
        )

        # Verify output dim
        actual_d = self._model.config.hidden_size
        assert actual_d == d_output, (
            f"[MoLFormerEncoder] d_output mismatch: expected {d_output}, "
            f"model has hidden_size={actual_d}. Update D_MOLFORMER constant."
        )

        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            f"[MoLFormerEncoder] Loaded {model_name}  "
            f"d_output={d_output}  params={n_params/1e6:.1f}M"
        )

        self._frozen = False

    # ------------------------------------------------------------------
    # Freeze / LoRA helpers
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze ALL parameters in the underlying MoLFormer model."""
        for param in self._model.parameters():
            param.requires_grad_(False)
        self._frozen = True
        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "[MoLFormerEncoder] All parameters frozen. "
            "(%.1fM params, 0 trainable)", n_params / 1e6,
        )

    def apply_lora(
        self,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: list[str],
    ) -> None:
        """Wrap the frozen MoLFormer with LoRA adapters (peft).

        Must be called AFTER freeze() — peft adds new trainable adapters
        on top of frozen base weights.
        """
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            inference_mode=False,
        )
        self._model = get_peft_model(self._model, lora_config)

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        pct = 100.0 * trainable / total if total > 0 else 0.0
        logger.info(
            "[MoLFormerEncoder] LoRA applied  r=%d  alpha=%d  dropout=%.2f  "
            "trainable=%d (%.2f%%)",
            r, alpha, dropout, trainable, pct,
        )

    def lora_parameters(self):
        """Yield only parameters where requires_grad is True.

        Used by kicm_module.py to build the molformer_lora optimizer group.
        """
        for param in self._model.parameters():
            if param.requires_grad:
                yield param

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, smiles_list: list[str]) -> Tensor:
        """
        Args:
            smiles_list: list[str] of length B

        Returns:
            Tensor [B, d_output] float32 — molecular embeddings
        """
        device = next(self._model.parameters()).device

        # Tokenize
        inputs = self._tokenizer(
            smiles_list,
            padding    = True,
            truncation = True,
            max_length = self.max_length,
            return_tensors = "pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward under autocast (matches ESM-3 precision)
        with torch.autocast(device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            outputs = self._model(**inputs)

        # Extract pooled molecular representation
        if self.use_pooler and outputs.pooler_output is not None:
            embedding = outputs.pooler_output   # [B, d]
        else:
            # Attention-masked mean pool
            mask   = inputs["attention_mask"].unsqueeze(-1).float()   # [B, S, 1]
            hidden = outputs.last_hidden_state.float()                 # [B, S, d]
            embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return embedding.float()   # ensure float32 for downstream

    def extra_repr(self) -> str:
        return (
            f"model={self.model_name}  "
            f"d_output={self.d_output}  "
            f"use_pooler={self.use_pooler}  "
            f"max_length={self.max_length}"
        )
