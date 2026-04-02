"""ESM-3 encoder with last-layer extraction via forward hook.

Uses EvolutionaryScale ESM-3 (esm==3.2.3):
    Model:    esm3_sm_open_v1
    d_model:  1536
    Layers:   48 transformer blocks (model.transformer.blocks)
    Hook:     model.transformer.blocks[-1] — last block

D2 LoRA: ESM-3 backbone is frozen; only LoRA adapters (via PEFT) are trainable.
The tokenizer (EsmSequenceTokenizer) is intentionally excluded from this module.
The collator is responsible for tokenisation; this encoder receives pre-tokenised
sequence_tokens [B, L].
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

# ESM-3 architecture constants
_ESM3_D_MODEL: int = 1536
_ESM3_N_LAYERS: int = 48
_ESM3_MODEL_NAME: str = "esm3_sm_open_v1"


# ---------------------------------------------------------------------------
# ESM3Encoder
# ---------------------------------------------------------------------------


class ESM3Encoder(nn.Module):
    """Wrap ESM-3 and extract hidden states from the last transformer layer.

    The last layer is accessed via a forward hook registered on
    ``model.transformer.blocks[-1]``. ESM-3 blocks return a single tensor
    (not a tuple), so the hook captures ``output`` directly.

    Args:
        model_name: ESM-3 model identifier. Default: ``"esm3_sm_open_v1"``.
        d_model: Hidden dimension. Default: 1536 (ESM-3 small).

    Component Contracts:
        forward(sequence_tokens [B,L], padding_mask [B,L]) -> (hidden [B,L,1536], padding_mask [B,L])
        freeze() — freeze all ESM-3 parameters
        apply_lora(r, alpha, dropout, target_modules) — wrap ESM-3 with LoRA adapters
        lora_parameters() — yield only LoRA adapter parameters
    """

    def __init__(
        self,
        model_name: str = _ESM3_MODEL_NAME,
        d_model: int = _ESM3_D_MODEL,
    ) -> None:
        super().__init__()

        from esm.models.esm3 import ESM3  # noqa: PLC0415 — deferred heavy import

        self.esm3: ESM3 = ESM3.from_pretrained(model_name)
        self.d_model: int = d_model

        # Last-layer hidden states captured by forward hook
        self._penultimate_hidden: Optional[Tensor] = None
        self._hook = self.esm3.transformer.blocks[-1].register_forward_hook(
            self._capture_penultimate
        )

        logger.info(
            "[ESM3Encoder] Loaded %s  d_model=%d  n_layers=%d  hook on block[-1]",
            model_name,
            d_model,
            len(self.esm3.transformer.blocks),
        )

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def _capture_penultimate(
        self,
        module: nn.Module,  # noqa: ARG002
        input: tuple,       # noqa: ARG002
        output: Tensor,
    ) -> None:
        """Forward hook: store penultimate block output.

        ESM-3 transformer blocks return a single Tensor [B, L, D],
        not a tuple — capture it directly.
        """
        self._penultimate_hidden = output

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sequence_tokens: Tensor,
        padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run ESM-3 and return penultimate-layer hidden states.

        Args:
            sequence_tokens: LongTensor [B, L] — pre-tokenised sequences
                (BOS and EOS added by EsmSequenceTokenizer in the collator).
            padding_mask: BoolTensor [B, L] — True at padding positions
                (derived from attention_mask == 0 in the collator).

        Returns:
            Tuple of:
                hidden — FloatTensor [B, L, d_model] from penultimate block
                padding_mask — BoolTensor [B, L] (passed through unchanged)
        """
        self._penultimate_hidden = None

        # Move inputs to ESM-3 device (auto-loads to CUDA if available)
        _esm_device = next(self.esm3.parameters()).device
        _input_device = sequence_tokens.device
        sequence_tokens = sequence_tokens.to(_esm_device)

        # ESM-3 loads as bfloat16 on CUDA but internally creates some intermediate
        # tensors (average_plddt) as float32 — autocast resolves the dtype mismatch.
        # On CPU (float32 model) autocast is a no-op.
        _autocast_device = _esm_device.type  # "cuda" or "cpu"
        with torch.autocast(_autocast_device, dtype=torch.bfloat16, enabled=(_esm_device.type == "cuda")):
            _ = self.esm3(sequence_tokens=sequence_tokens)

        if self._penultimate_hidden is None:
            raise RuntimeError(
                "[ESM3Encoder] Last-layer hook did not fire. "
                "Check that model.transformer.blocks[-1] exists."
            )

        # Cast to float32 and move back to the caller's device so downstream
        # modules (AttentivePooling, projections) don't hit device mismatches.
        hidden = self._penultimate_hidden.float().to(_input_device)  # [B, L, d_model]
        return hidden, padding_mask

    # ------------------------------------------------------------------
    # Freezing & LoRA
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze all ESM-3 parameters."""
        for p in self.esm3.parameters():
            p.requires_grad_(False)
        logger.info("[ESM3Encoder] All parameters frozen.")

    def apply_lora(
        self,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ) -> None:
        """Wrap ESM-3 with LoRA adapters via PEFT.

        Must be called AFTER freeze(). The hook on blocks[-1] survives
        because PEFT wraps linear layers inside the block, not the block itself.

        Args:
            r: LoRA rank.
            alpha: LoRA scaling factor.
            dropout: Dropout on LoRA layers.
            target_modules: Linear layer names to adapt. Default: ["q_proj", "v_proj"].
        """
        if target_modules is None:
            target_modules = ["layernorm_qkv.1"]  # verified from named_modules scan

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.esm3 = get_peft_model(self.esm3, lora_config)

        n_lora = sum(p.numel() for p in self.esm3.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.esm3.parameters())
        logger.info(
            "[ESM3Encoder] LoRA applied: r=%d alpha=%d dropout=%.2f targets=%s  "
            "trainable=%d / %d (%.2f%%)",
            r, alpha, dropout, target_modules,
            n_lora, n_total, 100.0 * n_lora / n_total,
        )

    def lora_parameters(self):
        """Yield only LoRA adapter parameters (for optimizer param groups)."""
        for name, param in self.esm3.named_parameters():
            if param.requires_grad:
                yield param

    def __del__(self) -> None:
        """Remove the forward hook when the encoder is garbage-collected."""
        if hasattr(self, "_hook"):
            self._hook.remove()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)

    enc = ESM3Encoder()

    enc.freeze()
    enc.apply_lora(r=16, alpha=32, dropout=0.05)

    # Verify hook survived PEFT wrapping
    assert enc._penultimate_hidden is None, "Hook buffer should be None before forward"

    # Verify LoRA parameters are trainable
    lora_params = list(enc.lora_parameters())
    assert len(lora_params) > 0, "No LoRA parameters found"
    print(f"LoRA trainable params: {sum(p.numel() for p in lora_params)}")

    # Determine the device the model loaded onto (GPU if available, else CPU)
    _device = next(enc.esm3.parameters()).device

    B, L = 2, 50
    tokens = torch.randint(0, 32, (B, L), device=_device)
    padding_mask = torch.zeros(B, L, dtype=torch.bool, device=_device)

    with torch.no_grad():
        hidden, mask = enc(tokens, padding_mask)

    print(f"hidden shape: {hidden.shape}")   # [2, 50, 1536]

    assert hidden.shape == (B, L, enc.d_model), f"Unexpected hidden shape: {hidden.shape}"
    print("esm3_encoder.py smoke test PASSED (LoRA + hook survival verified)")