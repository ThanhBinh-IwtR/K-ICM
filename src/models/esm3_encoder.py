"""ESM-3 encoder with last-layer extraction via forward hook.

Uses EvolutionaryScale ESM-3 (esm==3.2.3):
    Model:    esm3_sm_open_v1
    d_model:  1536
    Layers:   48 transformer blocks (model.transformer.blocks)
    Hook:     model.transformer.blocks[-1] — last block

Single last-layer hook captures the most refined, highest-level
representation from the final transformer block. ESM-3 is frozen.

The tokenizer (EsmSequenceTokenizer) is intentionally excluded from this module.
The collator is responsible for tokenisation; this encoder receives pre-tokenised
sequence_tokens [B, L].
"""
from __future__ import annotations

import logging

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
        forward(sequence_tokens [B,L], esm_pad_mask [B,L]) -> (hidden [B,L,1536], esm_pad_mask [B,L])
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

        # Multi-scale hooks on last 4 transformer blocks, averaged for OOD generalization.
        # ESM-3 is frozen → zero VRAM cost (no gradients through hooked blocks).
        self._hidden_states: dict[int, Tensor] = {}
        self._hooks: list = []

        # Scalar Mix weights for 4 multi-scale hooks (S12-02)
        # Initialized to zeros → softmax gives uniform 0.25 per layer at start
        # Equivalent to current mean() — model learns optimal weights during training
        self.layer_weights = nn.Parameter(torch.zeros(4))

        for hook_idx in [-4, -3, -2, -1]:
            block = self.esm3.transformer.blocks[hook_idx]

            def _make_hook(idx: int):
                return lambda m, inp, out: self._hidden_states.__setitem__(idx, out)

            self._hooks.append(block.register_forward_hook(_make_hook(hook_idx)))

        logger.info(
            "[ESM3Encoder] Loaded %s  d_model=%d  n_layers=%d  hooks on blocks[-4:] (multi-scale)",
            model_name,
            d_model,
            len(self.esm3.transformer.blocks),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sequence_tokens: Tensor,
        esm_pad_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run ESM-3 and return multi-scale averaged hidden states.

        Averages the outputs of the last 4 transformer blocks for better
        OOD generalization. Zero VRAM cost since ESM-3 is frozen.

        Mask convention (M-01):
            esm_pad_mask: [B, L] bool, True = PADDING position

        Args:
            sequence_tokens: LongTensor [B, L] — pre-tokenised sequences
                (BOS and EOS added by EsmSequenceTokenizer in the collator).
            esm_pad_mask: BoolTensor [B, L] — True at padding positions
                (derived from attention_mask == 0 in the collator).

        Returns:
            Tuple of:
                hidden — FloatTensor [B, L, d_model] averaged from blocks[-4:]
                esm_pad_mask — BoolTensor [B, L] (passed through unchanged)
        """
        self._hidden_states.clear()

        _esm_device = next(self.esm3.parameters()).device
        _input_device = sequence_tokens.device
        sequence_tokens = sequence_tokens.to(_esm_device)

        _autocast_device = _esm_device.type
        with torch.autocast(_autocast_device, dtype=torch.bfloat16, enabled=(_esm_device.type == "cuda")):
            _ = self.esm3(sequence_tokens=sequence_tokens)

        if len(self._hidden_states) != 4:
            raise RuntimeError(
                f"[ESM3Encoder] Expected 4 hook outputs, got {len(self._hidden_states)}. "
                "Check that model.transformer.blocks[-4:] exist."
            )

        # Scalar Mix: learned weighted combination of 4 ESM-3 layers (S12-02)
        # softmax ensures weights sum to 1 and are always positive.
        # OUTSIDE autocast: layer_weights are float32; cast each hidden state to
        # float32 before weighting to avoid bfloat16 precision loss. (S13-03)
        weights = torch.softmax(self.layer_weights, dim=0)  # [4] float32
        hidden = sum(
            weights[i] * self._hidden_states[k].float()
            for i, k in enumerate([-4, -3, -2, -1])
        )
        return hidden.float().to(_input_device), esm_pad_mask

    # ------------------------------------------------------------------
    # Freezing & LoRA
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze all ESM-3 parameters and set eval mode."""
        for p in self.esm3.parameters():
            p.requires_grad_(False)
        self.esm3.eval()
        logger.info("[ESM3Encoder] All parameters frozen, model set to eval() mode.")

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

    def remove_hooks(self) -> None:
        """Explicitly remove all forward hooks. Call before model deletion or reuse."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._hidden_states.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)

    enc = ESM3Encoder()
    enc.freeze()

    # Verify 4 multi-scale hooks registered
    assert len(enc._hooks) == 4, f"Expected 4 hooks, got {len(enc._hooks)}"
    print(f"Multi-scale hooks registered: {len(enc._hooks)} ✓")

    # Apply LoRA and verify
    enc.apply_lora(r=16, alpha=32, dropout=0.05)
    lora_params = list(enc.lora_parameters())
    assert len(lora_params) > 0, "No LoRA parameters found"
    print(f"LoRA trainable params: {sum(p.numel() for p in lora_params)}")

    # Determine the device the model loaded onto (GPU if available, else CPU)
    _device = next(enc.esm3.parameters()).device

    B, L = 2, 50
    tokens = torch.randint(0, 32, (B, L), device=_device)
    esm_pad_mask = torch.zeros(B, L, dtype=torch.bool, device=_device)

    with torch.no_grad():
        hidden, mask = enc(tokens, esm_pad_mask)

    print(f"hidden shape: {hidden.shape}")   # [2, 50, 1536]
    assert hidden.shape == (B, L, enc.d_model), f"Unexpected hidden shape: {hidden.shape}"
    print("esm3_encoder.py smoke test PASSED (multi-scale hooks + LoRA verified)")
