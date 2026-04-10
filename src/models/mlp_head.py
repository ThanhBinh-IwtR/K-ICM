"""MLP prediction head for kcat regression."""
from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


def _get_activation(name: str) -> nn.Module:
    """Return an activation module by name.

    Args:
        name: One of ``'gelu'``, ``'relu'``, ``'silu'``.

    Returns:
        Activation module instance.

    Raises:
        ValueError: If name is not recognised.
    """
    activations = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
    }
    name = name.lower()
    if name not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: {list(activations)}"
        )
    return activations[name]


class MLPHead(nn.Module):
    """Multi-layer perceptron regression head with optional residual connection.

    Architecture (from config, d_in=512):
        LayerNorm(d_in=512)
        Linear(512, 512) → GELU → Dropout(0.3) + residual + LayerNorm
        Linear(512, 256)  → GELU → Dropout(0.2)
        Linear(256, 128)  → GELU → Dropout(0.1)
        Linear(128, 1)

    When d_in == hidden_dims[0], a residual connection is added around the
    first layer to improve gradient flow (especially for ESM-3 fine-tuning
    where gradients must flow back through cross-attention into MLP).

    Input is the fused protein+substrate embedding [B, d_in].
    Output is a scalar log10(kcat) prediction [B].

    Args:
        d_in: Input dimensionality (= d_proj = 512 with cross-attention).
        hidden_dims: Sequence of hidden layer widths (default [512, 256, 128]).
        dropouts: Dropout probability after each hidden layer except the last.
            Must have same length as hidden_dims. Last entry may be 0.0.
        activation: Activation function name. Defaults to ``'gelu'``.

    Component Contract:
        forward([B, d_in]) -> [B]  (scalar predictions, no squeeze to 1D)
    """

    def __init__(
        self,
        d_in: int = 512,
        hidden_dims: Sequence[int] = (512, 256, 128),
        dropouts: Sequence[float] = (0.2, 0.1, 0.0),
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        if len(dropouts) != len(hidden_dims):
            raise ValueError(
                f"len(dropouts)={len(dropouts)} must equal "
                f"len(hidden_dims)={len(hidden_dims)}"
            )

        # Input normalization
        self.input_norm = nn.LayerNorm(d_in)

        # First layer with optional residual (only when d_in == hidden_dims[0])
        self.use_residual = (d_in == hidden_dims[0])
        first_drop = dropouts[0]
        layer0_parts: list[nn.Module] = [
            nn.Linear(d_in, hidden_dims[0]),
            _get_activation(activation),
        ]
        if first_drop > 0.0:
            layer0_parts.append(nn.Dropout(p=first_drop))
        self.layer0 = nn.Sequential(*layer0_parts)

        if self.use_residual:
            self.residual_norm = nn.LayerNorm(hidden_dims[0])

        # Remaining hidden layers (no residual — dims change)
        remaining: list[nn.Module] = []
        for i in range(1, len(hidden_dims)):
            remaining.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            remaining.append(_get_activation(activation))
            if dropouts[i] > 0.0:
                remaining.append(nn.Dropout(p=dropouts[i]))
        self.remaining = nn.Sequential(*remaining)

        # Output head: hidden_dims[-1] → 1
        self.out = nn.Linear(hidden_dims[-1], 1)

        logger.debug(
            "[MLPHead] d_in=%d  hidden=%s  dropouts=%s  activation=%s  residual=%s",
            d_in,
            list(hidden_dims),
            list(dropouts),
            activation,
            self.use_residual,
        )

    def forward_hidden(self, x: Tensor) -> Tensor:
        """Run all hidden layers, return intermediate representation.

        Args:
            x: FloatTensor [B, d_in].

        Returns:
            FloatTensor [B, hidden_dims[-1]] — hidden representation before output linear.
        """
        x = self.input_norm(x)
        h = self.layer0(x)
        if self.use_residual:
            h = self.residual_norm(x + h)
        h = self.remaining(h)
        return h

    def forward_output(self, h: Tensor) -> Tensor:
        """Apply final output linear to hidden representation.

        Args:
            h: FloatTensor [B, hidden_dims[-1]].

        Returns:
            FloatTensor [B] — scalar predictions.
        """
        return self.out(h).squeeze(-1)  # [B, 1] → [B]

    def forward(self, x: Tensor) -> Tensor:
        """Predict log10(kcat) from fused embedding.

        Args:
            x: FloatTensor [B, d_in].

        Returns:
            FloatTensor [B] — scalar predictions (not [B, 1]).
        """
        return self.forward_output(self.forward_hidden(x))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)

    head = MLPHead(d_in=512, hidden_dims=(512, 256, 128), dropouts=(0.2, 0.1, 0.0))
    x = torch.randn(8, 512)
    out = head(x)
    print("MLPHead output shape:", out.shape)   # [8]
    assert out.shape == (8,), f"Expected [8], got {out.shape}"
    assert head.use_residual, "Residual should be enabled for d_in=512, hidden[0]=512"

    # Test config-driven dims (no residual when dims differ)
    head2 = MLPHead(d_in=512, hidden_dims=(256,), dropouts=(0.1,))
    out2 = head2(torch.randn(4, 512))
    assert out2.shape == (4,)
    assert not head2.use_residual, "Residual should be disabled for d_in=512, hidden[0]=256"

    print("mlp_head.py smoke test PASSED")
