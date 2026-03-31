"""Projection layer: Linear → GELU for substrate branch embedding."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class ProjectionLayer(nn.Module):
    """Single-layer projection: (optional LayerNorm →) Linear(d_in, d_out) → GELU activation.

    D1 usage: substrate_proj = ProjectionLayer(512, 512, normalize_input=False)
              (Linear(512→512) + GELU, no LayerNorm — Uni-Mol output already normalized)
    Only substrate_proj is used — ESM-3 residues go directly to CrossAttentionFusion K/V.

    Args:
        d_in: Input dimensionality.
        d_out: Output dimensionality. Defaults to 512.
        normalize_input: When True, apply LayerNorm(d_in) before the Linear.
            Stabilises training when encoder outputs have varying scale
            (recommended: True). Defaults to False for backward compatibility.

    Component Contract:
        forward([B, d_in]) -> [B, d_out]
    """

    def __init__(self, d_in: int, d_out: int = 512, normalize_input: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if normalize_input:
            layers.append(nn.LayerNorm(d_in))
        layers.extend([nn.Linear(d_in, d_out), nn.GELU()])
        self.proj = nn.Sequential(*layers)
        logger.debug(
            "[ProjectionLayer] d_in=%d  d_out=%d  normalize_input=%s",
            d_in, d_out, normalize_input,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project input tensor.

        Args:
            x: FloatTensor [B, d_in].

        Returns:
            FloatTensor [B, d_out].
        """
        return self.proj(x)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)

    proj_plm = ProjectionLayer(d_in=1280, d_out=512)
    proj_sub = ProjectionLayer(d_in=300, d_out=512)

    x_plm = torch.randn(4, 1280)
    x_sub = torch.randn(4, 300)

    out_plm = proj_plm(x_plm)
    out_sub = proj_sub(x_sub)

    print("pLM projection output:", out_plm.shape)      # [4, 512]
    print("substrate projection output:", out_sub.shape)  # [4, 512]
    assert out_plm.shape == (4, 512)
    assert out_sub.shape == (4, 512)
    print("projection.py smoke test PASSED")
