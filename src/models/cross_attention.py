"""
CrossAttentionFusion — B1 unidirectional (substrate → protein).

Architecture:
  Q = h_sub [B, 1, d_sub]   — substrate queries protein
  K = V = protein_residues [B, L, d_prot]
  ctx = LayerNorm(h_sub + CrossAttn(Q, K, V))  — residual preserves substrate identity
  output = ctx [B, d_sub]   → directly into MLP head

Output shape: [B, d_sub].
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        d_sub:   int   = 512,
        d_prot:  int   = 1536,
        n_heads: int   = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_sub  = d_sub
        self.d_prot = d_prot

        self.protein_norm = nn.LayerNorm(d_prot)

        self.sub_to_prot = nn.MultiheadAttention(
            embed_dim=d_sub,
            num_heads=n_heads,
            kdim=d_prot,
            vdim=d_prot,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_sub)

    def forward(
        self,
        h_sub:            Tensor,  # [B, d_sub]
        protein_residues: Tensor,  # [B, L, d_prot]
        protein_pad_mask: Tensor,  # [B, L] True=padding
    ) -> Tensor:
        """
        Returns:
            ctx: [B, d_sub] — fused representation, ready for MLP head.
        """
        q = h_sub.unsqueeze(1)                              # [B, 1, d_sub]
        prot_normed = self.protein_norm(protein_residues)    # [B, L, d_prot]
        attn_out, _ = self.sub_to_prot(
            query=q,
            key=prot_normed,
            value=prot_normed,
            key_padding_mask=protein_pad_mask,
        )                                                    # [B, 1, d_sub]
        ctx = self.norm1(q + attn_out).squeeze(1)            # [B, d_sub]
        return ctx

    def extra_repr(self) -> str:
        return f"d_sub={self.d_sub}, d_prot={self.d_prot}"
