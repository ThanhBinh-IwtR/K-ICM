"""
CrossAttentionFusion — B1 unidirectional (substrate → protein).

Architecture:
  Q = h_sub [B, n_queries, d_sub]  — substrate queries protein (multi-query)
  K = V = protein_residues [B, L, d_prot]
  ctx = LayerNorm(Q + CrossAttn(Q, K, V))  — residual preserves substrate identity
  output = mean(ctx, dim=1) → [B, d_sub] → directly into MLP head

When n_queries=1 (default), behaviour is identical to the original single-query
implementation.  n_queries>1 projects h_sub into multiple learned queries via a
Linear layer, attends each query over the protein residues, then mean-pools
the query outputs back to [B, d_sub].

Output shape: [B, d_sub].
"""

from __future__ import annotations

import logging

import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        d_sub:     int   = 512,
        d_prot:    int   = 1536,
        n_heads:   int   = 8,
        dropout:   float = 0.1,
        n_queries: int   = 1,
    ):
        super().__init__()
        self.d_sub     = d_sub
        self.d_prot    = d_prot
        self.n_queries = n_queries

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

        # Multi-query projection: only when n_queries > 1
        if n_queries > 1:
            self.query_proj = nn.Linear(d_sub, n_queries * d_sub, bias=True)
            nn.init.normal_(self.query_proj.weight, std=0.02)
            nn.init.zeros_(self.query_proj.bias)

        logger.info("[CrossAttentionFusion] n_queries=%d", n_queries)

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
        B = h_sub.shape[0]

        if self.n_queries == 1:
            q = h_sub.unsqueeze(1)                              # [B, 1, d_sub]
        else:
            q = self.query_proj(h_sub)                          # [B, n_queries * d_sub]
            q = q.view(B, self.n_queries, self.d_sub)           # [B, n_queries, d_sub]

        prot_normed = self.protein_norm(protein_residues)    # [B, L, d_prot]
        attn_out, _ = self.sub_to_prot(
            query=q,
            key=prot_normed,
            value=prot_normed,
            key_padding_mask=protein_pad_mask,
        )                                                    # [B, n_queries, d_sub]
        ctx_residual = self.norm1(q + attn_out)              # [B, n_queries, d_sub]
        ctx = ctx_residual.mean(dim=1)                       # [B, d_sub]
        return ctx

    def extra_repr(self) -> str:
        return f"d_sub={self.d_sub}, d_prot={self.d_prot}, n_queries={self.n_queries}"
