"""
CrossAttentionFusion — multi-query B1 unidirectional attention.

Substrate token sequence queries protein residues. Each SMILES token
(representing a molecular fragment) attends independently to protein
residues, then a learned pooling aggregates token-level context.

This replaces the single-vector design (pooler_output as Q) with
per-token queries from BioT5+ last_hidden_state.
"""
import torch
import torch.nn as nn
from torch import Tensor


class CrossAttentionFusion(nn.Module):
    """
    Multi-query B1 cross-attention: substrate tokens query protein residues.

    Args:
        d_sub:    substrate token dim (= d_proj = 512)
        d_prot:   protein hidden dim  (= d_esm3 = 1536)
        n_heads:  attention heads
        dropout:  attention dropout
    """

    def __init__(
        self,
        d_sub:   int   = 512,
        d_prot:  int   = 1536,
        n_heads: int   = 8,
        dropout: float = 0.1,
        ffn_dropout: float = None,
    ):
        super().__init__()
        self.d_sub  = d_sub
        self.d_prot = d_prot

        # Protein residue pre-projection (before MHA)
        # Reduces 1536→512 with non-linearity, matching substrate branch treatment
        # Decomposed from nn.Sequential so LayerNorm gets a named parameter path
        # (prevents weight-decay leakage via numeric "0" from nn.Sequential — S15-01)
        self.protein_proj_norm   = nn.LayerNorm(d_prot)
        self.protein_proj_linear = nn.Linear(d_prot, d_sub)
        self.protein_proj_act    = nn.GELU()
        self.sub_to_prot  = nn.MultiheadAttention(
            embed_dim   = d_sub,
            kdim        = d_sub,
            vdim        = d_sub,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm1 = nn.LayerNorm(d_sub)

        # Post-attention FFN (standard transformer block completion)
        _ffn_dropout = ffn_dropout if ffn_dropout is not None else dropout
        self.ffn = nn.Sequential(
            nn.Linear(d_sub, 4 * d_sub),
            nn.GELU(),
            nn.Dropout(_ffn_dropout),
            nn.Linear(4 * d_sub, d_sub),
        )
        self.ffn_norm = nn.LayerNorm(d_sub)

        # Learned pooling: score each substrate token by its
        # interaction relevance with protein, then softmax-weight sum.
        self.pool_proj = nn.Linear(d_sub, 1, bias=False)
        nn.init.normal_(self.pool_proj.weight, std=0.02)

    def forward(
        self,
        h_sub_seq:        Tensor,   # [B, S, d_sub]  substrate token sequence
        protein_res:      Tensor,   # [B, L, d_prot] protein residues
        protein_pad:      Tensor,   # [B, L]         True = padding position  (ESM convention)
        biot5_valid_mask: Tensor,   # [B, S]         True = valid substrate token (CA convention)
    ) -> Tensor:
        """
        Mask conventions (M-01):
            protein_pad:      [B, L] bool, True = PADDING (ESM-3 convention)
            biot5_valid_mask:  [B, S] bool, True = VALID   (CrossAttention convention)

        Returns:
            ctx [B, d_sub] — substrate-protein context
        """
        # NaN-safe pooling (C-01) handles all-masked edge case gracefully

        # Cross-attention: substrate tokens attend to projected protein
        kv = self.protein_proj_act(self.protein_proj_linear(self.protein_proj_norm(protein_res)))  # [B, L, d_sub]  # S15-01
        attn_out, _ = self.sub_to_prot(
            query            = h_sub_seq,
            key              = kv,
            value            = kv,
            key_padding_mask = protein_pad,
        )                                                  # [B, S, d_sub]
        x = self.norm1(h_sub_seq + attn_out)               # [B, S, d_sub]

        # Zero padding positions before FFN to prevent LayerNorm contamination
        if biot5_valid_mask is not None:
            x = x * biot5_valid_mask.unsqueeze(-1).float()

        # FFN block (standard transformer completion)
        ffn_out = self.ffn(x)
        if biot5_valid_mask is not None:
            ffn_out = ffn_out * biot5_valid_mask.unsqueeze(-1).float()  # zero padding BEFORE norm
        x = self.ffn_norm(x + ffn_out)                     # [B, S, d_sub]

        # Zero padding again after FFN+norm (norm can leak nonzero at padding positions)
        if biot5_valid_mask is not None:
            x = x * biot5_valid_mask.unsqueeze(-1).float()

        # Learned attention pooling over substrate tokens
        logits = self.pool_proj(x).squeeze(-1)              # [B, S]
        logits = logits.masked_fill(~biot5_valid_mask, float('-inf'))  # mask padding

        # NaN guard: if all tokens masked (edge case), fall back to uniform attention
        # over valid tokens. Prevents softmax(all -inf) = NaN corrupting gradients.
        all_masked = (~biot5_valid_mask).all(dim=-1, keepdim=True)  # [B, 1]
        logits = torch.where(
            all_masked.expand_as(logits),
            torch.zeros_like(logits).masked_fill(~biot5_valid_mask, float('-inf')),
            logits
        )
        # Second guard: replace any remaining full-mask rows with uniform (0.0) logits
        # so softmax produces 1/S uniform weights rather than NaN
        safe_logits = logits.clone()
        safe_logits[all_masked.squeeze(-1)] = 0.0           # uniform for all-masked rows

        weights = torch.softmax(safe_logits, dim=-1)        # [B, S]
        ctx = (weights.unsqueeze(-1) * x).sum(dim=1)        # [B, d_sub]

        if self.training:
            assert not ctx.isnan().any(), (
                "CrossAttentionFusion: NaN detected in ctx after pooling. "
                f"biot5_valid_mask valid counts: {biot5_valid_mask.sum(dim=-1).tolist()}"
            )

        return ctx

    def extra_repr(self) -> str:
        return f"d_sub={self.d_sub}, d_prot={self.d_prot}, protein_proj_norm+linear={self.d_prot}→{self.d_sub}, ffn=4x{self.d_sub}"
