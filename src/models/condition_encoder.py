"""ConditionEncoder and FiLMLayer for pH/temperature conditioning.

FiLM (Feature-wise Linear Modulation) applies a learned per-feature scale
and shift to the fused protein-substrate context vector:

    ctx_conditioned = γ(pH, temp) ⊙ ctx + β(pH, temp)

where γ and β are predicted by ConditionEncoder from the condition inputs.

FiLMLayer is initialized to identity (γ=1, β=0) so at epoch 0 the model
predictions are identical to the unconditioned baseline. The model gradually
learns to use pH/temp to shift predictions toward the correct experimental
regime (e.g., optimal pH for a specific enzyme family).

Missing values:
    ~94% of training samples have both pH and temperature.
    ~6% are missing one or both. Rather than dropping these samples or using
    hard masking (which discards all information about the missing condition),
    ConditionEncoder uses a learned nn.Parameter (zero-initialized) for each
    unknown condition. These "unknown embeddings" start at zero (matching the
    known-value embedding at the neutral reference point) and diverge only as
    training reveals a systematic pattern for truly unknown conditions.
    Weight decay is NOT applied to these embeddings — L2 would shrink them
    back to zero each step, preventing learning. They must be routed to the
    task_no_decay optimizer group (see kicm_module.py configure_optimizers).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation.

    Applies elementwise affine transform:  output = γ ⊙ x + β
    where γ, β ∈ ℝ^d_ctx are predicted from a d_cond-dimensional condition vector.

    Initialization:
        γ starts at 1.0 (identity scale) — achieved via zero weight + ones bias
        β starts at 0.0 (no shift)       — achieved via zero weight + zero bias
    This means at epoch 0 the layer is a no-op, and training is not destabilized.

    Args:
        d_ctx:  Dimensionality of the tensor being modulated (= d_fusion = 512).
        d_cond: Dimensionality of the conditioning vector from ConditionEncoder.
    """

    def __init__(self, d_ctx: int, d_cond: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(d_cond, d_ctx)
        self.beta_proj  = nn.Linear(d_cond, d_ctx)

        # Identity initialization: γ ≈ 1.0, β ≈ 0.0 at epoch 0
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply FiLM modulation.

        Args:
            x:    FloatTensor [B, d_ctx]  — tensor to modulate (fused ctx vector).
            cond: FloatTensor [B, d_cond] — condition vector from ConditionEncoder.

        Returns:
            FloatTensor [B, d_ctx] — modulated tensor, same shape as input.
        """
        gamma = self.gamma_proj(cond)  # [B, d_ctx]
        beta  = self.beta_proj(cond)   # [B, d_ctx]
        return gamma * x + beta

    def extra_repr(self) -> str:
        d_cond = self.gamma_proj.in_features
        d_ctx  = self.gamma_proj.out_features
        return f"d_cond={d_cond}, d_ctx={d_ctx}, init=identity"


class ConditionEncoder(nn.Module):
    """Encode pH, temperature, and cofactor fingerprint into a conditioning vector for FiLMLayer.

    For each condition (pH, temp) independently:
        known=True  → embed the scalar value via a small Linear+GELU MLP
        known=False → use a learned nn.Parameter (zero-initialized)

    cofactor_fp [B, 512] is projected to [B, 64] via cofactor_proj (always applied).

    The ph_emb(32), temp_emb(32), cofactor_proj(64) vectors are concatenated → [B, 128]
    and projected to d_cond via out_proj.

    Scalar normalization (applied before embedding):
        pH:   (ph - 7.0) / 4.0    — centers on neutral pH 7, range ~[-1.5, 1.1]
        temp: (temp - 37.0) / 20.0 — centers on physiological temp, signed range ~[-2, 3]

    Args:
        d_cond:      Output conditioning dimension fed to FiLMLayer.
        d_mlp_hidden: Hidden width of the scalar-to-embedding MLPs.

    Note on ph_unknown / temp_unknown:
        These nn.Parameters must be routed to the task_no_decay optimizer group.
        L2 weight decay would shrink them back to zero each step, preventing
        them from learning to represent the "unknown condition" state.
        See kicm_module.py configure_optimizers for the explicit routing block.
    """

    def __init__(self, d_cond: int = 64, d_mlp_hidden: int = 32) -> None:
        super().__init__()

        # Scalar → d_mlp_hidden embeddings for known values
        self.ph_embed   = nn.Sequential(nn.Linear(1, d_mlp_hidden), nn.GELU())
        self.temp_embed = nn.Sequential(nn.Linear(1, d_mlp_hidden), nn.GELU())

        # Learned embeddings for unknown conditions (zero-initialized, no weight decay)
        self.ph_unknown   = nn.Parameter(torch.zeros(d_mlp_hidden))
        self.temp_unknown = nn.Parameter(torch.zeros(d_mlp_hidden))

        # Cofactor fingerprint projection: [B, 512] → [B, 64]
        self.cofactor_proj = nn.Sequential(nn.Linear(512, 64), nn.GELU())

        # Combine all embeddings: ph(32) + temp(32) + cofactor(64) = 128 → d_cond
        self.out_proj = nn.Sequential(
            nn.Linear(2 * d_mlp_hidden + 64, d_cond),
            nn.GELU(),
        )

    def forward(
        self,
        ph:          Tensor,  # [B] float32 — 0.0 if unknown
        temp:        Tensor,  # [B] float32 — 0.0 if unknown
        ph_known:    Tensor,  # [B] bool
        temp_known:  Tensor,  # [B] bool
        cofactor_fp: Tensor,  # [B, 512] float32 — binary Morgan fingerprint
    ) -> Tensor:              # [B, d_cond]
        """Encode conditions into a d_cond-dimensional vector.

        For each sample in the batch, selects the scalar embedding (known) or
        the learned unknown embedding (unknown) per condition.

        Args:
            ph:          Raw pH values.  0.0 placeholder when ph_known=False.
            temp:        Raw temperature values (°C). 0.0 when temp_known=False.
            ph_known:    True if pH was measured for this sample.
            temp_known:  True if temperature was measured for this sample.
            cofactor_fp: Binary Morgan fingerprint [B, 512] for cofactor-aware conditioning.

        Returns:
            FloatTensor [B, d_cond] — condition vector for FiLMLayer.
        """
        B = ph.shape[0]

        # Normalize scalars to roughly [-1.5, 1.5] range before embedding
        ph_norm   = ((ph   - 7.0) / 4.0).unsqueeze(-1)           # [B, 1]
        temp_norm = ((temp - 37.0) / 20.0).unsqueeze(-1)          # [B, 1]

        # Per-sample selection: scalar embedding if known, else learned unknown
        ph_emb = torch.where(
            ph_known.unsqueeze(-1),                           # [B, 1] broadcast
            self.ph_embed(ph_norm),                           # [B, d_mlp_hidden]
            self.ph_unknown.unsqueeze(0).expand(B, -1),       # [B, d_mlp_hidden]
        )

        temp_emb = torch.where(
            temp_known.unsqueeze(-1),
            self.temp_embed(temp_norm),
            self.temp_unknown.unsqueeze(0).expand(B, -1),
        )

        # Cofactor fingerprint projection
        cofactor_emb = self.cofactor_proj(cofactor_fp)  # [B, 64]

        # Concatenate and project
        cond = torch.cat([ph_emb, temp_emb, cofactor_emb], dim=-1)  # [B, 2*d_mlp_hidden + 64]
        return self.out_proj(cond)                                    # [B, d_cond]

    def extra_repr(self) -> str:
        d_mlp_hidden = self.ph_unknown.shape[0]
        d_cond = self.out_proj[0].out_features
        return f"d_mlp_hidden={d_mlp_hidden}, d_cond={d_cond}"
