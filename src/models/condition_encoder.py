"""
ConditionEncoder — encodes pH and temperature as FiLM conditioning.

FiLM (Feature-wise Linear Modulation):
  ctx_conditioned = gamma * ctx + beta
  where gamma, beta are predicted from experimental conditions (pH, temp)

Missing values use LEARNED unknown embeddings — NOT imputed means.
This is critical: the model must learn what "unknown condition" means,
not assume standard conditions (pH=7, T=25C).

Init: gamma=1, beta=0 → identity at training start (no initial distortion)
"""
import torch
import torch.nn as nn
from torch import Tensor


class ConditionEncoder(nn.Module):
    """
    Encode pH and temperature into FiLM gamma/beta for ctx modulation.

    Double-stage FiLM:
      Stage 1: modulates ctx [B, d_ctx] after cross-attention
      Stage 2: modulates hidden [B, d_mlp_hidden] before MLP output layer

    Args:
        d_ctx:  dimension of ctx tensor to modulate (512)
        d_cond: internal condition embedding dimension (256)
        d_mlp_hidden: dimension of MLP hidden repr for stage 2 (None = single-stage)
    """

    PH_MIN, PH_MAX       = 0.0,   14.0
    TEMP_MIN_K, TEMP_MAX_K = 270.0, 385.0   # Kelvin — covers [0°C, 105°C] with margin

    def __init__(self, d_ctx: int = 512, d_cond: int = 256, d_mlp_hidden: int | None = None,
                 cofactor_fp_dim: int = 512):
        super().__init__()
        self.cofactor_fp_dim = cofactor_fp_dim

        # Encoders for known values (MLP on normalized scalar)
        self.ph_encoder = nn.Sequential(
            nn.Linear(1, d_cond), nn.SiLU(), nn.Linear(d_cond, d_cond)
        )
        self.temp_encoder = nn.Sequential(
            nn.Linear(1, d_cond), nn.SiLU(), nn.Linear(d_cond, d_cond)
        )

        # Learnable unknown embeddings — model learns what "missing" means
        self.ph_unknown   = nn.Parameter(torch.zeros(d_cond))
        self.temp_unknown = nn.Parameter(torch.zeros(d_cond))

        # NEW: cofactor fingerprint encoder (S12-06)
        # Input: [B, 512] binary Morgan fingerprint
        # Zero input (no cofactor detected) → near-zero embedding at init
        self.cofactor_encoder = nn.Sequential(
            nn.Linear(cofactor_fp_dim, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

        # FiLM stage 1: [3*d_cond] → [2*d_ctx] (gamma + beta)
        self.film_net = nn.Sequential(
            nn.Linear(d_cond * 3, d_ctx),
            nn.SiLU(),
            nn.Linear(d_ctx, d_ctx * 2),
        )

        # Init: output gamma=1, beta=0 → identity at start
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)
        self.film_net[-1].bias.data[:d_ctx] = 1.0   # gamma init = 1

        # FiLM stage 2 (optional): [3*d_cond] → [2*d_mlp_hidden]
        self.d_mlp_hidden = d_mlp_hidden
        if d_mlp_hidden is not None:
            self.film_net2 = nn.Sequential(
                nn.Linear(d_cond * 3, d_mlp_hidden),
                nn.SiLU(),
                nn.Linear(d_mlp_hidden, d_mlp_hidden * 2),
            )
            nn.init.zeros_(self.film_net2[-1].weight)
            nn.init.zeros_(self.film_net2[-1].bias)
            self.film_net2[-1].bias.data[:d_mlp_hidden] = 1.0  # gamma init = 1

        self.d_ctx  = d_ctx
        self.d_cond = d_cond

    def _normalize_ph(self, ph: Tensor) -> Tensor:
        ph_norm = (ph - self.PH_MIN) / (self.PH_MAX - self.PH_MIN)
        return ph_norm.clamp(0.0, 1.0)

    def _normalize_temp(self, temp: Tensor) -> Tensor:
        # temp expected in Celsius — convert to Kelvin then normalize
        temp_kelvin = temp + 273.15
        temp_norm = (temp_kelvin - self.TEMP_MIN_K) / (self.TEMP_MAX_K - self.TEMP_MIN_K)
        return temp_norm.clamp(0.0, 1.0)

    def forward(
        self,
        ph:         Tensor,          # [B] float — pH value (0-14)
        temp:       Tensor,          # [B] float — temperature in Celsius
        ph_known:   Tensor,          # [B] bool  — True = actual measured value
        temp_known: Tensor,          # [B] bool  — True = actual measured value
        cofactor_fp: Tensor,         # [B, 512] binary Morgan FP; zeros if no cofactor
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns (double-stage when d_mlp_hidden is set):
            gamma1 [B, d_ctx], beta1 [B, d_ctx] — stage 1 (after cross-attention)
            gamma2 [B, d_mlp_hidden], beta2 [B, d_mlp_hidden] — stage 2 (before MLP output)
        Returns (single-stage):
            gamma [B, d_ctx], beta [B, d_ctx]
        """
        B = ph.shape[0]

        ph_norm   = self._normalize_ph(ph).unsqueeze(-1).float()      # [B,1]
        temp_norm = self._normalize_temp(temp).unsqueeze(-1).float()   # [B,1]

        ph_enc   = self.ph_encoder(ph_norm)      # [B, d_cond]
        temp_enc = self.temp_encoder(temp_norm)  # [B, d_cond]

        ph_feat = torch.where(
            ph_known.unsqueeze(-1),
            ph_enc,
            self.ph_unknown.unsqueeze(0).expand(B, -1),
        )
        temp_feat = torch.where(
            temp_known.unsqueeze(-1),
            temp_enc,
            self.temp_unknown.unsqueeze(0).expand(B, -1),
        )

        # Cofactor encoding (S12-06)
        # Zero input → near-zero output initially (identity behavior preserved)
        h_cofactor = self.cofactor_encoder(cofactor_fp.float())  # [B, d_cond]

        cond = torch.cat([ph_feat, temp_feat, h_cofactor], dim=-1)   # [B, 3*d_cond]

        # Stage 1: gamma/beta for ctx modulation
        film_out1 = self.film_net(cond)                    # [B, 2*d_ctx]
        gamma1, beta1 = film_out1.chunk(2, dim=-1)         # [B, d_ctx] each

        if self.d_mlp_hidden is not None:
            # Stage 2: gamma/beta for MLP hidden modulation
            film_out2 = self.film_net2(cond)                # [B, 2*d_mlp_hidden]
            gamma2, beta2 = film_out2.chunk(2, dim=-1)      # [B, d_mlp_hidden] each
            return gamma1, beta1, gamma2, beta2

        return gamma1, beta1

    def extra_repr(self) -> str:
        return f"d_ctx={self.d_ctx}  d_cond={self.d_cond}  d_mlp_hidden={self.d_mlp_hidden}  cofactor_fp_dim={self.cofactor_fp_dim}"
