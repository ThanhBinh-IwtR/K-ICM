"""Full KICMModel assembly: ESM-3 (LoRA) + MoLFormer + CrossAttention + Projection + MLPHead.

D2 architecture — single substrate encoder (MoLFormer-XL), B1 unidirectional cross-attention.
ESM-3 backbone is frozen with LoRA adapters as the only trainable protein-side parameters.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

from .esm3_encoder import ESM3Encoder
from .molformer_encoder import MoLFormerEncoder
from .mlp_head import MLPHead
from .projection import ProjectionLayer
from .cross_attention import CrossAttentionFusion

logger = logging.getLogger(__name__)


class KICMModel(nn.Module):
    """End-to-end model for kinetic parameter prediction.

    Data flow (D2 — MoLFormer + LoRA ESM-3 + B1 cross-attention):

        sequence_tokens [B,L] + padding_mask [B,L]
            -> ESM3Encoder (frozen + LoRA) -> hidden [B, L, d_esm3]

        smiles list[str]
            -> MoLFormerEncoder -> h_sub [B, d_substrate=768]
            -> substrate_proj -> [B, d_proj]

        CrossAttentionFusion(Q=h_sub, K=V=hidden) -> ctx [B, d_proj]
            -> MLPHead -> predictions [B]

    Args:
        config: Dict with keys from train_config.yaml model section.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        model_cfg = config.get("model", config)

        esm_variant: str = model_cfg["esm3_variant"]
        d_esm: int = model_cfg["d_esm3"]
        d_substrate: int = model_cfg["d_substrate"]
        d_proj: int = model_cfg["d_proj"]
        d_fusion: int = model_cfg["d_fusion"]
        mlp_cfg: dict = model_cfg.get("mlp", {})
        normalize_proj_input: bool = model_cfg.get("projection_normalize_input", False)
        ca_cfg: dict = model_cfg["cross_attention"]
        lora_cfg: dict = model_cfg.get("esm3_lora", {})

        # Protein encoder (residue-level output for cross-attention)
        self.esm3_encoder = ESM3Encoder(model_name=esm_variant, d_model=d_esm)
        self.esm3_encoder.freeze()
        self.esm3_encoder.apply_lora(
            r=lora_cfg.get("r", 16),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", None),
        )

        # Substrate encoder (MoLFormer-XL)
        self.substrate_encoder = MoLFormerEncoder(
            model_name=model_cfg.get("molformer_model", "ibm/MoLFormer-XL-both-10pct"),
            max_length=model_cfg.get("molformer_max_length", 202),
            use_pooler=model_cfg.get("molformer_use_pooler", True),
            d_output=d_substrate,
        )

        # MoLFormer freeze / LoRA (backward-compatible: default = fully trainable)
        molformer_frozen: bool = model_cfg.get("molformer_frozen", False)
        mol_lora_cfg: dict = model_cfg.get("molformer_lora", {})
        mol_lora_enabled: bool = mol_lora_cfg.get("enabled", False)

        if molformer_frozen:
            self.substrate_encoder.freeze()

        if mol_lora_enabled:
            if not molformer_frozen:
                raise ValueError(
                    "molformer_lora.enabled requires molformer_frozen: true"
                )
            self.substrate_encoder.apply_lora(
                r=mol_lora_cfg.get("r", 8),
                alpha=mol_lora_cfg.get("alpha", 16),
                dropout=mol_lora_cfg.get("dropout", 0.05),
                target_modules=mol_lora_cfg.get("target_modules", []),
            )

        # Projection: d_substrate → d_proj
        self.substrate_proj = ProjectionLayer(
            d_in=d_substrate, d_out=d_proj, normalize_input=normalize_proj_input,
        )

        # Cross-attention fusion: substrate queries protein residues (B1 unidirectional)
        self.cross_attn = CrossAttentionFusion(
            d_sub=ca_cfg["d_sub"],
            d_prot=ca_cfg["d_prot"],
            n_heads=ca_cfg["n_heads"],
            dropout=ca_cfg["dropout"],
        )

        # MLP prediction head
        self.mlp_head = MLPHead(
            d_in=d_fusion,
            hidden_dims=tuple(mlp_cfg.get("hidden_dims", [512, 256, 128])),
            dropouts=tuple(mlp_cfg.get("dropouts", [0.2, 0.1, 0.0])),
            activation=mlp_cfg.get("activation", "gelu"),
        )

        mol_status = "frozen" if molformer_frozen else "trainable"
        if mol_lora_enabled:
            mol_status += "+lora"

        logger.info(
            "[KICMModel] Constructed  "
            "esm=%s  d_esm=%d  d_sub=%d  d_proj=%d  d_fusion=%d  "
            "lora_r=%s  cross_attn=B1(uni)  molformer=%s",
            esm_variant, d_esm, d_substrate, d_proj, d_fusion,
            lora_cfg.get("r", "none"), mol_status,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sequence_tokens: Tensor,
        padding_mask: Tensor,
        smiles_list: list[str],
    ) -> Tensor:
        """Predict target kinetic parameter for a batch.

        Args:
            sequence_tokens: LongTensor [B, L] — ESM-3 tokenised sequences.
            padding_mask: BoolTensor [B, L] — True at padding positions.
            smiles_list: list[str] of length B — substrate SMILES.

        Returns:
            FloatTensor [B] — predicted log10 target values.
        """
        # Protein branch — residue-level hidden states
        hidden, mask = self.esm3_encoder(sequence_tokens, padding_mask)

        # Substrate branch
        h_substrate = self.substrate_encoder(smiles_list)  # [B, d_substrate]
        h_sub = self.substrate_proj(h_substrate)            # [B, d_proj]

        # Cross-attention fusion
        ctx = self.cross_attn(
            h_sub=h_sub,
            protein_residues=hidden,
            protein_pad_mask=mask,
        )  # [B, d_proj]

        # MLP prediction
        predictions = self.mlp_head(ctx)  # [B]
        return predictions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def count_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    import os as _os
    import logging as _logging

    _sys.path.insert(
        0,
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    )

    _logging.basicConfig(level=_logging.INFO)

    cfg = {
        "model": {
            "esm3_variant": "esm3_sm_open_v1",
            "d_esm3": 1536,
            "d_substrate": 768,
            "d_proj": 512,
            "d_fusion": 512,
            "molformer_model": "ibm/MoLFormer-XL-both-10pct",
            "molformer_max_length": 202,
            "molformer_use_pooler": True,
            "projection_normalize_input": True,
            "esm3_lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["layernorm_qkv.1"],
            },
            "cross_attention": {
                "d_sub": 512,
                "d_prot": 1536,
                "n_heads": 8,
                "dropout": 0.1,
            },
            "mlp": {
                "hidden_dims": [512, 256, 128],
                "dropouts": [0.2, 0.1, 0.0],
                "activation": "gelu",
            },
        }
    }

    model = KICMModel(cfg)

    print(
        f"Total params: {model.count_total_params():,}  "
        f"Trainable: {model.count_trainable_params():,}"
    )

    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    tok = EsmSequenceTokenizer()
    encoded = tok(["MKTLL", "ACDE"], return_tensors="pt", padding=True)
    tokens = encoded["input_ids"]
    padding_mask = encoded["attention_mask"].eq(0)

    smiles_list = ["CC(=O)O", "c1ccccc1"]

    model.eval()
    with torch.no_grad():
        preds = model(tokens, padding_mask, smiles_list)

    print(f"Predictions shape: {preds.shape}")  # [2]
    assert preds.shape == (2,), f"Expected [2], got {preds.shape}"
    print("kicm_model.py smoke test PASSED")
