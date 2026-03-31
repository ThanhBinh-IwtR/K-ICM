"""End-to-end smoke test for the kcat_predictor D2 LoRA pipeline.

Run from kcat_predictor/ directory:
    python smoke_test.py

Tests:
    1. All model components instantiate correctly
    2. Forward pass produces correct output shapes
    3. Loss is finite
    4. Backward pass completes without error (LoRA + task grads flow)
    5. Collator tokenises correctly
    6. Dataset loads and filters correctly
"""
from __future__ import annotations

import logging
import sys
import os
import tempfile

# Add kcat_predictor root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("smoke_test")


def test_seed():
    from src.utils.seed import set_seed
    set_seed(42)
    x = torch.rand(3)
    set_seed(42)
    y = torch.rand(3)
    assert torch.allclose(x, y), "Seed not reproducible"
    log.info("[PASS] seed")


def test_metrics():
    from src.training.metrics import compute_metrics
    rng = np.random.default_rng(42)
    t = rng.standard_normal(50)
    p = t + 0.1 * rng.standard_normal(50)
    m = compute_metrics(p, t)
    assert m["r2"] > 0.95
    assert all(k in m for k in ["r2", "rmse", "mae", "spearman", "pearson"])
    log.info("[PASS] metrics  r2=%.4f", m["r2"])


def test_dataset():
    from src.data.dataset import KICMDataset
    csv = "sequence,reactant_smiles,log10_value\nMKTLL,CC(=O)O,3.5\nACDE,c1ccccc1,2.1\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv)
        path = f.name
    ds = KICMDataset(path, max_seq_len=1024)
    assert len(ds) == 2
    item = ds[0]
    assert isinstance(item["sequence"], str)
    assert item["target"].dtype == torch.float32
    log.info("[PASS] dataset  len=%d", len(ds))


def test_projection():
    from src.models.projection import ProjectionLayer
    p = ProjectionLayer(d_in=1280, d_out=512)
    x = torch.randn(4, 1280)
    out = p(x)
    assert out.shape == (4, 512)
    log.info("[PASS] projection")


def test_mlp_head():
    from src.models.mlp_head import MLPHead
    head = MLPHead(d_in=512)
    x = torch.randn(8, 512)
    out = head(x)
    assert out.shape == (8,)
    assert torch.isfinite(out).all()
    log.info("[PASS] mlp_head")


def test_esm3_encoder():
    """Test ESM-3 encoder forward with LoRA."""
    from src.models.esm3_encoder import ESM3Encoder
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    enc = ESM3Encoder()

    tokenizer = EsmSequenceTokenizer()
    encoded = tokenizer(
        ["MKTLL", "ACDE"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    tokens = encoded["input_ids"]
    mask = encoded["attention_mask"].eq(0)  # True = padding position

    enc.freeze()
    enc.apply_lora(r=16, alpha=32, dropout=0.05)
    n_lora = sum(p.numel() for p in enc.lora_parameters())
    assert n_lora > 0, "No LoRA parameters found"

    with torch.no_grad():
        hidden, m = enc(tokens, mask)
    assert hidden.shape[-1] == enc.d_model
    assert hidden.shape[0] == 2
    log.info("[PASS] esm3_encoder  hidden=%s  lora_params=%d", hidden.shape, n_lora)


def test_collator():
    from src.data.collator import KICMCollator
    col = KICMCollator()  # uses EsmSequenceTokenizer internally
    batch = [
        {"sequence": "MKTLL", "smiles": "CC(=O)O", "target": torch.tensor(3.5)},
        {"sequence": "ACDE", "smiles": "c1ccccc1", "target": torch.tensor(2.1)},
    ]
    out = col(batch)
    assert out["sequence_tokens"].dtype == torch.int64
    assert out["padding_mask"].dtype == torch.bool
    assert out["targets"].shape == (2,)
    log.info("[PASS] collator  tokens=%s", out["sequence_tokens"].shape)


def _d2_config():
    """Minimal D2 LoRA config for smoke tests."""
    return {
        "model": {
            "esm3_variant": "esm3_sm_open_v1",
            "d_esm3": 1536,
            "substrate_encoder": "molformer",
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
                "enabled": True,
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


def test_full_forward():
    """End-to-end forward pass through KICMModel (D2 LoRA)."""
    from src.models.kicm_model import KICMModel
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    cfg = _d2_config()
    model = KICMModel(cfg)

    tokenizer = EsmSequenceTokenizer()
    encoded = tokenizer(
        ["MKTLL", "ACDE"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    tokens = encoded["input_ids"]
    mask = encoded["attention_mask"].eq(0)
    smiles = ["CC(=O)O", "c1ccccc1"]
    targets = torch.tensor([3.5, 2.1])

    model.eval()
    with torch.no_grad():
        preds = model(tokens, mask, smiles)

    assert preds.shape == (2,), f"Expected [2], got {preds.shape}"
    assert torch.isfinite(preds).all(), "Predictions contain NaN or Inf"

    # Test loss is finite
    criterion = torch.nn.SmoothL1Loss(beta=1.0)
    loss = criterion(preds, targets)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    log.info(
        "[PASS] full_forward  preds=%s  loss=%.4f  trainable_params=%d",
        preds.shape,
        loss.item(),
        model.count_trainable_params(),
    )


def test_backward():
    """Verify gradients flow through LoRA + task parameters."""
    from src.models.kicm_model import KICMModel
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    cfg = _d2_config()
    model = KICMModel(cfg)
    model.train()

    tokenizer = EsmSequenceTokenizer()
    encoded = tokenizer(
        ["MKTLL", "ACDE"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    tokens = encoded["input_ids"]
    mask = encoded["attention_mask"].eq(0)
    smiles = ["CC(=O)O", "c1ccccc1"]
    targets = torch.tensor([3.5, 2.1])

    preds = model(tokens, mask, smiles)
    loss = torch.nn.SmoothL1Loss()(preds, targets)
    loss.backward()

    # LoRA params must have grads
    lora_grads = [
        p.grad for p in model.esm3_encoder.lora_parameters()
        if p.grad is not None
    ]
    assert len(lora_grads) > 0, "LoRA params must have gradients"

    # MLP head must have grads
    mlp_grads = [
        p.grad for p in model.mlp_head.parameters()
        if p.grad is not None
    ]
    assert len(mlp_grads) > 0, "MLP head must have gradients"

    # Frozen ESM-3 backbone (non-LoRA) must NOT have grads
    lora_ids = {id(p) for p in model.esm3_encoder.lora_parameters()}
    frozen_grads = [
        p.grad for p in model.esm3_encoder.esm3.parameters()
        if p.grad is not None and id(p) not in lora_ids
    ]
    assert len(frozen_grads) == 0, "Frozen ESM-3 backbone should have no grads"

    log.info(
        "[PASS] backward  lora_grads=%d  mlp_grads=%d  frozen_grads=%d (expected 0)",
        len(lora_grads), len(mlp_grads), len(frozen_grads),
    )


if __name__ == "__main__":
    log.info("=" * 55)
    log.info("   kicm_predictor D2 LoRA smoke test suite")
    log.info("=" * 55)

    tests = [
        test_seed,
        test_metrics,
        test_dataset,
        test_projection,
        test_mlp_head,
        test_esm3_encoder,
        test_collator,
        test_full_forward,
        test_backward,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            log.error("[FAIL] %s: %s", t.__name__, e)
            import traceback; traceback.print_exc()
            failed += 1

    log.info("=" * 55)
    log.info("   Results: %d passed  %d failed", passed, failed)
    log.info("=" * 55)

    if failed > 0:
        sys.exit(1)
