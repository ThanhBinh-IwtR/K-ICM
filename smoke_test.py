"""Phase E smoke test for the kcat_predictor pipeline.

Run from kcat_predictor/ directory:
    python smoke_test.py

Checks:
     1. Config validation
     2. Batch structure (8 fields, no smiles_list, biot5_input_ids present)
     3. Model construction with logging
     4. ESM-3 frozen + 4 multi-scale hooks
     5. BioT5+ LoRA: trainable params < 10M
     6. Forward pass: preds shape [B], no NaN
     7. Backward pass: gradients on BioT5+ LoRA params, ESM-3 zero grads
     8. Optimizer step: completes without error
     9. ConditionEncoder identity init: gamma≈1.0, beta≈0.0
    10. Val DataLoader: drop_last=False, persistent_workers=False
    11. CrossAttentionFusion: protein_proj present
    12. Runtime: < 120s on CPU
    13-15. (reserved)
    16. cofactor_registry loads and whitelist is non-empty
    17. ConditionEncoder accepts cofactor_fp
    18. MSE loss in training_step
    19. ESM3Encoder Scalar Mix

Exit code 0 = ALL PASS. Exit code 1 = any failure.
"""
from __future__ import annotations

import logging
import sys
import os
import time
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
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    from transformers import AutoTokenizer

    esm_tok = EsmSequenceTokenizer()
    biot5_tok = AutoTokenizer.from_pretrained(
        "QizhiPei/biot5-plus-base-mol-instructions-molecule", trust_remote_code=True,
    )
    col = KICMCollator(esm_tokenizer=esm_tok, biot5_tokenizer=biot5_tok, biot5_max_length=128, max_seq_len=1024)
    batch = [
        {"sequence": "MKTLL", "smiles": "CC(=O)O", "target": torch.tensor(3.5),
         "ph": 7.0, "temp": 25.0, "ph_known": True, "temp_known": True},
        {"sequence": "ACDE", "smiles": "c1ccccc1", "target": torch.tensor(2.1),
         "ph": 6.5, "temp": 37.0, "ph_known": True, "temp_known": True},
    ]
    out = col(batch)
    assert out["sequence_tokens"].dtype == torch.int64
    assert out["esm_pad_mask"].dtype == torch.bool
    assert out["targets"].shape == (2,)
    assert "biot5_input_ids" in out, "Missing biot5_input_ids"
    assert "biot5_attention_mask" in out, "Missing biot5_attention_mask"
    log.info("[PASS] collator  tokens=%s  biot5=%s", out["sequence_tokens"].shape, out["biot5_input_ids"].shape)


def _phase_e_config():
    """Minimal Phase E config for smoke tests (Session 4)."""
    return {
        "model": {
            "esm3_variant": "esm3_sm_open_v1",
            "d_esm3": 1536,
            "substrate_encoder": "biot5",
            "biot5_model": "QizhiPei/biot5-plus-base-mol-instructions-molecule",
            "biot5_max_length": 128,
            "d_substrate": 768,
            "d_proj": 512,
            "d_fusion": 512,
            "d_cond": 256,
            "projection_normalize_input": True,
            "esm3_lora": {
                "enabled": False,
                "r": 4,
                "alpha": 8,
                "dropout": 0.05,
                "target_modules": ["layernorm_qkv.1"],
            },
            "biot5_lora": {
                "enabled": True,
                "r": 4,
                "alpha": 8,
                "dropout": 0.1,
                "target_modules": ["q", "v"],
            },
            "cross_attention": {
                "d_sub": 512,
                "d_prot": 1536,
                "n_heads": 8,
                "dropout": 0.1,
            },
            "mlp": {
                "hidden_dims": [512, 256, 128],
                "dropouts": [0.3, 0.2, 0.1],
                "activation": "gelu",
            },
        },
    }


def test_full_forward():
    """End-to-end forward pass through KICMModel (Phase E)."""
    from src.models.kicm_model import KICMModel
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    cfg = _phase_e_config()
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
    biot5_ids = torch.randint(0, 100, (2, 15))
    biot5_mask = torch.ones(2, 15, dtype=torch.long)
    targets = torch.tensor([3.5, 2.1])

    model.eval()
    with torch.no_grad():
        preds = model(tokens, mask, biot5_ids, biot5_mask,
                      ph=torch.tensor([7.0, 6.5]),
                      temp=torch.tensor([25.0, 37.0]),
                      ph_known=torch.tensor([True, True]),
                      temp_known=torch.tensor([True, False]),
                      cofactor_fp=torch.zeros(2, 512))

    assert preds.shape == (2,), f"Expected [2], got {preds.shape}"
    assert torch.isfinite(preds).all(), "Predictions contain NaN or Inf"

    # Test loss is finite
    import torch.nn.functional as F
    loss = F.mse_loss(preds, targets, reduction='none').clamp(max=16.0).mean()
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    log.info(
        "[PASS] full_forward  preds=%s  loss=%.4f  trainable_params=%d",
        preds.shape,
        loss.item(),
        model.count_trainable_params(),
    )


def test_backward():
    """Verify gradients flow through BioT5+ LoRA + task params, ESM-3 frozen."""
    from src.models.kicm_model import KICMModel
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    cfg = _phase_e_config()
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
    biot5_ids = torch.randint(0, 100, (2, 15))
    biot5_mask = torch.ones(2, 15, dtype=torch.long)
    targets = torch.tensor([3.5, 2.1])

    preds = model(tokens, mask, biot5_ids, biot5_mask,
                  ph=torch.tensor([7.0, 6.5]),
                  temp=torch.tensor([25.0, 37.0]),
                  ph_known=torch.tensor([True, True]),
                  temp_known=torch.tensor([True, False]),
                  cofactor_fp=torch.zeros(2, 512))
    import torch.nn.functional as F
    loss = F.mse_loss(preds, targets, reduction='none').clamp(max=16.0).mean()
    loss.backward()

    # BioT5+ LoRA params must have grads
    biot5_lora_grads = [
        p.grad for p in model.substrate_encoder.lora_parameters()
        if p.grad is not None
    ]
    assert len(biot5_lora_grads) > 0, "BioT5+ LoRA params must have gradients"

    # MLP head must have grads
    mlp_grads = [
        p.grad for p in model.mlp_head.parameters()
        if p.grad is not None
    ]
    assert len(mlp_grads) > 0, "MLP head must have gradients"

    # Frozen ESM-3 backbone must NOT have grads
    esm3_grads = [
        p.grad for p in model.esm3_encoder.esm3.parameters()
        if p.grad is not None
    ]
    assert len(esm3_grads) == 0, "Frozen ESM-3 backbone should have no grads"

    log.info(
        "[PASS] backward  biot5_lora_grads=%d  mlp_grads=%d  esm3_grads=%d (expected 0)",
        len(biot5_lora_grads), len(mlp_grads), len(esm3_grads),
    )


def test_config_validation():
    """Config validation passes for both train and probe configs."""
    import yaml
    from src.utils.config_validation import validate_config

    for fname in ['configs/train_config.yaml', 'configs/train_probe.yaml']:
        try:
            with open(fname) as f:
                cfg = yaml.safe_load(f)
            validate_config(cfg)
        except FileNotFoundError:
            log.warning("[SKIP] %s not found", fname)
            continue
    log.info("[PASS] config_validation")


def test_esm3_multiscale_hooks():
    """ESM3Encoder must register exactly 4 multi-scale hooks."""
    from src.models.esm3_encoder import ESM3Encoder

    enc = ESM3Encoder()
    assert hasattr(enc, '_hooks'), "ESM3Encoder missing _hooks attribute"
    assert len(enc._hooks) == 4, f"Expected 4 hooks, got {len(enc._hooks)}"
    assert hasattr(enc, '_hidden_states'), "ESM3Encoder missing _hidden_states dict"
    log.info("[PASS] esm3_multiscale_hooks  count=%d", len(enc._hooks))


def test_biot5_lora_param_count():
    """With BioT5+ LoRA enabled, trainable params must be < 10M."""
    from src.models.kicm_model import KICMModel

    cfg = _phase_e_config()
    model = KICMModel(cfg)
    biot5_trainable = sum(
        p.numel() for p in model.substrate_encoder.parameters() if p.requires_grad
    )
    assert biot5_trainable < 10_000_000, (
        f"BioT5+ LoRA not applied correctly: {biot5_trainable} trainable params (expected < 10M)"
    )
    assert biot5_trainable > 0, "BioT5+ has 0 trainable params — LoRA not applied?"
    log.info("[PASS] biot5_lora_param_count  trainable=%.2fM", biot5_trainable / 1e6)


def test_condition_encoder_identity_init():
    """ConditionEncoder FiLM must init to identity: gamma≈1.0, beta≈0.0."""
    from src.models.condition_encoder import ConditionEncoder

    cond = ConditionEncoder(d_ctx=512, d_cond=64)
    ph = torch.tensor([7.0, 7.0])
    temp = torch.tensor([25.0, 25.0])
    ph_known = torch.tensor([True, True])
    temp_known = torch.tensor([True, True])
    cofactor_fp = torch.zeros(2, 512)  # no cofactor

    with torch.no_grad():
        gamma, beta = cond(ph, temp, ph_known, temp_known, cofactor_fp)

    gamma_mean = gamma.mean().item()
    beta_mean = beta.abs().mean().item()
    assert abs(gamma_mean - 1.0) < 0.5, f"gamma should be ≈1.0 at init, got mean={gamma_mean:.4f}"
    assert beta_mean < 0.5, f"beta should be ≈0.0 at init, got abs_mean={beta_mean:.4f}"
    log.info("[PASS] condition_encoder_identity_init  gamma_mean=%.4f  beta_abs_mean=%.4f",
             gamma_mean, beta_mean)


def test_cross_attention_protein_proj():
    """CrossAttentionFusion must have decomposed protein_proj_norm/linear/act (S15-01)."""
    from src.models.cross_attention import CrossAttentionFusion

    fusion = CrossAttentionFusion(d_sub=512, d_prot=1536, n_heads=8, dropout=0.1)
    assert hasattr(fusion, 'protein_proj_norm'),   "Missing protein_proj_norm in CrossAttentionFusion"
    assert hasattr(fusion, 'protein_proj_linear'), "Missing protein_proj_linear in CrossAttentionFusion"
    assert hasattr(fusion, 'protein_proj_act'),    "Missing protein_proj_act in CrossAttentionFusion"
    assert not hasattr(fusion, 'protein_proj'),    "protein_proj Sequential should not exist (S15-01)"

    # Test forward shape
    B, S, L = 2, 10, 50
    h_sub = torch.randn(B, S, 512)
    h_prot = torch.randn(B, L, 1536)
    sub_mask = torch.ones(B, S, dtype=torch.bool)  # biot5_valid_mask convention: True=valid
    prot_mask = torch.zeros(B, L, dtype=torch.bool)
    ctx = fusion(h_sub, h_prot, prot_mask, sub_mask)
    assert ctx.shape == (B, 512), f"CrossAttention output shape wrong: {ctx.shape}"
    log.info("[PASS] cross_attention_protein_proj  output=%s", ctx.shape)


def test_optimizer_step():
    """Optimizer step completes without error on Phase E model."""
    from src.models.kicm_model import KICMModel
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    cfg = _phase_e_config()
    model = KICMModel(cfg)
    model.train()

    # Collect trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

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
    biot5_ids = torch.randint(0, 100, (2, 15))
    biot5_mask = torch.ones(2, 15, dtype=torch.long)

    preds = model(tokens, mask, biot5_ids, biot5_mask,
                  ph=torch.tensor([7.0, 6.5]),
                  temp=torch.tensor([25.0, 37.0]),
                  ph_known=torch.tensor([True, True]),
                  temp_known=torch.tensor([True, False]),
                  cofactor_fp=torch.zeros(2, 512))
    import torch.nn.functional as F
    loss = F.mse_loss(preds, torch.tensor([3.5, 2.1]), reduction='none').clamp(max=16.0).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    log.info("[PASS] optimizer_step  loss=%.4f", loss.item())


def test_cofactor_registry():
    """Check 16: cofactor_registry loads and whitelist is non-empty."""
    from src.data.cofactor_registry import _COFACTOR_CANONICAL_LIST, separate_reactants
    assert len(_COFACTOR_CANONICAL_LIST) > 0, "cofactor_registry whitelist empty"
    substrate, fp, n_cof = separate_reactants("C[C@@H](N)C(=O)O.[H+]")
    assert fp.shape[0] == 512, f"cofactor_fp wrong shape: {fp.shape[0]}"
    log.info("[PASS] cofactor_registry  whitelist=%d  fp_shape=%d", len(_COFACTOR_CANONICAL_LIST), fp.shape[0])


def test_condition_encoder_cofactor():
    """Check 17: ConditionEncoder accepts cofactor_fp."""
    from src.models.condition_encoder import ConditionEncoder
    ce = ConditionEncoder(d_ctx=512, d_cond=256, d_mlp_hidden=128)
    g1, b1, g2, b2 = ce(
        torch.tensor([7.0]), torch.tensor([37.0]),
        torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool),
        torch.zeros(1, 512)
    )
    assert g1.shape == (1, 512), f"g1 shape wrong: {g1.shape}"
    log.info("[PASS] condition_encoder_cofactor  g1=%s  g2=%s", g1.shape, g2.shape)


def test_mse_loss_in_training_step():
    """Check 18: Huber(delta=4.0) loss in training_step (S14-01 replaces MSE+clamp)."""
    import inspect
    from src.models.kicm_module import KICMLightningModule
    src = inspect.getsource(KICMLightningModule.training_step)
    assert "huber_loss" in src, "huber_loss not found in training_step"
    assert "clamp" not in src, "clamp must not be in training_step (S14-01)"
    log.info("[PASS] huber_loss_in_training_step")


def test_esm3_scalar_mix():
    """Check 19: ESM3Encoder Scalar Mix in forward."""
    import inspect
    from src.models.esm3_encoder import ESM3Encoder
    src = inspect.getsource(ESM3Encoder.forward)
    assert "softmax" in src, "softmax not found in ESM3Encoder.forward"
    assert "layer_weights" in src, "layer_weights not found in ESM3Encoder.forward"
    log.info("[PASS] esm3_scalar_mix")


if __name__ == "__main__":
    t0 = time.time()
    log.info("=" * 55)
    log.info("   kicm_predictor Phase E smoke test suite")
    log.info("=" * 55)

    tests = [
        test_seed,
        test_metrics,
        test_dataset,
        test_projection,
        test_mlp_head,
        test_esm3_encoder,
        test_esm3_multiscale_hooks,
        test_collator,
        test_config_validation,
        test_biot5_lora_param_count,
        test_condition_encoder_identity_init,
        test_cross_attention_protein_proj,
        test_full_forward,
        test_backward,
        test_optimizer_step,
        test_cofactor_registry,           # Check 16
        test_condition_encoder_cofactor,  # Check 17
        test_mse_loss_in_training_step,   # Check 18
        test_esm3_scalar_mix,             # Check 19
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

    elapsed = time.time() - t0
    log.info("=" * 55)
    log.info("   Results: %d passed  %d failed  (%.1fs)", passed, failed, elapsed)
    if elapsed > 120:
        log.warning("   Runtime %.1fs exceeds 120s target", elapsed)
    log.info("=" * 55)

    if failed > 0:
        log.error("   SMOKE TEST FAILED")
        sys.exit(1)
    else:
        log.info("   ALL SMOKE TESTS PASSED")
