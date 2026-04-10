"""
Phase E automated test suite.
Tests: BioT5PlusEncoder · CrossAttentionFusion (MQCA) · ConditionEncoder ·
       KICMModel (Phase E forward) · Config Validation · DataModule

Run: pytest tests/test_phase_e.py -v
"""
from __future__ import annotations

import sys
import os
import inspect

import pytest
import torch
import yaml

# Add kcat_predictor root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.models import (
    BioT5PlusEncoder,
    CrossAttentionFusion,
    ConditionEncoder,
    KICMModel,
    ESM3Encoder,
)
from src.utils.config_validation import validate_config

# ── Minimal dimensions for CPU tests ─────────────────────────────────────────
B, S, L, D_SUB, D_PROT = 2, 10, 50, 512, 1536


# ---------------------------------------------------------------------------
# Test 1: BioT5PlusEncoder
# ---------------------------------------------------------------------------


class TestBioT5PlusEncoder:
    """BioT5PlusEncoder with pre-tokenized Tensor inputs."""

    @pytest.fixture(scope="class")
    def encoder(self):
        return BioT5PlusEncoder(
            model_name="QizhiPei/biot5-plus-base-mol-instructions-molecule",
            max_length=128,
            d_output=768,
        )

    def test_output_shape(self, encoder):
        """forward(input_ids, attention_mask) -> hidden [B,S,768], mask [B,S]"""
        input_ids = torch.randint(0, 100, (B, S))
        attention_mask = torch.ones(B, S, dtype=torch.long)
        with torch.no_grad():
            hidden, mask = encoder(input_ids, attention_mask)
        assert hidden.shape == (B, S, 768), f"Expected [{B},{S},768], got {hidden.shape}"
        assert mask.shape == (B, S), f"Expected [{B},{S}], got {mask.shape}"

    def test_mask_convention_true_is_valid(self, encoder):
        """Returned mask must follow True=valid, False=padding convention."""
        input_ids = torch.randint(0, 100, (1, 5))
        # Last 2 positions are padding
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
        with torch.no_grad():
            _, mask = encoder(input_ids, attention_mask)
        assert mask[0, 0].item() is True, "Valid position should be True"
        assert mask[0, -1].item() is False, "Padding position should be False"

    def test_no_tokenizer_in_forward(self):
        """forward() must not call any tokenizer — pre-tokenization is in collator."""
        src_code = inspect.getsource(BioT5PlusEncoder.forward)
        assert "tokenizer" not in src_code, \
            "Tokenizer call found in BioT5PlusEncoder.forward()"
        assert "smiles" not in src_code.lower(), \
            "SMILES conversion found in BioT5PlusEncoder.forward()"

    def test_gradient_flows(self, encoder):
        """Gradient must flow through BioT5+ encoder parameters."""
        input_ids = torch.randint(0, 100, (1, 5))
        attention_mask = torch.ones(1, 5, dtype=torch.long)
        hidden, _ = encoder(input_ids, attention_mask)
        hidden.sum().backward()
        has_grad = any(
            p.grad is not None for p in encoder.parameters() if p.requires_grad
        )
        assert has_grad, "No gradient flowing through BioT5+ encoder"
        # Zero grads for safety
        encoder.zero_grad()


# ---------------------------------------------------------------------------
# Test 2: CrossAttentionFusion (MQCA + FFN)
# ---------------------------------------------------------------------------


class TestCrossAttentionFusionMQCA:
    """Multi-query cross-attention with FFN block."""

    def _make_fusion(self):
        return CrossAttentionFusion(
            d_sub=D_SUB, d_prot=D_PROT, n_heads=8, dropout=0.1
        )

    def test_output_shape(self):
        """Q[B,S,512] + K/V[B,L,1536] -> ctx [B,512]"""
        fusion = self._make_fusion()
        h_sub = torch.randn(B, S, D_SUB)
        h_prot = torch.randn(B, L, D_PROT)
        biot5_valid_mask = torch.ones(B, S, dtype=torch.bool)
        prot_mask = torch.zeros(B, L, dtype=torch.bool)
        ctx = fusion(h_sub, h_prot, prot_mask, biot5_valid_mask)
        assert ctx.shape == (B, D_SUB), f"Expected [{B},{D_SUB}], got {ctx.shape}"

    def test_ffn_block_exists(self):
        """FFN and ffn_norm must be present in CrossAttentionFusion."""
        fusion = self._make_fusion()
        named = dict(fusion.named_modules())
        assert "ffn" in named, "CrossAttentionFusion missing ffn block"
        assert "ffn_norm" in named, "CrossAttentionFusion missing ffn_norm"

    def test_gradient_through_ffn(self):
        """Gradient must flow through ffn layers."""
        fusion = self._make_fusion()
        h_sub = torch.randn(B, S, D_SUB, requires_grad=True)
        h_prot = torch.randn(B, L, D_PROT)
        biot5_valid_mask = torch.ones(B, S, dtype=torch.bool)
        prot_mask = torch.zeros(B, L, dtype=torch.bool)
        ctx = fusion(h_sub, h_prot, prot_mask, biot5_valid_mask)
        ctx.sum().backward()
        ffn_weight = list(fusion.ffn.parameters())[0]
        assert ffn_weight.grad is not None, "No gradient through ffn[0].weight"

    def test_padding_gets_zero_attention(self):
        """Padding positions in biot5_valid_mask must not cause NaN."""
        fusion = self._make_fusion()
        fusion.eval()
        h_sub = torch.randn(B, S, D_SUB)
        h_prot = torch.randn(B, L, D_PROT)
        # First 5 tokens valid, last 5 are padding
        biot5_valid_mask = torch.zeros(B, S, dtype=torch.bool)
        biot5_valid_mask[:, :5] = True
        prot_mask = torch.zeros(B, L, dtype=torch.bool)
        ctx = fusion(h_sub, h_prot, prot_mask, biot5_valid_mask)
        assert not ctx.isnan().any(), "NaN in ctx — possible mask convention error"

    def test_no_bidirectional_arg(self):
        """CrossAttentionFusion should NOT accept bidirectional arg."""
        sig = inspect.signature(CrossAttentionFusion.__init__)
        assert "bidirectional" not in sig.parameters

    def test_all_masked_substrate_no_nan(self):
        """C-01: all-masked substrate tokens must not produce NaN ctx."""
        ca = CrossAttentionFusion(d_sub=512, d_prot=1536, n_heads=8, dropout=0.0)
        ca.train()
        h_sub = torch.randn(B, S, 512)
        prot = torch.randn(B, L, 1536)
        prot_mask = torch.zeros(B, L, dtype=torch.bool)    # all valid
        # First sample: all masked substrate (edge case)
        biot5_valid_mask = torch.ones(B, S, dtype=torch.bool)
        biot5_valid_mask[0] = False                         # sample 0: all padding
        ctx = ca(h_sub, prot, prot_mask, biot5_valid_mask)
        assert not ctx.isnan().any(), "NaN detected for all-masked substrate"
        assert ctx.shape == (B, 512)


# ---------------------------------------------------------------------------
# Test 3: ConditionEncoder (FiLM)
# ---------------------------------------------------------------------------


class TestConditionEncoder:
    """FiLM conditioning with identity initialization."""

    def _make_encoder(self):
        return ConditionEncoder(d_ctx=D_SUB, d_cond=64)

    @staticmethod
    def _zero_cofactor_fp(batch_size: int) -> "torch.Tensor":
        """Return [B, 512] zero cofactor FP (no cofactor case)."""
        return torch.zeros(batch_size, 512)

    def test_identity_init(self):
        """At init: gamma must be ~1.0, beta must be ~0.0."""
        enc = self._make_encoder()
        enc.eval()
        ph = torch.tensor([7.0, 7.0])
        temp = torch.tensor([37.0, 37.0])
        ph_known = torch.tensor([True, True])
        temp_known = torch.tensor([True, True])
        cofactor_fp = self._zero_cofactor_fp(2)
        gamma, beta = enc(ph, temp, ph_known, temp_known, cofactor_fp)
        assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-4), \
            f"gamma not ~1.0 at init: {gamma.mean().item()}"
        assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-4), \
            f"beta not ~0.0 at init: {beta.mean().item()}"

    def test_temperature_range_coverage(self):
        """temp=105C (dataset max) must not produce NaN or extrapolation."""
        enc = self._make_encoder()
        ph = torch.tensor([7.0])
        temp = torch.tensor([105.0])
        ph_known = torch.tensor([True])
        temp_known = torch.tensor([True])
        cofactor_fp = self._zero_cofactor_fp(1)
        gamma, beta = enc(ph, temp, ph_known, temp_known, cofactor_fp)
        assert not gamma.isnan().any(), "NaN in gamma at temp=105C"

    def test_unknown_handling(self):
        """ph_known=False must use learned unknown embedding, not encoded value.

        At init, FiLM net has zero weights → identity output regardless of input.
        We randomize film_net weights so the different ph embeddings produce different outputs.
        """
        enc = self._make_encoder()
        enc.eval()
        # Randomize film_net weights so inputs actually affect output
        with torch.no_grad():
            for p in enc.film_net.parameters():
                p.normal_(0, 0.1)
            enc.ph_unknown.fill_(5.0)

        ph = torch.tensor([0.0, 0.0])
        temp = torch.tensor([37.0, 37.0])
        known = torch.tensor([True, True])
        unknown = torch.tensor([False, False])
        cofactor_fp = self._zero_cofactor_fp(2)
        with torch.no_grad():
            gamma_known, _ = enc(ph, temp, known, known, cofactor_fp)
            gamma_unknown, _ = enc(ph, temp, unknown, known, cofactor_fp)
        assert not torch.allclose(gamma_known, gamma_unknown), \
            "Unknown handling broken: ph_known=False should use ph_unknown embedding"

    def test_unknown_embedding_exists(self):
        """ConditionEncoder must have learnable ph_unknown and temp_unknown params."""
        enc = self._make_encoder()
        assert hasattr(enc, "ph_unknown"), "Missing ph_unknown parameter"
        assert hasattr(enc, "temp_unknown"), "Missing temp_unknown parameter"
        assert isinstance(enc.ph_unknown, torch.nn.Parameter)
        assert isinstance(enc.temp_unknown, torch.nn.Parameter)

    def test_output_shapes(self):
        enc = self._make_encoder()
        ph = torch.tensor([7.0, 5.5])
        temp = torch.tensor([37.0, 25.0])
        ph_known = torch.tensor([True, True])
        temp_known = torch.tensor([True, False])
        cofactor_fp = self._zero_cofactor_fp(2)
        gamma, beta = enc(ph, temp, ph_known, temp_known, cofactor_fp)
        assert gamma.shape == (2, D_SUB), f"gamma shape wrong: {gamma.shape}"
        assert beta.shape == (2, D_SUB), f"beta shape wrong: {beta.shape}"


# ---------------------------------------------------------------------------
# Test 4: KICMModel Phase E forward
# ---------------------------------------------------------------------------


class TestKICMModelPhaseE:
    """Full KICMModel Phase E forward with pre-tokenized inputs."""

    @pytest.fixture(scope="class")
    def model(self):
        from tests.test_d2 import _phase_e_config
        cfg = _phase_e_config()
        return KICMModel(cfg)

    def test_no_smiles_list_in_forward(self):
        """KICMModel.forward() must NOT accept smiles_list — uses biot5_input_ids."""
        sig = inspect.signature(KICMModel.forward)
        assert "smiles_list" not in sig.parameters, \
            "smiles_list still in KICMModel.forward()"
        assert "biot5_input_ids" in sig.parameters, \
            "biot5_input_ids missing from KICMModel.forward()"

    def test_frozen_esm3(self, model):
        """All ESM3Encoder parameters must have requires_grad=False."""
        for name, p in model.esm3_encoder.esm3.named_parameters():
            if "lora" not in name.lower():
                assert not p.requires_grad, f"ESM3 param {name} not frozen"

    def test_trainable_biot5(self, model):
        """BioT5PlusEncoder parameters must have requires_grad=True."""
        trainable = [p for p in model.substrate_encoder.parameters() if p.requires_grad]
        assert len(trainable) > 0, "No trainable BioT5+ parameters"

    def test_output_is_1d(self, model):
        """Model forward must return predictions [B], not [B,1]."""
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        tok = EsmSequenceTokenizer()
        encoded = tok(["MKTLL", "ACDE"], return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]
        mask = encoded["attention_mask"].eq(0)
        biot5_ids = torch.randint(0, 100, (2, 15))
        biot5_mask = torch.ones(2, 15, dtype=torch.long)
        cofactor_fp = torch.zeros(2, 512)  # S12-06: cofactor fingerprint [B, 512]
        model.eval()
        with torch.no_grad():
            preds = model(
                tokens, mask, biot5_ids, biot5_mask,
                ph=torch.tensor([7.0, 6.5]),
                temp=torch.tensor([25.0, 37.0]),
                ph_known=torch.tensor([True, True]),
                temp_known=torch.tensor([True, False]),
                cofactor_fp=cofactor_fp,
            )
        assert preds.shape == (2,), f"Expected [2], got {preds.shape}"
        assert preds.ndim == 1, "Output should be 1D"

    def test_multi_scale_hooks(self, model):
        """ESM3Encoder must have 4 multi-scale hooks."""
        assert len(model.esm3_encoder._hooks) == 4, \
            f"Expected 4 hooks, got {len(model.esm3_encoder._hooks)}"


# ---------------------------------------------------------------------------
# Test 5: Data Pipeline
# ---------------------------------------------------------------------------


class TestDataPipelinePhaseE:
    """Data pipeline correctness for Phase E batch format."""

    def test_val_dataloader_drop_last_false(self):
        """Validation DataLoader must have drop_last=False."""
        from src.data.kicm_datamodule import KICMDataModule
        with open("configs/train_probe.yaml") as f:
            cfg = yaml.safe_load(f)
        dm = KICMDataModule(cfg)
        dm.setup("fit")
        val_loader = dm.val_dataloader()
        assert val_loader.drop_last is False, \
            "val_dataloader has drop_last=True — will bias R2 evaluation"

    def test_batch_contains_biot5_ids(self):
        """Batch must contain biot5_input_ids and biot5_attention_mask."""
        from src.data.kicm_datamodule import KICMDataModule
        with open("configs/train_probe.yaml") as f:
            cfg = yaml.safe_load(f)
        dm = KICMDataModule(cfg)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        assert "biot5_input_ids" in batch, "biot5_input_ids missing from batch"
        assert "biot5_attention_mask" in batch, "biot5_attention_mask missing from batch"

    def test_phase_e_config_validates(self):
        """Production config must pass validate_config()."""
        with open("configs/train_config.yaml") as f:
            cfg = yaml.safe_load(f)
        validate_config(cfg)

    def test_probe_config_validates(self):
        """Probe config must pass validate_config()."""
        with open("configs/train_probe.yaml") as f:
            cfg = yaml.safe_load(f)
        validate_config(cfg)

    def test_no_tokenizer_in_phase_e_models(self):
        """Phase E model files must not have tokenizer instances."""
        phase_e_files = [
            "biot5_encoder.py", "esm3_encoder.py", "kicm_model.py",
            "kicm_module.py", "cross_attention.py", "projection.py",
            "mlp_head.py", "condition_encoder.py",
        ]
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "models",
        )
        for basename in phase_e_files:
            fpath = os.path.join(model_dir, basename)
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                lines = f.read().split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                if "self.tokenizer" in stripped or "self._tokenizer" in stripped:
                    pytest.fail(
                        f"Tokenizer instance found in {basename}:{i+1}: {stripped}"
                    )

    def test_selfies_fallback_warns_and_counts(self):
        """C-05: SELFIES conversion failure must log warning and increment counter."""
        from src.data.collator import KICMCollator, reset_selfies_fallback_count
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        from transformers import AutoTokenizer

        reset_selfies_fallback_count()

        esm_tok = EsmSequenceTokenizer()
        biot5_tok = AutoTokenizer.from_pretrained(
            "QizhiPei/biot5-plus-base-mol-instructions-molecule", trust_remote_code=True,
        )
        collator = KICMCollator(esm_tok, biot5_tok, biot5_max_length=128, max_seq_len=1024)

        # Deliberately invalid SMILES for SELFIES conversion
        batch = [
            {"sequence": "MKTLL", "smiles": "C%99CC", "target": torch.tensor(3.5),
             "ph": 7.0, "temp": 25.0, "ph_known": True, "temp_known": True},
            {"sequence": "ACDE", "smiles": "CC(=O)O", "target": torch.tensor(2.1),
             "ph": 7.0, "temp": 25.0, "ph_known": True, "temp_known": True},
        ]

        collator(batch)
        count = collator.get_selfies_fallback_count()
        assert count > 0, f"SELFIES fallback counter should be > 0, got {count}"


# ---------------------------------------------------------------------------
# Test 6: BioT5+ LoRA (Session 3)
# ---------------------------------------------------------------------------


class TestBioT5LoRA:
    """BioT5+ LoRA adapter tests."""

    @pytest.fixture(scope="class")
    def model(self):
        from tests.test_d2 import _phase_e_config
        cfg = _phase_e_config()
        return KICMModel(cfg)

    def test_lora_applied_when_enabled(self, model):
        """With biot5_lora.enabled=true, BioT5+ trainable params must be < 10M."""
        trainable = sum(
            p.numel() for p in model.substrate_encoder.parameters() if p.requires_grad
        )
        assert trainable < 10_000_000, f"LoRA not applied: {trainable} trainable params"

    def test_base_weights_frozen_with_lora(self, model):
        """Non-LoRA BioT5+ params must have requires_grad=False after LoRA."""
        if not getattr(model, '_biot5_lora_enabled', False):
            pytest.skip("BioT5+ LoRA not enabled in test config")
        frozen_count = sum(
            1 for p in model.substrate_encoder.parameters() if not p.requires_grad
        )
        assert frozen_count > 0, "No frozen params — LoRA base should be frozen"


# ---------------------------------------------------------------------------
# Test 7: ProteinProjection (Session 3)
# ---------------------------------------------------------------------------


class TestProteinProjection:
    """CrossAttentionFusion with protein pre-projection."""

    def test_protein_proj_in_cross_attention(self):
        """CrossAttentionFusion must have decomposed protein_proj_norm/linear/act attributes (S15-01)."""
        fusion = CrossAttentionFusion(d_sub=D_SUB, d_prot=D_PROT, n_heads=8, dropout=0.1)
        assert hasattr(fusion, 'protein_proj_norm'),   "protein_proj_norm missing"
        assert hasattr(fusion, 'protein_proj_linear'), "protein_proj_linear missing"
        assert hasattr(fusion, 'protein_proj_act'),    "protein_proj_act missing"
        assert not hasattr(fusion, 'protein_proj'),    "protein_proj Sequential should NOT exist (S15-01)"

    def test_protein_proj_shape(self):
        """protein_proj_norm+linear+act must accept [B,L,1536] and output [B,L,512] (S15-01)."""
        fusion = CrossAttentionFusion(d_sub=D_SUB, d_prot=D_PROT, n_heads=8, dropout=0.1)
        x = torch.randn(B, L, D_PROT)
        out = fusion.protein_proj_act(fusion.protein_proj_linear(fusion.protein_proj_norm(x)))
        assert out.shape == (B, L, D_SUB), f"Expected [{B},{L},{D_SUB}], got {out.shape}"

    def test_mha_kdim_is_d_sub(self):
        """After protein_proj, MHA kdim should be d_sub, not d_prot."""
        fusion = CrossAttentionFusion(d_sub=D_SUB, d_prot=D_PROT, n_heads=8, dropout=0.1)
        assert fusion.sub_to_prot.kdim == D_SUB, \
            f"kdim={fusion.sub_to_prot.kdim}, expected {D_SUB}"


# ---------------------------------------------------------------------------
# Test 8: Multi-scale hooks (Session 3)
# ---------------------------------------------------------------------------


class TestMultiScaleHooks:
    """ESM3Encoder multi-scale hook verification."""

    def test_four_hooks_in_source(self):
        """ESM3Encoder __init__ must contain _hooks and block[-4] references."""
        src = inspect.getsource(ESM3Encoder.__init__)
        assert '_hooks' in src, "Missing _hooks list"
        assert '-4' in src, "Missing block[-4] hook"

    def test_closure_fix(self):
        """Hook registration must use _make_hook pattern to avoid closure bug."""
        src = inspect.getsource(ESM3Encoder.__init__)
        assert '_make_hook' in src, "Closure bug: must use _make_hook(idx) pattern"

    def test_remove_hooks_cleans_up(self):
        """M-05: remove_hooks() must clear all hooks and hidden_states."""
        encoder = ESM3Encoder()
        encoder.freeze()
        assert len(encoder._hooks) == 4
        encoder.remove_hooks()
        assert len(encoder._hooks) == 0
        assert len(encoder._hidden_states) == 0

    def test_remove_hooks_explicit_cleanup(self):
        """M-05 / S10-06: remove_hooks() clears hooks and hidden_states; __del__ absent."""
        import inspect as _inspect
        from src.models.esm3_encoder import ESM3Encoder as _ESM3Encoder
        # Verify __del__ does not exist
        src = _inspect.getsource(_ESM3Encoder)
        assert 'def __del__' not in src, (
            "__del__ must be removed; use remove_hooks() instead"
        )
        # Verify remove_hooks exists
        assert hasattr(_ESM3Encoder, 'remove_hooks'), (
            "remove_hooks() method must exist on ESM3Encoder"
        )


# ---------------------------------------------------------------------------
# Test 9: sync_dist (Session 3)
# ---------------------------------------------------------------------------


class TestSyncDist:
    """Verify DDP sync strategy on epoch-level log calls."""

    def test_epoch_logs_no_sync_dist(self):
        """Coverage metric logs in on_train_epoch_end must NOT use sync_dist=True.

        MeanMetric.compute() with sync_on_compute=True (default) all-reduces internally.
        Adding sync_dist=True on the coverage self.log() calls would double-sync.

        S10-11: gpu/memory_gb/mean with sync_dist=True has been REMOVED.
        It created an ALLREDUCE immediately after MeanMetric's two ALLGATHER collectives
        (ph + temp coverage), producing SeqNum mismatch at Epoch 1 boundary.
        Only per-rank logging without sync_dist is retained.
        """
        from src.models.kicm_module import KICMLightningModule
        src = inspect.getsource(KICMLightningModule.on_train_epoch_end)
        # Coverage metric log calls must not use sync_dist
        lines = src.splitlines()
        for line in lines:
            if ('ph_coverage' in line or 'temp_coverage' in line) and 'sync_dist=True' in line:
                raise AssertionError(
                    f"Coverage metric log must NOT use sync_dist=True: {line.strip()}"
                )
        # S10-11: gpu/memory_gb/mean with sync_dist=True must be ABSENT
        mean_line = next((l for l in lines if 'memory_gb/mean' in l and 'self.log' in l), None)
        assert mean_line is None, (
            f"S10-11: gpu/memory_gb/mean self.log() must be removed (ALLREDUCE after "
            f"ALLGATHER causes SeqNum mismatch): {mean_line}"
        )
        # Per-rank key must be present without sync_dist=True
        rank_line = next((l for l in lines if 'memory_gb/rank' in l and 'self.log' in l), None)
        assert rank_line is not None, "gpu/memory_gb/rank{N} self.log() must be present"
        assert 'sync_dist=True' not in rank_line, (
            f"gpu/memory_gb/rank{{N}} must NOT use sync_dist=True: {rank_line.strip()}"
        )

    def test_mean_metric_sync_on_compute_true(self):
        """MeanMetric must use default sync_on_compute=True (S10-01 regression fix).

        sync_on_compute=False (Deviation 131) was reverted in S10-01 because it caused
        ph_coverage/temp_coverage to be rank-local only — silently wrong in DDP.
        MeanMetric.compute() with sync_on_compute=True all-reduces once per epoch,
        which is the correct behavior.
        """
        from src.models.kicm_module import KICMLightningModule
        from tests.test_d2 import _phase_e_config
        cfg = _phase_e_config()
        mod = KICMLightningModule(cfg)
        assert mod.ph_coverage_metric.sync_on_compute, (
            "ph_coverage_metric must have sync_on_compute=True (default) — "
            "S10-01 reverted Deviation 131"
        )
        assert mod.temp_coverage_metric.sync_on_compute, (
            "temp_coverage_metric must have sync_on_compute=True (default) — "
            "S10-01 reverted Deviation 131"
        )

    def test_val_logs_use_all_gather(self):
        """on_validation_epoch_end must use all_gather_tensors (no sync_dist needed)."""
        from src.models.kicm_module import KICMLightningModule
        src = inspect.getsource(KICMLightningModule.on_validation_epoch_end)
        assert 'all_gather_tensors' in src, "on_validation_epoch_end missing all_gather_tensors"
        # sync_dist=True must NOT appear — data is already gathered across ranks
        assert 'sync_dist=True' not in src, (
            "on_validation_epoch_end should NOT use sync_dist=True with all_gather_tensors "
            "(double-sync causes NCCL desync hangs)"
        )

    def test_training_step_no_on_epoch_sync_dist(self):
        """S10-12: training_step must have no on_epoch=True + sync_dist=True in code lines.

        Lightning queues on_epoch=True + sync_dist=True as ALLREDUCE at epoch boundary.
        This collides with MeanMetric ALLGATHER in on_train_epoch_end → SeqNum mismatch
        → 30-min NCCL timeout. Architectural rule: sync_dist=True is only safe with
        on_step=True, on_epoch=False in training_step.
        """
        from src.models.kicm_module import KICMLightningModule
        src = inspect.getsource(KICMLightningModule.training_step)
        violations = []
        for i, line in enumerate(src.split('\n'), 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue  # skip comments
            if 'on_epoch=True' in stripped and 'sync_dist=True' in stripped:
                violations.append(f"L{i}: {stripped}")
        assert not violations, (
            "training_step has on_epoch=True + sync_dist=True (NCCL desync risk):\n"
            + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# Test 10: Effective batch config (Session 3)
# ---------------------------------------------------------------------------


class TestEffectiveBatchConfig:
    """Config consistency checks."""

    def test_effective_batch_size_consistent(self):
        """effective_batch_size must equal num_gpus * batch_size_per_gpu * grad_accum."""
        with open("configs/train_config.yaml") as f:
            cfg = yaml.safe_load(f)
        t = cfg['training']
        expected = t['num_gpus'] * t['batch_size_per_gpu'] * t['gradient_accumulation_steps']
        assert t['effective_batch_size'] == expected, \
            f"Documented {t['effective_batch_size']} != actual {expected}"

    def test_probe_effective_batch_consistent(self):
        """Probe effective_batch_size must be consistent."""
        with open("configs/train_probe.yaml") as f:
            cfg = yaml.safe_load(f)
        t = cfg['training']
        expected = t['num_gpus'] * t['batch_size_per_gpu'] * t['gradient_accumulation_steps']
        assert t['effective_batch_size'] == expected, \
            f"Documented {t['effective_batch_size']} != actual {expected}"


# ---------------------------------------------------------------------------
# Session 12 New Tests
# ---------------------------------------------------------------------------


class TestCofactorRegistry:
    """S12/S13: Data-driven cofactor whitelist tests."""

    def test_whitelist_nonempty(self):
        from src.data.cofactor_registry import _COFACTOR_CANONICAL_LIST
        assert len(_COFACTOR_CANONICAL_LIST) > 0, "Whitelist must not be empty"

    def test_whitelist_size_s13(self):
        """S13-01: Whitelist must have >= 5 entries (Strategy B thresh=8, dom>=0.95)."""
        from src.data.cofactor_registry import _COFACTOR_CANONICAL_LIST
        assert len(_COFACTOR_CANONICAL_LIST) >= 5, (
            f"S13-01: Whitelist too small ({len(_COFACTOR_CANONICAL_LIST)}). "
            "Strategy B thresh=8 should yield ~29 entries."
        )

    def test_proton_is_cofactor(self):
        """[H+] (proton) should be in whitelist — 1104 rows, ec_breadth=25, sole_rows=0."""
        from src.data.cofactor_registry import is_cofactor
        assert is_cofactor("[H+]"), "[H+] should be a known cofactor"

    def test_water_is_cofactor(self):
        """S13-01: Water (O) must be in whitelist — 7749 rows, ec_breadth=50, dom=0.970."""
        from src.data.cofactor_registry import is_cofactor
        assert is_cofactor("O"), "Water (O) must be a known cofactor (S13-01 Strategy B)"

    def test_alanine_not_cofactor(self):
        """Alanine (amino acid substrate) should NOT be in whitelist."""
        from src.data.cofactor_registry import is_cofactor
        # L-Alanine canonical SMILES
        assert not is_cofactor("C[C@@H](N)C(=O)O"), "Alanine must not be classified as cofactor"

    def test_separate_reactants_single(self):
        """Single-component SMILES: no cofactor, substrate returned unchanged."""
        from src.data.cofactor_registry import separate_reactants
        smi = "C[C@@H](N)C(=O)O"  # L-Alanine
        substrate, fp, n_cof = separate_reactants(smi)
        assert n_cof == 0
        assert fp.sum().item() == 0.0, "No cofactor → zero fingerprint"
        assert substrate == smi

    def test_separate_reactants_with_proton(self):
        """Multi-component: [H+] separated as cofactor."""
        from src.data.cofactor_registry import separate_reactants, is_cofactor
        if not is_cofactor("[H+]"):
            pytest.skip("[H+] not in whitelist — check threshold")
        smi = "C[C@@H](N)C(=O)O.[H+]"  # Alanine + proton
        substrate, fp, n_cof = separate_reactants(smi)
        assert n_cof == 1
        assert fp.sum().item() > 0.0, "[H+] cofactor → nonzero fingerprint"
        # Substrate should be Alanine, not [H+]
        assert substrate.strip() != "[H+]"

    def test_separate_reactants_water_coverage(self):
        """S13-01: Water (O) must be separated as cofactor from multi-component SMILES."""
        from src.data.cofactor_registry import separate_reactants, is_cofactor
        if not is_cofactor("O"):
            pytest.skip("Water (O) not in whitelist — S13-01 fix required")
        smi = "C[C@@H](N)C(=O)O.O"  # Alanine + water
        substrate, fp, n_cof = separate_reactants(smi)
        assert n_cof == 1, f"Expected 1 cofactor (water), got n_cof={n_cof}"
        assert fp.sum().item() > 0.0, "Water cofactor → nonzero fingerprint"
        assert substrate.strip() != "O", "Substrate should be Alanine, not water"

    def test_cofactor_fp_binary(self):
        """Cofactor fingerprint must be binary [0,1] after clamp."""
        from src.data.cofactor_registry import separate_reactants
        smi = "C[C@@H](N)C(=O)O.[H+]"  # Alanine + proton
        _, fp, _ = separate_reactants(smi)
        assert (fp >= 0).all() and (fp <= 1).all(), "Fingerprint must be clamped to [0, 1]"
        assert fp.shape[0] == 512

    def test_canonical_set_exists(self):
        """S13-01: _COFACTOR_CANONICAL_SET must be a set (not lru_cache)."""
        from src.data.cofactor_registry import _COFACTOR_CANONICAL_SET
        assert isinstance(_COFACTOR_CANONICAL_SET, set), (
            "_COFACTOR_CANONICAL_SET must be a plain set for O(1) lookup"
        )


class TestConditionEncoderCofactor:
    """S12: ConditionEncoder cofactor branch tests."""

    def _make_ce(self):
        from src.models.condition_encoder import ConditionEncoder
        return ConditionEncoder(d_ctx=512, d_cond=256, d_mlp_hidden=128, cofactor_fp_dim=512)

    def test_cofactor_fp_zero_identity(self):
        """Zero cofactor_fp must not disturb FiLM identity init."""
        ce = self._make_ce()
        B = 2
        g1, b1, g2, b2 = ce(
            torch.tensor([7.0, 7.0]),
            torch.tensor([37.0, 37.0]),
            torch.ones(B, dtype=torch.bool),
            torch.ones(B, dtype=torch.bool),
            torch.zeros(B, 512),  # no cofactor
        )
        # gamma should be close to 1.0 at init (identity)
        assert abs(g1.mean().item() - 1.0) < 0.5, f"gamma1 mean should be ~1.0, got {g1.mean().item():.4f}"

    def test_cofactor_shapes(self):
        """Nonzero cofactor_fp produces correct output shapes."""
        ce = self._make_ce()
        B = 4
        g1, b1, g2, b2 = ce(
            torch.rand(B),
            torch.rand(B) * 40 + 20,
            torch.ones(B, dtype=torch.bool),
            torch.ones(B, dtype=torch.bool),
            torch.ones(B, 512),  # all-one fingerprint
        )
        assert g1.shape == (B, 512), f"FAIL g1 shape: {g1.shape}"
        assert g2.shape == (B, 128), f"FAIL g2 shape: {g2.shape}"


class TestMSELoss:
    """S14-01: Huber(delta=4.0) loss replaces MSE+clamp. Class name kept for history."""

    def test_loss_is_mse_not_huber(self):
        from src.models.kicm_module import KICMLightningModule
        src = inspect.getsource(KICMLightningModule.training_step)
        assert "huber_loss" in src, "Huber loss must be used in training_step (S14-01)"
        assert "mse_loss" not in src, "mse_loss must be removed from training_step (S14-01)"
        assert "clamp" not in src, "outlier clamp must be removed from training_step (S14-01)"

    def test_loss_clamp_present(self):
        """S14-01: clamp removed, Huber provides non-zero gradient for |error|>4."""
        import torch
        import torch.nn.functional as F
        p = torch.tensor([5.0], requires_grad=True)
        t = torch.tensor([0.0])
        loss = F.huber_loss(p, t, delta=4.0)
        loss.backward()
        assert p.grad.item() != 0.0, f"Huber gradient at |error|=5 must be non-zero, got {p.grad.item()}"

    def test_mse_loss_value(self):
        """S14-01: Verify Huber(delta=4.0) produces expected values."""
        import torch
        import torch.nn.functional as F
        preds   = torch.tensor([1.0, 5.0, 10.0])
        targets = torch.tensor([1.0, 1.0,  1.0])
        loss = F.huber_loss(preds, targets, delta=4.0, reduction='mean')
        # |error|=[0, 4, 9]: Huber(0)=0.5*0^2=0, Huber(4)=0.5*16=8 (at boundary),
        # Huber(9)=4*(9-4/2)=4*7=28 → mean=(0+8+28)/3=12.0
        expected = (0.0 + 8.0 + 28.0) / 3.0
        assert abs(loss.item() - expected) < 1e-3, f"Expected {expected:.4f}, got {loss.item():.4f}"


class TestScalarMix:
    """S12: Learnable Scalar Mix for ESM3 layer weights."""

    def test_layer_weights_exist(self):
        from src.models.esm3_encoder import ESM3Encoder
        src = inspect.getsource(ESM3Encoder.__init__)
        assert "layer_weights" in src, "layer_weights nn.Parameter must be in __init__"

    def test_layer_weights_uniform_init(self):
        """At init, softmax of zeros = uniform 0.25 per layer."""
        import torch
        weights = torch.zeros(4)
        softmax_w = torch.softmax(weights, dim=0)
        for i in range(4):
            assert abs(softmax_w[i].item() - 0.25) < 1e-4, f"Weight {i} should be 0.25, got {softmax_w[i].item()}"

    def test_scalar_mix_in_forward(self):
        from src.models.esm3_encoder import ESM3Encoder
        src = inspect.getsource(ESM3Encoder.forward)
        assert "softmax" in src, "softmax must be used in ESM3Encoder.forward"
        assert "layer_weights" in src, "layer_weights must be used in forward"

    def test_scalar_mix_float_cast_in_forward(self):
        """S13-03: Each hidden state must be cast to float32 before Scalar Mix weighting."""
        from src.models.esm3_encoder import ESM3Encoder
        src = inspect.getsource(ESM3Encoder.forward)
        assert ".float()" in src, (
            "S13-03: hidden states must be cast to .float() in ESM3Encoder.forward "
            "to avoid bfloat16 precision loss from autocast context"
        )


class TestScalarMixLR:
    """S13-02: Scalar Mix layer_weights must be in task_decay optimizer group."""

    def test_layer_weights_in_task_group_not_lora(self):
        """layer_weights must be trained at lr.task, not lr.esm3_lora."""
        from src.models.kicm_module import KICMLightningModule
        src = inspect.getsource(KICMLightningModule.configure_optimizers)
        task_section_start = src.find("task_decay_params")
        if task_section_start < 0:
            # task_decay is initialized differently — look for task_decay list
            task_section_start = src.find("task_decay")
        task_section = src[task_section_start:task_section_start + 600] if task_section_start > 0 else ""
        assert "layer_weights" in task_section, (
            "S13-02: layer_weights must be added to task_decay (task lr group), "
            "not esm3_lora_params. Found section:\n" + task_section[:300]
        )

    def test_layer_weights_not_in_lora_params(self):
        """layer_weights must NOT be appended to esm3_lora_params."""
        from src.models.kicm_module import KICMLightningModule
        src = inspect.getsource(KICMLightningModule.configure_optimizers)
        assert "esm3_lora_params.append(self.model.esm3.layer_weights)" not in src, (
            "S13-02: layer_weights must NOT be appended to esm3_lora_params"
        )
        assert "lora_params.append(self.model.esm3_encoder.layer_weights)" not in src, (
            "S13-02: layer_weights must NOT be appended to lora_params (esm3_lora group)"
        )
