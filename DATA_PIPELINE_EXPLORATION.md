# Data Pipeline & Architecture Deep Dive — ESM3-MoLFormer D2 FiLM
**Branch:** `binhpht/feat/esm3-molformer-kcat-CA-LoRA`  
**Date:** 2026-04-11  
**Conda env:** kicmv2  
**Purpose:** Complete ground-truth documentation of the forward pass, data pipeline,
conditioning mechanism, cofactor handling, and optimizer design. Built from a
line-by-line read of every source file.

---

## Part A — Full Pipeline ASCII Diagram

```
Raw CSV (train.csv / val.csv / test.csv)
   │  columns: sequence, reactant_smiles, log10_value, ph, temperature
   │
   ▼
KICMDataset.__getitem__()  ──────────────────────────────────────────────
   │  • Drop NaN rows (sequence, smiles, target)
   │  • Drop sequences > max_seq_len=1024
   │  • Drop invalid SMILES (RDKit parse check)
   │  • Per-sample NaN check → ph_known, temp_known booleans
   │  • ph=0.0 / temp=0.0 placeholder when unknown
   │
   │  Returns KICMSample dict:
   │    sequence:    str
   │    smiles:      str  (raw, may include cofactor components)
   │    target:      Tensor[]  float32  (log10 value)
   │    ph:          float  (0.0 if unknown)
   │    temp:        float  (0.0 if unknown)
   │    ph_known:    bool
   │    temp_known:  bool
   │    cofactor_fp: Tensor[512] float32  (zeros placeholder)
   │
   ▼
KICMCollator.__call__(batch: list[KICMSample])  ─────────────────────────
   │
   │  ── Protein branch ──────────────────────────────────────────────────
   │    EsmSequenceTokenizer(sequences, padding=True, truncation=True,
   │                          max_length=1024)
   │      → sequence_tokens  [B, L_max]  int64   (BOS + residues + EOS)
   │      → attention_mask   [B, L_max]  int64   (1=real, 0=pad)
   │      → esm_pad_mask     [B, L_max]  bool    (True = padding, inverted)
   │
   │  ── Substrate / cofactor branch ─────────────────────────────────────
   │    for each sample:
   │      separate_reactants(raw_smiles, fp_bits=512)
   │        → substrate_smiles  str        (heaviest non-cofactor component)
   │        → cofactor_fp       Tensor[512] (binary Morgan FP, radius=2)
   │        → n_cofactors       int
   │    stack → cofactor_fp_batch  [B, 512]  float32
   │    collect → substrate_smiles_list  list[str]  length B
   │
   │  ── Condition tensors ───────────────────────────────────────────────
   │    ph        [B]  float32
   │    temp      [B]  float32
   │    ph_known  [B]  bool
   │    temp_known[B]  bool
   │    targets   [B]  float32
   │
   │  Returns batch dict (9 keys):
   │    sequence_tokens [B, L_max]  int64
   │    esm_pad_mask    [B, L_max]  bool
   │    smiles          list[str]   length B
   │    cofactor_fp     [B, 512]    float32
   │    targets         [B]         float32
   │    ph              [B]         float32
   │    temp            [B]         float32
   │    ph_known        [B]         bool
   │    temp_known      [B]         bool
   │
   ▼
KICMLightningModule.training_step(batch, batch_idx)  ────────────────────
   │
   │  sequence_tokens  = batch["sequence_tokens"]   [B, L_max]
   │  padding_mask     = batch["esm_pad_mask"]       [B, L_max]
   │  smiles_list      = batch["smiles"]             list[str]
   │  ph               = batch["ph"]                 [B]
   │  temp             = batch["temp"]               [B]
   │  ph_known         = batch["ph_known"]           [B]
   │  temp_known       = batch["temp_known"]         [B]
   │  cofactor_fp      = batch["cofactor_fp"]        [B, 512]
   │  targets          = batch["targets"].squeeze(-1) [B]
   │
   ▼
KICMModel.forward(sequence_tokens, padding_mask, smiles_list,
                   ph, temp, ph_known, temp_known, cofactor_fp)  ─────────
   │
   │  ┌─── PROTEIN BRANCH ─────────────────────────────────────────────┐
   │  │  ESM3Encoder.forward(sequence_tokens, padding_mask)             │
   │  │    autocast(bfloat16)                                           │
   │  │    ESM3.forward(sequence_tokens)   [fires hook on blocks[-1]]  │
   │  │    hook captures: hidden_states  [B, L, 1536]                  │
   │  │    cast float32 + move to caller device                        │
   │  │  → hidden  [B, L, 1536]  float32                               │
   │  │  → mask    [B, L]        bool     (passed through unchanged)   │
   │  └────────────────────────────────────────────────────────────────┘
   │                             │
   │  ┌─── SUBSTRATE BRANCH ───────────────────────────────────────────┐
   │  │  MoLFormerEncoder.forward(smiles_list)                         │
   │  │    AutoTokenizer(smiles_list, padding=True, max_length=202)    │
   │  │    autocast(bfloat16)                                          │
   │  │    AutoModel(**inputs)                                         │
   │  │    mask_expanded = attention_mask.unsqueeze(-1).float()        │
   │  │    embedding = (last_hidden_state * mask_expanded).sum(1)      │
   │  │                / mask_expanded.sum(1).clamp(min=1)             │
   │  │    .float()  → unconditional attention-masked mean pool        │
   │  │  → h_substrate  [B, 768]  float32                              │
   │  │                                                                 │
   │  │  ProjectionLayer.forward(h_substrate)                          │
   │  │    LayerNorm(768) → Linear(768→512) → GELU                    │
   │  │  → h_sub  [B, 512]  float32                                    │
   │  └────────────────────────────────────────────────────────────────┘
   │                             │
   │  ┌─── CROSS-ATTENTION FUSION ────────────────────────────────────┐
   │  │  CrossAttentionFusion.forward(h_sub, hidden, mask)             │
   │  │    q = h_sub.unsqueeze(1)              [B, 1, 512]             │
   │  │    prot_normed = LayerNorm(1536)(hidden) [B, L, 1536]          │
   │  │    MHA(Q=q, K=prot_normed, V=prot_normed,                     │
   │  │        key_padding_mask=mask)          [B, 1, 512]             │
   │  │    ctx = LayerNorm(512)(q + attn_out).squeeze(1)  [B, 512]     │
   │  └────────────────────────────────────────────────────────────────┘
   │                             │
   │  ┌─── FILM CONDITIONING ─────────────────────────────────────────┐
   │  │  ConditionEncoder.forward(ph, temp, ph_known, temp_known,      │
   │  │                            cofactor_fp)                         │
   │  │    ph_norm   = (ph - 7.0) / 4.0     → [B, 1]                  │
   │  │    temp_norm = (temp - 37.0) / 20.0 → [B, 1]  (physiological) │
   │  │    ph_emb = where(ph_known,                                    │
   │  │              ph_embed(ph_norm),       [B, 32]                  │
   │  │              ph_unknown.expand(B,-1)) [B, 32]                  │
   │  │    temp_emb = where(temp_known, ...) → [B, 32]                 │
   │  │    cofactor_emb = cofactor_proj(cofactor_fp)  → [B, 64]        │
   │  │      cofactor_proj: Sequential(Linear(512,64), GELU)           │
   │  │    cond = cat([ph_emb, temp_emb, cofactor_emb]) → [B, 128]    │
   │  │    out_proj(Linear(128,64)+GELU)                → [B, 64]      │
   │  │  → cond_vec  [B, 64]                                           │
   │  │                                                                 │
   │  │  FiLMLayer.forward(ctx, cond_vec)                              │
   │  │    γ = gamma_proj(cond_vec)   Linear(64→512)  [B, 512]        │
   │  │    β = beta_proj(cond_vec)    Linear(64→512)  [B, 512]        │
   │  │    output = γ ⊙ ctx + β                       [B, 512]        │
   │  └────────────────────────────────────────────────────────────────┘
   │                             │
   │  ┌─── MLP HEAD ──────────────────────────────────────────────────┐
   │  │  MLPHead.forward(ctx)                                          │
   │  │    LayerNorm(512)                               [B, 512]       │
   │  │    layer0: Linear(512→512) + GELU + Dropout(0.2) [B, 512]     │
   │  │    residual: LayerNorm(512)(x + h)              [B, 512]       │
   │  │    remaining:                                                   │
   │  │      Linear(512→256) + GELU + Dropout(0.1)     [B, 256]       │
   │  │      Linear(256→128) + GELU                    [B, 128]       │
   │  │    out: Linear(128→1).squeeze(-1)               [B]            │
   │  └────────────────────────────────────────────────────────────────┘
   │
   ▼
predictions  [B]  float32   (log10 kinetic values)
   │
   ▼
SmoothL1Loss(beta=1.0)  ─────────────────────────────────────────────────
   │  preds [B]  vs  targets [B]
   │  loss = mean(huber_element)  — scalar
   │
   ▼
loss.backward()  →  gradient_clip_val=1.0  →  AdamW.step()
   │
   ▼
LambdaLR.step()   (per optimizer step)
```

---

## Part B — Per-Module Detail Boxes

```
┌──────────────────────────────────────────────────────────────────────┐
│  ESM3Encoder                                                         │
│  Class: ESM3Encoder  (file: src/models/esm3_encoder.py)             │
│  Input:  sequence_tokens  [B, L]     int64   (pre-tokenized)        │
│          padding_mask     [B, L]     bool    (True = pad)            │
│  Output: hidden           [B, L, 1536]  float32                     │
│          padding_mask     [B, L]        bool  (pass-through)        │
│  Frozen:  All ESM-3 backbone parameters (690M+)                     │
│  Trainable: LoRA adapters only                                       │
│    • r=12, α=24, dropout=0.05                                        │
│    • Targets: layernorm_qkv.1, out_proj, ffn.1                      │
│      – layernorm_qkv.1: Linear [4608, 1536] — QKV proj in Seq(LN,L)│
│      – out_proj:        Linear [1536, 1536] — attention output proj │
│      – ffn.1:           Linear [8192, 1536] — FFN up-projection     │
│    • Applied via PEFT get_peft_model()                               │
│    • All LoRA params are 2D weight matrices (asserted in apply_lora) │
│  Architecture: esm3_sm_open_v1 — 48 transformer blocks, d=1536      │
│  Hook: blocks[-1].register_forward_hook() captures last-layer       │
│        hidden states. Hook survives PEFT wrapping because PEFT      │
│        wraps linear layers inside blocks, not the blocks themselves. │
│  Autocast: bfloat16 on CUDA (resolves ESM-3 dtype mixed ops)       │
│  How it learns: LoRA matrices inject task-relevant sequence         │
│    representations while preserving pretrained knowledge            │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  MoLFormerEncoder                                                    │
│  Class: MoLFormerEncoder  (file: src/models/molformer_encoder.py)   │
│  Input:  smiles_list  list[str]  length B  (substrate SMILES)       │
│  Output: embedding    [B, 768]   float32                            │
│  Frozen:  All MoLFormer backbone parameters (~47M)                  │
│  Trainable: LoRA adapters                                            │
│    • r=12, α=16, dropout=0.05                                        │
│    • Targets: attention.self.query/key/value,                        │
│               attention.output.dense, intermediate.dense,           │
│               output.dense (6 modules)                              │
│  Tokenization: Internal — AutoTokenizer with padding=True,         │
│    truncation=True, max_length=202                                   │
│  Pooling: Unconditional attention-masked mean pool over             │
│    last_hidden_state — always applied regardless of model output.   │
│    mask_expanded = attention_mask.unsqueeze(-1).float()  [B, S, 1] │
│    embedding = (last_hidden_state * mask_expanded).sum(1)           │
│                / mask_expanded.sum(1).clamp(min=1)       [B, 768]  │
│    (pooler_output is not used — it is NSP-objective and not         │
│     suitable for regression alignment)                              │
│  Autocast: bfloat16 on CUDA                                         │
│  How it learns: LoRA adapts SMILES transformer representations      │
│    toward kinetic property prediction from frozen chemical LM       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  SubstrateProjection                                                 │
│  Class: ProjectionLayer  (file: src/models/projection.py)           │
│  Input:  x  [B, 768]   float32  (MoLFormer embedding)              │
│  Output: x  [B, 512]   float32                                      │
│  Frozen:  Nothing frozen                                             │
│  Trainable: LayerNorm(768), Linear(768→512) weight+bias, GELU       │
│    (normalize_input=True enabled in current config)                 │
│  Architecture: Sequential(LayerNorm(768), Linear(768→512), GELU)   │
│  How it learns: Aligns MoLFormer's 768-dim space to the 512-dim     │
│    fusion space; LayerNorm stabilizes varying MoLFormer output      │
│    magnitudes during LoRA-driven representation shifts              │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  CrossAttentionFusion                                                │
│  Class: CrossAttentionFusion  (file: src/models/cross_attention.py) │
│  Input:  h_sub            [B, 512]        float32  (substrate vec)  │
│          protein_residues [B, L, 1536]    float32  (ESM-3 hidden)   │
│          protein_pad_mask [B, L]          bool     (True=pad)        │
│  Output: ctx              [B, 512]        float32                   │
│  Frozen:  Nothing frozen                                             │
│  Trainable: LayerNorm(1536) on protein input;                        │
│    MultiheadAttention(embed_dim=512, num_heads=8, kdim=1536,        │
│      vdim=1536, batch_first=True); LayerNorm(512) output norm       │
│  Architecture: B1 unidirectional — substrate is QUERY,              │
│    protein residues are KEY and VALUE. One attention step.          │
│    Residual: ctx = LayerNorm(q + attn_out).squeeze(1)               │
│  How it learns: Attends to the protein residues most relevant to    │
│    the substrate; residual preserves substrate identity while       │
│    injecting protein context for kinetic prediction                 │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  ConditionEncoder                                                    │
│  Class: ConditionEncoder  (file: src/models/condition_encoder.py)   │
│  Input:  ph          [B]       float32  (0.0 if unknown)            │
│          temp        [B]       float32  (0.0 if unknown)            │
│          ph_known    [B]       bool                                  │
│          temp_known  [B]       bool                                  │
│          cofactor_fp [B, 512]  float32  (binary Morgan FP)          │
│  Output: cond_vec    [B, 64]   float32                              │
│  Frozen:  Nothing frozen                                             │
│  Trainable:                                                          │
│    ph_embed:      Sequential(Linear(1,32), GELU)                    │
│    temp_embed:    Sequential(Linear(1,32), GELU)                    │
│    ph_unknown:    nn.Parameter [32]  (zero-init, no weight decay)   │
│    temp_unknown:  nn.Parameter [32]  (zero-init, no weight decay)   │
│    cofactor_proj: Sequential(Linear(512,64), GELU)                  │
│    out_proj:      Sequential(Linear(128,64), GELU)                  │
│  Normalization before embedding:                                     │
│    pH:   (ph - 7.0) / 4.0        → centers on neutral, ~[-1.5,1.1]│
│    temp: (temp - 37.0) / 20.0    → centers on 37°C, signed range   │
│           physiological center; ~[-2, 3] for 0–100°C               │
│  Concat: cat([ph_emb[B,32], temp_emb[B,32], cofactor_emb[B,64]])   │
│          → [B, 128]  →  out_proj(Linear(128,64))  →  [B, 64]       │
│  How it learns: Produces condition vectors from pH, temperature AND │
│    cofactor identity that shift the fused representation toward the │
│    correct experimental regime (pH/temp optima + cofactor context   │
│    vary systematically by enzyme family)                            │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  FiLMLayer                                                           │
│  Class: FiLMLayer  (file: src/models/condition_encoder.py)          │
│  Input:  x         [B, 512]  float32  (CrossAttention output ctx)   │
│          cond      [B, 64]   float32  (from ConditionEncoder)       │
│  Output: modulated [B, 512]  float32                                │
│  Frozen:  Nothing frozen                                             │
│  Trainable: gamma_proj: Linear(64→512); beta_proj: Linear(64→512)  │
│  Initialization:                                                     │
│    gamma_proj.weight = zeros,  gamma_proj.bias = ones  → γ ≈ 1.0  │
│    beta_proj.weight  = zeros,  beta_proj.bias  = zeros → β ≈ 0.0  │
│    → At epoch 0 the layer is a no-op identity transform             │
│  Operation: output = γ(cond) ⊙ x + β(cond)                        │
│  How it learns: Learns per-feature scale and shift conditioned on   │
│    pH/temperature/cofactor; identity init prevents destabilization  │
│    while allowing gradual conditioning signal to emerge             │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  MLPHead                                                             │
│  Class: MLPHead  (file: src/models/mlp_head.py)                     │
│  Input:  x   [B, 512]  float32  (FiLM-conditioned ctx vector)      │
│  Output: y   [B]       float32  (log10 kinetic parameter)           │
│  Frozen:  Nothing frozen                                             │
│  Trainable: All parameters                                           │
│    input_norm: LayerNorm(512)                                        │
│    layer0: Sequential(Linear(512→512), GELU, Dropout(0.2))         │
│    residual_norm: LayerNorm(512)  [d_in==hidden[0] → residual ON]  │
│    remaining: Sequential(                                            │
│      Linear(512→256), GELU, Dropout(0.1),                          │
│      Linear(256→128), GELU)                                         │
│    out: Linear(128→1)  → squeeze(-1) → [B]                         │
│  Residual: h = residual_norm(x_normalized + layer0(x_normalized))  │
│    (only when d_in == hidden_dims[0] == 512, which is true here)   │
│  How it learns: Regresses the final kinetic value from the fused,   │
│    pH/temp/cofactor-conditioned protein-substrate representation    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Part C — FiLM Conditioning Deep Dive

### C.1 What γ and β mean mathematically

FiLM (Feature-wise Linear Modulation) applies an **elementwise affine transformation**
to a feature map conditioned on auxiliary inputs:

```
output_i = γ_i(cond) · x_i + β_i(cond)

where:
  x      ∈ ℝ^512  — the fused protein-substrate context vector
  cond   ∈ ℝ^64   — the condition vector from ConditionEncoder
  γ      ∈ ℝ^512  — predicted per-feature scale (from Linear(64→512))
  β      ∈ ℝ^512  — predicted per-feature shift (from Linear(64→512))
  output ∈ ℝ^512  — the modulated context vector
```

**Intuition:** γ controls how much each feature dimension is amplified or suppressed
depending on the pH/temperature/cofactor context. β shifts the representation toward
the correct "operating point" for the given experimental conditions. Both are predicted
jointly from the same 64-dimensional condition vector, meaning the scale and shift for
all 512 features are determined by the same pH + temperature + cofactor signal.

### C.2 Identity initialization and why it prevents destabilization

At initialization:
```python
nn.init.zeros_(self.gamma_proj.weight)   # W_γ = 0
nn.init.ones_(self.gamma_proj.bias)      # b_γ = 1  →  γ(cond) = 0·cond + 1 = 1
nn.init.zeros_(self.beta_proj.weight)    # W_β = 0
nn.init.zeros_(self.beta_proj.bias)      # b_β = 0  →  β(cond) = 0·cond + 0 = 0
```

**At epoch 0:** `output = 1 · x + 0 = x` — the FiLM layer is a strict no-op.

**Why this matters:** If γ and β were randomly initialized, they would immediately corrupt
the carefully learned fused representation `ctx`, causing large training loss spikes and
potentially unstable gradients early in training. By starting as identity, the model first
learns the base protein-substrate task (through the other modules), and only once the main
gradients are stable does FiLM's conditioning begin to diverge from identity. The weight
matrices gradually shift from zero as conditioning signal proves useful.

### C.3 How `torch.where` selects known vs unknown embeddings

```python
ph_emb = torch.where(
    ph_known.unsqueeze(-1),              # [B, 1]  broadcast to [B, d_mlp_hidden]
    self.ph_embed(ph_norm),              # [B, 32]  — scalar embedding for known pH
    self.ph_unknown.unsqueeze(0).expand(B, -1),  # [B, 32]  — learned unknown embedding
)
```

`torch.where(condition, x, y)` selects element-wise: `x[i]` where `condition[i]=True`,
`y[i]` where `condition[i]=False`. The `unsqueeze(-1)` on `ph_known` broadcasts the
scalar boolean `[B, 1]` across the 32-dimensional embedding space.

**Key property:** Both branches are computed for all samples, but only the selected branch
contributes to the output. Crucially, `self.ph_embed(ph_norm)` runs on all samples even
those with `ph_known=False` — however the result is discarded by `where`. Similarly,
`ph_unknown.expand(B, -1)` is computed for all samples, but only used for unknown ones.
This means gradients flow back to **both** the MLP and the unknown parameter at every step,
but the parameter only gets meaningful gradient from the unknown-condition samples.

### C.4 What `ph_unknown` / `temp_unknown` learn over training

These are `nn.Parameter(torch.zeros(d_mlp_hidden))` — 32-dimensional zero-initialized
learnable vectors that represent the **expected embedding for an enzyme/substrate pair
where the pH (or temperature) was not recorded**.

At initialization they are zero, which matches the known-value embedding at the neutral
reference point (pH 7.0 → `ph_norm = 0.0` → embedding near zero after ReLU/GELU).

Over training, these vectors diverge from zero only if there is a systematic pattern in
the samples with unknown conditions. For example, if enzymes with unknown pH tend to be
from sources with a characteristic pH, the `ph_unknown` vector learns to represent that
"typical unknown" state. If there is no pattern, they stay near zero and the conditioning
has no effect for unknown-condition samples (matching the zero-masking baseline).

### C.5 Why `ph_unknown` / `temp_unknown` must be in `task_no_decay`

**Weight decay** applies L2 regularization: at each optimizer step, weights are shrunk by
`weight_decay × parameter_value`. For ordinary weight matrices this prevents overfitting by
keeping weights small. For these learned unknown embeddings it is harmful:

- Start at zero → L2 penalty = 0 initially, which is fine
- Once training begins to push them away from zero (to represent unknown-condition patterns),
  L2 immediately starts pulling them back toward zero
- The learning signal (gradient from unknown-condition samples, ~6% of data) competes with
  a constant decay force that fires on every batch (100% of steps)
- Net result: the embeddings can never escape the zero-initialization basin, which is
  functionally equivalent to hard zero-masking

By routing them to `task_no_decay` (weight_decay=0.0), they are free to move wherever the
gradient pushes them, learning a stable representation for the unknown-condition state.

---

## Part D — Cofactor Handling

### D.1 What `separate_reactants` does

`separate_reactants(reactant_smiles, fp_bits=512)` in `src/data/cofactor_registry.py`
processes a potentially multi-component SMILES string (components separated by `.`) into:
1. A single substrate SMILES string for MoLFormer
2. A binary fingerprint summarizing all cofactors present

**Strategy (Strategy B, analysis date 2026-04-06):**
- 29 canonical SMILES in the whitelist, covering 83.5% of multi-component rows
- Threshold: molecule appears as cofactor in ≥ 95% of its appearances AND across ≥ 8 distinct EC classes
- Includes water (O), O2, ATP, NAD+/NADH variants, H⁺, SAM, CoA, pyruvate, phosphate, etc.

**Algorithm:**
```
1. Split reactant_smiles on '.'  → components
2. For each component:
   a. Parse with RDKit → canonical SMILES
   b. If canonical in _COFACTOR_CANONICAL_SET:
      - Compute Morgan FP (radius=2, fp_bits=512)
      - Add to cofactor_fp accumulator
      - Increment n_cofactors
   c. Else:
      - Compute molecular weight (Descriptors.MolWt)
      - Append (mw, smiles) to substrate_candidates
3. clamp(cofactor_fp, 0, 1) → binary (handles shared bits from multiple cofactors)
4. If substrate_candidates is empty (all components are cofactors):
   → Edge case: use heaviest component as fallback substrate
5. Else:
   → primary substrate = max(substrate_candidates, key=mw)
6. Return: substrate_smiles, cofactor_fp, n_cofactors
```

**Heaviest-by-MW heuristic** for substrate selection: In enzyme kinetics, the true
substrate is almost always the largest molecule in the reaction — cofactors tend to be
smaller energy-carrying molecules. This is a principled default that avoids needing
substrate/cofactor annotations in the raw data.

### D.2 How `cofactor_fp [B, 512]` is computed

Each cofactor component gets a binary Morgan fingerprint:
```python
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_bits)
cofactor_fp += torch.tensor(list(fp), dtype=torch.float32)
```

After accumulation over all cofactor components, the fingerprint is clamped to [0, 1]:
```python
cofactor_fp = cofactor_fp.clamp(0.0, 1.0)
```

This produces a **binary** fingerprint (not a count fingerprint) even when multiple cofactors
share bits. The resulting `[B, 512]` tensor is assembled in the collator:
```python
cofactor_fp_batch = torch.stack(cofactor_fps, dim=0)  # [B, 512] float32
```

### D.3 How `cofactor_fp` flows into `KICMModel.forward()`

`cofactor_fp [B, 512]` is computed in `KICMCollator`, included in the batch dict, and
actively consumed by `ConditionEncoder` to provide cofactor-aware FiLM conditioning.

**Data flow:**
```
KICMCollator  →  batch["cofactor_fp"]  [B, 512]
   ↓
KICMLightningModule.training_step / validation_step / test_step
   →  cofactor_fp = batch["cofactor_fp"]
   ↓
KICMModel.forward(..., cofactor_fp)
   ↓
ConditionEncoder.forward(ph, temp, ph_known, temp_known, cofactor_fp)
   │  cofactor_emb = cofactor_proj(cofactor_fp)      [B, 512] → [B, 64]
   │    cofactor_proj: Sequential(Linear(512,64), GELU)
   │  cond = cat([ph_emb[B,32], temp_emb[B,32], cofactor_emb[B,64]])
   │       → [B, 128]
   │  cond_vec = out_proj(Linear(128,64)+GELU)        → [B, 64]
   ↓
FiLMLayer — γ and β predicted from cond_vec incorporating cofactor identity
```

**Rationale:** The cofactor fingerprint encodes which energy-carrying or electron-transfer
molecules are present in the reaction (ATP, NAD+, CoA, etc.). Different cofactor contexts
correspond to fundamentally different reaction mechanisms and therefore different kcat ranges.
By injecting this signal into the FiLM conditioning vector, the model can modulate the
fused protein-substrate representation differently depending on whether the enzyme is using
ATP hydrolysis vs NADH reduction vs direct transfer, without conflating this signal with
the substrate identity (which is separately represented in the MoLFormer branch).

---

## Part E — Optimizer Param Groups Table

| Group name | LR | Weight Decay | What's in it | Condition for existence |
|---|---|---|---|---|
| `task_decay` | 2.0e-4 | 1.0e-2 | Projection weights, CrossAttn weights, ConditionEncoder known-embedding weights & biases, FiLM projection weights, MLPHead weights — any task module param where name doesn't end in `.bias` and no segment contains `"norm"` | Always (when non-empty) |
| `task_no_decay` | 2.0e-4 | 0.0 | Bias params + LayerNorm params in task modules; **explicitly: `cond_encoder.ph_unknown [32]` and `cond_encoder.temp_unknown [32]`** | Always (when non-empty) |
| `molformer_decay` | 5.0e-5 | 1.0e-2 | MoLFormer weight matrices (non-norm, non-bias) | Only when `molformer_frozen=False` |
| `molformer_no_decay` | 5.0e-5 | 0.0 | MoLFormer bias + norm params | Only when `molformer_frozen=False` |
| `molformer_lora` | 5.0e-5 | 1.0e-2 | MoLFormer LoRA adapter matrices (all 2D, `bias="none"`) | Only when `molformer_frozen=True` AND `molformer_lora.enabled=True` |
| `esm3_lora` | 7.0e-6 | 1.0e-2 | ESM-3 LoRA adapter matrices (all 2D, `bias="none"`) | Always present (LoRA always active) |

**Active groups in production config** (`molformer_frozen=True`, `molformer_lora.enabled=True`):
`task_decay`, `task_no_decay`, `molformer_lora`, `esm3_lora` = **4 groups total**

**Active groups with fully trainable MoLFormer** (`molformer_frozen=False`):
`task_decay`, `task_no_decay`, `molformer_decay`, `molformer_no_decay`, `esm3_lora` = **5 groups**

**No-decay criterion** (`_is_no_decay_param`):
```python
return name.endswith(".bias") or any("norm" in part.lower() for part in name.split("."))
```

**LR schedule:** Single `LambdaLR` with the same lambda applied to all groups:
- Steps [0, warmup_steps): linear ramp from 0 to 1
- Steps [warmup_steps, total_steps): cosine decay from 1 to `min_lr_ratio=0.01`
- Steps ≥ total_steps: floor at `min_lr_ratio=0.01`

---

## Part F — Generalization Design Rationale

### F.1 Freezing + LoRA: preventing catastrophic forgetting while enabling adaptation

**Catastrophic forgetting** occurs when fine-tuning a pretrained model on a narrow task
(e.g., kcat prediction) causes the model to lose its broad, transferable representations.
For ESM-3 in particular, the general protein language model representations are the primary
source of generalization to OOD sequences — sequences outside the CD-HIT 40% training set.
If ESM-3 were fully fine-tuned, the training distribution would overwrite the pretraining
distribution, and OOD performance would degrade.

**LoRA (Low-Rank Adaptation)** constrains adaptation to a low-rank subspace (r=12):
```
W_adapted = W_frozen + (α/r) · B · A
where A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}
```
The number of trainable parameters in ESM-3 via LoRA is ~4–6M vs ~690M total — less than 1%.
This ensures the backbone retains its pretrained representations while a small residual
adaptation captures the kcat-specific information. The low learning rate for LoRA (7.0e-6
vs 2.0e-4 for task params) further slows the adaptation to prevent overwriting.

### F.2 FiLM conditioning: shifting predictions toward the correct experimental regime

Kinetic parameters are strongly pH- and temperature-dependent. Kcat at pH 5.0 can differ
by 1–2 log10 units from the same enzyme at pH 8.0 (a 10–100× difference). Without pH/temp
conditioning, a model trained on mixed conditions would learn the average behavior and
predict poorly for any specific condition.

FiLM addresses this by learning pH- and temperature-dependent per-feature modulation of the
fused representation. Rather than simply concatenating pH/temp scalars to the input
(which is weak — a 2-dimensional signal competing against 512 dimensions of protein-substrate
information), FiLM uses pH/temp to predict multiplicative scale and additive shift for all
512 features simultaneously. This is a much richer conditioning mechanism.

The temperature normalization `(temp - 37.0) / 20.0` is centered on 37°C (human physiological
temperature, the most common assay condition in the CatPred-DB training data), producing a
signed range where measurements above physiological are positive and below are negative. This
contrasts with the previous `temp / 50.0` (non-centered, always positive), which conflated
temperature direction and magnitude and set an asymmetric scale around a non-meaningful origin.

**Why it improves OOD generalization specifically:** OOD test enzymes (CD-HIT 40%) may have
different pH/temperature optima than their training-distribution neighbors. FiLM allows the
model to shift its prediction for a novel enzyme based on the experimental condition, rather
than relying solely on sequence similarity to interpolate conditions.

### F.3 Why learned unknown embeddings are better than zero-masking

**Zero-masking** (setting `ph=0`, `ph_known_weight=0`) has two failure modes:

1. **Semantic collision:** `ph=0.0` after normalization becomes `(0-7)/4 = -1.75`, which is
   indistinguishable from a sample with a genuine pH of 0 (extremely acidic). The embedding
   network therefore cannot distinguish "unknown" from "very acidic."

2. **Loss of distributional signal:** If ~6% of training samples have unknown pH, they all
   receive the same embedding vector (the scalar embedding of 0.0). The model sees the same
   input for all unknown-pH samples regardless of the enzyme family, and cannot learn that
   "unknown" is a meaningful category that might have systematic properties.

**Learned unknown embeddings** solve both problems:
- They occupy a distinct region of embedding space (not tied to any scalar value)
- They can learn to represent the distributional average of unknown-condition samples
- Weight decay is disabled so they can diverge from zero and establish a stable learned state
- They start at zero (matching initialization of known-value embeddings at neutral conditions)
  and only diverge if the training data contains a systematic pattern for unknown conditions

If the ~6% unknown-condition subset has no systematic structure different from the known
subset, the embeddings converge to approximately zero and the system is equivalent to
zero-masking — but with the additional flexibility to diverge when it would help.
