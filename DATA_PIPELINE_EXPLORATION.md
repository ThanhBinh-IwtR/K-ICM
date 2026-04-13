# Data Pipeline & Architecture Deep Dive — ESM3-MoLFormer D2

**Branch:** `binhpht/feat/esm3-molformer-kcat-CA-LoRA`
**Date:** 2026-04-13 (updated)
**Conda env:** kicmv2
**Purpose:** Complete ground-truth documentation of the forward pass, data pipeline,
conditioning mechanism, cofactor handling, and optimizer design. Built from a
line-by-line read of every source file.

**Current config state:** `conditioning.use_conditioning: false` (FiLM ablation),
`molformer_frozen: true + molformer_lora.enabled: true`, `esm3_lora.enabled: true`
with `lora_start_block: 36` (last 12 of 48 blocks only).

---

## Part A — Full Pipeline ASCII Diagram

```
Raw CSV (train.csv / val.csv / test.csv)
   |  columns: sequence, reactant_smiles, log10_value, ph, temperature
   |
   v
KICMDataset.__getitem__()  ──────────────────────────────────────────────
   |  * Drop NaN rows (sequence, smiles, target)
   |  * Drop sequences > max_seq_len=1024
   |  * Drop invalid SMILES (RDKit parse check)
   |  * Per-sample NaN check -> ph_known, temp_known booleans
   |  * ph=0.0 / temp=0.0 placeholder when unknown
   |
   |  Returns KICMSample dict:
   |    sequence:    str
   |    smiles:      str  (raw, may include cofactor components)
   |    target:      Tensor[]  float32  (log10 value)
   |    ph:          float  (0.0 if unknown)
   |    temp:        float  (0.0 if unknown)
   |    ph_known:    bool
   |    temp_known:  bool
   |    cofactor_fp: Tensor[512] float32  (zeros placeholder)
   |
   v
KICMCollator.__call__(batch: list[KICMSample])  ─────────────────────────
   |
   |  -- Protein branch ──────────────────────────────────────────────────
   |    EsmSequenceTokenizer(sequences, padding=True, truncation=True,
   |                          max_length=1024)
   |      -> sequence_tokens  [B, L_max]  int64   (BOS + residues + EOS)
   |      -> attention_mask   [B, L_max]  int64   (1=real, 0=pad)
   |      -> esm_pad_mask     [B, L_max]  bool    (True = padding, inverted)
   |
   |  -- Substrate / cofactor branch ─────────────────────────────────────
   |    for each sample:
   |      separate_reactants(raw_smiles, fp_bits=512)
   |        -> substrate_smiles  str        (heaviest non-cofactor component)
   |        -> cofactor_fp       Tensor[512] (binary Morgan FP, radius=2)
   |        -> n_cofactors       int
   |    stack -> cofactor_fp_batch  [B, 512]  float32
   |    collect -> substrate_smiles_list  list[str]  length B
   |
   |  -- Condition tensors ───────────────────────────────────────────────
   |    ph        [B]  float32
   |    temp      [B]  float32
   |    ph_known  [B]  bool
   |    temp_known[B]  bool
   |    targets   [B]  float32
   |
   |  Returns batch dict (9 keys):
   |    sequence_tokens [B, L_max]  int64
   |    esm_pad_mask    [B, L_max]  bool
   |    smiles          list[str]   length B
   |    cofactor_fp     [B, 512]    float32
   |    targets         [B]         float32
   |    ph              [B]         float32
   |    temp            [B]         float32
   |    ph_known        [B]         bool
   |    temp_known      [B]         bool
   |
   v
KICMLightningModule.training_step(batch, batch_idx)  ────────────────────
   |
   |  sequence_tokens  = batch["sequence_tokens"]   [B, L_max]
   |  padding_mask     = batch["esm_pad_mask"]       [B, L_max]
   |  smiles_list      = batch["smiles"]             list[str]
   |  ph               = batch["ph"]                 [B]
   |  temp             = batch["temp"]               [B]
   |  ph_known         = batch["ph_known"]           [B]
   |  temp_known       = batch["temp_known"]         [B]
   |  cofactor_fp      = batch["cofactor_fp"]        [B, 512]
   |  targets          = batch["targets"].squeeze(-1) [B]
   |
   v
KICMModel.forward(sequence_tokens, padding_mask, smiles_list,
                   ph, temp, ph_known, temp_known, cofactor_fp)  ─────────
   |
   |  +--- PROTEIN BRANCH ─────────────────────────────────────────────+
   |  |  ESM3Encoder.forward(sequence_tokens, padding_mask)             |
   |  |    autocast(bfloat16)                                           |
   |  |    ESM3.forward(sequence_tokens)   [fires hook on blocks[-1]]  |
   |  |    hook captures: hidden_states  [B, L, 1536]                  |
   |  |    cast float32 + move to caller device                        |
   |  |  -> hidden  [B, L, 1536]  float32                               |
   |  |  -> mask    [B, L]        bool     (passed through unchanged)   |
   |  +────────────────────────────────────────────────────────────────+
   |                             |
   |  +--- SUBSTRATE BRANCH ───────────────────────────────────────────+
   |  |  MoLFormerEncoder.forward(smiles_list)                         |
   |  |    AutoTokenizer(smiles_list, padding=True, max_length=202)    |
   |  |    autocast(bfloat16)                                          |
   |  |    AutoModel(**inputs)                                         |
   |  |    Pooling: pooler_output preferred if available;              |
   |  |      fallback: attention-masked mean pool over last_hidden_state|
   |  |    .float()                                                     |
   |  |  -> h_substrate  [B, 768]  float32                              |
   |  |                                                                 |
   |  |  ProjectionLayer.forward(h_substrate)                          |
   |  |    LayerNorm(768) -> Linear(768->512) -> GELU                  |
   |  |  -> h_sub  [B, 512]  float32                                    |
   |  +────────────────────────────────────────────────────────────────+
   |                             |
   |  +--- CROSS-ATTENTION FUSION (Multi-Query, n_queries=4) ─────────+
   |  |  CrossAttentionFusion.forward(h_sub, hidden, mask)             |
   |  |    q = query_proj(h_sub)                [B, 4*512]             |
   |  |    q = q.view(B, 4, 512)                [B, 4, 512]            |
   |  |    prot_normed = LayerNorm(1536)(hidden) [B, L, 1536]          |
   |  |    MHA(Q=q, K=prot_normed, V=prot_normed,                     |
   |  |        key_padding_mask=mask)            [B, 4, 512]           |
   |  |    ctx_residual = LayerNorm(512)(q + attn_out) [B, 4, 512]     |
   |  |    ctx = ctx_residual.mean(dim=1)        [B, 512]              |
   |  +────────────────────────────────────────────────────────────────+
   |                             |
   |  +--- FILM CONDITIONING (CURRENTLY DISABLED) ────────────────────+
   |  |  use_conditioning: false  -> this block is SKIPPED             |
   |  |  cond_encoder = None, film = None                              |
   |  |  (When enabled: ConditionEncoder -> FiLMLayer -> modulate ctx) |
   |  +────────────────────────────────────────────────────────────────+
   |                             |
   |  +--- MLP HEAD ──────────────────────────────────────────────────+
   |  |  MLPHead.forward(ctx)                                          |
   |  |    LayerNorm(512)                               [B, 512]       |
   |  |    layer0: Linear(512->512) + GELU + Dropout(0.2) [B, 512]    |
   |  |    residual: LayerNorm(512)(x + h)              [B, 512]       |
   |  |    remaining:                                                   |
   |  |      Linear(512->256) + GELU + Dropout(0.1)     [B, 256]      |
   |  |      Linear(256->128) + GELU                    [B, 128]      |
   |  |    out: Linear(128->1).squeeze(-1)               [B]           |
   |  +────────────────────────────────────────────────────────────────+
   |
   v
predictions  [B]  float32   (log10 kinetic values)
   |
   v
SmoothL1Loss(beta=0.3)  ────────────────────────────────────────────────
   |  preds [B]  vs  targets [B]
   |  Huber loss with delta=0.3 (errors < 0.3 are L2, errors > 0.3 are L1)
   |  loss = mean(huber_element)  -- scalar
   |
   v
loss.backward()  ->  gradient_clip_val=1.0  ->  AdamW.step()
   |
   v
LambdaLR.step()   (per optimizer step)
```

---

## Part B — Per-Module Detail Boxes

```
+----------------------------------------------------------------------+
|  ESM3Encoder                                                         |
|  Class: ESM3Encoder  (file: src/models/esm3_encoder.py)             |
|  Input:  sequence_tokens  [B, L]     int64   (pre-tokenized)        |
|          padding_mask     [B, L]     bool    (True = pad)            |
|  Output: hidden           [B, L, 1536]  float32                     |
|          padding_mask     [B, L]        bool  (pass-through)        |
|  Frozen:  All ESM-3 backbone parameters (~690M+)                    |
|  Trainable: LoRA adapters only (last 12 blocks)                     |
|    * r=12, alpha=24, dropout=0.05                                    |
|    * lora_start_block=36 -> blocks 0-35 LoRA FROZEN, 36-47 ACTIVE   |
|    * Targets: layernorm_qkv.1, out_proj, ffn.1                      |
|      - layernorm_qkv.1: Linear [4608, 1536] -- QKV proj in Seq(LN,L)|
|      - out_proj:        Linear [1536, 1536] -- attention output proj |
|      - ffn.1:           Linear [8192, 1536] -- FFN up-projection    |
|    * Applied via PEFT get_peft_model()                               |
|    * All LoRA params are 2D weight matrices (asserted in apply_lora) |
|  Architecture: esm3_sm_open_v1 -- 48 transformer blocks, d=1536     |
|  Hook: blocks[-1].register_forward_hook() captures last-layer       |
|        hidden states. Hook survives PEFT wrapping because PEFT      |
|        wraps linear layers inside blocks, not the blocks themselves. |
|  Autocast: bfloat16 on CUDA (resolves ESM-3 dtype mixed ops)       |
|  How it learns: LoRA matrices inject task-relevant sequence         |
|    representations while preserving pretrained knowledge.           |
|    Only last 12 blocks are adapted -- earlier blocks retain general  |
|    protein representations for better OOD generalization.            |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
|  MoLFormerEncoder                                                    |
|  Class: MoLFormerEncoder  (file: src/models/molformer_encoder.py)   |
|  Input:  smiles_list  list[str]  length B  (substrate SMILES)       |
|  Output: embedding    [B, 768]   float32                            |
|  Frozen:  All MoLFormer backbone parameters (~47M)                  |
|  Trainable: LoRA adapters                                            |
|    * r=16, alpha=16, dropout=0.05                                    |
|    * Targets: attention.self.query, attention.self.key,              |
|               attention.self.value, attention.output.dense,          |
|               intermediate.dense, output.dense (6 module types)      |
|  Tokenization: Internal -- AutoTokenizer with padding=True,         |
|    truncation=True, max_length=202                                   |
|  Pooling: Prefers pooler_output when available (HuggingFace output); |
|    fallback to attention-masked mean pool:                           |
|      mask_expanded = attention_mask.unsqueeze(-1).float()  [B, S, 1] |
|      embedding = (last_hidden_state * mask_expanded).sum(1)         |
|                  / mask_expanded.sum(1).clamp(min=1)       [B, 768]  |
|  Autocast: bfloat16 on CUDA                                         |
|  How it learns: LoRA adapts SMILES transformer representations      |
|    toward kinetic property prediction from frozen chemical LM       |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
|  SubstrateProjection                                                 |
|  Class: ProjectionLayer  (file: src/models/projection.py)           |
|  Input:  x  [B, 768]   float32  (MoLFormer embedding)              |
|  Output: x  [B, 512]   float32                                      |
|  Frozen:  Nothing frozen                                             |
|  Trainable: LayerNorm(768), Linear(768->512) weight+bias, GELU      |
|    (normalize_input=True enabled in current config)                  |
|  Architecture: Sequential(LayerNorm(768), Linear(768->512), GELU)   |
|  How it learns: Aligns MoLFormer's 768-dim space to the 512-dim     |
|    fusion space; LayerNorm stabilizes varying MoLFormer output      |
|    magnitudes during LoRA-driven representation shifts              |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
|  CrossAttentionFusion                                                |
|  Class: CrossAttentionFusion  (file: src/models/cross_attention.py) |
|  Input:  h_sub            [B, 512]        float32  (substrate vec)  |
|          protein_residues [B, L, 1536]    float32  (ESM-3 hidden)   |
|          protein_pad_mask [B, L]          bool     (True=pad)        |
|  Output: ctx              [B, 512]        float32                   |
|  Frozen:  Nothing frozen                                             |
|  Trainable:                                                          |
|    query_proj:  Linear(512, 4*512=2048)  (multi-query expansion)    |
|    protein_norm: LayerNorm(1536)                                     |
|    sub_to_prot: MultiheadAttention(embed_dim=512, num_heads=8,       |
|      kdim=1536, vdim=1536, dropout=0.1, batch_first=True)           |
|    norm1: LayerNorm(512) output normalization                        |
|  Architecture: B1 unidirectional multi-query cross-attention         |
|    * n_queries=4 learned substrate queries                           |
|    * substrate is QUERY, protein residues are KEY and VALUE          |
|    * query_proj(h_sub) -> [B, 4*512] -> view [B, 4, 512]           |
|    * protein_norm(protein_residues) -> [B, L, 1536]                 |
|    * MHA(Q=[B,4,512], K=[B,L,1536], V=[B,L,1536]) -> [B,4,512]    |
|    * Residual: ctx = norm1(q + attn_out) -> [B, 4, 512]            |
|    * Mean pool over queries: ctx.mean(dim=1) -> [B, 512]           |
|  Internal MHA projections (PyTorch nn.MultiheadAttention):           |
|    * in_proj_weight splits into: W_q [512,512], W_k [512,1536],    |
|      W_v [512,1536]                                                  |
|    * out_proj: Linear(512, 512)                                      |
|    * Per-head dim = 512/8 = 64                                       |
|  How it learns: Each of 4 queries learns to attend to different      |
|    aspects of protein-substrate interaction (e.g., active site       |
|    residues, binding pocket geometry, allosteric sites, surface).    |
|    Mean-pooling aggregates these complementary views into a single   |
|    fused representation for the MLP head.                            |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
|  ConditionEncoder  (CURRENTLY NOT INSTANTIATED -- use_conditioning   |
|                     is false)                                         |
|  Class: ConditionEncoder  (file: src/models/condition_encoder.py)   |
|  Input:  ph          [B]       float32  (0.0 if unknown)            |
|          temp        [B]       float32  (0.0 if unknown)            |
|          ph_known    [B]       bool                                  |
|          temp_known  [B]       bool                                  |
|          cofactor_fp [B, 512]  float32  (binary Morgan FP)          |
|  Output: cond_vec    [B, 64]   float32                              |
|  Architecture:                                                       |
|    ph_embed:      Sequential(Linear(1,32), GELU)                    |
|    temp_embed:    Sequential(Linear(1,32), GELU)                    |
|    ph_unknown:    nn.Parameter [32]  (zero-init, no weight decay)   |
|    temp_unknown:  nn.Parameter [32]  (zero-init, no weight decay)   |
|    cofactor_proj: Sequential(Linear(512,64), GELU)                  |
|    out_proj:      Sequential(Linear(128,64), GELU)                  |
|  Normalization:                                                      |
|    pH:   (ph - 7.0) / 4.0        -> centers on neutral              |
|    temp: (temp - 37.0) / 20.0    -> centers on 37C physiological    |
|  Status: cond_encoder=None, film=None when use_conditioning=false   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
|  FiLMLayer  (CURRENTLY NOT INSTANTIATED)                             |
|  Class: FiLMLayer  (file: src/models/condition_encoder.py)          |
|  Input:  x         [B, 512]  float32  (CrossAttention output ctx)   |
|          cond      [B, 64]   float32  (from ConditionEncoder)       |
|  Output: modulated [B, 512]  float32                                |
|  Trainable: gamma_proj: Linear(64->512); beta_proj: Linear(64->512) |
|  Initialization:                                                     |
|    gamma_proj.weight = zeros,  gamma_proj.bias = ones  -> gamma ~ 1 |
|    beta_proj.weight  = zeros,  beta_proj.bias  = zeros -> beta ~ 0  |
|    -> At epoch 0 the layer is a no-op identity transform             |
|  Operation: output = gamma(cond) * x + beta(cond)                   |
|  Status: Not instantiated when use_conditioning=false                |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
|  MLPHead                                                             |
|  Class: MLPHead  (file: src/models/mlp_head.py)                     |
|  Input:  x   [B, 512]  float32  (cross-attention ctx vector)        |
|  Output: y   [B]       float32  (log10 kinetic parameter)           |
|  Frozen:  Nothing frozen                                             |
|  Trainable: All parameters                                           |
|    input_norm: LayerNorm(512)                                        |
|    layer0: Sequential(Linear(512->512), GELU, Dropout(0.2))        |
|    residual_norm: LayerNorm(512)  [d_in==hidden[0] -> residual ON]  |
|    remaining: Sequential(                                            |
|      Linear(512->256), GELU, Dropout(0.1),                          |
|      Linear(256->128), GELU)                                         |
|    out: Linear(128->1)  -> squeeze(-1) -> [B]                       |
|  Residual: h = residual_norm(x_normalized + layer0(x_normalized))   |
|    (only when d_in == hidden_dims[0] == 512, which is true here)    |
|  How it learns: Regresses the final kinetic value from the fused     |
|    protein-substrate representation via dimensionality reduction     |
|    512 -> 512 (residual) -> 256 -> 128 -> 1                         |
+----------------------------------------------------------------------+
```

---

## Part C — FiLM Conditioning Deep Dive (Reference — Currently Disabled)

> **Note:** As of the current config (`conditioning.use_conditioning: false`), FiLM
> conditioning is completely disabled. The `ConditionEncoder` and `FiLMLayer` modules
> are not instantiated (`cond_encoder=None`, `film=None`). The batch still carries
> `ph`, `temp`, `ph_known`, `temp_known`, and `cofactor_fp` tensors but they are
> accepted and ignored in `KICMModel.forward()`. This section documents the mechanism
> for reference when re-enabling conditioning.

### C.1 What gamma and beta mean mathematically

FiLM (Feature-wise Linear Modulation) applies an **elementwise affine transformation**
to a feature map conditioned on auxiliary inputs:

```
output_i = gamma_i(cond) * x_i + beta_i(cond)

where:
  x      in R^512  -- the fused protein-substrate context vector
  cond   in R^64   -- the condition vector from ConditionEncoder
  gamma  in R^512  -- predicted per-feature scale (from Linear(64->512))
  beta   in R^512  -- predicted per-feature shift (from Linear(64->512))
  output in R^512  -- the modulated context vector
```

**Intuition:** gamma controls how much each feature dimension is amplified or suppressed
depending on the pH/temperature/cofactor context. beta shifts the representation toward
the correct "operating point" for the given experimental conditions. Both are predicted
jointly from the same 64-dimensional condition vector, meaning the scale and shift for
all 512 features are determined by the same pH + temperature + cofactor signal.

### C.2 Identity initialization and why it prevents destabilization

At initialization:
```python
nn.init.zeros_(self.gamma_proj.weight)   # W_gamma = 0
nn.init.ones_(self.gamma_proj.bias)      # b_gamma = 1  ->  gamma(cond) = 0*cond + 1 = 1
nn.init.zeros_(self.beta_proj.weight)    # W_beta = 0
nn.init.zeros_(self.beta_proj.bias)      # b_beta = 0  ->  beta(cond) = 0*cond + 0 = 0
```

**At epoch 0:** `output = 1 * x + 0 = x` -- the FiLM layer is a strict no-op.

**Why this matters:** If gamma and beta were randomly initialized, they would immediately corrupt
the carefully learned fused representation `ctx`, causing large training loss spikes and
potentially unstable gradients early in training. By starting as identity, the model first
learns the base protein-substrate task (through the other modules), and only once the main
gradients are stable does FiLM's conditioning begin to diverge from identity.

### C.3 How `torch.where` selects known vs unknown embeddings

```python
ph_emb = torch.where(
    ph_known.unsqueeze(-1),              # [B, 1]  broadcast to [B, d_mlp_hidden]
    self.ph_embed(ph_norm),              # [B, 32]  -- scalar embedding for known pH
    self.ph_unknown.unsqueeze(0).expand(B, -1),  # [B, 32]  -- learned unknown embedding
)
```

`torch.where(condition, x, y)` selects element-wise: `x[i]` where `condition[i]=True`,
`y[i]` where `condition[i]=False`. The `unsqueeze(-1)` on `ph_known` broadcasts the
scalar boolean `[B, 1]` across the 32-dimensional embedding space.

### C.4 Why `ph_unknown` / `temp_unknown` must be in `task_no_decay`

**Weight decay** applies L2 regularization: at each optimizer step, weights are shrunk by
`weight_decay * parameter_value`. For these learned unknown embeddings it is harmful:

- Start at zero -> L2 penalty = 0 initially, which is fine
- Once training begins to push them away from zero, L2 immediately pulls them back
- The learning signal (gradient from unknown-condition samples, ~6% of data) competes with
  a constant decay force that fires on every batch (100% of steps)
- Net result: the embeddings can never escape the zero-initialization basin

By routing them to `task_no_decay` (weight_decay=0.0), they are free to move wherever the
gradient pushes them.

---

## Part D — Cofactor Handling

### D.1 What `separate_reactants` does

`separate_reactants(reactant_smiles, fp_bits=512)` in `src/data/cofactor_registry.py`
processes a potentially multi-component SMILES string (components separated by `.`) into:
1. A single substrate SMILES string for MoLFormer
2. A binary fingerprint summarizing all cofactors present

**Strategy (Strategy B, analysis date 2026-04-06):**
- 29 canonical SMILES in the whitelist, covering 83.5% of multi-component rows
- Threshold: molecule appears as cofactor in >= 95% of its appearances AND across >= 8 distinct EC classes
- Includes water (O), O2, ATP, NAD+/NADH variants, H+, SAM, CoA, pyruvate, phosphate, etc.

**Algorithm:**
```
1. Split reactant_smiles on '.'  -> components
2. For each component:
   a. Parse with RDKit -> canonical SMILES
   b. If canonical in _COFACTOR_CANONICAL_SET:
      - Compute Morgan FP (radius=2, fp_bits=512)
      - Add to cofactor_fp accumulator
      - Increment n_cofactors
   c. Else:
      - Compute molecular weight (Descriptors.MolWt)
      - Append (mw, smiles) to substrate_candidates
3. clamp(cofactor_fp, 0, 1) -> binary (handles shared bits from multiple cofactors)
4. If substrate_candidates is empty (all components are cofactors):
   -> Edge case: use heaviest component as fallback substrate
5. Else:
   -> primary substrate = max(substrate_candidates, key=mw)
6. Return: substrate_smiles, cofactor_fp, n_cofactors
```

**Heaviest-by-MW heuristic** for substrate selection: In enzyme kinetics, the true
substrate is almost always the largest molecule in the reaction -- cofactors tend to be
smaller energy-carrying molecules.

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

When `use_conditioning=true`, `cofactor_fp [B, 512]` is consumed by `ConditionEncoder`:
```
KICMCollator  ->  batch["cofactor_fp"]  [B, 512]
   |
KICMModel.forward(..., cofactor_fp)
   |
ConditionEncoder.forward(ph, temp, ph_known, temp_known, cofactor_fp)
   |  cofactor_emb = cofactor_proj(cofactor_fp)      [B, 512] -> [B, 64]
   |  cond = cat([ph_emb[B,32], temp_emb[B,32], cofactor_emb[B,64]])
   |       -> [B, 128]
   |  cond_vec = out_proj(Linear(128,64)+GELU)        -> [B, 64]
   |
FiLMLayer -- gamma and beta predicted from cond_vec
```

**When `use_conditioning=false` (current state):** `cofactor_fp` is passed through
the function signature but never consumed. The collator still computes it so that
re-enabling conditioning requires no data pipeline changes.

---

## Part E — Optimizer Param Groups Table

### E.1 Current active configuration

With `molformer_frozen=true`, `molformer_lora.enabled=true`, `use_conditioning=false`:

| Group name | LR | Weight Decay | What's in it | Count |
|---|---|---|---|---|
| `task_decay` | 2.0e-4 | 1.0e-2 | Projection weights, CrossAttn weights (query_proj, MHA, norms), MLPHead weights -- any task module param where name doesn't end in `.bias` and no segment contains `"norm"` | varies |
| `task_no_decay` | 2.0e-4 | 0.0 | Bias params + LayerNorm params in task modules | varies |
| `molformer_lora` | 5.0e-5 | 1.0e-2 | MoLFormer LoRA adapter matrices (all 2D, `bias="none"`) across 6 target module types | varies |
| `esm3_lora` | 2.0e-5 | 1.0e-2 | ESM-3 LoRA adapter matrices for blocks 36-47 (all 2D, `bias="none"`) | varies |

**Active groups:** 4 total (`task_decay`, `task_no_decay`, `molformer_lora`, `esm3_lora`)

**Note:** When `use_conditioning=false`, there are NO `cond_encoder.ph_unknown` or
`cond_encoder.temp_unknown` parameters to route (they don't exist), so the explicit
routing block in `configure_optimizers()` is safely skipped.

### E.2 All possible configurations

| Group name | LR | Weight Decay | Condition for existence |
|---|---|---|---|
| `task_decay` | 2.0e-4 | 1.0e-2 | Always (when non-empty) |
| `task_no_decay` | 2.0e-4 | 0.0 | Always (when non-empty) |
| `molformer_decay` | 5.0e-5 | 1.0e-2 | Only when `molformer_frozen=False` |
| `molformer_no_decay` | 5.0e-5 | 0.0 | Only when `molformer_frozen=False` |
| `molformer_lora` | 5.0e-5 | 1.0e-2 | Only when `molformer_frozen=True` AND `molformer_lora.enabled=True` |
| `esm3_lora` | 2.0e-5 | 1.0e-2 | When `esm3_lora.enabled=True` (default) |

### E.3 No-decay criterion

```python
@staticmethod
def _is_no_decay_param(name: str) -> bool:
    if name.endswith(".bias"):
        return True
    parts = name.split(".")
    return any("norm" in part.lower() for part in parts)
```

### E.4 LR schedule

Single `LambdaLR` with the same lambda applied to all groups:
- Steps [0, warmup_steps=100): linear ramp from 0 to 1
- Steps [warmup_steps, total_steps): cosine decay from 1 to `min_lr_ratio=0.01`
- Steps >= total_steps: floor at `min_lr_ratio=0.01`

Each group's effective LR = base_lr * lambda(step). So at warmup end:
- task: 2.0e-4 * 1.0 = 2.0e-4
- esm3_lora: 2.0e-5 * 1.0 = 2.0e-5
- molformer_lora: 5.0e-5 * 1.0 = 5.0e-5

At min_lr_ratio:
- task: 2.0e-4 * 0.01 = 2.0e-6
- esm3_lora: 2.0e-5 * 0.01 = 2.0e-7
- molformer_lora: 5.0e-5 * 0.01 = 5.0e-7

---

## Part F — Detailed Tensor Shape Trace

Complete shape trace through a single forward pass with B=6 (batch_size_per_gpu),
max_seq_len=1024:

```
Input:
  sequence_tokens:  [6, L]       int64     L <= 1026 (1024 + BOS + EOS)
  esm_pad_mask:     [6, L]       bool      True at padding positions
  smiles:           list[str]    len=6     substrate SMILES only (cofactors separated)
  cofactor_fp:      [6, 512]     float32   binary Morgan FP
  ph:               [6]          float32
  temp:             [6]          float32
  ph_known:         [6]          bool
  temp_known:       [6]          bool
  targets:          [6]          float32   log10 kinetic values

ESM3Encoder:
  sequence_tokens -> ESM3.forward() -> hook captures blocks[-1] output
  hidden:           [6, L, 1536]   float32  (cast from bfloat16)
  mask:             [6, L]         bool     (pass-through)

MoLFormerEncoder:
  smiles -> AutoTokenizer -> input_ids [6, S], attention_mask [6, S]
    (S = max token length in batch, S <= 202)
  AutoModel(**inputs) -> pooler_output [6, 768] or mean-pool fallback
  h_substrate:      [6, 768]     float32  (cast from bfloat16)

ProjectionLayer:
  h_substrate [6, 768]
  -> LayerNorm(768)  [6, 768]
  -> Linear(768,512) [6, 512]
  -> GELU            [6, 512]
  h_sub:              [6, 512]     float32

CrossAttentionFusion (n_queries=4):
  h_sub [6, 512]
  -> query_proj(h_sub) [6, 2048]  -> view [6, 4, 512]
  q:                  [6, 4, 512]  float32

  protein_residues [6, L, 1536]
  -> protein_norm(LayerNorm(1536)) [6, L, 1536]
  prot_normed:      [6, L, 1536]  float32

  MHA internals:
    Q projected:    [6, 4, 512]  -> per-head: [6, 8, 4, 64]
    K projected:    [6, L, 512]  -> per-head: [6, 8, L, 64]  (kdim=1536->512)
    V projected:    [6, L, 512]  -> per-head: [6, 8, L, 64]  (vdim=1536->512)
    attn weights:   [6, 8, 4, L]  (softmax over L dim, masked by protein_pad_mask)
    attn values:    [6, 8, 4, 64] -> concat -> [6, 4, 512]
    out_proj:       [6, 4, 512]

  attn_out:         [6, 4, 512]  float32
  ctx_residual = LayerNorm(q + attn_out):  [6, 4, 512]
  ctx = mean(dim=1):  [6, 512]   float32

[FiLM -- SKIPPED when use_conditioning=false]
  ctx passes through unchanged: [6, 512]

MLPHead:
  input_norm(LayerNorm(512)):    [6, 512]
  layer0(Linear+GELU+Drop):     [6, 512]
  residual_norm(x + h):         [6, 512]   (residual ON: d_in==hidden[0]==512)
  Linear(512,256)+GELU+Drop:    [6, 256]
  Linear(256,128)+GELU:         [6, 128]
  Linear(128,1).squeeze(-1):    [6]

Loss:
  predictions:  [6]  float32
  targets:      [6]  float32
  SmoothL1Loss(beta=0.3):  scalar
```

---

## Part G — Training Configuration Reference

### G.1 Current config values (train_config.yaml)

```yaml
project:
  name: "esm3-molformer-kcat-CA-MolLoRA"
  seed: 42

data:
  max_seq_len:  1024
  num_workers:  4

model:
  esm3_variant:   "esm3_sm_open_v1"
  d_esm3:         1536
  substrate_encoder: "molformer"
  d_substrate:    768
  d_proj:         512
  d_fusion:       512
  molformer_model: "ibm/MoLFormer-XL-both-10pct"
  molformer_max_length: 202
  molformer_frozen: true
  molformer_lora:
    enabled: true
    r: 16, alpha: 16, dropout: 0.05
    targets: [query, key, value, attn_out, intermediate, output] (6 types)
  esm3_lora:
    enabled: true
    r: 12, alpha: 24, dropout: 0.05
    targets: [layernorm_qkv.1, out_proj, ffn.1]
    lora_start_block: 36  (blocks 36-47 only)
  cross_attention:
    d_sub: 512, d_prot: 1536, n_heads: 8, dropout: 0.1, n_queries: 4
  mlp:
    hidden_dims: [512, 256, 128], dropouts: [0.2, 0.1, 0.0], activation: gelu

conditioning:
  use_conditioning: false  # FiLM + ConditionEncoder DISABLED
  d_cond: 64, d_mlp_hidden: 32

training:
  num_gpus: 2
  precision: "bf16-mixed"
  batch_size_per_gpu: 6
  gradient_accumulation_steps: 8
  effective_batch_size: 96  (6 * 2 GPUs * 8 accum)
  max_epochs: 100
  loss: "huber"
  huber_beta: 0.3
  gradient_clip_norm: 1.0
  lr:
    task: 2.0e-4
    esm3_lora: 2.0e-5
    molformer_lora: 5.0e-5
  weight_decay: 1.0e-2
  scheduler:
    warmup_steps: 100
    min_lr_ratio: 0.01
  early_stopping:
    monitor: "val/r2", mode: "max", patience: 15, min_delta: 0.0005
  checkpoint:
    save_top_k: 3, monitor: "val/r2", mode: "max"
```

### G.2 DDP configuration

- 2 GPUs, NCCL backend
- bf16-mixed precision
- `find_unused_parameters: false` (all parameters participate in forward)
- Gradient accumulation = 8 -> effective batch = 6 * 2 * 8 = 96
- Validation: `all_gather` to collect predictions from all ranks before computing metrics
- Test: runs on single GPU (rank 0 only) via `devices=1` in train.py

---

## Part H — Generalization Design Rationale

### H.1 Freezing + LoRA: preventing catastrophic forgetting while enabling adaptation

**Catastrophic forgetting** occurs when fine-tuning a pretrained model on a narrow task
causes the model to lose its broad, transferable representations. For ESM-3, the general
protein language model representations are the primary source of generalization to OOD
sequences -- sequences outside the CD-HIT 40% training set.

**LoRA (Low-Rank Adaptation)** constrains adaptation to a low-rank subspace:
```
W_adapted = W_frozen + (alpha/r) * B * A
where A in R^{r x d_in}, B in R^{d_out x r}
```

The number of trainable parameters in ESM-3 via LoRA is ~4-6M vs ~690M total -- less than 1%.
The low learning rate for LoRA (2.0e-5 vs 2.0e-4 for task params) further slows adaptation.

**Block restriction (lora_start_block=36):** Only the last 12 of 48 blocks have trainable
LoRA. Earlier blocks contain more general protein representations (secondary structure,
fold-level features); later blocks contain more task-specific features. By restricting
LoRA to later blocks, we preserve general features while adapting task-specific ones.

### H.2 Multi-query cross-attention: richer protein-substrate interaction

With n_queries=4, the substrate vector is projected into 4 independent queries, each of
which attends to the full protein sequence. This allows the model to capture multiple
aspects of protein-substrate interaction simultaneously:

- Query 1 might learn to focus on active site residues
- Query 2 might attend to binding pocket geometry
- Query 3 might capture allosteric regulation sites
- Query 4 might focus on surface accessibility

After attention, the 4 query outputs are mean-pooled back to a single [B, 512] vector.
This is richer than a single query (which must compress all interaction information into
one attention pattern) while remaining computationally efficient.

### H.3 Huber loss (SmoothL1, beta=0.3)

The Huber loss combines L2 for small errors (|error| < beta=0.3) and L1 for large errors:
```
L(error) = 0.5 * error^2 / beta   if |error| < beta
         = |error| - 0.5 * beta    if |error| >= beta
```

**Why beta=0.3:** In log10 space, an error of 0.3 corresponds to a 2x factor in the
original kinetic parameter. Errors smaller than 2x are penalized quadratically (smooth
optimization), while errors larger than 2x are penalized linearly (robustness to outliers).
This is particularly important for enzyme kinetics where some extreme measurements may be
noisy or represent genuinely different conditions.

---

## Part I — Known Bottlenecks and Architectural Observations

### I.1 Information bottleneck at cross-attention K/V projection

The internal MHA projects protein representations from 1536 -> 512 dimensions for K and V:
```
K_proj: [B, L, 1536] -> [B, L, 512]   (67% compression)
V_proj: [B, L, 1536] -> [B, L, 512]   (67% compression)
```

This loses 1024 dimensions of per-residue information before any attention occurs.
ESM-3's 1536-dim hidden states encode rich structural and evolutionary information
that may be partially discarded by this projection.

**Potential fix:** Increase cross-attention embed_dim to 768 or 1024 to reduce compression,
or add a learned residue-level projection before cross-attention.

### I.2 Single-layer cross-attention (B1)

The current architecture uses a single cross-attention layer. Multi-layer cross-attention
(e.g., 2-3 layers with residual connections) could capture more complex protein-substrate
interactions but at increased computational cost.

### I.3 MoLFormer pooling to CLS/pooler vs sequence

MoLFormer's output is pooled to a single vector [B, 768], discarding all token-level
information about the substrate. For larger substrates, token-level attention (where
protein queries attend to substrate tokens AND vice versa) could capture finer-grained
interactions.

### I.4 FiLM conditioning disabled

The current ablation disables FiLM conditioning entirely, meaning pH, temperature, and
cofactor identity have NO influence on predictions. This simplifies the model to test
whether the base ESM3+MoLFormer+CrossAttention+MLP architecture is sufficient, but
means the model cannot adapt its predictions based on experimental conditions.

If re-enabling, consider:
- Whether the conditioning signal is strong enough to justify the additional parameters
- Whether cofactor information should be injected at a different point (e.g., concatenated
  with substrate embedding rather than via FiLM modulation)
- Whether the val-test gap (R^2 0.57 vs 0.52) was caused by FiLM overfitting

### I.5 Effective batch size and gradient accumulation

Effective batch = 96 (6 per GPU * 2 GPUs * 8 accumulation). This is relatively large
for a dataset of this size and may smooth out informative gradients from rare enzyme
families. Consider reducing if training plateaus early.

---

## Part J — File Reference

| File | Path | Role |
|---|---|---|
| `train_config.yaml` | `configs/train_config.yaml` | Full YAML configuration |
| `train.py` | `train.py` | Entry point, arg parsing, Trainer construction |
| `kicm_model.py` | `src/models/kicm_model.py` | Top-level nn.Module assembly |
| `kicm_module.py` | `src/models/kicm_module.py` | Lightning Module (train/val/test steps, optimizer) |
| `esm3_encoder.py` | `src/models/esm3_encoder.py` | ESM-3 encoder with hook + LoRA |
| `molformer_encoder.py` | `src/models/molformer_encoder.py` | MoLFormer-XL encoder with LoRA |
| `projection.py` | `src/models/projection.py` | Substrate projection (768->512) |
| `cross_attention.py` | `src/models/cross_attention.py` | B1 multi-query cross-attention |
| `condition_encoder.py` | `src/models/condition_encoder.py` | ConditionEncoder + FiLMLayer |
| `mlp_head.py` | `src/models/mlp_head.py` | MLP prediction head |
| `dataset.py` | `src/data/dataset.py` | CSV -> KICMDataset |
| `collator.py` | `src/data/collator.py` | Batch assembly + tokenization |
| `cofactor_registry.py` | `src/data/cofactor_registry.py` | 29-molecule cofactor whitelist |
| `kicm_datamodule.py` | `src/data/kicm_datamodule.py` | Lightning DataModule |
| `metrics.py` | `src/training/metrics.py` | R2, RMSE, MAE, Pearson, Spearman |
