# ESM2-DMPNN KICM Predictor

Predict enzyme kinetic parameters (log10 target) from protein sequence and substrate SMILES.

## DEVIATION NOTICE

The blueprint specified **ESM-3** (`esm>=3.0.0`, `esm3_sm_open_v1`, d_model=1536). The active `kicmv1` environment has **`fair-esm==2.0.0` (ESM-2 only)**. All files marked `# DEVIATION:` use ESM-2 (`esm2_t33_650M_UR50D`, d_model=1280, 33 layers) with the same API contract.

To upgrade to ESM-3 when available:
1. `pip install esm>=3.0.0`
2. Change `esm3_variant` and `d_esm3` in `configs/train_config.yaml`
3. Update `ESM2Encoder` in `src/models/esm3_encoder.py` to use `ESM3.from_pretrained()` and `esm3.transformer.blocks`

Similarly, the blueprint specified **chemprop>=2.0.0**. The installed version is **chemprop==1.6.1** (v1 API). The D-MPNN encoder uses `MPNEncoder + BatchMolGraph` from v1 with identical output shape [B, 300].

---

## Architecture

```
sequence_tokens [B,L] -> ESM-2 (penultimate layer) -> [B,L,1280] -> AttentivePooling -> [B,1280]
                                                                                          |
                                                                                   Projection A -> [B,512]
                                                                                          |
smiles list[str] -> D-MPNN (Chemprop v1) -> [B,300] -> Projection B -> [B,512] -> Concatenate -> [B,1024]
                                                                                          |
                                                                                     LayerNorm
                                                                                          |
                                                                                    MLP Head -> [B] (log10 target)
```

### 3-Phase Training

| Phase | Epochs | ESM-2 | ESM LR | Task LR |
|-------|--------|-------|--------|---------|
| 1 | 1-5 | Frozen | 0 | 1e-4 |
| 2 | 6-15 | Last 2 blocks unfrozen | 5e-6 | 1e-4 |
| 3 | 16-150 | Fully unfrozen | 1e-5 | 1e-4 |

---

## Setup

```bash
conda activate kicmv1
pip install -r requirements.txt
```

---

## Data Format

CSV with columns:
- `sequence` — amino acid sequence (1-letter code)
- `reactant_smiles` — substrate canonical SMILES
- `log10_value` — target in log10 space

Sequences > 1024 AA and invalid SMILES are filtered at dataset load time.

---

## Training

```bash
# Multi-GPU (4 x L40S)
torchrun --nproc_per_node=4 --master_port=29500 train.py \
    --config configs/train_config.yaml

# Single GPU / debug
python train.py --config configs/train_config.yaml
```

Checkpoints are saved to `outputs/checkpoints/`. W&B logging is enabled by default (set `logging.backend: none` to disable).

---

## Evaluation

```bash
python evaluate.py \
    --config configs/train_config.yaml \
    --checkpoint outputs/checkpoints/epoch010-r20.7123.pt \
    --test-csv data/test_ood_40.csv \
    --split-name test_ood_40 \
    --output outputs/metrics_ood40.json
```

---

## Project Structure

```
kcat_predictor/
├── configs/
│   └── train_config.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py       # KICMDataset (CSV -> sequence/smiles/target)
│   │   └── collator.py      # KICMCollator (ESM-2 tokenisation + padding)
│   ├── models/
│   │   ├── esm3_encoder.py  # ESM2Encoder + AttentivePooling
│   │   ├── dmpnn_encoder.py # DMPNNEncoder (Chemprop v1)
│   │   ├── projection.py    # ProjectionLayer
│   │   ├── mlp_head.py      # MLPHead
│   │   └── kicm_model.py    # KICMModel (full assembly)
│   ├── training/
│   │   ├── trainer.py       # DDP training loop + phase management
│   │   ├── metrics.py       # R2, RMSE, MAE, Spearman, Pearson
│   │   └── callbacks.py     # EarlyStopping, ModelCheckpoint
│   └── utils/
│       ├── seed.py          # set_seed()
│       └── logging_utils.py # W&B + file logger setup
├── train.py                 # torchrun entry point
├── evaluate.py              # standalone evaluation
├── requirements.txt
└── README.md
```

---

## Monitoring Risks (from blueprint Section 9)

1. If `val_r2 < 0.50` after Phase 1 (epoch 5): check ESM tokenisation and padding mask
2. If `train_r2 - val_r2 > 0.15` in Phase 3: increase MLP dropout to 0.3/0.2
3. Loss spikes in epochs 1-3: increase `warmup_steps` to 1000
4. Huber beta miscalibration: plot `|pred - target|` histogram after epoch 1

Expected performance (from blueprint):
| Metric | Pessimistic | Realistic | Optimistic |
|--------|-------------|-----------|------------|
| Val R2 | 0.62 | 0.72 | 0.80 |
| Val RMSE (log10) | 0.85 | 0.70 | 0.58 |
