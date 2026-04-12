# Data Pipeline Exploration Index

## Overview
This directory contains a comprehensive read-only exploration of the kcat predictor data pipeline, specifically focusing on pH and temperature conditioning for FiLM-based adaptation.

**Project:** Kcat Predictor (D2 LoRA Architecture)  
**Branch:** `binhpht/feat/esm3-molformer-kcat-CA-LoRA`  
**Exploration Date:** 2026-04-07

---

## Documentation Files

### 1. **EXPLORATION_SUMMARY.txt** (15 KB) ⭐ START HERE
Quick reference guide with:
- Executive summary of findings
- Missing value statistics
- Data pipeline status checklist
- Identified bugs (2 blocking issues)
- Implementation checklist (6 phases)
- Key file locations with line numbers
- One-page data statistics

**Best for:** Getting oriented quickly, understanding the big picture

### 2. **DATA_PIPELINE_EXPLORATION.md** (23 KB) 📖 DETAILED REFERENCE
Comprehensive 737-line technical report with:
- Section 1: Raw data files & file locations
- Section 2: Detailed missing value analysis
- Section 3: Dataset class (dataset.py) code walkthrough
- Section 4: Collator class (collator.py) code walkthrough
- Section 5: Configuration file analysis
- Section 6: Model architecture (kicm_model.py)
- Section 7: Training loop (kicm_module.py)
- Section 8: Existing conditioning infrastructure survey
- Section 9-10: Missing data handling strategies (3 options)
- Section 11-12: Bugs and validation infrastructure
- Section 13-20: Statistics, recommendations, and implementation plans

**Best for:** Deep dives into specific code sections, detailed planning, reference material

---

## Quick Facts

### Missing Value Statistics
| Dataset | Total | Both Conditions | % Complete | Neither | % Missing Both |
|---------|-------|----------------|------------|---------|----------------|
| Train | 18,751 | 17,693 | 94.36% | 244 | 1.30% |
| Val | 2,084 | 2,012 | 96.5% | N/A | N/A |
| Test | 2,316 | ~2,160 | 93% | N/A | N/A |

### Data Pipeline Status
```
✅ 80% Complete
✅ Dataset reads pH/temp with NaN detection
✅ Collator extracts & batches conditions
❌ Config missing ph_col, temp_col (BLOCKER)
❌ Model doesn't accept conditions yet
❌ FiLM module doesn't exist
```

### Critical Issues Found
1. **BUG #1:** Config missing `ph_col` and `temp_col` → Will crash at training
2. **BUG #2:** DataModule uses hard-coded dict access without defaults

---

## Key Files Referenced

### Data Pipeline (Already Implemented ✅)
- **src/data/dataset.py** (210 lines)
  - Lines 20-31: KICMSample TypedDict (has ph, temp, ph_known, temp_known!)
  - Lines 160-175: NaN detection and missing value handling
  
- **src/data/collator.py** (231 lines)
  - Lines 91-95: Condition tensor creation
  - Lines 169-180: Batch dict output (includes all conditions!)

- **src/data/kicm_datamodule.py** (163 lines)
  - Lines 68-69: Config access (BUG: no defaults!)

### Model Architecture (Needs Modification)
- **src/models/kicm_model.py** (249 lines)
  - Lines 132-164: forward() signature (needs conditions!)
  - D2 LoRA architecture: ESM-3 + MoLFormer + CrossAttention + MLP

- **src/models/mlp_head.py** (162 lines)
  - Regression head where FiLM output feeds in

- **src/models/cross_attention.py** (66 lines)
  - B1 unidirectional fusion (output: [B, 512])

### Training Loop (Needs Modification)
- **src/models/kicm_module.py** (220+ lines)
  - Lines 72-85: training_step() (missing condition extraction!)
  - Lines 91-99: validation_step()
  - Lines 140-148: test_step()

### Configuration (Missing Entries)
- **configs/train_config.yaml** (142 lines)
  - Lines 16-27: data section (MISSING ph_col, temp_col!)
  - Lines 29-73: model section (ADD film config here)

---

## Implementation Roadmap

### Phase 1: Fix Blocking Bugs ⚠️ DO THIS FIRST
- [ ] Add `ph_col: "ph"` to configs/train_config.yaml
- [ ] Add `temp_col: "temperature"` to configs/train_config.yaml
- [ ] Update src/data/kicm_datamodule.py to use `.get()` with defaults
- [ ] Verify with smoke_test.py

### Phase 2: Create FiLM Module (NEW)
- [ ] Create src/models/film.py
- [ ] Implement FiLMLayer class
- [ ] Implement ConditionEncoder class
- [ ] Handle ph_known/temp_known flags

### Phase 3: Integrate FiLM into Model
- [ ] Modify KICMModel.forward() signature
- [ ] Insert FiLM layer after cross_attn
- [ ] Wire to MLPHead input

### Phase 4: Wire into Training Loop
- [ ] Extract conditions in training_step()
- [ ] Pass conditions to model
- [ ] Update validation/test steps similarly

### Phase 5: Configuration
- [ ] Add FiLM config section to train_config.yaml

### Phase 6: Testing
- [ ] Add FiLM smoke tests
- [ ] Test all-known, mixed, all-unknown scenarios
- [ ] Verify gradient flow

---

## Missing Data Strategy: Masking (Recommended)

For samples with missing condition values:

1. **Data Pipeline** (already done!)
   - Set condition value to 0.0 (placeholder)
   - Set _known flag to False
   - Include in training (no data loss)

2. **FiLM Layer** (to implement)
   - If ph_known=False: use identity transform (gamma=1.0, beta=0.0)
   - If temp_known=False: use identity transform
   - Otherwise: apply learned modulation

3. **Result**
   - Model learns conditions when available (94% of data)
   - Model learns robustness when unavailable (~6% of data)
   - Natural regularization built-in

---

## Data Statistics Summary

### Missing Conditions (Training Data)
- pH missing: 594 samples (3.17%)
- Temperature missing: 708 samples (3.78%)
- Both missing: 244 samples (1.30%)

### Value Ranges
- pH: 1.0 to 11.5 (mostly ~7.2, neutral/slightly basic)
- Temperature: 0.0 to 105.0°C (mostly 25-30°C)

### CSV Format (43 total columns)
- Required: sequence, reactant_smiles, log10_value
- Conditions: temperature, ph
- Other: uniprot, ec, taxonomy_id, reaction info, clustering

---

## Recommended Reading Order

1. **First:** EXPLORATION_SUMMARY.txt (15 min)
   - Get the overview
   - Understand the bugs
   - See the checklist

2. **Then:** DATA_PIPELINE_EXPLORATION.md (30 min)
   - Deep dive into specific code sections
   - Understand the data flow
   - Review implementation options

3. **Reference:** Keep both files handy during implementation
   - Use line numbers to find code quickly
   - Check implementation checklist for progress
   - Refer to data flow diagrams

---

## Key Insights

### The Good News ✅
- **80% of infrastructure already exists!**
- Dataset already reads and tracks pH/temp
- Collator already batches conditions with flags
- Missing value handling strategy already in place (0.0 with flags)
- Data is sparse (1.3% both missing) - masking strategy perfect fit

### The Challenges ⚠️
- Config is missing required entries (BLOCKER)
- Model doesn't accept condition parameters yet
- FiLM module needs to be created
- Training loop needs to pass conditions through

### The Opportunity 🎯
- Easy 20% to finish the implementation
- Medium complexity (200-300 lines of new code)
- High impact: enables pH/temperature-aware predictions
- Clean architecture: FiLM sits naturally after cross-attention

---

## Testing Strategy

### Smoke Tests to Add
1. FiLMLayer instantiation with various d_features
2. Forward pass with known conditions
3. Forward pass with unknown conditions (all flags False)
4. Forward pass with mixed known/unknown conditions
5. Gradient flow through FiLM to condition encoder
6. Identity transform verification (when conditions unknown)

### Integration Tests to Add
1. Full pipeline with all conditions known
2. Full pipeline with 50% conditions known
3. Full pipeline with 0% conditions known
4. Batch with mixed condition availability
5. Compare predictions: with vs without FiLM

---

## Architecture Overview

### Current Data Flow
```
CSV file (pH, temp) 
    ↓
Dataset (reads, detects NaN)
    ↓
KICMSample (ph, temp, ph_known, temp_known)
    ↓
Collator (batches)
    ↓
Batch dict [B] tensors
    ↓
[LOST HERE - NOT PASSED TO MODEL YET!] ❌
```

### Target Data Flow (Post-Implementation)
```
CSV file (pH, temp)
    ↓
Dataset (reads, detects NaN)
    ↓
Collator (batches with flags)
    ↓
Training step (extracts conditions)
    ↓
Model.forward(sequence_tokens, padding_mask, smiles_list, 
              ph, temp, ph_known, temp_known)
    ↓
ESM3 encoder → MoLFormer encoder → CrossAttention → ctx [B, 512]
    ↓
FiLM layer (modulates ctx based on pH/temp + flags)
    ↓
MLPHead (regression)
    ↓
Predictions [B]
```

---

## Reference Materials

### FiLM Background
- Paper: "FiLM: Visual Reasoning with a General Conditioning Layer"
- Concept: Feature-wise Linear Modulation
- Formula: output = (gamma * input) + beta
- Gamma/beta are learned functions of conditioning variables

### Missing Data Strategies
1. **Masking** (recommended): Identity transform when unknown
2. **Dropout**: Randomly ignore conditions during training
3. **Imputation**: Train separate model to predict missing values

### PyTorch Lightning
- See src/models/kicm_module.py for training_step patterns
- Batch dict handling examples in collator.py

---

## Questions to Ask Yourself

When reading the code:
1. Where should FiLM be inserted? (Answer: after cross_attn, before mlp_head)
2. How do we handle unknown conditions? (Answer: identity transform with flags)
3. What if test data has no conditions? (Answer: FiLM returns identity, model still works)
4. Why is masking strategy better than filtering? (Answer: uses all data, naturally robust)
5. How do we verify gradients flow correctly? (Answer: check requires_grad on FiLM params)

---

## Next Steps for Implementation

1. **Read** EXPLORATION_SUMMARY.txt (this session)
2. **Review** key code locations from the reports
3. **Fix** the 2 config bugs (Phase 1)
4. **Test** with smoke_test.py
5. **Implement** FiLM module (Phase 2)
6. **Wire** into model and training (Phases 3-4)
7. **Configure** FiLM settings (Phase 5)
8. **Test** extensively (Phase 6)

---

## Contact & Notes

- **Explored by:** Claude Code (READ-ONLY exploration)
- **Date:** 2026-04-07
- **Files modified:** None (read-only mode)
- **Recommendations:** See EXPLORATION_SUMMARY.txt for implementation checklist

---

**Last Updated:** 2026-04-10 17:42 UTC  
**Files:** 
- DATA_PIPELINE_EXPLORATION.md (23 KB)
- EXPLORATION_SUMMARY.txt (15 KB)
- EXPLORATION_INDEX.md (this file)
