# Task Plan: Calibration Analysis for Active Learning

## Goal
Test the causal chain: **Miscalibration → Bad Uncertainty → AL Fails**

Does fixing calibration (via temperature scaling) improve uncertainty estimates and rescue active learning?

## Current Phase
**ALL PHASES COMPLETE**

---

## Phases

### Phase 1: Baseline Calibration Metrics - `complete`
**Objective:** Quantify how miscalibrated the current model is.

| Task | Status |
|------|--------|
| Load trained model and test set | complete |
| Compute ECE (Expected Calibration Error, 15 bins) | complete |
| Compute MCE (Maximum Calibration Error) | complete |
| Generate reliability diagram (confidence vs accuracy per bin) | complete |
| Generate confidence histogram | complete |

**Input:** Trained model, validation/test set predictions (logits + labels)
**Output:** ECE, MCE, reliability diagram, confidence histogram

**Results (MC Dropout, T=15):**
| Metric | Value | vs Deterministic |
|--------|-------|------------------|
| ECE | **17.67%** | ↓ from 24.18% |
| MCE | **27.45%** | ↓ from 30.87% |
| Overall Accuracy | 64.67% | ~same |
| Mean Confidence | 82.34% | ↓ from 89.25% |

**Key Finding:** MC Dropout helps slightly (ECE 24%→18%) but model is still severely **overconfident**. Mean confidence 82% vs 65% accuracy. Temperature scaling needed.

---

### Phase 2: Baseline Uncertainty-Error Correlation - `complete`
**Objective:** Measure if uncertainty predicts error (Gleave's diagnostic).

| Task | Status |
|------|--------|
| Run MC Dropout predictions (15 passes) | complete |
| Compute BALD uncertainty | complete |
| Compute Spearman correlation: uncertainty vs error (0/1) | complete |
| Generate scatter plot with trend line | complete |

**Results:**
| Metric | Spearman ρ | p-value |
|--------|------------|---------|
| BALD | **0.215** | 4.09e-17 |
| Entropy | 0.231 | 1.39e-19 |
| Least Confidence | 0.231 | 1.39e-19 |

**Interpretation:** Moderate positive correlation - uncertainty DOES predict errors, but weakly.

---

### Phase 3: Temperature Scaling - `complete`
**Objective:** Find optimal temperature T to calibrate predictions.

| Task | Status |
|------|--------|
| Extract validation set logits + labels | complete |
| Optimize T to minimize NLL on validation set | complete |
| Apply calibration: `p = softmax(logit / T)` | complete |
| Report optimal T and validation NLL before/after | complete |

**Results:**
| Metric | Value |
|--------|-------|
| **Optimal T** | 5.78 |
| NLL before | 1.163 |
| NLL after | 0.618 |
| **NLL reduction** | 46.9% |

**Interpretation:** High T value needed because logit_scale=100 makes outputs too peaked.

---

### Phase 4: Post-Calibration Evaluation - `complete`
**Objective:** Check if calibration improved.

| Task | Status |
|------|--------|
| Compute ECE/MCE after calibration | complete |
| Generate reliability diagram after calibration | complete |
| Compute uncertainty-error correlation after calibration | complete |
| Create comparison table: baseline vs calibrated | complete |

**Results:**
| Metric | Baseline | Calibrated | Change |
|--------|----------|------------|--------|
| **ECE** | 18.12% | **1.99%** | -89% |
| MCE | 29.11% | 7.36% | -75% |
| Mean Confidence | 82.27% | 64.74% | ↓ |
| BALD-Error ρ | 0.215 | **0.104** | -51% |

**Key Finding:** Calibration fixes ECE but WORSENS uncertainty-error correlation!

---

### Phase 5: Acquisition Score Comparison - `complete`
**Objective:** Analyze how calibration affects acquisition scores.

| Task | Status |
|------|--------|
| Compare baseline vs calibrated acquisition scores | complete |
| Compute rank correlation between baseline/calibrated | complete |
| Analyze top-k sample overlap | complete |

**Results:**
| Acquisition | Rank Correlation (baseline vs calibrated) |
|-------------|-------------------------------------------|
| BALD | 0.651 |
| Entropy | 0.985 |
| Least Confidence | 0.985 |

**Top-100 BALD sample overlap:** 51% (calibration changes which samples are "most uncertain")

**Conclusion:** Calibration significantly changes BALD rankings but NOT Entropy/LC rankings.

---

## Key Questions to Answer

| # | Question | Threshold | Status |
|---|----------|-----------|--------|
| 1 | Is the model miscalibrated? | ECE > 5%? | **YES - ECE=18.12%** |
| 2 | Does temperature scaling reduce ECE? | | **YES - 18%→2% (89% reduction)** |
| 3 | Does calibration improve uncertainty-error correlation? | | **NO - ρ drops 0.215→0.104** |
| 4 | Does calibrated AL outperform baseline AL? | | **UNLIKELY** (see Q3) |
| 5 | Does calibrated BALD beat random? | **Key question** | **UNLIKELY** - calibration hurts uncertainty signal |

---

## Implementation Notes

- Temperature scaling is post-hoc — don't retrain model, just scale logits
- For MC Dropout with temperature: apply T to each forward pass before computing variance
- Handle empty bins in ECE gracefully

---
---

# Previous Task Plan: Pipeline Explanation & Bug Audit (COMPLETE)

## Goal
Comprehensively explain this repository's pipeline/architecture and simultaneously check for any remaining or new bugs.

## Status
**COMPLETE**

## Phases

### Phase 1: Codebase Exploration - `complete`
| Task | Status |
|------|--------|
| Map repository structure | complete |
| Identify entry points | complete |
| Trace data flow | complete |
| Trace training flow | complete |
| Trace active learning flow | complete |

### Phase 2: Pipeline Documentation - `complete`
| Task | Status |
|------|--------|
| Document data pipeline | complete |
| Document model architecture | complete |
| Document training loop | complete |
| Document active learning loop | complete |
| Create architecture diagram | complete |

### Phase 3: Bug Audit - `complete`
| Task | Status |
|------|--------|
| Review previous bug findings | complete |
| Check if fixes are in place | complete |
| Identify new/remaining bugs | complete |
| Document all findings | complete |

### Phase 4: Summary - `complete`
| Task | Status |
|------|--------|
| Create comprehensive explanation | complete |
| Summarize bugs found | complete |
| Update findings.md | complete |

---

## Summary

### Files Read
- `experiments/experiment.py` - Main entry point
- `configs/experiment_configs.py` - Configuration presets
- `src/active_learning/data/` - All data handling modules
- `src/active_learning/models/` - Model implementations
- `src/active_learning/training/` - Training and AL loop
- `baselines/` - Evaluation scripts

### Key Findings

1. **Pipeline is well-structured**: Clear separation between data, models, and training.

2. **Previously identified bugs are FIXED**:
   - Metric calculation (`s0 - s1` vs `s0 - 0.5`)
   - MC Dropout enabling during inference
   - Collate function initialization
   - torch.load() security
   - Index caching for performance

3. **Additional fixes applied**:
   - Optimizer now resets at each AL iteration (fresh momentum/adaptive states)
   - Added cosine learning rate scheduler with warmup
   - CoresetKCenter now functional (computes embeddings automatically)

4. **Remaining minor issues** (by design, not bugs):
   - Sequential MC sampling (performance, not correctness)
   - Test set doubles as validation set (common in AL)

4. **Experimental results**: Random sampling outperforms uncertainty-based methods (BALD, Entropy, Least Confidence).

Full documentation available in `findings.md`.
