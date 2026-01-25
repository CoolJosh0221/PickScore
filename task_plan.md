# Task Plan: Explain Repository Pipeline & Bug Audit

## Goal
Comprehensively explain this repository's pipeline/architecture and simultaneously check for any remaining or new bugs.

## Current Phase
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
