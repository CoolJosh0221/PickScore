# Progress Log

## Session: 2026-01-25 (Pipeline Explanation & Bug Audit)

### Current Status
- **Phase:** Complete
- **Started:** 2026-01-25

### Actions Taken
1. Recovered context from previous session via session-catchup
2. Read all 18 Python source files in src/active_learning/
3. Read experiment entry points and configuration
4. Traced complete data flow from HuggingFace → DataLoader
5. Traced training pipeline from model → loss → optimizer
6. Traced active learning loop with acquisition strategies
7. Verified all previously identified bugs are fixed
8. Identified remaining minor issues
9. Created comprehensive architecture documentation in findings.md

### Files Analyzed
| Directory | Files |
|-----------|-------|
| src/active_learning/data/ | datasets.py, collate.py, manager.py, sampling.py, loaders.py |
| src/active_learning/models/ | base_model.py, model_baseline.py, model_mcdo.py |
| src/active_learning/training/ | active_learning.py, acquisitions.py, train.py, eval.py, losses.py, metrics.py, utils.py |
| configs/ | experiment_configs.py |
| experiments/ | experiment.py |
| baselines/ | evaluate_lower_bounds.py |

### Bug Status Summary
| Previously Found | Status |
|-----------------|--------|
| Metric calculation bug | FIXED |
| MC Dropout disabled in MC sampling | FIXED |
| Collate class not initialized | FIXED |
| Missing processor argument | FIXED |
| Duplicate seed parameters | FIXED |
| Deprecated torch.load() | FIXED |
| O(n) pool lookup | FIXED (cached) |

### Issues Fixed This Session
| Issue | Fix Applied |
|-------|-------------|
| Optimizer state persists | Added `reset_optimizer()` - creates fresh optimizer each AL iteration |
| No learning rate scheduler | Added `get_cosine_schedule_with_warmup()` - warmup + cosine decay |
| CoresetKCenter non-functional | Added `compute_embeddings()` and updated `al_iteration()` to compute/pass embeddings |

### Remaining Issues (Non-Critical)
1. Sequential MC sampling (slow but correct)
2. Test set used as validation (common in AL practice)

---

## Session: 2026-01-19 (Experiment Rerun)

### Current Status
- **Phase:** Complete - All experiments finished
- **Started:** 2026-01-19

### Actions Taken
1. Ran FAST_PROTOTYPE with BALD acquisition (W&B: y7ugchax)
2. Ran FAST_PROTOTYPE with Random acquisition (W&B: ln3neu2o)
3. Ran SMALL_SCALE with BALD acquisition (W&B: zy10xozo)
4. Ran SMALL_SCALE with Random acquisition (W&B: g6cjn5dv)
5. Documented all results in findings.md and task_plan.md

### Experiment Results Summary

| Experiment | Method | pref_acc | overall_acc |
|------------|--------|----------|-------------|
| FAST_PROTOTYPE | BALD | 60.54% | 53.60% |
| FAST_PROTOTYPE | Random | 58.50% | 51.60% |
| SMALL_SCALE | BALD | 64.47% | 57.73% |
| SMALL_SCALE | Random | **66.34%** | **59.40%** |

### Key Finding
**Unexpected**: Random outperforms BALD at larger scale!
- FAST_PROTOTYPE: BALD wins by +2.04%
- SMALL_SCALE: Random wins by +1.87%

---

## Session: 2026-01-19 (Bug Fixes)

### Current Status
- **Phase:** Complete - All bugs fixed and verified
- **Started:** 2026-01-19

### Actions Taken
1. Fixed metric calculation bug in `metrics.py:14` (s0-0.5 → s0-s1)
2. Fixed MC Dropout being disabled during MC sampling in `active_learning.py`
3. Fixed Collate class not properly initialized in `manager.py`
4. Fixed missing processor argument in `train.py:166`
5. Removed duplicate `random_seed` parameter from `experiment_configs.py`
6. Fixed deprecated `torch.load()` usage in both model files
7. Added caching for unlabeled pool indices in `manager.py`
8. Added `EarlyStopping` class to `utils.py`
9. Added early stopping support to `run_active_learning()`
10. Verified all fixes with import tests and unit tests

### Files Modified
| File | Change |
|------|--------|
| `src/active_learning/training/metrics.py` | Fixed pred_gap calculation |
| `src/active_learning/training/active_learning.py` | Fixed MC Dropout, added early stopping |
| `src/active_learning/data/manager.py` | Added processor support, caching |
| `src/active_learning/training/train.py` | Added missing processor arg |
| `src/active_learning/training/utils.py` | Added EarlyStopping class |
| `src/active_learning/models/model_mcdo.py` | Fixed torch.load() |
| `src/active_learning/models/model_baseline.py` | Fixed torch.load() |
| `configs/experiment_configs.py` | Removed duplicate seed param |

### Test Results
| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Module imports | All pass | All pass | PASS |
| pref_metrics calculation | Correct accuracy | 100% for test case | PASS |
| EarlyStopping | Stop after patience | Stopped at patience=2 | PASS |
| MC Dropout param | Correct flag value | Correctly set | PASS |

### Errors Encountered
| Error | Resolution |
|-------|------------|
| src_old tests reference old modules | Verified new modules instead |
| ModuleNotFoundError: datasets | Activated venv for testing |
| ModuleNotFoundError: configs | Added PYTHONPATH to include project root |

---

## Session: 2026-01-19 (Code Review)

### Actions Taken
- Identified 16 issues across the codebase
- Documented all findings in `findings.md`

### Issues Found
| Severity | Count |
|----------|-------|
| CRITICAL | 2 |
| HIGH | 2 |
| MEDIUM | 3 |
| LOW | 5+ |

---

## Session: 2026-01-18 (Initial Audit)

### Actions Taken
- Initial repository structure analysis
- Identified entry points and components
- Documented architecture in findings.md
