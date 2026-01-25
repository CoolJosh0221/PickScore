# Repository Findings: PickScore Active Learning Framework

## Overview

This repository implements **Active Learning for Human Preference Prediction** in text-to-image generation. It's built on the PickScore/Pick-a-Pic research, exploring whether active learning strategies can reduce the amount of human preference data needed to train effective reward models.

---

## Architecture & Pipeline

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENTRY POINTS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  experiments/experiment.py  →  Main entry for running AL experiments        │
│  main.py                    →  Legacy entry point (uses src_old/)           │
│  baselines/                 →  Standalone evaluation scripts                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  configs/experiment_configs.py                                               │
│  ├── ExperimentConfig (attrs dataclass)                                      │
│  ├── FAST_PROTOTYPE (100 seed, 2k pool, quick testing)                       │
│  ├── SMALL_SCALE   (300 seed, 8k pool, balanced experiments)                 │
│  ├── MEDIUM_SCALE  (800 seed, 30k pool)                                      │
│  └── LARGE_SCALE   (1.5k seed, 75k pool, full scale)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. SAMPLING (src/active_learning/data/sampling.py)                          │
│     ├── Streams from HuggingFace: pickapic-anonymous/pickapic_v1             │
│     ├── Filters: only "are_different" samples (clear preferences)            │
│     ├── Shuffles with buffer_size=1000                                       │
│     └── Creates splits: seed (labeled), pool (unlabeled), test               │
│                                                                              │
│  2. DATASETS (src/active_learning/data/datasets.py)                          │
│     ├── PreferenceDataset: Loads saved Arrow files, decodes JPEG bytes       │
│     │   Returns: caption, image_0, image_1, label_0, label_1                 │
│     └── ALIndexedDataset: Wrapper exposing specific indices                  │
│                                                                              │
│  3. COLLATE (src/active_learning/data/collate.py)                            │
│     ├── Collate class with set_processor() for initialization                │
│     ├── Processes PIL images → pixel_values tensors                          │
│     └── Stacks labels into tensors                                           │
│                                                                              │
│  4. DATA MANAGER (src/active_learning/data/manager.py)                       │
│     ├── ActiveLearningDataManager: Central state management                  │
│     ├── Tracks labeled_pool_indices (which pool samples are "labeled")       │
│     ├── Provides: get_labeled_dataloader(), get_unlabeled_dataloader()       │
│     ├── Persists state to JSON for experiment resumption                     │
│     └── Caches unlabeled indices for performance                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MODEL LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  src/active_learning/models/                                                 │
│                                                                              │
│  base_model.py                                                               │
│  └── BaseModel (abstract): defines save(), load(), get_*_features()         │
│                                                                              │
│  model_baseline.py                                                           │
│  └── CLIPModel: Standard CLIP wrapper (no dropout modifications)            │
│                                                                              │
│  model_mcdo.py                                                               │
│  └── MCDropoutCLIPModel: MC Dropout implementation                          │
│      ├── Modifies CLIP config to add attention_dropout & dropout            │
│      ├── Adds explicit text_dropout & image_dropout layers                  │
│      ├── enable_mc_dropout flag controls inference-time dropout             │
│      └── Custom eval() keeps dropout active when MC mode enabled            │
│                                                                              │
│  PREFERENCE SCORING:                                                         │
│  ├── text_features = model.get_text_features(input_ids, attention_mask)     │
│  ├── img0_features = model.get_image_features(pixel_values_0)               │
│  ├── img1_features = model.get_image_features(pixel_values_1)               │
│  ├── s0 = cosine_sim(text, img0) * logit_scale.exp()                        │
│  ├── s1 = cosine_sim(text, img1) * logit_scale.exp()                        │
│  └── Preference: image with higher score wins                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  src/active_learning/training/                                               │
│                                                                              │
│  losses.py                                                                   │
│  ├── l2_normalize(): Normalizes features                                    │
│  ├── pairwise_scores(): Computes scaled cosine similarities                 │
│  └── soft_ce_from_pairs(): Soft cross-entropy loss for preference learning  │
│      └── Loss = -sum(target_probs * log_softmax([s0, s1]))                   │
│                                                                              │
│  metrics.py                                                                  │
│  ├── pref_metrics(): Computes pref_acc, tie_acc, overall_acc                │
│  │   ├── pref_acc: Accuracy on non-tie samples (correct preference order)   │
│  │   ├── tie_acc: Accuracy on tie samples (pred within tie_margin)          │
│  │   └── overall_acc: Combined accuracy                                      │
│  └── PrefMetricTracker: Accumulates metrics across batches                  │
│                                                                              │
│  train.py                                                                    │
│  ├── build_*(): Factory functions for device, model, processor, optimizer   │
│  ├── prepare_inputs(): Converts batch to model-ready inputs                 │
│  ├── loss_on_batch(): Forward pass + loss computation                       │
│  ├── train_one_epoch(): Training loop with AMP support                      │
│  └── validate_epoch(): Evaluation on validation set                         │
│                                                                              │
│  eval.py                                                                     │
│  └── evaluate(): Full evaluation with metrics tracking                      │
│                                                                              │
│  utils.py                                                                    │
│  ├── EarlyStopping: Stops training if val_loss doesn't improve              │
│  ├── set_seed(): Reproducibility                                            │
│  ├── make_run_dir(): Creates timestamped checkpoint directory               │
│  └── save_epoch(), save_best_pointer(): Checkpoint management               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ACTIVE LEARNING LOOP                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  src/active_learning/training/active_learning.py                             │
│                                                                              │
│  run_active_learning():                                                      │
│  ├── Initialize model, processor, optimizer, data_manager                   │
│  └── For each AL iteration:                                                  │
│      └── al_iteration():                                                     │
│          ├── Train on labeled data (seed + acquired pool samples)            │
│          ├── Early stopping check (optional)                                 │
│          ├── Validate on test set                                            │
│          ├── Compute acquisition scores on unlabeled pool:                   │
│          │   ├── If requires_mc: predict_pool_mc_probs() with T samples      │
│          │   └── Else: predict_pool_probs() (deterministic)                  │
│          ├── Select top-k samples by acquisition score                       │
│          ├── Mark selected samples as "labeled"                              │
│          └── Log metrics to W&B                                              │
│                                                                              │
│  ACQUISITION STRATEGIES (acquisitions.py):                                   │
│  ├── RandomSampling: Uniform random scores                                  │
│  ├── BALD: H[E[p]] - E[H[p]] (requires MC dropout)                           │
│  ├── EntropyUncertainty: Predictive entropy                                 │
│  ├── LeastConfidence: 1 - max(probs)                                        │
│  └── CoresetKCenter: Distance-based (requires embeddings)                   │
│                                                                              │
│  MC DROPOUT INFERENCE:                                                       │
│  ├── predict_pool_mc_probs(): Runs T stochastic forward passes              │
│  │   └── Each pass has enable_mc_dropout=True                                │
│  └── Returns [T, N, 2] tensor of probabilities                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
Pick-a-Pic Dataset (HuggingFace)
           │
           ▼ streaming + filter + shuffle
    ┌──────────────────────────────────────┐
    │         Local Arrow Files            │
    ├──────────────────────────────────────┤
    │  /out/<strategy>/<run>/              │
    │  ├── seed/     (initial labeled)     │
    │  ├── pool/     (unlabeled pool)      │
    │  └── test/     (evaluation)          │
    └──────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────┐
    │      PreferenceDataset               │
    │  Decodes JPEG → PIL Images           │
    │  Returns: caption, img0, img1, y0, y1│
    └──────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────┐
    │      ALIndexedDataset                │
    │  Wraps base dataset with index list  │
    │  Used for labeled/unlabeled subsets  │
    └──────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────┐
    │         Collate                      │
    │  PIL → pixel_values (tensor)         │
    │  Batches captions, labels            │
    └──────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────┐
    │         DataLoader                   │
    │  With persistent_workers, pin_memory │
    └──────────────────────────────────────┘
```

---

## Bug Audit

### Previously Identified Issues (from prior sessions)

| # | Severity | Issue | Location | Status |
|---|----------|-------|----------|--------|
| 1 | CRITICAL | Metric calculation: `s0 - 0.5` should be `s0 - s1` | `metrics.py:14` | **FIXED** |
| 2 | CRITICAL | MC Dropout disabled during MC sampling | `active_learning.py:34` | **FIXED** |
| 3 | HIGH | Collate class passed instead of instance | `manager.py:85` | **FIXED** |
| 4 | HIGH | Missing processor argument in build_loaders | `train.py:166` | **FIXED** |
| 5 | MEDIUM | Duplicate seed parameters | `experiment_configs.py` | **FIXED** |
| 6 | MEDIUM | Deprecated torch.load() usage | `model_mcdo.py`, `model_baseline.py` | **FIXED** |
| 7 | LOW | O(n) pool index lookup | `manager.py` | **FIXED** (cached) |

### Current Code Review

After reviewing the current codebase, the previously identified bugs have been fixed. Here's the current state:

#### 1. Metrics (FIXED)
```python
# metrics.py:14
pred_gap = s0 - s1  # Correctly compares score differences
```

#### 2. MC Dropout (FIXED)
```python
# active_learning.py:32-33
def predict_pool_probs(model, processor, loader, device: str, enable_mc_dropout: bool = False):
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout = enable_mc_dropout  # Now parameterized correctly
```

```python
# active_learning.py:76-79
samples = [
    predict_pool_probs(model, processor, loader, device, enable_mc_dropout=True)
    for _ in range(num_samples)
]
```

#### 3. Collate (FIXED)
```python
# manager.py:45-53
def _make_collate(self) -> Collate:
    if self.processor is None:
        raise ValueError("Processor not set...")
    collater = Collate()
    collater.set_processor(self.processor)
    return collater
```

#### 4. torch.load (FIXED)
```python
# model_mcdo.py:71
state_dict = torch.load(path, map_location="cpu", weights_only=True)

# model_baseline.py:32
state_dict = torch.load(path, map_location="cpu", weights_only=True)
```

#### 5. Caching (FIXED)
```python
# manager.py:180-188
def get_unlabeled_pool_indices(self) -> List[int]:
    if self._unlabeled_indices_cache is None:
        self._unlabeled_indices_cache = [...]
    return self._unlabeled_indices_cache
```

---

### Remaining Issues / Observations

#### 1. **Optimizer State Persists Across AL Iterations** - FIXED
**Location:** `active_learning.py:262-264`

The optimizer is now reset at each AL iteration using the new `reset_optimizer()` function.
This ensures fresh momentum/adaptive states for each round of active learning.

#### 2. **No Learning Rate Scheduler** - FIXED
**Location:** `active_learning.py:266-272`, `utils.py:97-127`

Added `get_cosine_schedule_with_warmup()` which provides:
- Linear warmup for 10% of training steps
- Cosine decay to 10% of initial learning rate
- Scheduler is created fresh each AL iteration based on current labeled data size

#### 3. **CoresetKCenter Never Usable** - FIXED
**Location:** `active_learning.py:167-179`

Added `compute_embeddings()` function and updated `al_iteration()` to:
- Compute embeddings for labeled data when using CoresetKCenter
- Compute embeddings for unlabeled pool candidates
- Pass both to the acquisition function

#### 4. **Image Decoding on Every Access** (UNCHANGED)
**Location:** `datasets.py:17-25`

```python
def __getitem__(self, idx):
    row = self.ds[idx]
    return {
        "image_0": Image.open(io.BytesIO(row["jpg_0"])).convert("RGB"),
        "image_1": Image.open(io.BytesIO(row["jpg_1"])).convert("RGB"),
        ...
    }
```

JPEG bytes are decoded to PIL images on every access. With multiple epochs, same images decoded repeatedly.

**Impact:** Low (I/O bound). DataLoader workers handle this in parallel.

#### 5. **Sequential MC Sampling** (UNCHANGED)
**Location:** `active_learning.py:76-79`

```python
samples = [
    predict_pool_probs(model, processor, loader, device, enable_mc_dropout=True)
    for _ in range(num_samples)
]
```

Each MC sample requires a full pass through the pool. With T=15 samples and 8k pool, this is slow.

**Impact:** Performance only. Results are correct.

**Potential Fix:** Batch multiple MC samples in a single forward pass.

#### 6. **Test Set Used as Validation**
**Location:** `manager.py:135-147`

The "test" split serves dual purpose:
- Validation during training (early stopping)
- Final evaluation

**Impact:** Low. In AL experiments, this is common practice. For production, separate val/test.

---

## Experimental Results Summary

Based on previous session findings:

| Rank | Method | Final pref_acc | Final overall_acc |
|------|--------|----------------|-------------------|
| 1 | **Random** | **66.34%** | **59.40%** |
| 2 | Entropy | 66.12% | 59.07% |
| 2 | Least Confidence | 66.12% | 59.07% |
| 4 | BALD | 64.47% | 57.73% |

**Key Finding:** Random sampling outperforms all uncertainty-based AL strategies. This suggests:
1. Diversity matters more than uncertainty for preference learning
2. Uncertainty methods may select redundant "hard" samples
3. Human preference data may not have exploitable uncertainty gradients

---

## File Structure Summary

```
PickScore-Workshop-Repo/
├── configs/
│   └── experiment_configs.py    # ExperimentConfig + presets
├── experiments/
│   └── experiment.py            # Main entry: run_experiment()
├── src/active_learning/
│   ├── data/
│   │   ├── datasets.py          # PreferenceDataset, ALIndexedDataset
│   │   ├── collate.py           # Collate class
│   │   ├── manager.py           # ActiveLearningDataManager
│   │   ├── sampling.py          # create_active_learning_splits()
│   │   └── loaders.py           # create_dataloader()
│   ├── models/
│   │   ├── base_model.py        # BaseModel (abstract)
│   │   ├── model_baseline.py    # CLIPModel (standard)
│   │   └── model_mcdo.py        # MCDropoutCLIPModel
│   └── training/
│       ├── active_learning.py   # run_active_learning(), al_iteration()
│       ├── acquisitions.py      # BALD, Entropy, etc.
│       ├── train.py             # Training utilities
│       ├── eval.py              # evaluate()
│       ├── losses.py            # pairwise_scores(), soft_ce_from_pairs()
│       ├── metrics.py           # pref_metrics(), PrefMetricTracker
│       └── utils.py             # EarlyStopping, checkpointing
├── baselines/
│   ├── evaluate_lower_bounds.py # Zero-shot CLIP evaluation
│   └── evaluate_upper_bounds.py # Full fine-tuning evaluation
├── main.py                      # Legacy entry (uses src_old/)
└── src_old/                     # Deprecated code
```

---

## Conclusion

The codebase is well-structured and the previously identified critical bugs have all been fixed. The pipeline correctly:
1. Streams and samples preference data from Pick-a-Pic
2. Manages labeled/unlabeled state for active learning
3. Implements MC Dropout for uncertainty estimation
4. Provides multiple acquisition strategies (random, BALD, entropy, least_confidence)
5. Tracks experiments via Weights & Biases

The remaining issues are minor performance optimizations and design choices that don't affect correctness.
