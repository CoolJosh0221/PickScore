# Active Learning Pipeline - Complete Guide

## Overview

The active learning pipeline has been completely reorganized and refactored for better maintainability, clarity, and ease of use. This document provides a comprehensive guide to the new structure.

## Table of Contents

1. [What Changed](#what-changed)
2. [New Architecture](#new-architecture)
3. [Quick Start](#quick-start)
4. [Detailed Usage](#detailed-usage)
5. [Migration Guide](#migration-guide)
6. [Examples](#examples)

## What Changed

### Before (Old Structure)
```
PickScore/
├── inference.py                    # Mixed inference and uncertainty code
├── inference_score_only.py         # Duplicate inference code
├── evaluation/
│   └── generate_data_for_evaluation.py  # Hard-coded evaluation
├── test.py                         # Ad-hoc testing
└── trainer/
    └── models/
        └── clip_model.py          # Mixed concerns, commented code
```

**Problems:**
- Code scattered across multiple files
- Duplication and inconsistency
- Hard to extend with new acquisition functions
- No clear separation of concerns
- Mixed evaluation and active learning logic

### After (New Structure)
```
PickScore/
├── active_learning/                # New organized module
│   ├── __init__.py                # Clean exports
│   ├── acquisition_functions.py   # All acquisition functions
│   ├── uncertainty.py             # Uncertainty estimation
│   ├── sample_selector.py         # Sample selection strategies
│   ├── config.py                  # Configuration classes
│   ├── pipeline.py                # Main orchestrator
│   └── README.md                  # Module documentation
├── run_active_learning.py         # Main CLI script
├── run_evaluation.py              # Evaluation script
├── test_active_learning.py        # Comprehensive tests
└── trainer/
    └── models/
        └── clip_model.py          # Cleaned up, focused on model
```

**Benefits:**
- Clear separation of concerns
- Easy to extend (add new acquisition functions)
- Reusable components
- Well-documented
- Type hints and error handling
- Comprehensive testing

## New Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Active Learning Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────┐         ┌──────────────────┐           │
│  │  CLIP Model    │────────▶│  Uncertainty     │           │
│  │  (MC Dropout)  │         │  Estimator       │           │
│  └────────────────┘         └──────────────────┘           │
│         │                            │                       │
│         │                            ▼                       │
│         │                   ┌──────────────────┐           │
│         │                   │  Acquisition     │           │
│         │                   │  Functions       │           │
│         │                   └──────────────────┘           │
│         │                            │                       │
│         ▼                            ▼                       │
│  ┌────────────────┐         ┌──────────────────┐           │
│  │  Processor     │         │  Sample          │           │
│  │  (CLIP)        │         │  Selector        │           │
│  └────────────────┘         └──────────────────┘           │
│         │                            │                       │
│         └──────────┬─────────────────┘                      │
│                    ▼                                         │
│           ┌──────────────────┐                              │
│           │  Pipeline        │                              │
│           │  Orchestrator    │                              │
│           └──────────────────┘                              │
│                    │                                         │
│                    ▼                                         │
│           ┌──────────────────┐                              │
│           │  Results &       │                              │
│           │  Visualization   │                              │
│           └──────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

#### 1. **acquisition_functions.py**
Provides various uncertainty quantification methods:

- **Base Class**: `AcquisitionFunction` (abstract)
- **Implementations**:
  - `VarianceAcquisition`: σ²
  - `StdDevAcquisition`: σ (recommended)
  - `EntropyAcquisition`: H(p)
  - `BALDAcquisition`: I(y; θ|x)
  - `CoefficientOfVariationAcquisition`: σ/μ
  - `MADAcquisition`: Median absolute deviation
  - `IQRAcquisition`: Interquartile range
  - `ConfidenceIntervalAcquisition`: CI width

**Key Features:**
- Easy to add new acquisition functions
- Factory method for getting functions by name
- Unified interface
- Comprehensive uncertainty metrics

#### 2. **uncertainty.py**
Handles MC Dropout inference and uncertainty estimation:

- **`UncertaintyEstimator`**: Main class for uncertainty estimation
  - `estimate_score_uncertainty()`: Single image-prompt
  - `estimate_preference_uncertainty()`: Multiple images (preference)
  - `batch_estimate_uncertainty()`: Batch processing
  - `enable_mc_dropout()` / `disable_mc_dropout()`: Control dropout

**Key Features:**
- Automatic MC dropout management
- Flexible input handling
- Progress bars for long operations
- Comprehensive error handling

#### 3. **sample_selector.py**
Implements various selection strategies:

- **`SampleSelector`**: Main selector class
  - `select_top_k()`: Select k most uncertain
  - `select_above_threshold()`: Threshold-based
  - `select_proportional()`: Probability-based sampling
  - `select_diverse()`: Balance uncertainty and diversity
  - `get_uncertainty_statistics()`: Compute statistics

**Key Features:**
- Multiple selection strategies
- Diversity-aware selection
- Statistical analysis
- Flexible input types (list, numpy, torch)

#### 4. **config.py**
Configuration dataclasses:

- **`ActiveLearningConfig`**: Main configuration
  - Model settings
  - MC Dropout settings
  - Selection strategy settings
  - Data and device settings
  - Output settings

- **`EvaluationConfig`**: Evaluation-specific configuration
  - Test prompts by category
  - Image generation settings
  - Model settings

**Key Features:**
- Type-safe configuration
- Validation in `__post_init__`
- Easy to extend
- Clear defaults

#### 5. **pipeline.py**
Main orchestrator:

- **`ActiveLearningPipeline`**: Complete workflow
  - `run()`: Full pipeline execution
  - `estimate_dataset_uncertainty()`: Batch uncertainty estimation
  - `select_samples()`: Apply selection strategy
  - `evaluate_on_test_prompts()`: Evaluation mode
  - `_save_results()`: Save outputs

**Key Features:**
- End-to-end workflow
- Progress tracking
- Automatic result saving
- Evaluation support

## Quick Start

### 1. Basic Usage (Python API)

```python
from PIL import Image
from transformers import CLIPProcessor
from trainer.models.clip_model import CLIPModel, ClipModelConfig
from active_learning import ActiveLearningConfig, ActiveLearningPipeline

# Load model
model_config = ClipModelConfig(
    pretrained_model_name_or_path="openai/clip-vit-base-patch32",
    dropout_rate=0.1,
)
model = CLIPModel(model_config).to("cuda")

# Load processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Configure pipeline
config = ActiveLearningConfig(
    n_mc_samples=20,
    acquisition_function="std_dev",
    selection_strategy="top_k",
    num_samples_to_select=100,
)

# Create and run pipeline
pipeline = ActiveLearningPipeline(model, processor, config)
results = pipeline.run(prompts, images, image_ids)
```

### 2. Command Line Usage

**Active Learning Mode:**
```bash
python run_active_learning.py \
    --model-name openai/clip-vit-base-patch32 \
    --checkpoint outputs/checkpoint-final/pytorch_model.bin \
    --data-file samples.json \
    --n-samples 20 \
    --acquisition-function std_dev \
    --num-select 100 \
    --output-dir results
```

**Evaluation Mode:**
```bash
python run_active_learning.py \
    --eval-mode \
    --checkpoint outputs/checkpoint-final/pytorch_model.bin \
    --test-images-dir test_images/real_images \
    --n-samples 20 \
    --output-dir eval_results
```

**Standalone Evaluation:**
```bash
python run_evaluation.py \
    --checkpoint outputs/checkpoint-final/pytorch_model.bin \
    --test-images-dir test_images/real_images \
    --output-file results.json \
    --n-samples 20
```

## Detailed Usage

### Using Individual Components

#### 1. Uncertainty Estimation Only

```python
from active_learning import UncertaintyEstimator

estimator = UncertaintyEstimator(
    model=model,
    acquisition_function="bald",
    n_samples=30,
    device="cuda"
)

# Single sample
result = estimator.estimate_score_uncertainty(
    prompt="a beautiful landscape",
    image=image,
    processor=processor,
    compute_all_metrics=True,
    show_progress=True,
)

print(f"Mean score: {result['mean']:.4f}")
print(f"Uncertainty (BALD): {result['bald']:.4f}")
```

#### 2. Sample Selection Only

```python
from active_learning import SampleSelector

selector = SampleSelector(strategy="top_k")

# You have uncertainties from somewhere
uncertainties = [0.1, 0.5, 0.3, 0.8, 0.2, ...]

# Select top 100
top_indices = selector.select_top_k(uncertainties, k=100)

# Or with diversity
diverse_indices = selector.select_diverse(
    uncertainties=uncertainties,
    features=image_features,  # Need feature vectors
    k=100,
    diversity_weight=0.5,
)

# Get statistics
stats = selector.get_uncertainty_statistics(uncertainties)
print(f"Mean: {stats['mean']:.4f}")
print(f"Std: {stats['std']:.4f}")
```

#### 3. Custom Acquisition Function

```python
from active_learning import AcquisitionFunction, UncertaintyEstimator
import torch

class RangeAcquisition(AcquisitionFunction):
    """Range (max - min) as uncertainty measure."""

    def __call__(self, samples: torch.Tensor, dim: int = 0) -> torch.Tensor:
        max_val = torch.max(samples, dim=dim)[0]
        min_val = torch.min(samples, dim=dim)[0]
        return max_val - min_val

    def name(self) -> str:
        return "range"

# Use it
estimator = UncertaintyEstimator(
    model=model,
    acquisition_function=RangeAcquisition(),
    n_samples=20,
)
```

### Data Format

For `run_active_learning.py --data-file`:

```json
{
  "samples": [
    {
      "id": "sample_001",
      "prompt": "a photo of a sunset over mountains",
      "image_path": "images/sunset.jpg"
    },
    {
      "id": "sample_002",
      "prompt": "a photo of a city at night",
      "image_path": "images/city.jpg"
    }
  ]
}
```

## Migration Guide

### From Old Code to New Code

#### Old: `inference.py`
```python
# Old way
mc_results = model.calc_probs_with_uncertainty(
    prompt=prompt,
    images=images,
    processor=processor,
    n_samples=30,
    device=device,
)
```

#### New: Using `UncertaintyEstimator`
```python
# New way
from active_learning import UncertaintyEstimator

estimator = UncertaintyEstimator(
    model=model,
    acquisition_function="std_dev",
    n_samples=30,
    device=device,
)

result = estimator.estimate_preference_uncertainty(
    prompt=prompt,
    images=images,
    processor=processor,
)
```

#### Old: `generate_data_for_evaluation.py`
```python
# Old way - hard-coded loops
for style in EvaluationConfig.prompts:
    for prompt_id in range(len(EvaluationConfig.prompts[style])):
        for i in range(EvaluationConfig.NUM_IMAGES):
            # Load image
            # Run inference
            # Collect results
```

#### New: Using `Pipeline`
```python
# New way - use the pipeline
from active_learning import ActiveLearningPipeline

pipeline = ActiveLearningPipeline(model, processor, config)
results = pipeline.evaluate_on_test_prompts(test_images_dir)
```

## Examples

### Example 1: Find Most Uncertain Samples

```python
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor
from trainer.models.clip_model import CLIPModel, ClipModelConfig
from active_learning import UncertaintyEstimator

# Setup
model_config = ClipModelConfig(pretrained_model_name_or_path="openai/clip-vit-base-patch32", dropout_rate=0.1)
model = CLIPModel(model_config).to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

estimator = UncertaintyEstimator(model, "std_dev", n_samples=20, device="cuda")

# Load your data
image_dir = Path("my_images")
prompts_file = "prompts.txt"

with open(prompts_file) as f:
    prompts = [line.strip() for line in f]

uncertainties = []
for img_path in image_dir.glob("*.jpg"):
    img = Image.open(img_path).convert("RGB")
    prompt = prompts[int(img_path.stem)]  # Assuming numbered images

    result = estimator.estimate_score_uncertainty(
        prompt, img, processor, show_progress=False
    )

    uncertainties.append({
        "path": str(img_path),
        "prompt": prompt,
        "uncertainty": result["std_dev_score"],
    })

# Sort by uncertainty
uncertainties.sort(key=lambda x: x["uncertainty"], reverse=True)

# Print top 10 most uncertain
print("Top 10 most uncertain samples:")
for i, item in enumerate(uncertainties[:10], 1):
    print(f"{i}. {item['path']}: {item['uncertainty']:.4f}")
```

### Example 2: Compare Acquisition Functions

```python
from active_learning import get_acquisition_function
import torch
import matplotlib.pyplot as plt

# Simulate MC samples
torch.manual_seed(42)
samples = torch.randn(30, 100)  # 30 MC samples, 100 data points

# Try different acquisition functions
functions = ["std_dev", "variance", "mad", "iqr", "cv"]
results = {}

for func_name in functions:
    func = get_acquisition_function(func_name)
    uncertainty = func(samples, dim=0)
    results[func_name] = uncertainty.numpy()

# Plot comparison
fig, axes = plt.subplots(len(functions), 1, figsize=(12, 10))
for ax, (name, values) in zip(axes, results.items()):
    ax.plot(values)
    ax.set_title(f"Acquisition Function: {name}")
    ax.set_ylabel("Uncertainty")

plt.tight_layout()
plt.savefig("acquisition_comparison.png")
print("Saved comparison to acquisition_comparison.png")
```

### Example 3: Full Pipeline with Checkpointing

```python
from active_learning import ActiveLearningConfig, ActiveLearningPipeline
from trainer.models.clip_model import CLIPModel, ClipModelConfig
from transformers import CLIPProcessor
import json

# Load model with checkpoint
model_config = ClipModelConfig(
    pretrained_model_name_or_path="openai/clip-vit-base-patch32",
    dropout_rate=0.1,
)
model = CLIPModel(model_config).to("cuda")

checkpoint_path = "outputs/checkpoint-final/pytorch_model.bin"
state_dict = torch.load(checkpoint_path)
model.model.load_state_dict(state_dict)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Configure for aggressive uncertainty sampling
config = ActiveLearningConfig(
    n_mc_samples=50,  # More samples for better estimates
    acquisition_function="bald",  # BALD for classification
    selection_strategy="top_k",
    num_samples_to_select=200,
    output_dir="aggressive_al_results",
)

# Run pipeline
pipeline = ActiveLearningPipeline(model, processor, config)
results = pipeline.run(prompts, images, image_ids)

# Analyze results
print(f"\nSelected {len(results['selected_samples'])} samples")
print(f"Uncertainty range: [{results['summary_statistics']['min']:.4f}, {results['summary_statistics']['max']:.4f}]")

# Save selected image IDs for labeling
selected_ids = [s['image_id'] for s in results['selected_samples']]
with open("to_label.json", "w") as f:
    json.dump(selected_ids, f)
```

## Troubleshooting

See the main [active_learning/README.md](active_learning/README.md) for detailed troubleshooting.

## Next Steps

1. **Run Tests**: `python test_active_learning.py`
2. **Try Evaluation**: `python run_evaluation.py --eval-mode`
3. **Read Module Docs**: See `active_learning/README.md`
4. **Customize**: Add your own acquisition functions or selection strategies

## Summary

The refactored active learning pipeline provides:

✅ **Clean Architecture**: Modular, reusable components
✅ **Easy to Extend**: Add new acquisition functions easily
✅ **Well Documented**: Comprehensive docs and examples
✅ **Type Safe**: Type hints throughout
✅ **Tested**: Comprehensive test suite
✅ **CLI & API**: Use from command line or Python
✅ **Flexible**: Many configuration options
✅ **Production Ready**: Error handling, logging, validation

The pipeline is now ready for serious active learning experiments!
