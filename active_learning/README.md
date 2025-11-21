# Active Learning Module for PickScore

This module provides a comprehensive framework for active learning with uncertainty estimation using Monte Carlo Dropout on CLIP models.

## Overview

The active learning pipeline consists of several components:

1. **Acquisition Functions** (`acquisition_functions.py`): Various methods to quantify uncertainty
2. **Uncertainty Estimator** (`uncertainty.py`): MC Dropout-based uncertainty estimation
3. **Sample Selector** (`sample_selector.py`): Strategies for selecting informative samples
4. **Pipeline** (`pipeline.py`): Main orchestrator for the active learning workflow
5. **Configuration** (`config.py`): Configuration classes for experiments

## Quick Start

### 1. Basic Usage

```python
import torch
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
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Create configuration
config = ActiveLearningConfig(
    n_mc_samples=20,
    acquisition_function="std_dev",
    selection_strategy="top_k",
    num_samples_to_select=100,
    output_dir="results",
)

# Create pipeline
pipeline = ActiveLearningPipeline(model, processor, config)

# Run on your data
prompts = ["a photo of a cat", "a photo of a dog"]
images = [Image.open("cat.jpg"), Image.open("dog.jpg")]
results = pipeline.run(prompts, images)
```

### 2. Command-Line Interface

Run active learning on a dataset:

```bash
python run_active_learning.py \
    --model-name openai/clip-vit-base-patch32 \
    --checkpoint outputs/checkpoint-final/pytorch_model.bin \
    --data-file data.json \
    --n-samples 20 \
    --acquisition-function std_dev \
    --num-select 100 \
    --output-dir results
```

Run evaluation on test prompts:

```bash
python run_active_learning.py \
    --model-name openai/clip-vit-base-patch32 \
    --checkpoint outputs/checkpoint-final/pytorch_model.bin \
    --eval-mode \
    --test-images-dir test_images/real_images \
    --n-samples 20 \
    --output-dir eval_results
```

### 3. Evaluation Script

Evaluate uncertainty on predefined test prompts:

```bash
python run_evaluation.py \
    --model-name openai/clip-vit-base-patch32 \
    --checkpoint outputs/checkpoint-final/pytorch_model.bin \
    --test-images-dir test_images/real_images \
    --output-file results.json \
    --n-samples 20
```

## Acquisition Functions

The module supports multiple acquisition functions for uncertainty quantification:

### Basic Statistics
- **`std_dev`**: Standard deviation across MC samples (recommended)
- **`variance`**: Variance across MC samples
- **`cv`**: Coefficient of variation (std / mean)

### Probabilistic Measures
- **`entropy`**: Predictive entropy for classification
- **`bald`**: Bayesian Active Learning by Disagreement (mutual information)

### Robust Statistics
- **`mad`**: Median Absolute Deviation
- **`iqr`**: Interquartile Range (Q3 - Q1)
- **`ci90`**: 90% confidence interval width
- **`ci95`**: 95% confidence interval width
- **`ci99`**: 99% confidence interval width

### Usage Example

```python
from active_learning import UncertaintyEstimator

# Create estimator with specific acquisition function
estimator = UncertaintyEstimator(
    model=model,
    acquisition_function="bald",  # or any other function
    n_samples=30,
    device="cuda"
)

# Estimate uncertainty
result = estimator.estimate_score_uncertainty(
    prompt="a beautiful landscape",
    image=image,
    processor=processor,
    compute_all_metrics=True  # Get all available metrics
)
```

## Selection Strategies

Three strategies for selecting samples:

### 1. Top-K Selection
Select the k samples with highest uncertainty:

```python
config = ActiveLearningConfig(
    selection_strategy="top_k",
    num_samples_to_select=100,
)
```

### 2. Threshold Selection
Select all samples above an uncertainty threshold:

```python
config = ActiveLearningConfig(
    selection_strategy="threshold",
    uncertainty_threshold=0.5,
)
```

### 3. Proportional Selection
Sample proportionally to uncertainty (exploration):

```python
config = ActiveLearningConfig(
    selection_strategy="proportional",
    num_samples_to_select=100,
)
```

### 4. Diverse Selection
Balance uncertainty and diversity:

```python
from active_learning import SampleSelector

selector = SampleSelector()
selected_indices = selector.select_diverse(
    uncertainties=uncertainties,
    features=image_features,
    k=100,
    diversity_weight=0.5  # 0=only uncertainty, 1=only diversity
)
```

## Module Components

### UncertaintyEstimator

Estimates uncertainty using Monte Carlo Dropout:

```python
from active_learning import UncertaintyEstimator

estimator = UncertaintyEstimator(
    model=model,
    acquisition_function="std_dev",
    n_samples=20,
    device="cuda"
)

# Single image-prompt pair
result = estimator.estimate_score_uncertainty(
    prompt="a cat",
    image=image,
    processor=processor,
)

# Multiple images with same prompt (preference)
result = estimator.estimate_preference_uncertainty(
    prompt="a cat",
    images=[image1, image2],
    processor=processor,
)

# Batch processing
results = estimator.batch_estimate_uncertainty(
    prompts=["prompt1", "prompt2"],
    images=[image1, image2],
    processor=processor,
)
```

### SampleSelector

Selects informative samples:

```python
from active_learning import SampleSelector

selector = SampleSelector(strategy="top_k")

# Top-k selection
indices = selector.select_top_k(uncertainties, k=100)

# Threshold selection
indices = selector.select_above_threshold(uncertainties, threshold=0.5)

# Proportional sampling
indices = selector.select_proportional(uncertainties, k=100, temperature=1.0)

# Get statistics
stats = selector.get_uncertainty_statistics(uncertainties)
```

### ActiveLearningPipeline

Orchestrates the complete workflow:

```python
from active_learning import ActiveLearningPipeline, ActiveLearningConfig

# Create pipeline
config = ActiveLearningConfig(...)
pipeline = ActiveLearningPipeline(model, processor, config)

# Run on dataset
results = pipeline.run(prompts, images, image_ids)

# Evaluate on test prompts
eval_results = pipeline.evaluate_on_test_prompts(test_images_dir)
```

## Output Format

The pipeline saves results in JSON format:

### uncertainties.json
```json
[
  {
    "image_id": "img_001",
    "prompt": "a photo of a cat",
    "mean_score": 23.45,
    "std": 1.23,
    "var": 1.51,
    "cv": 0.052,
    "mad": 0.98,
    "iqr": 1.45,
    "ci90": 2.01,
    "ci95": 2.35
  },
  ...
]
```

### selected_samples.json
```json
[
  {
    "rank": 1,
    "selection_index": 42,
    "image_id": "img_042",
    "prompt": "an ambiguous scene",
    "mean_score": 25.67,
    "std": 2.45,
    ...
  },
  ...
]
```

### summary.json
```json
{
  "config": {
    "pretrained_model": "openai/clip-vit-base-patch32",
    "n_mc_samples": 20,
    "acquisition_function": "std_dev",
    ...
  },
  "summary_statistics": {
    "mean": 1.23,
    "std": 0.45,
    "min": 0.12,
    "max": 3.45,
    "median": 1.15,
    ...
  },
  "num_selected": 100
}
```

## Data Format

For the `--data-file` argument, use this JSON format:

```json
{
  "samples": [
    {
      "id": "img_001",
      "prompt": "a photo of a cat",
      "image_path": "path/to/image1.jpg"
    },
    {
      "id": "img_002",
      "prompt": "a photo of a dog",
      "image_path": "path/to/image2.jpg"
    }
  ]
}
```

## Best Practices

1. **MC Dropout Samples**: Start with 20-30 samples for good uncertainty estimates
2. **Acquisition Function**: `std_dev` works well for most cases; use `bald` for classification
3. **Selection Strategy**: `top_k` is simple and effective; use `proportional` for more exploration
4. **Batch Size**: Process in batches to avoid memory issues
5. **Checkpoint**: Always save and version your model checkpoints

## Advanced Usage

### Custom Acquisition Function

```python
from active_learning import AcquisitionFunction

class MyAcquisition(AcquisitionFunction):
    def __call__(self, samples, dim=0):
        # Your custom logic here
        return my_uncertainty_metric(samples, dim)

    def name(self):
        return "my_custom_metric"

# Use it
estimator = UncertaintyEstimator(
    model=model,
    acquisition_function=MyAcquisition(),
    n_samples=20,
)
```

### Programmatic Pipeline

```python
# Create components manually for fine-grained control
from active_learning import (
    UncertaintyEstimator,
    SampleSelector,
    get_acquisition_function,
)

# Custom acquisition function
acq_func = get_acquisition_function("bald")

# Estimator with custom settings
estimator = UncertaintyEstimator(
    model=model,
    acquisition_function=acq_func,
    n_samples=50,
    device="cuda"
)

# Estimate uncertainties
uncertainties = []
for prompt, image in zip(prompts, images):
    result = estimator.estimate_score_uncertainty(
        prompt, image, processor
    )
    uncertainties.append(result["std_dev_score"])

# Select samples with custom strategy
selector = SampleSelector(strategy="top_k")
selected = selector.select_diverse(
    uncertainties=uncertainties,
    features=image_features,
    k=100,
    diversity_weight=0.3,
)
```

## Troubleshooting

### Out of Memory
- Reduce `n_mc_samples`
- Use smaller batch sizes
- Enable `--fp16` mode
- Process images sequentially

### Slow Performance
- Reduce `n_mc_samples` (try 10-15)
- Use GPU (`--device cuda`)
- Process in batches
- Use simpler acquisition functions (std_dev vs bald)

### Low Uncertainty Estimates
- Increase `dropout_rate` (try 0.2-0.3)
- Check that MC dropout is enabled
- Verify model is in eval mode
- Ensure dropout layers exist in model

## Citation

If you use this active learning module, please cite:

```bibtex
@software{pickscore_active_learning,
  title={Active Learning Module for PickScore},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/PickScore}
}
```
