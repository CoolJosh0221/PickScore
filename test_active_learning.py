#!/usr/bin/env python3
"""
Test script for active learning pipeline.

This script performs basic tests to verify the pipeline works correctly.
"""

import sys
import torch
from PIL import Image
import numpy as np

print("Testing active learning module...")
print("=" * 80)

# Test imports
print("\n1. Testing imports...")
try:
    from trainer.models.clip_model import CLIPModel, ClipModelConfig
    from transformers import CLIPProcessor
    from active_learning import (
        ActiveLearningConfig,
        ActiveLearningPipeline,
        UncertaintyEstimator,
        SampleSelector,
        get_acquisition_function,
    )
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test acquisition functions
print("\n2. Testing acquisition functions...")
try:
    # Test getting acquisition functions by name
    acq_functions = ["std_dev", "variance", "entropy", "bald", "cv", "mad", "iqr"]
    for name in acq_functions:
        acq_func = get_acquisition_function(name)
        print(f"  ✓ {name}: {acq_func.name()}")

    # Test on dummy data
    dummy_samples = torch.randn(10, 5)  # 10 MC samples, 5 predictions
    for name in acq_functions:
        acq_func = get_acquisition_function(name)
        result = acq_func(dummy_samples, dim=0)
        print(f"  ✓ {name} computation: shape {result.shape}")

    print("✓ All acquisition functions work")
except Exception as e:
    print(f"✗ Acquisition function test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test sample selector
print("\n3. Testing sample selector...")
try:
    selector = SampleSelector(strategy="top_k")

    # Create dummy uncertainties
    uncertainties = np.random.rand(100)

    # Test top-k selection
    selected = selector.select_top_k(uncertainties, k=10)
    assert len(selected) == 10, "Top-k should return 10 samples"
    print(f"  ✓ Top-k selection: {len(selected)} samples")

    # Test threshold selection
    selected = selector.select_above_threshold(uncertainties, threshold=0.5)
    print(f"  ✓ Threshold selection: {len(selected)} samples")

    # Test statistics
    stats = selector.get_uncertainty_statistics(uncertainties)
    print(f"  ✓ Statistics: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    print("✓ Sample selector works")
except Exception as e:
    print(f"✗ Sample selector test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model loading
print("\n4. Testing model loading...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    model_config = ClipModelConfig(
        pretrained_model_name_or_path="openai/clip-vit-base-patch32",
        dropout_rate=0.1,
        enable_mc_dropout=False,
    )
    model = CLIPModel(model_config)
    model = model.to(device)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("  ✓ Model loaded successfully")
    print(f"  ✓ Model device: {next(model.parameters()).device}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test uncertainty estimator
print("\n5. Testing uncertainty estimator...")
try:
    estimator = UncertaintyEstimator(
        model=model,
        acquisition_function="std_dev",
        n_samples=5,  # Small number for testing
        device=device,
    )
    print("  ✓ Uncertainty estimator created")

    # Create a dummy image
    dummy_image = Image.new("RGB", (224, 224), color="red")
    dummy_prompt = "a test image"

    # Test uncertainty estimation
    print("  Running MC dropout inference (5 samples)...")
    result = estimator.estimate_score_uncertainty(
        prompt=dummy_prompt,
        image=dummy_image,
        processor=processor,
        compute_all_metrics=True,
        show_progress=False,
    )

    print(f"  ✓ Uncertainty estimation completed")
    print(f"    - Mean score: {result.get('mean', 0):.4f}")
    print(f"    - Std: {result.get('std', 0):.4f}")
    print(f"    - Available metrics: {', '.join(result.keys())}")

    print("✓ Uncertainty estimator works")
except Exception as e:
    print(f"✗ Uncertainty estimator test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test configuration
print("\n6. Testing configuration...")
try:
    config = ActiveLearningConfig(
        pretrained_model_name="openai/clip-vit-base-patch32",
        n_mc_samples=10,
        acquisition_function="std_dev",
        selection_strategy="top_k",
        num_samples_to_select=5,
        output_dir="test_output",
    )
    print("  ✓ Configuration created")
    print(f"    - MC samples: {config.n_mc_samples}")
    print(f"    - Acquisition: {config.acquisition_function}")
    print(f"    - Strategy: {config.selection_strategy}")
    print("✓ Configuration works")
except Exception as e:
    print(f"✗ Configuration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test full pipeline (simplified)
print("\n7. Testing full pipeline...")
try:
    # Create pipeline
    pipeline = ActiveLearningPipeline(
        model=model,
        processor=processor,
        config=config,
    )
    print("  ✓ Pipeline created")

    # Create dummy data
    dummy_images = [Image.new("RGB", (224, 224), color=c) for c in ["red", "green", "blue"]]
    dummy_prompts = ["test prompt 1", "test prompt 2", "test prompt 3"]
    dummy_ids = ["img1", "img2", "img3"]

    print("  Running pipeline on 3 dummy samples...")
    results = pipeline.run(
        prompts=dummy_prompts,
        images=dummy_images,
        image_ids=dummy_ids,
        save_results=False,  # Don't save for test
    )

    print(f"  ✓ Pipeline completed")
    print(f"    - Total samples: {len(results['all_uncertainties'])}")
    print(f"    - Selected samples: {len(results['selected_samples'])}")

    print("✓ Full pipeline works")
except Exception as e:
    print(f"✗ Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All tests passed successfully!")
print("=" * 80)
print("\nThe active learning pipeline is working correctly.")
print("You can now use:")
print("  - run_active_learning.py for active learning")
print("  - run_evaluation.py for evaluation")
