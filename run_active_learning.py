#!/usr/bin/env python3
"""
Main script for running active learning pipeline.

This script provides a command-line interface for running active learning
experiments with PickScore models.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPProcessor

from trainer.models.clip_model import CLIPModel, ClipModelConfig
from active_learning import ActiveLearningConfig, ActiveLearningPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run active learning pipeline for PickScore"
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (optional)",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate for MC dropout",
    )

    # MC Dropout arguments
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of MC dropout samples",
    )
    parser.add_argument(
        "--acquisition-function",
        type=str,
        default="std_dev",
        choices=["std_dev", "variance", "entropy", "bald", "cv", "mad", "iqr"],
        help="Acquisition function for uncertainty estimation",
    )

    # Selection arguments
    parser.add_argument(
        "--selection-strategy",
        type=str,
        default="top_k",
        choices=["top_k", "threshold", "proportional"],
        help="Sample selection strategy",
    )
    parser.add_argument(
        "--num-select",
        type=int,
        default=100,
        help="Number of samples to select",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Uncertainty threshold (for threshold strategy)",
    )

    # Data arguments
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to JSON file with prompts and image paths",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Run evaluation on test prompts",
    )
    parser.add_argument(
        "--test-images-dir",
        type=str,
        default="test_images/real_images",
        help="Directory with test images (for eval mode)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="active_learning_outputs",
        help="Output directory for results",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    return parser.parse_args()


def load_model(
    model_name: str,
    checkpoint_path: Optional[str],
    dropout_rate: float,
    device: str,
):
    """
    Load CLIP model with MC dropout support.

    Args:
        model_name: Pretrained model name
        checkpoint_path: Optional path to checkpoint
        dropout_rate: Dropout rate
        device: Device to load model on

    Returns:
        Tuple of (model, processor)
    """
    print(f"\n{'=' * 80}")
    print(f"Loading model: {model_name}")
    print(f"{'=' * 80}\n")

    # Create model configuration
    model_config = ClipModelConfig(
        pretrained_model_name_or_path=model_name,
        dropout_rate=dropout_rate,
        enable_mc_dropout=False,  # Will be enabled by uncertainty estimator
    )

    # Create model
    model = CLIPModel(model_config)
    model = model.to(device)

    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if "model" in state_dict:
            model.model.load_state_dict(state_dict["model"])
        elif "state_dict" in state_dict:
            model.model.load_state_dict(state_dict["state_dict"])
        else:
            model.model.load_state_dict(state_dict)

        print("Checkpoint loaded successfully")

    # Load processor
    processor = CLIPProcessor.from_pretrained(model_name)

    print(f"Model loaded on device: {device}")
    print(f"Dropout rate: {dropout_rate}")

    return model, processor


def load_data_from_file(data_file: str):
    """
    Load prompts and images from a JSON file.

    Expected JSON format:
    {
        "samples": [
            {"id": "img1", "prompt": "...", "image_path": "..."},
            {"id": "img2", "prompt": "...", "image_path": "..."},
            ...
        ]
    }

    Args:
        data_file: Path to JSON file

    Returns:
        Tuple of (prompts, images, image_ids)
    """
    print(f"\nLoading data from: {data_file}")

    with open(data_file, "r") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    print(f"Found {len(samples)} samples")

    prompts = []
    images = []
    image_ids = []

    for sample in samples:
        image_path = sample.get("image_path")
        prompt = sample.get("prompt")
        img_id = sample.get("id", Path(image_path).stem)

        if not image_path or not prompt:
            print(f"Warning: Skipping invalid sample: {sample}")
            continue

        try:
            img = Image.open(image_path).convert("RGB")
            images.append(img)
            prompts.append(prompt)
            image_ids.append(img_id)
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")

    print(f"Loaded {len(images)} valid samples")
    return prompts, images, image_ids


def main():
    """Main entry point."""
    args = parse_args()

    # Load model
    model, processor = load_model(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        dropout_rate=args.dropout_rate,
        device=args.device,
    )

    # Create configuration
    config = ActiveLearningConfig(
        pretrained_model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        dropout_rate=args.dropout_rate,
        n_mc_samples=args.n_samples,
        acquisition_function=args.acquisition_function,
        selection_strategy=args.selection_strategy,
        num_samples_to_select=args.num_select,
        uncertainty_threshold=args.threshold,
        device=args.device,
        output_dir=args.output_dir,
    )

    # Create pipeline
    pipeline = ActiveLearningPipeline(
        model=model,
        processor=processor,
        config=config,
    )

    # Run pipeline
    if args.eval_mode:
        # Evaluation mode: run on test prompts
        test_images_dir = Path(args.test_images_dir)
        if not test_images_dir.exists():
            print(f"Error: Test images directory not found: {test_images_dir}")
            sys.exit(1)

        results = pipeline.evaluate_on_test_prompts(
            test_images_dir=test_images_dir,
            save_results=True,
        )

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"Evaluation Summary")
        print(f"{'=' * 80}\n")

        for category, category_results in results.items():
            print(f"\nCategory: {category}")
            for prompt_id, prompt_results in category_results.items():
                summary = prompt_results["summary"]
                print(f"  Prompt {prompt_id}:")
                print(f"    Num images: {prompt_results['num_images']}")
                print(f"    Mean uncertainty: {summary['mean']:.4f}")
                print(f"    Std uncertainty: {summary['std']:.4f}")

    else:
        # Active learning mode: run on provided data
        if not args.data_file:
            print("Error: --data-file is required for active learning mode")
            print("Use --eval-mode to run evaluation on test prompts")
            sys.exit(1)

        # Load data
        prompts, images, image_ids = load_data_from_file(args.data_file)

        if not images:
            print("Error: No valid images found")
            sys.exit(1)

        # Run pipeline
        results = pipeline.run(
            prompts=prompts,
            images=images,
            image_ids=image_ids,
            save_results=True,
        )

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"Active Learning Summary")
        print(f"{'=' * 80}\n")
        print(f"Total samples: {len(images)}")
        print(f"Selected samples: {len(results['selected_samples'])}")
        print(f"\nTop 10 most uncertain samples:")
        for i, sample in enumerate(results["selected_samples"][:10], 1):
            print(f"  {i}. {sample['image_id']}")
            print(f"     Prompt: {sample['prompt'][:60]}...")
            acq_func_name = config.acquisition_function
            if acq_func_name == "std_dev":
                uncertainty_key = "std_dev_score"
            else:
                uncertainty_key = f"{acq_func_name}_score"
            uncertainty = sample.get(uncertainty_key, sample.get("std", 0))
            print(f"     Uncertainty: {uncertainty:.4f}")


if __name__ == "__main__":
    main()
