#!/usr/bin/env python3
"""
Evaluation script for uncertainty estimation on test prompts.

This script evaluates model uncertainty on predefined test prompts
and generates comprehensive uncertainty metrics.
"""

import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPProcessor
from tqdm.rich import tqdm

from trainer.models.clip_model import CLIPModel, ClipModelConfig
from active_learning import UncertaintyEstimator, ActiveLearningConfig
from evaluation.evaluation_config import EvaluationConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty on test prompts"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoint-final/pytorch_model.bin",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test-images-dir",
        type=str,
        default="test_images/real_images",
        help="Directory with test images",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="uncertainty_evaluation_results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of MC dropout samples",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Specific categories to evaluate (default: all)",
    )

    return parser.parse_args()


def load_model(model_name: str, checkpoint_path: str, dropout_rate: float, device: str):
    """Load model and processor."""
    print(f"\n{'=' * 80}")
    print(f"Loading model: {model_name}")
    print(f"{'=' * 80}\n")

    # Create model
    model_config = ClipModelConfig(
        pretrained_model_name_or_path=model_name,
        dropout_rate=dropout_rate,
        enable_mc_dropout=False,
    )
    model = CLIPModel(model_config)
    model = model.to(device)

    # Load checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using pretrained weights")

    # Load processor
    processor = CLIPProcessor.from_pretrained(model_name)

    return model, processor


def evaluate_category_prompt(
    category: str,
    prompt_id: int,
    prompt: str,
    test_images_dir: Path,
    uncertainty_estimator: UncertaintyEstimator,
    processor,
    num_images: int = 25,
):
    """Evaluate uncertainty for one prompt."""
    print(f"  Evaluating prompt {prompt_id}: {prompt[:60]}...")

    # Load images
    image_dir = test_images_dir / category / str(prompt_id)
    if not image_dir.exists():
        print(f"    Warning: Directory not found: {image_dir}")
        return None

    uncertainties = []

    for i in range(num_images):
        img_path = image_dir / f"{i}.png"
        if not img_path.exists():
            print(f"    Warning: Image not found: {img_path}")
            continue

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                result = uncertainty_estimator.estimate_score_uncertainty(
                    prompt=prompt,
                    image=img,
                    processor=processor,
                    compute_all_metrics=True,
                    show_progress=False,
                )
                result["image_id"] = i
                uncertainties.append(result)
        except Exception as e:
            print(f"    Warning: Error processing {img_path}: {e}")

    if not uncertainties:
        return None

    # Compute mean uncertainty scores across all images
    mean_scores = {}
    for key in uncertainties[0].keys():
        if key != "image_id":
            values = [u[key] for u in uncertainties]
            mean_scores[f"mean_{key}"] = sum(values) / len(values)

    return {
        "prompt": prompt,
        "num_images": len(uncertainties),
        "mean_scores": mean_scores,
        "per_image_uncertainties": uncertainties,
    }


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

    # Create uncertainty estimator
    uncertainty_estimator = UncertaintyEstimator(
        model=model,
        acquisition_function="std_dev",
        n_samples=args.n_samples,
        device=args.device,
    )

    # Get test images directory
    test_images_dir = Path(args.test_images_dir)
    if not test_images_dir.exists():
        print(f"Error: Test images directory not found: {test_images_dir}")
        return

    # Get categories to evaluate
    eval_config = EvaluationConfig()
    categories = args.categories or list(eval_config.prompts.keys())

    print(f"\n{'=' * 80}")
    print(f"Evaluating uncertainty on test prompts")
    print(f"Categories: {', '.join(categories)}")
    print(f"MC samples: {args.n_samples}")
    print(f"{'=' * 80}\n")

    # Evaluate each category
    all_results = {}

    for category in categories:
        if category not in eval_config.prompts:
            print(f"Warning: Unknown category '{category}', skipping")
            continue

        print(f"\nCategory: {category}")
        prompts = eval_config.prompts[category]
        category_results = {}

        for prompt_id, prompt in enumerate(prompts):
            result = evaluate_category_prompt(
                category=category,
                prompt_id=prompt_id,
                prompt=prompt,
                test_images_dir=test_images_dir,
                uncertainty_estimator=uncertainty_estimator,
                processor=processor,
                num_images=eval_config.NUM_IMAGES,
            )

            if result:
                category_results[prompt_id] = result

        all_results[category] = category_results

    # Save results
    output_path = Path(args.output_file)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 80}\n")

    # Print summary
    print("\nSummary:")
    for category, category_results in all_results.items():
        print(f"\n{category}:")
        for prompt_id, result in category_results.items():
            mean_std = result["mean_scores"].get("mean_std", 0)
            print(f"  Prompt {prompt_id}: Mean std = {mean_std:.4f}")


if __name__ == "__main__":
    main()
