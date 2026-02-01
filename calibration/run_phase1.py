#!/usr/bin/env python
"""
Phase 1: Baseline Calibration Metrics
Computes ECE, MCE, reliability diagram using MC Dropout predictions.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import CLIPProcessor

from active_learning.data.loaders import create_dataloader
from active_learning.models.model_mcdo import MCDropoutCLIPModel

from calibration.collect import collect_mc_predictions
from calibration.metrics import compute_ece_mce
from calibration.plots import plot_reliability_diagram


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline Calibration Metrics")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="calibration_results_v2")
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-mc-samples", type=int, default=15)
    parser.add_argument("--pretrained", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--mc-dropout-p", type=float, default=0.1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.model_path}")
    model = MCDropoutCLIPModel(
        pretrained_model_name_or_path=args.pretrained,
        mc_dropout_p=args.mc_dropout_p
    )
    model.load(args.model_path)
    model.to(device)

    processor = CLIPProcessor.from_pretrained(args.pretrained)

    # Load test data
    print(f"Loading test data from {args.data_dir}")
    test_loader = create_dataloader(
        data_dir=Path(args.data_dir),
        split="test",
        batch_size=args.batch_size,
        num_workers=4,
        processor=processor,
        shuffle=False,
    )

    # Collect MC predictions
    print(f"\nCollecting predictions with MC Dropout (T={args.num_mc_samples})...")
    mc_probs, mean_probs, labels = collect_mc_predictions(
        model, processor, test_loader, device, args.num_mc_samples
    )
    print(f"Collected {len(mean_probs)} samples")

    # Compute calibration metrics
    print(f"\nComputing calibration metrics with {args.n_bins} bins...")
    metrics = compute_ece_mce(mean_probs, labels, n_bins=args.n_bins)

    print("\n" + "=" * 60)
    print("PHASE 1: BASELINE CALIBRATION RESULTS")
    print("=" * 60)
    print(f"ECE: {metrics['ece']*100:.2f}%")
    print(f"MCE: {metrics['mce']*100:.2f}%")
    print(f"Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
    print(f"Mean Confidence: {metrics['mean_confidence']*100:.2f}%")
    print("=" * 60)

    # Plot
    plot_reliability_diagram(
        metrics["bins"],
        output_dir / "phase1_reliability_baseline.png",
        title=f"Baseline Reliability (ECE={metrics['ece']*100:.2f}%)"
    )

    # Save results
    results = {
        "phase": 1,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "n_bins": args.n_bins,
            "num_mc_samples": args.num_mc_samples,
        },
        "metrics": {
            "ece": metrics["ece"],
            "mce": metrics["mce"],
            "overall_accuracy": metrics["overall_accuracy"],
            "mean_confidence": metrics["mean_confidence"],
        },
        "bins": metrics["bins"],
    }

    results_path = output_dir / "phase1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")

    # Save tensors for later phases
    torch.save({
        "mc_probs": mc_probs,
        "mean_probs": mean_probs,
        "labels": labels,
    }, output_dir / "phase1_tensors.pt")
    print(f"Saved tensors to {output_dir / 'phase1_tensors.pt'}")


if __name__ == "__main__":
    main()
