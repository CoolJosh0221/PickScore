#!/usr/bin/env python
"""
Phase 3: Temperature Scaling
Finds optimal temperature T to calibrate predictions.
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

from calibration.collect import collect_mc_logits
from calibration.temperature import find_optimal_temperature, compute_nll, apply_temperature


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Temperature Scaling")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="calibration_results_v2")
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

    # Collect MC logits
    print(f"\nCollecting logits with MC Dropout (T={args.num_mc_samples})...")
    mc_logits, labels = collect_mc_logits(
        model, processor, test_loader, device, args.num_mc_samples
    )
    print(f"Collected {mc_logits.shape[1]} samples")

    # Find optimal temperature
    print("\nFinding optimal temperature...")
    optimal_T = find_optimal_temperature(mc_logits, labels)

    # Compute NLL before and after
    mean_logits = mc_logits.mean(dim=0)
    nll_before = compute_nll(mean_logits, labels, T=1.0)
    nll_after = compute_nll(mean_logits, labels, T=optimal_T)

    print("\n" + "=" * 60)
    print("PHASE 3: TEMPERATURE SCALING")
    print("=" * 60)
    print(f"Optimal Temperature: T = {optimal_T:.4f}")
    print(f"NLL before (T=1.0): {nll_before:.4f}")
    print(f"NLL after (T={optimal_T:.2f}): {nll_after:.4f}")
    print(f"NLL reduction: {(nll_before - nll_after) / nll_before * 100:.2f}%")
    print("=" * 60)

    # Apply temperature and save calibrated tensors
    mc_probs_calibrated, mean_probs_calibrated = apply_temperature(mc_logits, optimal_T)

    # Save results
    results = {
        "phase": 3,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "num_mc_samples": args.num_mc_samples,
        },
        "optimal_temperature": optimal_T,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "nll_reduction_pct": (nll_before - nll_after) / nll_before * 100,
    }

    results_path = output_dir / "phase3_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")

    # Save tensors for Phase 4
    torch.save({
        "mc_logits": mc_logits,
        "labels": labels,
        "optimal_T": optimal_T,
        "mc_probs_calibrated": mc_probs_calibrated,
        "mean_probs_calibrated": mean_probs_calibrated,
    }, output_dir / "phase3_tensors.pt")
    print(f"Saved tensors to {output_dir / 'phase3_tensors.pt'}")


if __name__ == "__main__":
    main()
