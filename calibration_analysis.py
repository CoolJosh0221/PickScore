"""
Calibration Analysis for Active Learning
Phase 1: Baseline Calibration Metrics (ECE, MCE, reliability diagram, confidence histogram)
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import CLIPProcessor

from active_learning.data.loaders import create_dataloader
from active_learning.models.model_mcdo import MCDropoutCLIPModel


def collect_predictions_mc(model, processor, loader, device: str, num_mc_samples: int = 15):
    """Collect predictions using MC Dropout (multiple stochastic forward passes)."""
    model.enable_mc_dropout = True
    model.eval()  # This keeps dropout active when enable_mc_dropout=True

    all_mean_probs = []  # Will store mean probabilities across MC samples
    all_labels = []  # Will store [y0, y1] pairs

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Collecting MC predictions ({num_mc_samples} samples)"):
            imgs0 = batch["image_0"].to(device, non_blocking=True)
            imgs1 = batch["image_1"].to(device, non_blocking=True)
            y0 = batch["label_0"]
            y1 = batch["label_1"]

            txt = processor(
                text=batch["caption"],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            txt = {k: v.to(device, non_blocking=True) for k, v in txt.items()}

            # Run multiple MC samples
            batch_probs = []
            for _ in range(num_mc_samples):
                t = model.get_text_features(**txt)
                i0 = model.get_image_features(pixel_values=imgs0)
                i1 = model.get_image_features(pixel_values=imgs1)

                # Normalize and compute scores
                t = F.normalize(t, dim=-1)
                i0 = F.normalize(i0, dim=-1)
                i1 = F.normalize(i1, dim=-1)
                scale = model.logit_scale.exp()
                s0 = (t * i0).sum(-1) * scale
                s1 = (t * i1).sum(-1) * scale

                logits = torch.stack([s0, s1], dim=-1)  # [B, 2]
                probs = F.softmax(logits, dim=-1)  # [B, 2]
                batch_probs.append(probs)

            # Average probabilities across MC samples: E[p(y|x)]
            mc_probs = torch.stack(batch_probs, dim=0)  # [T, B, 2]
            mean_probs = mc_probs.mean(dim=0).cpu()  # [B, 2]

            labels = torch.stack([y0, y1], dim=-1)  # [B, 2]

            all_mean_probs.append(mean_probs)
            all_labels.append(labels)

    model.enable_mc_dropout = False  # Reset
    return torch.cat(all_mean_probs, dim=0), torch.cat(all_labels, dim=0)


def compute_calibration_metrics(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    For preference prediction: we treat it as binary classification where
    the model predicts P(image_0 preferred) = probs[0]

    Args:
        probs: [N, 2] tensor of probabilities (already softmaxed, possibly MC averaged)
        labels: [N, 2] tensor of labels
        n_bins: number of bins for calibration
    """
    # Get predicted probability for image_0 and actual label
    pred_probs = probs[:, 0]  # P(image_0 preferred)

    # For soft labels: if y0 > y1, image_0 is preferred (label=1)
    # For hard labels: directly use y0 as the target probability
    targets = labels[:, 0]  # y0 is the probability that image_0 is preferred

    # For binary accuracy: predict image_0 if P > 0.5
    predictions = (pred_probs > 0.5).float()
    actuals = (targets > 0.5).float()

    # Confidence is max(P, 1-P)
    confidences = torch.max(probs, dim=-1).values

    # Correctness: did we predict the right preference?
    correct = (predictions == actuals).float()

    # Bin by confidence
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean().item()
            avg_accuracy = correct[in_bin].mean().item()
            bin_size = in_bin.sum().item()

            calibration_gap = abs(avg_accuracy - avg_confidence)
            ece += prop_in_bin * calibration_gap
            mce = max(mce, calibration_gap)

            bin_data.append({
                "lower": bin_lower.item(),
                "upper": bin_upper.item(),
                "confidence": avg_confidence,
                "accuracy": avg_accuracy,
                "count": bin_size,
                "gap": calibration_gap
            })
        else:
            bin_data.append({
                "lower": bin_lower.item(),
                "upper": bin_upper.item(),
                "confidence": None,
                "accuracy": None,
                "count": 0,
                "gap": 0
            })

    return {
        "ece": ece,
        "mce": mce,
        "bins": bin_data,
        "overall_accuracy": correct.mean().item(),
        "mean_confidence": confidences.mean().item(),
    }


def plot_reliability_diagram(bin_data, output_path: Path, title: str = "Reliability Diagram"):
    """Generate reliability diagram (confidence vs accuracy per bin)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Filter non-empty bins
    non_empty = [b for b in bin_data if b["count"] > 0]

    if not non_empty:
        print("Warning: No non-empty bins for reliability diagram")
        return

    confidences = [b["confidence"] for b in non_empty]
    accuracies = [b["accuracy"] for b in non_empty]
    counts = [b["count"] for b in non_empty]
    gaps = [b["gap"] for b in non_empty]

    # Left plot: Reliability diagram
    ax1.bar(confidences, accuracies, width=0.05, alpha=0.7, label="Accuracy", color="blue", edgecolor="black")
    ax1.bar(confidences, gaps, width=0.05, alpha=0.5, bottom=accuracies, label="Gap", color="red", edgecolor="black")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Right plot: Confidence histogram
    bin_edges = np.linspace(0, 1, 16)
    bin_counts = np.zeros(15)
    for b in bin_data:
        idx = int(b["lower"] * 15)
        if idx < 15:
            bin_counts[idx] = b["count"]

    ax2.bar(bin_edges[:-1] + 0.033, bin_counts, width=0.066, alpha=0.7, color="green", edgecolor="black")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Confidence Histogram", fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reliability diagram to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibration Analysis - Phase 1: Baseline Metrics")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory with test split")
    parser.add_argument("--output-dir", type=str, default="calibration_results", help="Output directory")
    parser.add_argument("--n-bins", type=int, default=15, help="Number of bins for ECE/MCE")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--pretrained", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--mc-dropout-p", type=float, default=0.1, help="MC Dropout probability")
    parser.add_argument("--num-mc-samples", type=int, default=15, help="Number of MC Dropout samples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}")
    model = MCDropoutCLIPModel(
        pretrained_model_name_or_path=args.pretrained,
        mc_dropout_p=args.mc_dropout_p
    )
    model.load(args.model_path)
    model.to(device)
    model.eval()

    # Load processor
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

    # Collect predictions with MC Dropout
    print(f"Collecting predictions with MC Dropout ({args.num_mc_samples} samples)...")
    mean_probs, labels = collect_predictions_mc(model, processor, test_loader, device, num_mc_samples=args.num_mc_samples)
    print(f"Collected {len(mean_probs)} samples")

    # Compute calibration metrics
    print(f"Computing calibration metrics with {args.n_bins} bins...")
    metrics = compute_calibration_metrics(mean_probs, labels, n_bins=args.n_bins)

    print("\n" + "="*50)
    print("CALIBRATION RESULTS")
    print("="*50)
    print(f"Expected Calibration Error (ECE): {metrics['ece']*100:.2f}%")
    print(f"Maximum Calibration Error (MCE):  {metrics['mce']*100:.2f}%")
    print(f"Overall Accuracy:                 {metrics['overall_accuracy']*100:.2f}%")
    print(f"Mean Confidence:                  {metrics['mean_confidence']*100:.2f}%")
    print("="*50)

    # Generate reliability diagram
    plot_path = output_dir / "reliability_diagram_mc_baseline.png"
    plot_reliability_diagram(
        metrics["bins"],
        plot_path,
        title=f"MC Dropout Reliability Diagram (ECE={metrics['ece']*100:.2f}%, T={args.num_mc_samples})"
    )

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "n_bins": args.n_bins,
        "num_mc_samples": args.num_mc_samples,
        "ece": metrics["ece"],
        "mce": metrics["mce"],
        "overall_accuracy": metrics["overall_accuracy"],
        "mean_confidence": metrics["mean_confidence"],
        "bins": metrics["bins"],
    }

    results_path = output_dir / "calibration_mc_baseline.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")

    # Save mean probs and labels for later phases
    torch.save({
        "mean_probs": mean_probs,
        "labels": labels,
        "num_mc_samples": args.num_mc_samples,
    }, output_dir / "predictions_mc_baseline.pt")
    print(f"Saved predictions to {output_dir / 'predictions_mc_baseline.pt'}")


if __name__ == "__main__":
    main()
