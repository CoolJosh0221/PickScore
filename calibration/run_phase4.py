#!/usr/bin/env python
"""
Phase 4: Post-Calibration Evaluation
Compares baseline vs calibrated metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from calibration.metrics import compute_ece_mce
from calibration.correlation import compute_uncertainty_error_correlation
from calibration.plots import plot_reliability_diagram


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Post-Calibration Evaluation")
    parser.add_argument("--phase1-tensors", type=str, required=True,
                        help="Path to phase1_tensors.pt (baseline)")
    parser.add_argument("--phase3-tensors", type=str, required=True,
                        help="Path to phase3_tensors.pt (calibrated)")
    parser.add_argument("--output-dir", type=str, default="calibration_results_v2")
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline tensors
    print(f"Loading baseline tensors from {args.phase1_tensors}")
    baseline_data = torch.load(args.phase1_tensors, weights_only=False)
    mc_probs_baseline = baseline_data["mc_probs"]
    mean_probs_baseline = baseline_data["mean_probs"]
    labels = baseline_data["labels"]

    # Load calibrated tensors
    print(f"Loading calibrated tensors from {args.phase3_tensors}")
    calibrated_data = torch.load(args.phase3_tensors, weights_only=False)
    mc_probs_calibrated = calibrated_data["mc_probs_calibrated"]
    mean_probs_calibrated = calibrated_data["mean_probs_calibrated"]
    optimal_T = calibrated_data["optimal_T"]

    print(f"Optimal temperature: T = {optimal_T:.4f}")

    # Compute metrics - baseline
    print("\nComputing baseline metrics...")
    metrics_baseline = compute_ece_mce(mean_probs_baseline, labels, n_bins=args.n_bins)
    corr_baseline = compute_uncertainty_error_correlation(mc_probs_baseline, labels)

    # Compute metrics - calibrated
    print("Computing calibrated metrics...")
    metrics_calibrated = compute_ece_mce(mean_probs_calibrated, labels, n_bins=args.n_bins)
    corr_calibrated = compute_uncertainty_error_correlation(mc_probs_calibrated, labels)

    print("\n" + "=" * 60)
    print("PHASE 4: POST-CALIBRATION COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Baseline':<15} {'Calibrated':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'ECE':<25} {metrics_baseline['ece']*100:>12.2f}% {metrics_calibrated['ece']*100:>12.2f}% {(metrics_calibrated['ece']-metrics_baseline['ece'])*100:>+12.2f}%")
    print(f"{'MCE':<25} {metrics_baseline['mce']*100:>12.2f}% {metrics_calibrated['mce']*100:>12.2f}% {(metrics_calibrated['mce']-metrics_baseline['mce'])*100:>+12.2f}%")
    print(f"{'Mean Confidence':<25} {metrics_baseline['mean_confidence']*100:>12.2f}% {metrics_calibrated['mean_confidence']*100:>12.2f}% {(metrics_calibrated['mean_confidence']-metrics_baseline['mean_confidence'])*100:>+12.2f}%")
    print(f"{'BALD-Error ρ':<25} {corr_baseline['bald']['spearman']:>12.4f}  {corr_calibrated['bald']['spearman']:>12.4f}  {corr_calibrated['bald']['spearman']-corr_baseline['bald']['spearman']:>+12.4f}")
    print("=" * 60)

    ece_reduction = (metrics_baseline['ece'] - metrics_calibrated['ece']) / metrics_baseline['ece'] * 100
    print(f"\nECE reduction: {ece_reduction:.1f}%")

    bald_change = corr_calibrated['bald']['spearman'] - corr_baseline['bald']['spearman']
    if bald_change < 0:
        print(f"WARNING: BALD-Error correlation DECREASED by {abs(bald_change):.4f}")
    else:
        print(f"BALD-Error correlation increased by {bald_change:.4f}")

    # Plot reliability diagrams
    plot_reliability_diagram(
        metrics_baseline["bins"],
        output_dir / "phase4_reliability_baseline.png",
        title=f"Baseline (ECE={metrics_baseline['ece']*100:.2f}%)"
    )
    plot_reliability_diagram(
        metrics_calibrated["bins"],
        output_dir / "phase4_reliability_calibrated.png",
        title=f"Calibrated T={optimal_T:.2f} (ECE={metrics_calibrated['ece']*100:.2f}%)"
    )

    # Save results
    results = {
        "phase": 4,
        "timestamp": datetime.now().isoformat(),
        "optimal_temperature": optimal_T,
        "baseline": {
            "ece": metrics_baseline["ece"],
            "mce": metrics_baseline["mce"],
            "mean_confidence": metrics_baseline["mean_confidence"],
            "bald_error_spearman": corr_baseline["bald"]["spearman"],
        },
        "calibrated": {
            "ece": metrics_calibrated["ece"],
            "mce": metrics_calibrated["mce"],
            "mean_confidence": metrics_calibrated["mean_confidence"],
            "bald_error_spearman": corr_calibrated["bald"]["spearman"],
        },
        "ece_reduction_pct": ece_reduction,
        "bald_correlation_change": bald_change,
    }

    results_path = output_dir / "phase4_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
