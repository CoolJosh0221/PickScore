#!/usr/bin/env python
"""
Phase 5: Acquisition Score Comparison
Compares how calibration affects acquisition function rankings.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from scipy import stats

from calibration.bald import compute_bald, compute_entropy, compute_least_confidence


def spearman_rank_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Spearman rank correlation between two tensors."""
    corr, _ = stats.spearmanr(x.numpy(), y.numpy())
    return float(corr)


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Acquisition Score Comparison")
    parser.add_argument("--phase1-tensors", type=str, required=True,
                        help="Path to phase1_tensors.pt (baseline)")
    parser.add_argument("--phase3-tensors", type=str, required=True,
                        help="Path to phase3_tensors.pt (calibrated)")
    parser.add_argument("--output-dir", type=str, default="calibration_results_v2")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tensors
    print(f"Loading baseline tensors from {args.phase1_tensors}")
    baseline_data = torch.load(args.phase1_tensors, weights_only=False)
    mc_probs_baseline = baseline_data["mc_probs"]
    mean_probs_baseline = baseline_data["mean_probs"]

    print(f"Loading calibrated tensors from {args.phase3_tensors}")
    calibrated_data = torch.load(args.phase3_tensors, weights_only=False)
    mc_probs_calibrated = calibrated_data["mc_probs_calibrated"]
    mean_probs_calibrated = calibrated_data["mean_probs_calibrated"]
    optimal_T = calibrated_data["optimal_T"]

    # Compute acquisition scores - baseline
    print("\nComputing baseline acquisition scores...")
    bald_baseline = compute_bald(mc_probs_baseline)
    entropy_baseline = compute_entropy(mean_probs_baseline)
    lc_baseline = compute_least_confidence(mean_probs_baseline)

    # Compute acquisition scores - calibrated
    print("Computing calibrated acquisition scores...")
    bald_calibrated = compute_bald(mc_probs_calibrated)
    entropy_calibrated = compute_entropy(mean_probs_calibrated)
    lc_calibrated = compute_least_confidence(mean_probs_calibrated)

    # Rank correlations
    bald_rank_corr = spearman_rank_correlation(bald_baseline, bald_calibrated)
    entropy_rank_corr = spearman_rank_correlation(entropy_baseline, entropy_calibrated)
    lc_rank_corr = spearman_rank_correlation(lc_baseline, lc_calibrated)

    # Top-k overlap
    k = args.top_k
    top_k_baseline = set(bald_baseline.topk(k).indices.tolist())
    top_k_calibrated = set(bald_calibrated.topk(k).indices.tolist())
    overlap = len(top_k_baseline & top_k_calibrated)

    print("\n" + "=" * 60)
    print("PHASE 5: ACQUISITION SCORE COMPARISON")
    print("=" * 60)
    print(f"\nRank correlation (baseline vs calibrated T={optimal_T:.2f}):")
    print(f"  BALD:            {bald_rank_corr:.4f}")
    print(f"  Entropy:         {entropy_rank_corr:.4f}")
    print(f"  Least Confidence:{lc_rank_corr:.4f}")
    print(f"\nTop-{k} sample overlap (BALD): {overlap}/{k} ({overlap/k*100:.1f}%)")

    # BALD statistics
    print(f"\nBALD score statistics:")
    print(f"  Baseline:   mean={bald_baseline.mean():.4f}, std={bald_baseline.std():.4f}, max={bald_baseline.max():.4f}")
    print(f"  Calibrated: mean={bald_calibrated.mean():.4f}, std={bald_calibrated.std():.4f}, max={bald_calibrated.max():.4f}")
    print("=" * 60)

    # Interpretation
    if bald_rank_corr < 0.7:
        print("\nWARNING: Calibration significantly changes BALD rankings!")
        print("This means calibrated BALD would select different samples than baseline BALD.")

    if bald_calibrated.std() < bald_baseline.std() * 0.5:
        print("\nWARNING: BALD variance collapsed after calibration!")
        print("Temperature scaling compressed the uncertainty signal.")

    # Save results
    results = {
        "phase": 5,
        "timestamp": datetime.now().isoformat(),
        "optimal_temperature": optimal_T,
        "rank_correlations": {
            "bald": bald_rank_corr,
            "entropy": entropy_rank_corr,
            "least_confidence": lc_rank_corr,
        },
        "top_k_overlap": {
            "k": k,
            "overlap": overlap,
            "overlap_pct": overlap / k * 100,
        },
        "bald_stats": {
            "baseline": {
                "mean": bald_baseline.mean().item(),
                "std": bald_baseline.std().item(),
                "max": bald_baseline.max().item(),
            },
            "calibrated": {
                "mean": bald_calibrated.mean().item(),
                "std": bald_calibrated.std().item(),
                "max": bald_calibrated.max().item(),
            },
        },
    }

    results_path = output_dir / "phase5_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
