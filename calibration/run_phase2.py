#!/usr/bin/env python
"""
Phase 2: Uncertainty-Error Correlation
Measures if BALD/Entropy/LC uncertainty predicts errors (Gleave diagnostic).
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from calibration.correlation import compute_uncertainty_error_correlation, compute_errors
from calibration.bald import compute_bald
from calibration.plots import plot_uncertainty_error


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Uncertainty-Error Correlation")
    parser.add_argument("--tensors-path", type=str, required=True,
                        help="Path to phase1_tensors.pt")
    parser.add_argument("--output-dir", type=str, default="calibration_results_v2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tensors from Phase 1
    print(f"Loading tensors from {args.tensors_path}")
    data = torch.load(args.tensors_path, weights_only=False)
    mc_probs = data["mc_probs"]
    labels = data["labels"]
    print(f"Loaded {mc_probs.shape[1]} samples with T={mc_probs.shape[0]} MC samples")

    # Compute uncertainty-error correlation
    print("\nComputing uncertainty-error correlations...")
    correlations = compute_uncertainty_error_correlation(mc_probs, labels)

    print("\n" + "=" * 60)
    print("PHASE 2: UNCERTAINTY-ERROR CORRELATION")
    print("=" * 60)
    print(f"Error rate: {correlations['error_rate']*100:.2f}%")
    print(f"BALD ↔ Error:    ρ = {correlations['bald']['spearman']:.4f} (p={correlations['bald']['p_value']:.2e})")
    print(f"Entropy ↔ Error: ρ = {correlations['entropy']['spearman']:.4f} (p={correlations['entropy']['p_value']:.2e})")
    print(f"LC ↔ Error:      ρ = {correlations['least_confidence']['spearman']:.4f} (p={correlations['least_confidence']['p_value']:.2e})")
    print("=" * 60)

    # Interpretation
    bald_rho = abs(correlations['bald']['spearman'])
    if bald_rho < 0.1:
        interpretation = "Very weak"
    elif bald_rho < 0.2:
        interpretation = "Weak"
    elif bald_rho < 0.4:
        interpretation = "Moderate"
    elif bald_rho < 0.6:
        interpretation = "Strong"
    else:
        interpretation = "Very strong"
    print(f"Interpretation: {interpretation} correlation")

    # Plot
    mean_probs = mc_probs.mean(dim=0)
    errors = compute_errors(mean_probs, labels)
    bald_scores = compute_bald(mc_probs)

    plot_uncertainty_error(
        bald_scores.numpy(),
        errors.numpy(),
        output_dir / "phase2_uncertainty_error.png",
        title=f"BALD Uncertainty vs Error (ρ={correlations['bald']['spearman']:.3f})",
        correlation=correlations['bald']['spearman'],
        xlabel="BALD Uncertainty"
    )

    # Save results
    results = {
        "phase": 2,
        "timestamp": datetime.now().isoformat(),
        "correlations": correlations,
        "interpretation": interpretation,
    }

    results_path = output_dir / "phase2_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
