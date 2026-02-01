"""Calibration metrics: ECE, MCE."""

import torch
from typing import Dict, List, Any


def compute_calibration_bins(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15
) -> List[Dict[str, Any]]:
    """
    Compute calibration statistics per confidence bin.

    Args:
        probs: [N, 2] probabilities
        labels: [N, 2] ground truth labels
        n_bins: number of bins

    Returns:
        List of bin dictionaries with confidence, accuracy, count, gap
    """
    pred_probs = probs[:, 0]
    targets = labels[:, 0]

    predictions = (pred_probs > 0.5).float()
    actuals = (targets > 0.5).float()
    confidences = torch.max(probs, dim=-1).values
    correct = (predictions == actuals).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_data = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean().item()
            avg_accuracy = correct[in_bin].mean().item()
            bin_size = in_bin.sum().item()
            calibration_gap = abs(avg_accuracy - avg_confidence)
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

    return bin_data


def compute_ece_mce(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        probs: [N, 2] probabilities
        labels: [N, 2] ground truth labels
        n_bins: number of bins

    Returns:
        Dictionary with ece, mce, overall_accuracy, mean_confidence, bins
    """
    pred_probs = probs[:, 0]
    targets = labels[:, 0]

    predictions = (pred_probs > 0.5).float()
    actuals = (targets > 0.5).float()
    confidences = torch.max(probs, dim=-1).values
    correct = (predictions == actuals).float()

    bin_data = compute_calibration_bins(probs, labels, n_bins)

    ece = 0.0
    mce = 0.0
    total_samples = len(probs)

    for b in bin_data:
        if b["count"] > 0:
            prop_in_bin = b["count"] / total_samples
            ece += prop_in_bin * b["gap"]
            mce = max(mce, b["gap"])

    return {
        "ece": ece,
        "mce": mce,
        "overall_accuracy": correct.mean().item(),
        "mean_confidence": confidences.mean().item(),
        "bins": bin_data,
    }
