"""Uncertainty-error correlation analysis."""

import torch
import numpy as np
from scipy import stats
from typing import Dict, Tuple

from .bald import compute_bald, compute_entropy, compute_least_confidence


def compute_errors(mean_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute binary errors (0=correct, 1=wrong).

    Args:
        mean_probs: [N, K] mean probabilities
        labels: [N, K] ground truth labels

    Returns:
        errors: [N] binary error indicators
    """
    predictions = (mean_probs[:, 0] > 0.5).float()
    actuals = (labels[:, 0] > 0.5).float()
    return (predictions != actuals).float()


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation.

    Returns:
        (correlation, p_value)
    """
    corr, p_value = stats.spearmanr(x, y)
    return float(corr), float(p_value)


def compute_uncertainty_error_correlation(
    mc_probs: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    """
    Compute correlation between uncertainty metrics and prediction errors.

    This implements Gleave's diagnostic: good uncertainty should correlate
    with errors (high uncertainty → more likely to be wrong).

    Args:
        mc_probs: [T, N, K] probabilities from MC Dropout
        labels: [N, K] ground truth labels

    Returns:
        Dictionary with correlations for each uncertainty metric
    """
    mean_probs = mc_probs.mean(dim=0)
    errors = compute_errors(mean_probs, labels)
    errors_np = errors.numpy()

    # Compute uncertainty metrics
    bald_scores = compute_bald(mc_probs).numpy()
    entropy_scores = compute_entropy(mean_probs).numpy()
    lc_scores = compute_least_confidence(mean_probs).numpy()

    # Compute correlations
    bald_corr, bald_p = spearman_correlation(bald_scores, errors_np)
    entropy_corr, entropy_p = spearman_correlation(entropy_scores, errors_np)
    lc_corr, lc_p = spearman_correlation(lc_scores, errors_np)

    return {
        "bald": {"spearman": bald_corr, "p_value": bald_p},
        "entropy": {"spearman": entropy_corr, "p_value": entropy_p},
        "least_confidence": {"spearman": lc_corr, "p_value": lc_p},
        "error_rate": float(errors_np.mean()),
    }
