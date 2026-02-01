"""Uncertainty metrics: BALD, Entropy, Least Confidence."""

import torch


def _entropy(p: torch.Tensor) -> torch.Tensor:
    """Compute entropy along last dimension."""
    p = p.clamp_min(1e-9)
    return -(p * p.log()).sum(dim=-1)


def compute_bald(mc_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute BALD (Bayesian Active Learning by Disagreement) scores.

    BALD = H[E[p]] - E[H[p]]
         = Predictive entropy - Expected entropy under posterior

    Args:
        mc_probs: [T, N, K] probabilities from T MC Dropout passes

    Returns:
        bald_scores: [N] BALD uncertainty scores
    """
    mean_probs = mc_probs.mean(dim=0)  # [N, K]
    predictive_entropy = _entropy(mean_probs)  # [N]
    expected_entropy = _entropy(mc_probs).mean(dim=0)  # [N]
    return predictive_entropy - expected_entropy


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute predictive entropy.

    Args:
        probs: [N, K] mean probabilities

    Returns:
        entropy: [N] entropy scores
    """
    return _entropy(probs)


def compute_least_confidence(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute least confidence uncertainty (1 - max probability).

    Args:
        probs: [N, K] mean probabilities

    Returns:
        lc_scores: [N] least confidence scores
    """
    return 1.0 - probs.max(dim=-1).values
