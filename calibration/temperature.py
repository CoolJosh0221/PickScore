"""Temperature scaling for calibration."""

import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from typing import Tuple


def find_optimal_temperature(
    mc_logits: torch.Tensor,
    labels: torch.Tensor,
    bounds: Tuple[float, float] = (0.1, 10.0)
) -> float:
    """
    Find optimal temperature T that minimizes NLL on validation set.

    Args:
        mc_logits: [T, N, K] logits from MC Dropout
        labels: [N, K] ground truth labels (soft labels)
        bounds: (min_T, max_T) search bounds

    Returns:
        Optimal temperature value
    """
    mean_logits = mc_logits.mean(dim=0)  # [N, K]

    def nll_loss(T: float) -> float:
        scaled_logits = mean_logits / T
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        loss = -(labels * log_probs).sum(dim=-1).mean()
        return loss.item()

    result = minimize_scalar(nll_loss, bounds=bounds, method='bounded')
    return result.x


def apply_temperature(
    mc_logits: torch.Tensor,
    T: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply temperature scaling to MC logits.

    Args:
        mc_logits: [T, N, K] logits from MC Dropout
        T: temperature value

    Returns:
        mc_probs: [T, N, K] calibrated probabilities
        mean_probs: [N, K] mean calibrated probabilities
    """
    scaled_logits = mc_logits / T
    mc_probs = F.softmax(scaled_logits, dim=-1)
    mean_probs = mc_probs.mean(dim=0)
    return mc_probs, mean_probs


def compute_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    T: float = 1.0
) -> float:
    """
    Compute negative log-likelihood with optional temperature scaling.

    Args:
        logits: [N, K] logits
        labels: [N, K] ground truth labels (soft labels)
        T: temperature value

    Returns:
        NLL value
    """
    scaled_logits = logits / T
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    loss = -(labels * log_probs).sum(dim=-1).mean()
    return loss.item()
