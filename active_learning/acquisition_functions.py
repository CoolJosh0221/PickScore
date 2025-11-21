"""
Acquisition Functions for Active Learning

This module provides various acquisition functions for selecting samples
in active learning based on uncertainty estimates.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import numpy as np


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def __call__(self, samples: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Compute acquisition value from MC samples.

        Args:
            samples: Tensor of MC samples, shape (n_samples, ...)
            dim: Dimension along which to compute acquisition

        Returns:
            Acquisition values
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of the acquisition function."""
        pass


class VarianceAcquisition(AcquisitionFunction):
    """Variance-based acquisition function."""

    def __call__(self, samples: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Compute variance across samples."""
        return torch.var(samples, dim=dim)

    def name(self) -> str:
        return "variance"


class StdDevAcquisition(AcquisitionFunction):
    """Standard deviation-based acquisition function."""

    def __call__(self, samples: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Compute standard deviation across samples."""
        return torch.std(samples, dim=dim)

    def name(self) -> str:
        return "std_dev"


class EntropyAcquisition(AcquisitionFunction):
    """Entropy-based acquisition function for probability distributions."""

    def __call__(self, samples: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Compute predictive entropy.

        Args:
            samples: Probability samples, shape (n_samples, batch_size, n_classes)
            dim: Dimension along which to average

        Returns:
            Entropy values for each sample in the batch
        """
        # Average probabilities across MC samples
        mean_probs = torch.mean(samples, dim=dim)

        # Compute entropy: -sum(p * log(p))
        # Add small epsilon for numerical stability
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
        return entropy

    def name(self) -> str:
        return "entropy"


class BALDAcquisition(AcquisitionFunction):
    """
    BALD (Bayesian Active Learning by Disagreement) acquisition function.

    BALD measures the mutual information between predictions and model parameters.
    """

    def __call__(self, samples: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Compute BALD score.

        Args:
            samples: Probability samples, shape (n_samples, batch_size, n_classes)
            dim: Dimension along which to compute

        Returns:
            BALD scores for each sample in the batch
        """
        epsilon = 1e-10

        # Expected entropy: E[H[y|x,w]]
        # Entropy of each MC sample, then average
        sample_entropy = -torch.sum(samples * torch.log(samples + epsilon), dim=-1)
        expected_entropy = torch.mean(sample_entropy, dim=dim)

        # Entropy of expectation: H[E[y|x]]
        mean_probs = torch.mean(samples, dim=dim)
        entropy_of_expected = -torch.sum(mean_probs * torch.log(mean_probs + epsilon), dim=-1)

        # BALD = H[E[y|x]] - E[H[y|x,w]]
        bald_score = entropy_of_expected - expected_entropy
        return bald_score

    def name(self) -> str:
        return "bald"


class CoefficientOfVariationAcquisition(AcquisitionFunction):
    """Coefficient of Variation acquisition function."""

    def __call__(self, samples: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Compute coefficient of variation (std / mean)."""
        mean = torch.mean(samples, dim=dim)
        std = torch.std(samples, dim=dim)
        # Add small epsilon to avoid division by zero
        cv = std / (torch.abs(mean) + 1e-8)
        return cv

    def name(self) -> str:
        return "coefficient_of_variation"


class MADAcquisition(AcquisitionFunction):
    """Median Absolute Deviation acquisition function."""

    def __call__(self, samples: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Compute median absolute deviation."""
        median = torch.median(samples, dim=dim)[0]
        # Expand median to match samples shape for broadcasting
        if dim == 0:
            median_expanded = median.unsqueeze(0)
        else:
            median_expanded = median.unsqueeze(dim)

        abs_deviations = torch.abs(samples - median_expanded)
        mad = torch.median(abs_deviations, dim=dim)[0]
        return mad

    def name(self) -> str:
        return "mad"


class IQRAcquisition(AcquisitionFunction):
    """Interquartile Range acquisition function."""

    def __call__(self, samples: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Compute interquartile range (Q3 - Q1)."""
        q75 = torch.quantile(samples, 0.75, dim=dim)
        q25 = torch.quantile(samples, 0.25, dim=dim)
        iqr = q75 - q25
        return iqr

    def name(self) -> str:
        return "iqr"


class ConfidenceIntervalAcquisition(AcquisitionFunction):
    """Confidence Interval Width acquisition function."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize confidence interval acquisition.

        Args:
            confidence_level: Confidence level (e.g., 0.90, 0.95, 0.99)
        """
        self.confidence_level = confidence_level
        self.lower_quantile = (1 - confidence_level) / 2
        self.upper_quantile = 1 - self.lower_quantile

    def __call__(self, samples: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Compute confidence interval width."""
        upper = torch.quantile(samples, self.upper_quantile, dim=dim)
        lower = torch.quantile(samples, self.lower_quantile, dim=dim)
        ci_width = upper - lower
        return ci_width

    def name(self) -> str:
        return f"ci_{int(self.confidence_level * 100)}"


def get_acquisition_function(name: str, **kwargs) -> AcquisitionFunction:
    """
    Factory function to get acquisition function by name.

    Args:
        name: Name of the acquisition function
        **kwargs: Additional arguments for the acquisition function

    Returns:
        Acquisition function instance

    Raises:
        ValueError: If acquisition function name is not recognized
    """
    acquisition_functions = {
        "variance": VarianceAcquisition,
        "std_dev": StdDevAcquisition,
        "std": StdDevAcquisition,
        "entropy": EntropyAcquisition,
        "bald": BALDAcquisition,
        "cv": CoefficientOfVariationAcquisition,
        "coefficient_of_variation": CoefficientOfVariationAcquisition,
        "mad": MADAcquisition,
        "iqr": IQRAcquisition,
        "ci90": lambda: ConfidenceIntervalAcquisition(0.90),
        "ci95": lambda: ConfidenceIntervalAcquisition(0.95),
        "ci99": lambda: ConfidenceIntervalAcquisition(0.99),
    }

    name = name.lower()
    if name not in acquisition_functions:
        raise ValueError(
            f"Unknown acquisition function: {name}. "
            f"Available: {list(acquisition_functions.keys())}"
        )

    acq_func_class = acquisition_functions[name]
    if callable(acq_func_class) and not isinstance(acq_func_class, type):
        # It's a lambda function
        return acq_func_class()
    else:
        return acq_func_class(**kwargs)


def compute_all_uncertainty_metrics(
    samples: torch.Tensor, dim: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Compute all available uncertainty metrics.

    Args:
        samples: Tensor of MC samples
        dim: Dimension along which to compute metrics

    Returns:
        Dictionary mapping metric names to values
    """
    metrics = {}

    # Basic statistics
    metrics["mean"] = torch.mean(samples, dim=dim)
    metrics["std"] = torch.std(samples, dim=dim)
    metrics["var"] = torch.var(samples, dim=dim)

    # Coefficient of variation
    cv_acq = CoefficientOfVariationAcquisition()
    metrics["cv"] = cv_acq(samples, dim=dim)

    # Median-based metrics
    mad_acq = MADAcquisition()
    metrics["mad"] = mad_acq(samples, dim=dim)

    iqr_acq = IQRAcquisition()
    metrics["iqr"] = iqr_acq(samples, dim=dim)

    # Confidence intervals
    ci90_acq = ConfidenceIntervalAcquisition(0.90)
    metrics["ci90"] = ci90_acq(samples, dim=dim)

    ci95_acq = ConfidenceIntervalAcquisition(0.95)
    metrics["ci95"] = ci95_acq(samples, dim=dim)

    return metrics
