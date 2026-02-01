"""Calibration analysis module for Active Learning."""

from .collect import collect_mc_predictions, collect_mc_logits
from .metrics import compute_ece_mce, compute_calibration_bins
from .temperature import find_optimal_temperature, apply_temperature
from .correlation import compute_uncertainty_error_correlation
from .bald import compute_bald, compute_entropy, compute_least_confidence
from .plots import plot_reliability_diagram, plot_uncertainty_error

__all__ = [
    "collect_mc_predictions",
    "collect_mc_logits",
    "compute_ece_mce",
    "compute_calibration_bins",
    "find_optimal_temperature",
    "apply_temperature",
    "compute_uncertainty_error_correlation",
    "compute_bald",
    "compute_entropy",
    "compute_least_confidence",
    "plot_reliability_diagram",
    "plot_uncertainty_error",
]
