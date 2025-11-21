"""
Active Learning Module for PickScore

This module provides utilities for active learning with uncertainty estimation
using Monte Carlo Dropout on CLIP models.
"""

from .acquisition_functions import (
    AcquisitionFunction,
    VarianceAcquisition,
    StdDevAcquisition,
    EntropyAcquisition,
    BALDAcquisition,
    get_acquisition_function,
)
from .uncertainty import UncertaintyEstimator
from .sample_selector import SampleSelector
from .config import ActiveLearningConfig

__all__ = [
    "AcquisitionFunction",
    "VarianceAcquisition",
    "StdDevAcquisition",
    "EntropyAcquisition",
    "BALDAcquisition",
    "get_acquisition_function",
    "UncertaintyEstimator",
    "SampleSelector",
    "ActiveLearningConfig",
]
