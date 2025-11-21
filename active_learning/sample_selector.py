"""
Sample Selection for Active Learning

This module provides utilities for selecting samples based on uncertainty estimates.
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import torch


class SampleSelector:
    """
    Sample selector for active learning.

    Selects samples with highest uncertainty for labeling.
    """

    def __init__(self, strategy: str = "top_k"):
        """
        Initialize sample selector.

        Args:
            strategy: Selection strategy ('top_k', 'threshold', 'proportional')
        """
        self.strategy = strategy

    def select_top_k(
        self,
        uncertainties: Union[List[float], np.ndarray, torch.Tensor],
        k: int,
        return_indices: bool = True,
    ) -> Union[List[int], Tuple[List[int], List[float]]]:
        """
        Select top k samples with highest uncertainty.

        Args:
            uncertainties: Uncertainty scores
            k: Number of samples to select
            return_indices: If True, return indices; if False, return mask

        Returns:
            Indices of selected samples (and optionally their uncertainty scores)
        """
        # Convert to numpy array
        if isinstance(uncertainties, torch.Tensor):
            uncertainties = uncertainties.cpu().numpy()
        elif isinstance(uncertainties, list):
            uncertainties = np.array(uncertainties)

        # Get indices of top k uncertain samples
        top_k_indices = np.argsort(uncertainties)[-k:][::-1]

        if return_indices:
            return top_k_indices.tolist()
        else:
            mask = np.zeros(len(uncertainties), dtype=bool)
            mask[top_k_indices] = True
            return mask.tolist()

    def select_above_threshold(
        self,
        uncertainties: Union[List[float], np.ndarray, torch.Tensor],
        threshold: float,
        return_indices: bool = True,
    ) -> Union[List[int], List[bool]]:
        """
        Select samples with uncertainty above threshold.

        Args:
            uncertainties: Uncertainty scores
            threshold: Uncertainty threshold
            return_indices: If True, return indices; if False, return mask

        Returns:
            Indices or mask of selected samples
        """
        # Convert to numpy array
        if isinstance(uncertainties, torch.Tensor):
            uncertainties = uncertainties.cpu().numpy()
        elif isinstance(uncertainties, list):
            uncertainties = np.array(uncertainties)

        # Create mask for samples above threshold
        mask = uncertainties > threshold

        if return_indices:
            indices = np.where(mask)[0]
            return indices.tolist()
        else:
            return mask.tolist()

    def select_proportional(
        self,
        uncertainties: Union[List[float], np.ndarray, torch.Tensor],
        k: int,
        temperature: float = 1.0,
    ) -> List[int]:
        """
        Select samples with probability proportional to uncertainty.

        Args:
            uncertainties: Uncertainty scores
            k: Number of samples to select
            temperature: Temperature for softmax (higher = more random)

        Returns:
            Indices of selected samples
        """
        # Convert to numpy array
        if isinstance(uncertainties, torch.Tensor):
            uncertainties = uncertainties.cpu().numpy()
        elif isinstance(uncertainties, list):
            uncertainties = np.array(uncertainties)

        # Normalize uncertainties to probabilities
        uncertainties_scaled = uncertainties / temperature
        exp_uncertainties = np.exp(uncertainties_scaled - np.max(uncertainties_scaled))
        probabilities = exp_uncertainties / np.sum(exp_uncertainties)

        # Sample without replacement
        selected_indices = np.random.choice(
            len(uncertainties),
            size=min(k, len(uncertainties)),
            replace=False,
            p=probabilities,
        )

        return selected_indices.tolist()

    def select(
        self,
        uncertainties: Union[List[float], np.ndarray, torch.Tensor],
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> List[int]:
        """
        Select samples based on configured strategy.

        Args:
            uncertainties: Uncertainty scores
            k: Number of samples to select (for 'top_k' and 'proportional')
            threshold: Uncertainty threshold (for 'threshold')
            **kwargs: Additional arguments for specific strategies

        Returns:
            Indices of selected samples
        """
        if self.strategy == "top_k":
            if k is None:
                raise ValueError("k must be provided for top_k strategy")
            return self.select_top_k(uncertainties, k, return_indices=True)

        elif self.strategy == "threshold":
            if threshold is None:
                raise ValueError("threshold must be provided for threshold strategy")
            return self.select_above_threshold(uncertainties, threshold, return_indices=True)

        elif self.strategy == "proportional":
            if k is None:
                raise ValueError("k must be provided for proportional strategy")
            temperature = kwargs.get("temperature", 1.0)
            return self.select_proportional(uncertainties, k, temperature)

        else:
            raise ValueError(f"Unknown selection strategy: {self.strategy}")

    def select_diverse(
        self,
        uncertainties: Union[List[float], np.ndarray, torch.Tensor],
        features: Union[np.ndarray, torch.Tensor],
        k: int,
        diversity_weight: float = 0.5,
    ) -> List[int]:
        """
        Select diverse samples balancing uncertainty and feature diversity.

        Args:
            uncertainties: Uncertainty scores
            features: Feature vectors for diversity computation
            k: Number of samples to select
            diversity_weight: Weight for diversity vs uncertainty (0 to 1)

        Returns:
            Indices of selected samples
        """
        # Convert to numpy arrays
        if isinstance(uncertainties, torch.Tensor):
            uncertainties = uncertainties.cpu().numpy()
        elif isinstance(uncertainties, list):
            uncertainties = np.array(uncertainties)

        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        # Normalize uncertainties
        uncertainties_norm = (uncertainties - uncertainties.min()) / (
            uncertainties.max() - uncertainties.min() + 1e-8
        )

        selected_indices = []
        remaining_indices = list(range(len(uncertainties)))

        # Greedy selection
        for _ in range(min(k, len(uncertainties))):
            if not selected_indices:
                # First sample: highest uncertainty
                idx = remaining_indices[np.argmax(uncertainties_norm[remaining_indices])]
            else:
                # Subsequent samples: balance uncertainty and diversity
                scores = []
                for idx in remaining_indices:
                    # Uncertainty score
                    uncertainty_score = uncertainties_norm[idx]

                    # Diversity score (min distance to selected samples)
                    distances = np.linalg.norm(
                        features[idx] - features[selected_indices], axis=1
                    )
                    diversity_score = np.min(distances)

                    # Combined score
                    combined_score = (
                        (1 - diversity_weight) * uncertainty_score
                        + diversity_weight * diversity_score
                    )
                    scores.append(combined_score)

                # Select sample with highest combined score
                best_idx_in_remaining = np.argmax(scores)
                idx = remaining_indices[best_idx_in_remaining]

            selected_indices.append(idx)
            remaining_indices.remove(idx)

        return selected_indices

    def get_uncertainty_statistics(
        self, uncertainties: Union[List[float], np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute statistics of uncertainty distribution.

        Args:
            uncertainties: Uncertainty scores

        Returns:
            Dictionary with statistics
        """
        # Convert to numpy array
        if isinstance(uncertainties, torch.Tensor):
            uncertainties = uncertainties.cpu().numpy()
        elif isinstance(uncertainties, list):
            uncertainties = np.array(uncertainties)

        stats = {
            "mean": float(np.mean(uncertainties)),
            "std": float(np.std(uncertainties)),
            "min": float(np.min(uncertainties)),
            "max": float(np.max(uncertainties)),
            "median": float(np.median(uncertainties)),
            "q25": float(np.percentile(uncertainties, 25)),
            "q75": float(np.percentile(uncertainties, 75)),
            "iqr": float(np.percentile(uncertainties, 75) - np.percentile(uncertainties, 25)),
        }

        return stats
