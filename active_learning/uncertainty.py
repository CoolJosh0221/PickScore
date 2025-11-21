"""
Uncertainty Estimation with Monte Carlo Dropout

This module provides utilities for uncertainty estimation using MC Dropout.
"""

from typing import Dict, List, Optional, Callable, Union
import torch
from torch import nn
from PIL import Image
from tqdm.rich import tqdm

from .acquisition_functions import (
    AcquisitionFunction,
    get_acquisition_function,
    compute_all_uncertainty_metrics,
)


class UncertaintyEstimator:
    """
    Uncertainty estimator using Monte Carlo Dropout.

    This class provides methods to estimate uncertainty in model predictions
    using multiple forward passes with dropout enabled.
    """

    def __init__(
        self,
        model: nn.Module,
        acquisition_function: Union[str, AcquisitionFunction] = "std_dev",
        n_samples: int = 10,
        device: Union[str, torch.device] = "cuda",
    ):
        """
        Initialize uncertainty estimator.

        Args:
            model: The model to use for uncertainty estimation
            acquisition_function: Acquisition function name or instance
            n_samples: Number of MC dropout samples
            device: Device to run inference on
        """
        self.model = model
        self.n_samples = n_samples
        self.device = device

        if isinstance(acquisition_function, str):
            self.acquisition_function = get_acquisition_function(acquisition_function)
        else:
            self.acquisition_function = acquisition_function

    def enable_mc_dropout(self):
        """Enable MC dropout mode on the model."""
        if hasattr(self.model, "enable_mc_dropout"):
            self.model.enable_mc_dropout = True
        self.model.eval()

        # Ensure all dropout layers are in training mode
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def disable_mc_dropout(self):
        """Disable MC dropout mode on the model."""
        if hasattr(self.model, "enable_mc_dropout"):
            self.model.enable_mc_dropout = False
        self.model.eval()

    def estimate_score_uncertainty(
        self,
        prompt: str,
        image: Image.Image,
        processor,
        compute_all_metrics: bool = False,
        show_progress: bool = False,
    ) -> Dict[str, float]:
        """
        Estimate uncertainty for a single image-prompt pair.

        Args:
            prompt: Text prompt
            image: PIL Image
            processor: CLIP processor for preprocessing
            compute_all_metrics: If True, compute all uncertainty metrics
            show_progress: If True, show progress bar

        Returns:
            Dictionary with uncertainty metrics
        """
        # Enable MC dropout
        original_mc_state = getattr(self.model, "enable_mc_dropout", False)
        self.enable_mc_dropout()

        # Preprocess inputs
        image_inputs = processor(
            images=image,
            size={"shortest_edge": 224, "longest_edge": 224},
            return_tensors="pt",
        ).to(self.device)

        text_inputs = processor(
            text=prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        # Collect MC samples
        all_scores = []
        iterator = range(self.n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="MC Sampling")

        with torch.no_grad():
            for _ in iterator:
                # Get features
                image_features = self.model.get_image_features(**image_inputs)
                image_features = image_features / torch.norm(
                    image_features, dim=-1, keepdim=True
                )

                text_features = self.model.get_text_features(**text_inputs)
                text_features = text_features / torch.norm(
                    text_features, dim=-1, keepdim=True
                )

                # Calculate scores
                scores = self.model.logit_scale.exp() * (text_features @ image_features.T)[0]
                all_scores.append(scores.cpu())

        # Restore original MC dropout state
        if hasattr(self.model, "enable_mc_dropout"):
            self.model.enable_mc_dropout = original_mc_state
        if not original_mc_state:
            self.disable_mc_dropout()

        # Calculate metrics
        all_scores_tensor = torch.stack(all_scores)

        if compute_all_metrics:
            metrics = compute_all_uncertainty_metrics(all_scores_tensor, dim=0)
            # Convert tensors to scalars
            results = {k: v.item() if v.numel() == 1 else v[0].item() for k, v in metrics.items()}
        else:
            mean_score = torch.mean(all_scores_tensor, dim=0)[0]
            uncertainty = self.acquisition_function(all_scores_tensor, dim=0)[0]

            results = {
                "mean_score": mean_score.item(),
                f"{self.acquisition_function.name()}_score": uncertainty.item(),
            }

        return results

    def estimate_preference_uncertainty(
        self,
        prompt: str,
        images: List[Image.Image],
        processor,
        compute_all_metrics: bool = False,
        show_progress: bool = False,
    ) -> Dict[str, any]:
        """
        Estimate uncertainty for preference prediction between multiple images.

        Args:
            prompt: Text prompt
            images: List of PIL Images to compare
            processor: CLIP processor for preprocessing
            compute_all_metrics: If True, compute all uncertainty metrics
            show_progress: If True, show progress bar

        Returns:
            Dictionary with mean probabilities and uncertainty metrics
        """
        # Enable MC dropout
        original_mc_state = getattr(self.model, "enable_mc_dropout", False)
        self.enable_mc_dropout()

        # Preprocess inputs
        image_inputs = processor(
            images=images,
            size={"shortest_edge": 224, "longest_edge": 224},
            return_tensors="pt",
        ).to(self.device)

        text_inputs = processor(
            text=prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        # Collect MC samples
        all_probs = []
        iterator = range(self.n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="MC Sampling")

        with torch.no_grad():
            for _ in iterator:
                # Get features
                image_features = self.model.get_image_features(**image_inputs)
                image_features = image_features / torch.norm(
                    image_features, dim=-1, keepdim=True
                )

                text_features = self.model.get_text_features(**text_inputs)
                text_features = text_features / torch.norm(
                    text_features, dim=-1, keepdim=True
                )

                # Calculate scores and probabilities
                scores = self.model.logit_scale.exp() * (text_features @ image_features.T)[0]
                probs = torch.softmax(scores, dim=-1)
                all_probs.append(probs.cpu())

        # Restore original MC dropout state
        if hasattr(self.model, "enable_mc_dropout"):
            self.model.enable_mc_dropout = original_mc_state
        if not original_mc_state:
            self.disable_mc_dropout()

        # Calculate metrics
        all_probs_tensor = torch.stack(all_probs)
        mean_probs = torch.mean(all_probs_tensor, dim=0)

        if compute_all_metrics:
            # For probabilities, compute metrics for each image separately
            results = {
                "mean_probs": mean_probs.tolist(),
                "all_samples": all_probs_tensor.tolist(),
            }

            # Compute uncertainty metrics
            uncertainty = self.acquisition_function(all_probs_tensor, dim=0)
            results[f"{self.acquisition_function.name()}_uncertainty"] = uncertainty.tolist()

            # Additional metrics
            std = torch.std(all_probs_tensor, dim=0)
            results["std"] = std.tolist()

        else:
            uncertainty = self.acquisition_function(all_probs_tensor, dim=0)
            results = {
                "mean_probs": mean_probs.tolist(),
                f"{self.acquisition_function.name()}_uncertainty": uncertainty.tolist(),
            }

        return results

    def batch_estimate_uncertainty(
        self,
        prompts: List[str],
        images: List[Image.Image],
        processor,
        show_progress: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Estimate uncertainty for multiple image-prompt pairs.

        Args:
            prompts: List of text prompts
            images: List of PIL Images
            processor: CLIP processor
            show_progress: If True, show progress bar

        Returns:
            List of uncertainty dictionaries
        """
        if len(prompts) != len(images):
            raise ValueError("Number of prompts and images must match")

        results = []
        iterator = zip(prompts, images)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Processing samples")

        for prompt, image in iterator:
            uncertainty = self.estimate_score_uncertainty(
                prompt, image, processor, show_progress=False
            )
            results.append(uncertainty)

        return results
