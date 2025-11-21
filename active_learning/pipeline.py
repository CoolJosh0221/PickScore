"""
Active Learning Pipeline

This module provides the main pipeline orchestrator for active learning experiments.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from PIL import Image
from tqdm.rich import tqdm
import numpy as np

from .config import ActiveLearningConfig
from .uncertainty import UncertaintyEstimator
from .sample_selector import SampleSelector
from .acquisition_functions import get_acquisition_function


class ActiveLearningPipeline:
    """
    Main pipeline for active learning with uncertainty estimation.

    This class orchestrates the entire active learning workflow:
    1. Load model and data
    2. Estimate uncertainty for unlabeled samples
    3. Select most informative samples
    4. Save results and visualizations
    """

    def __init__(
        self,
        model: nn.Module,
        processor,
        config: ActiveLearningConfig,
    ):
        """
        Initialize active learning pipeline.

        Args:
            model: CLIP model with MC dropout support
            processor: CLIP processor for preprocessing
            config: Active learning configuration
        """
        self.model = model
        self.processor = processor
        self.config = config

        # Initialize components
        self.uncertainty_estimator = UncertaintyEstimator(
            model=model,
            acquisition_function=config.acquisition_function,
            n_samples=config.n_mc_samples,
            device=config.device,
        )

        self.sample_selector = SampleSelector(strategy=config.selection_strategy)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def estimate_dataset_uncertainty(
        self,
        prompts: List[str],
        images: List[Image.Image],
        image_ids: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> Tuple[List[Dict[str, float]], Dict[str, any]]:
        """
        Estimate uncertainty for a dataset of image-prompt pairs.

        Args:
            prompts: List of text prompts
            images: List of PIL Images
            image_ids: Optional list of image identifiers
            show_progress: If True, show progress bar

        Returns:
            Tuple of (uncertainty_results, summary_statistics)
        """
        if len(prompts) != len(images):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) and images ({len(images)}) must match"
            )

        if image_ids is None:
            image_ids = [f"image_{i}" for i in range(len(images))]

        print(f"\n{'=' * 80}")
        print(f"Estimating uncertainty for {len(images)} samples")
        print(f"MC samples: {self.config.n_mc_samples}")
        print(f"Acquisition function: {self.config.acquisition_function}")
        print(f"{'=' * 80}\n")

        results = []
        uncertainties = []

        iterator = zip(image_ids, prompts, images)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Processing samples")

        for img_id, prompt, image in iterator:
            uncertainty_dict = self.uncertainty_estimator.estimate_score_uncertainty(
                prompt=prompt,
                image=image,
                processor=self.processor,
                compute_all_metrics=True,
                show_progress=False,
            )

            result = {
                "image_id": img_id,
                "prompt": prompt,
                **uncertainty_dict,
            }
            results.append(result)

            # Extract primary uncertainty metric
            acq_func_name = self.uncertainty_estimator.acquisition_function.name()
            uncertainty_key = f"{acq_func_name}_score"
            uncertainties.append(uncertainty_dict.get(uncertainty_key, uncertainty_dict.get("std", 0)))

        # Compute summary statistics
        summary = self.sample_selector.get_uncertainty_statistics(uncertainties)
        summary["num_samples"] = len(results)
        summary["acquisition_function"] = self.config.acquisition_function

        print(f"\nUncertainty Statistics:")
        print(f"  Mean: {summary['mean']:.4f}")
        print(f"  Std:  {summary['std']:.4f}")
        print(f"  Min:  {summary['min']:.4f}")
        print(f"  Max:  {summary['max']:.4f}")
        print(f"  Median: {summary['median']:.4f}")

        return results, summary

    def select_samples(
        self,
        uncertainty_results: List[Dict[str, float]],
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, any]]:
        """
        Select most informative samples based on uncertainty.

        Args:
            uncertainty_results: List of uncertainty dictionaries
            k: Number of samples to select (overrides config)
            threshold: Uncertainty threshold (overrides config)

        Returns:
            List of selected sample dictionaries with rankings
        """
        k = k or self.config.num_samples_to_select
        threshold = threshold or self.config.uncertainty_threshold

        # Extract uncertainties
        acq_func_name = self.uncertainty_estimator.acquisition_function.name()
        uncertainty_key = f"{acq_func_name}_score"

        uncertainties = []
        for result in uncertainty_results:
            uncertainty = result.get(uncertainty_key, result.get("std", 0))
            uncertainties.append(uncertainty)

        print(f"\n{'=' * 80}")
        print(f"Selecting samples using strategy: {self.config.selection_strategy}")
        print(f"{'=' * 80}\n")

        # Select samples
        if self.config.selection_strategy == "top_k":
            selected_indices = self.sample_selector.select_top_k(uncertainties, k)
            print(f"Selected top {len(selected_indices)} samples")

        elif self.config.selection_strategy == "threshold":
            selected_indices = self.sample_selector.select_above_threshold(
                uncertainties, threshold
            )
            print(f"Selected {len(selected_indices)} samples above threshold {threshold:.4f}")

        elif self.config.selection_strategy == "proportional":
            selected_indices = self.sample_selector.select_proportional(uncertainties, k)
            print(f"Selected {len(selected_indices)} samples proportionally")

        else:
            raise ValueError(f"Unknown selection strategy: {self.config.selection_strategy}")

        # Create selected samples list with rankings
        selected_samples = []
        for rank, idx in enumerate(selected_indices, 1):
            sample = uncertainty_results[idx].copy()
            sample["rank"] = rank
            sample["selection_index"] = idx
            selected_samples.append(sample)

        return selected_samples

    def run(
        self,
        prompts: List[str],
        images: List[Image.Image],
        image_ids: Optional[List[str]] = None,
        save_results: bool = True,
    ) -> Dict[str, any]:
        """
        Run the complete active learning pipeline.

        Args:
            prompts: List of text prompts
            images: List of PIL Images
            image_ids: Optional list of image identifiers
            save_results: If True, save results to disk

        Returns:
            Dictionary with all results
        """
        print(f"\n{'=' * 80}")
        print(f"Starting Active Learning Pipeline")
        print(f"{'=' * 80}\n")

        # Step 1: Estimate uncertainty
        uncertainty_results, summary_stats = self.estimate_dataset_uncertainty(
            prompts=prompts,
            images=images,
            image_ids=image_ids,
            show_progress=True,
        )

        # Step 2: Select samples
        selected_samples = self.select_samples(uncertainty_results)

        # Prepare results
        results = {
            "config": {
                "pretrained_model": self.config.pretrained_model_name,
                "n_mc_samples": self.config.n_mc_samples,
                "acquisition_function": self.config.acquisition_function,
                "selection_strategy": self.config.selection_strategy,
                "num_samples_to_select": self.config.num_samples_to_select,
            },
            "summary_statistics": summary_stats,
            "all_uncertainties": uncertainty_results,
            "selected_samples": selected_samples,
        }

        # Step 3: Save results
        if save_results:
            self._save_results(results)

        print(f"\n{'=' * 80}")
        print(f"Pipeline completed successfully!")
        print(f"Results saved to: {self.config.output_dir}")
        print(f"{'=' * 80}\n")

        return results

    def _save_results(self, results: Dict[str, any]):
        """Save results to disk."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full results
        if self.config.save_uncertainties:
            uncertainties_path = output_dir / "uncertainties.json"
            with open(uncertainties_path, "w") as f:
                json.dump(results["all_uncertainties"], f, indent=2)
            print(f"Saved uncertainties to: {uncertainties_path}")

        # Save selected samples
        if self.config.save_selected_samples:
            selected_path = output_dir / "selected_samples.json"
            with open(selected_path, "w") as f:
                json.dump(results["selected_samples"], f, indent=2)
            print(f"Saved selected samples to: {selected_path}")

        # Save summary
        summary_path = output_dir / "summary.json"
        summary_data = {
            "config": results["config"],
            "summary_statistics": results["summary_statistics"],
            "num_selected": len(results["selected_samples"]),
        }
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved summary to: {summary_path}")

        # Save config
        config_path = output_dir / "config.json"
        config_dict = {
            k: v for k, v in vars(self.config).items()
            if not k.startswith("_")
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"Saved config to: {config_path}")

    def evaluate_on_test_prompts(
        self,
        test_images_dir: Path,
        save_results: bool = True,
    ) -> Dict[str, Dict[str, any]]:
        """
        Evaluate uncertainty on predefined test prompts.

        Args:
            test_images_dir: Directory containing test images organized by category/prompt
            save_results: If True, save results to disk

        Returns:
            Dictionary with results for each category
        """
        print(f"\n{'=' * 80}")
        print(f"Evaluating on test prompts")
        print(f"{'=' * 80}\n")

        all_results = {}

        for category, prompts in self.config.eval_prompts.items():
            print(f"\nProcessing category: {category}")
            category_results = {}

            for prompt_id, prompt in enumerate(prompts):
                print(f"  Prompt {prompt_id}: {prompt[:50]}...")

                # Load images for this prompt
                image_dir = test_images_dir / category / str(prompt_id)
                if not image_dir.exists():
                    print(f"    Warning: Directory not found: {image_dir}")
                    continue

                images = []
                image_paths = sorted(image_dir.glob("*.png"))

                for img_path in image_paths:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        images.append(img)
                    except Exception as e:
                        print(f"    Warning: Could not load {img_path}: {e}")

                if not images:
                    print(f"    Warning: No images found in {image_dir}")
                    continue

                # Estimate uncertainty for these images
                prompts_list = [prompt] * len(images)
                image_ids = [img_path.stem for img_path in image_paths]

                uncertainty_results, summary = self.estimate_dataset_uncertainty(
                    prompts=prompts_list,
                    images=images,
                    image_ids=image_ids,
                    show_progress=False,
                )

                category_results[prompt_id] = {
                    "prompt": prompt,
                    "num_images": len(images),
                    "summary": summary,
                    "uncertainties": uncertainty_results,
                }

            all_results[category] = category_results

        # Save results
        if save_results:
            output_path = Path(self.config.output_dir) / "eval_results.json"
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved evaluation results to: {output_path}")

        return all_results
