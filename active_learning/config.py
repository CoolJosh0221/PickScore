"""
Active Learning Configuration

This module provides configuration dataclasses for active learning experiments.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning experiments."""

    # Model configuration
    pretrained_model_name: str = "openai/clip-vit-base-patch32"
    checkpoint_path: Optional[str] = None
    dropout_rate: float = 0.1

    # MC Dropout configuration
    n_mc_samples: int = 20
    acquisition_function: str = "std_dev"  # Options: std_dev, variance, entropy, bald, cv, mad, iqr

    # Sample selection configuration
    selection_strategy: str = "top_k"  # Options: top_k, threshold, proportional
    num_samples_to_select: int = 100
    uncertainty_threshold: Optional[float] = None
    diversity_weight: float = 0.0  # 0 = only uncertainty, 1 = only diversity

    # Data configuration
    dataset_name: str = "kashif/pickascore"
    unlabeled_split: str = "train"
    batch_size: int = 32
    num_workers: int = 4

    # Device configuration
    device: str = "cuda"
    use_fp16: bool = False

    # Evaluation configuration
    eval_prompts: Dict[str, List[str]] = field(default_factory=lambda: {
        "clear": [
            "A red apple on a white table, professional photography, sharp focus",
            "Single origami crane, white paper, minimal shadow, studio lighting",
            "Chess piece (king) in dramatic lighting, black background",
        ],
        "abstract": [
            "Dreams melting into reality, surreal abstract art, Salvador Dali style",
            "The sound of jazz visualized, abstract expressionism, vibrant colors",
            "Mathematical beauty, fractals morphing into butterflies",
        ],
        "ambiguous": [
            "Something between a forest and an ocean",
            "The edge of understanding",
            "Almost but not quite a face",
        ],
        "complex": [
            "Busy Tokyo street at night, rain, neon reflections, crowds with umbrellas",
            "Steampunk airship battle above Victorian London, detailed, epic scale",
            "Underwater coral reef city with bioluminescent creatures",
        ],
    })

    # Output configuration
    output_dir: str = "active_learning_outputs"
    save_uncertainties: bool = True
    save_selected_samples: bool = True

    # Logging configuration
    log_interval: int = 10
    save_visualizations: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.selection_strategy == "top_k" and self.num_samples_to_select is None:
            raise ValueError("num_samples_to_select must be provided for top_k strategy")

        if self.selection_strategy == "threshold" and self.uncertainty_threshold is None:
            raise ValueError("uncertainty_threshold must be provided for threshold strategy")

        if not 0 <= self.diversity_weight <= 1:
            raise ValueError("diversity_weight must be between 0 and 1")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation experiments."""

    # Image generation configuration
    width: int = 224
    height: int = 224
    num_images: int = 25

    # Prompt categories
    prompts: Dict[str, List[str]] = field(default_factory=lambda: {
        "clear": [
            "A red apple on a white table, professional photography, sharp focus",
            "Single origami crane, white paper, minimal shadow, studio lighting",
            "Chess piece (king) in dramatic lighting, black background",
        ],
        "abstract": [
            "Dreams melting into reality, surreal abstract art, Salvador Dali style",
            "The sound of jazz visualized, abstract expressionism, vibrant colors",
            "Mathematical beauty, fractals morphing into butterflies",
        ],
        "ambiguous": [
            "Something between a forest and an ocean",
            "The edge of understanding",
            "Almost but not quite a face",
        ],
        "complex": [
            "Busy Tokyo street at night, rain, neon reflections, crowds with umbrellas",
            "Steampunk airship battle above Victorian London, detailed, epic scale",
            "Underwater coral reef city with bioluminescent creatures",
        ],
        "contradictory": [
            "A colorless green idea sleeping furiously",
            "Square circle floating in impossible space",
            "Transparent metal organic hybrid creature",
        ],
        "style_mixing": [
            "Mona Lisa painted by Picasso in anime style",
            "Van Gogh's Starry Night but it's a photograph",
            "Minecraft landscape with photorealistic textures",
        ],
    })

    # MC Dropout configuration
    num_mc_samples: int = 20
    dropout_rate: float = 0.1

    # Model configuration
    pretrained_model_name: str = "openai/clip-vit-base-patch32"
    checkpoint_path: Optional[str] = None

    # Device configuration
    device: str = "cuda"

    def get_by_category(self, category: str) -> List[str]:
        """Get prompts for a specific category."""
        if category in self.prompts:
            return self.prompts[category]
        else:
            raise KeyError(f"Category {category} not found")
