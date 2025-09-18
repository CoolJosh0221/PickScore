"""
Experiment configurations for active learning framework.
"""

from attrs import define


@define
class ExperimentConfig:
    """Base configuration for experiments."""

    # Data formats
    image_size: int = 224

    # Data
    dataset_name: str = "pickapic-anonymous/pickapic_v1"
    seed_size: int = 500
    pool_size: int = 20000
    test_size: int = 2000
    random_seed: int = 42

    # Model
    pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32"
    mc_dropout_p: float = 0.12

    # Training
    train_epochs: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    train_batch_size: int = 24
    eval_batch_size: int = 48

    # Active learning
    al_iterations: int = 12
    acquisition_batch_size: int = 150
    num_mc_samples: int = 20

    # Experiment
    wandb_project: str = "pickapic-active-learning"
    eval_with_mc_dropout: bool = True

    # Miscellaneous
    num_workers: int = 4
    tie_margin: float = 0.1
    seed: int = 42


# Pre-defined configurations

FAST_PROTOTYPE = ExperimentConfig(
    seed_size=100,
    pool_size=2000,
    test_size=500,
    train_epochs=1,
    al_iterations=5,
    acquisition_batch_size=50,
    num_mc_samples=10,
    mc_dropout_p=0.1,
    train_batch_size=16,
    eval_batch_size=32,
)

SMALL_SCALE = ExperimentConfig(
    seed_size=300,
    pool_size=8000,
    test_size=1500,
    train_epochs=2,
    al_iterations=8,
    acquisition_batch_size=100,
    num_mc_samples=15,
    mc_dropout_p=0.11,
    train_batch_size=20,
    eval_batch_size=40,
    learning_rate=1.2e-5,
)

MEDIUM_SCALE = ExperimentConfig(
    seed_size=800,
    pool_size=30000,
    test_size=2500,
    train_epochs=3,
    al_iterations=15,
    acquisition_batch_size=200,
    num_mc_samples=25,
    mc_dropout_p=0.15,
    train_batch_size=32,
    eval_batch_size=64,
    learning_rate=8e-6,
    weight_decay=1.5e-4,
)

LARGE_SCALE = ExperimentConfig(
    seed_size=1500,
    pool_size=75000,
    test_size=4000,
    train_epochs=4,
    al_iterations=25,
    acquisition_batch_size=300,
    num_mc_samples=30,
    mc_dropout_p=0.18,
    train_batch_size=40,
    eval_batch_size=80,
    learning_rate=5e-6,
    weight_decay=2e-4,
)
