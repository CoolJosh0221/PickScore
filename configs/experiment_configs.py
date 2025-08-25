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
    seed_size: int = 200
    pool_size: int = 10000
    test_size: int = 2000
    random_seed: int = 42
    
    # Model
    pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32"
    mc_dropout_p: float = 0.1
    
    # Training
    train_epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    train_batch_size: int = 16
    eval_batch_size: int = 32
    
    # Active learning
    acquisition_strategy: str = "bald"
    al_iterations: int = 10
    acquisition_batch_size: int = 100
    num_mc_samples: int = 20
    
    # Experiment
    wandb_project: str = "active-learning"
    eval_with_mc_dropout: bool = False

    # Miscellaneous
    num_workers: int = 4
    tie_margin: float = 0.1
    seed: int = 42

# Pre-defined configs
FAST_PROTOTYPE = ExperimentConfig(
    seed_size=100,
    pool_size=5000,
    test_size=1000,
    train_epochs=2,
    al_iterations=5,
    acquisition_batch_size=50,
    num_mc_samples=10,
)

SMALL_SCALE = ExperimentConfig()

MEDIUM_SCALE = ExperimentConfig(
    seed_size=500,
    pool_size=25000,
    test_size=3000,
    train_epochs=4,
    al_iterations=15,
    mc_dropout_p=0.15,
    num_mc_samples=25,
    train_batch_size=32,
    eval_batch_size=64
)

LARGE_SCALE = ExperimentConfig(
    seed_size=1000,
    pool_size=50000,
    test_size=5000,
    train_epochs=5,
    al_iterations=20,
    mc_dropout_p=0.2,
    num_mc_samples=30,
    train_batch_size=64,
    eval_batch_size=128,
    learning_rate=5e-5
)