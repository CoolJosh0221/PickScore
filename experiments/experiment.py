from pathlib import Path
from typing import Optional

from attrs import asdict

import wandb
from active_learning.data.sampling import create_active_learning_splits
from active_learning.training.active_learning import run_active_learning
from configs.experiment_configs import FAST_PROTOTYPE, MEDIUM_SCALE, ExperimentConfig


def run_experiment(
    config: ExperimentConfig,
    acquisition_strategy: str,
    run_id: int,
    out_dir: Optional[str | Path] = None,
) -> None:
    if out_dir is None:
        out_dir = f"out/{acquisition_strategy}/{run_id}/"
    out_dir = Path(out_dir)

    experiment_name = f"{acquisition_strategy}-{run_id}"

    with wandb.init(
        project=config.wandb_project,
        name=experiment_name,
        config={
            **asdict(config),
            "acquisition_strategy": acquisition_strategy,
            "run_id": run_id,
        },
        reinit="finish_previous",
    ) as run:
        assert wandb.run is not None
        if not out_dir.exists():
            create_active_learning_splits(
                out_dir,
                dataset_name=config.dataset_name,
                seed_size=config.seed_size,
                pool_size=config.pool_size,
                test_size=config.test_size,
            )

        run_active_learning(
            out_dir,
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            train_batch_size=config.train_batch_size,
            num_workers=config.num_workers,
            train_epochs=config.train_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            tie_margin=config.tie_margin,
            al_iterations=config.al_iterations,
            acquisition_batch_size=config.acquisition_batch_size,
            acquisition_strategy=acquisition_strategy,
            num_mc_samples=config.num_mc_samples,
            mc_dropout_p=config.mc_dropout_p,
            experiment_name=experiment_name,
            seed=config.seed,
        )


def main() -> None:
    # run_experiment(FAST_PROTOTYPE, acquisition_strategy="entropy", run_id=1)
    run_experiment(MEDIUM_SCALE, acquisition_strategy="entropy", run_id=2)


if __name__ == "__main__":
    main()
