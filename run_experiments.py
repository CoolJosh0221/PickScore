"""Run SMALL_SCALE experiments: Random vs BALD."""

from experiments.experiment import run_experiment
from configs.experiment_configs import SMALL_SCALE


def main():
    print("=" * 50)
    print("Running SMALL_SCALE Random (run_id=2)")
    print("=" * 50)
    run_experiment(SMALL_SCALE, acquisition_strategy="random", run_id=2)

    print("=" * 50)
    print("Running SMALL_SCALE BALD (run_id=2)")
    print("=" * 50)
    run_experiment(SMALL_SCALE, acquisition_strategy="bald", run_id=2)

    print("=" * 50)
    print("All experiments complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
