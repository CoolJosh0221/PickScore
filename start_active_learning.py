from pathlib import Path
from active_learning.training.active_learning import run_active_learning

import configs.experiment_configs as experiment_configs


def main() -> None:
    cfg = experiment_configs.SMALL_SCALE
    out_dir = Path("./runs/quick/")
    if not out_dir.exists():
        raise FileNotFoundError(
            "Make sure that the directory containing the data exists"
        )

    run_active_learning(
        out_dir=out_dir,
        pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
        train_batch_size=cfg.train_batch_size,
        num_workers=cfg.num_workers,
        train_epochs=cfg.train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        tie_margin=cfg.tie_margin,
        al_iterations=cfg.al_iterations,
        acquisition_batch_size=cfg.acquisition_batch_size,
        acquisition_strategy=cfg.acquisition_strategy,  # "bald" | "entropy" | "least_confidence" | "margin" | "tie"
        num_mc_samples=cfg.num_mc_samples,
        mc_dropout_p=cfg.mc_dropout_p,
        seed=cfg.seed,
    )


if __name__ == "__main__":
    main()


# prof_run.py
# from pathlib import Path
# from torch.profiler import profile, ProfilerActivity
# from active_learning.training.active_learning import run_active_learning
# import configs.experiment_configs as experiment_configs

# cfg = experiment_configs.FAST_PROTOTYPE
# out_dir = Path("./runs/quick")

# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     record_shapes=True,
#     profile_memory=True,
# ) as prof:
#     run_active_learning(
#         out_dir=out_dir,
#         pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
#         train_batch_size=cfg.train_batch_size,
#         num_workers=cfg.num_workers,
#         train_epochs=cfg.train_epochs,
#         learning_rate=cfg.learning_rate,
#         weight_decay=cfg.weight_decay,
#         tie_margin=cfg.tie_margin,
#         al_iterations=cfg.al_iterations,
#         acquisition_batch_size=cfg.acquisition_batch_size,
#         acquisition_strategy=cfg.acquisition_strategy,
#         num_mc_samples=cfg.num_mc_samples,
#         mc_dropout_p=cfg.mc_dropout_p,
#         seed=cfg.seed,
#     )

# prof.export_chrome_trace("prof_traces/whole.json")
# print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
