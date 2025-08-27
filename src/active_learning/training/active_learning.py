from pathlib import Path
from typing import Any, Dict, Optional

import torch
from tqdm.auto import tqdm
import wandb

from active_learning.data.manager import ActiveLearningDataManager
from active_learning.models.model_mcdo import MCDropoutCLIPModel
from active_learning.training.acquisitions import Acquisition, make_acquisition
from active_learning.training.losses import pairwise_scores
from active_learning.training.train import (
    build_device,
    build_model,
    build_optimizer,
    build_processor,
    build_scaler,
    prepare_inputs,
    train_one_epoch,
    validate_epoch,
)
from active_learning.training.utils import (
    make_run_dir,
    save_best_pointer,
    save_epoch,
    set_seed,
)


@torch.no_grad()
def predict_pool_probs(model, processor, loader, device: str) -> torch.Tensor:
    """Deterministic pool probabilities; returns [N,2] on CPU."""
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout = False
    model.eval()

    use_amp = torch.cuda.is_available() and device.startswith("cuda")
    probs_all = []

    for batch in tqdm(loader, desc="pool-scan", leave=True):
        txt_in, img0_in, img1_in, _, _ = prepare_inputs(processor, batch, device)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            t = model.get_text_features(**txt_in)
            i0 = model.get_image_features(**img0_in)
            i1 = model.get_image_features(**img1_in)
            s0, s1 = pairwise_scores(t, i0, i1, model.logit_scale)
            probs = torch.stack([s0, s1], dim=-1).softmax(-1)
        probs_all.append(probs)

    if not probs_all:
        return torch.empty(0, 2)

    return torch.cat(probs_all, dim=0).detach().cpu()


@torch.no_grad()
def predict_pool_mc_probs(
    model, processor, loader, device: str, num_samples: int
) -> torch.Tensor:
    """MC-Dropout via simple sequential passes; returns [T,N,2] on CPU."""
    if not hasattr(model, "enable_mc_dropout"):
        raise ValueError(
            "Model must have attribute `enable_mc_dropout` for MC inference"
        )

    model.enable_mc_dropout = True
    model.eval()

    samples = [
        predict_pool_probs(model, processor, loader, device) for _ in range(num_samples)
    ]
    return torch.stack(samples, dim=0)


def al_iteration(
    data_manager: ActiveLearningDataManager,
    *,
    model,
    processor,
    device: str,
    optimizer,
    scaler,
    train_epochs: int,
    train_batch_size: int,
    tie_margin: float,
    acquisition_batch_size: int,
    acquisition_strategy: str,
    num_mc_samples: int,
    acq_fit_kwargs: Optional[Dict[str, Any]] = None,
    acq_score_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    labeled_loader = data_manager.get_labeled_dataloader(
        batch_size=train_batch_size, shuffle=True
    )
    validation_loader = data_manager.get_validation_dataloader(
        batch_size=train_batch_size
    )

    for epoch in range(1, train_epochs + 1):
        _ = train_one_epoch(
            model, processor, labeled_loader, optimizer, scaler, device, epoch
        )

    val = validate_epoch(model, processor, validation_loader, device, tie_margin)

    pool_loader = data_manager.get_unlabeled_dataloader(
        batch_size=max(64, train_batch_size), shuffle=False
    )

    N_pool = len(pool_loader.dataset)
    if N_pool == 0:
        return {"val": val, "acquired": 0, "pool_before": 0, "pool_after": 0}

    acq: Acquisition = make_acquisition(acquisition_strategy)
    acq.fit(**(acq_fit_kwargs or {}))

    if getattr(acq, "requires_mc", False):
        if num_mc_samples <= 1:
            raise ValueError(f"{acquisition_strategy} requires num_mc_samples > 1")
        mc_probs = predict_pool_mc_probs(
            model, processor, pool_loader, device, num_samples=num_mc_samples
        )
        mean_probs = mc_probs.mean(dim=0)
        scores = acq.score(
            mean_probs=mean_probs, mc_probs=mc_probs, **(acq_score_kwargs or {})
        )
    else:
        mean_probs = predict_pool_probs(model, processor, pool_loader, device)
        scores = acq.score(mean_probs=mean_probs, **(acq_score_kwargs or {}))

    k = min(acquisition_batch_size, N_pool)
    selected_indices = torch.topk(scores, k=k, largest=True).indices.tolist()

    unlabeled_pool_indices = data_manager.get_unlabeled_pool_indices()
    actual_pool_indices = [unlabeled_pool_indices[i] for i in selected_indices]

    data_manager.label_samples(actual_pool_indices)

    return {"val": val, "acquired": k, "pool_before": N_pool, "pool_after": N_pool - k}


def run_active_learning(
    out_dir: Path,
    *,
    pretrained_model_name_or_path: str,
    train_batch_size: int,
    num_workers: int,
    train_epochs: int,
    learning_rate: float,
    weight_decay: float,
    tie_margin: float,
    al_iterations: int,
    acquisition_batch_size: int,
    acquisition_strategy: str = "bald",
    num_mc_samples: int = 20,
    mc_dropout_p: Optional[float] = None,
    experiment_name: str = "al_experiment",
    seed: int = 42,
) -> None:
    set_seed(seed)
    out_dir = Path(out_dir)
    device = build_device()
    print(f"Running on device {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    data_manager = ActiveLearningDataManager(
        data_root=str(out_dir), experiment_name=experiment_name, num_workers=num_workers
    )

    print("Initial stats:", data_manager.get_stats())

    assert mc_dropout_p is not None
    model = build_model(
        MCDropoutCLIPModel,
        device,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        mc_dropout_p=mc_dropout_p,
    )
    processor = build_processor(pretrained_model_name_or_path)
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    scaler = build_scaler(device)
    print("Setup finished")

    ckpt_root = make_run_dir(out_dir)
    best_overall = -1.0
    best_path = None

    for it in range(1, al_iterations + 1):
        print(f"\n--- AL Iteration {it} ---")

        info = al_iteration(
            data_manager,
            model=model,
            processor=processor,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            train_epochs=train_epochs,
            train_batch_size=train_batch_size,
            tie_margin=tie_margin,
            acquisition_batch_size=acquisition_batch_size,
            acquisition_strategy=acquisition_strategy,
            num_mc_samples=num_mc_samples,
        )

        val = info["val"]
        stats = data_manager.get_stats()

        print(
            f"AL iter {it} | val_loss {val['val_loss']:.4f} | "
            f"pref_acc {val['pref_acc']:.4f} | tie_acc {val['tie_acc']:.4f} | "
            f"overall {val['overall_acc']:.4f} | acquired {info['acquired']} | "
            f"progress {stats['progress']:.3f}"
        )

        assert wandb.run is not None
        wandb.log(
            {
                "iteration": it,
                "val_loss": val["val_loss"],
                "pref_acc": val["pref_acc"],
                "tie_acc": val["tie_acc"],
                "overall_acc": val["overall_acc"],
                "acquired": info["acquired"],
                "progress": stats["progress"],
                "labeled_samples": stats["total_labeled"],
                "pool_remaining": stats["pool_unlabeled"],
            }
        )

        ep_dir = save_epoch(model, out_dir, ckpt_root, it, val)
        if val["overall_acc"] > best_overall:
            best_overall = val["overall_acc"]
            best_path = ep_dir
            save_best_pointer(out_dir, best_path)

        data_manager.next_iteration()

        if not data_manager.has_unlabeled_data():
            print("No more unlabeled data available. Stopping AL.")
            break

    last_dir = out_dir / "last"
    last_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(last_dir / "model.pth"))
    if best_path is not None:
        model.save(str(best_path / "model.pth"))

    final_stats = data_manager.get_stats()
    print(f"\nFinal AL Statistics:")
    print(f"Total iterations: {final_stats['iteration']}")
    print(f"Total labeled: {final_stats['total_labeled']}")
    print(f"Progress: {final_stats['progress']:.3f}")
