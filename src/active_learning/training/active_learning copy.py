from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil
import torch
from tqdm.auto import tqdm
from datasets import Dataset as HFDataset, concatenate_datasets

from active_learning.data.loaders import create_dataloader
from active_learning.training.train import (
    build_device,
    build_model,
    build_processor,
    build_optimizer,
    build_scaler,
    train_one_epoch,
    validate_epoch,
    prepare_inputs,
)
from active_learning.training.losses import pairwise_scores
from active_learning.training.utils import (
    set_seed,
    make_run_dir,
    save_epoch,
    save_best_pointer,
)
from active_learning.training.acquisitions import make_acquisition, Acquisition
from active_learning.models.model_mcdo import MCDropoutCLIPModel


def _load_split(out_dir: Path, split: str) -> HFDataset:
    return HFDataset.load_from_disk(str(Path(out_dir) / split))


def _save_split(ds: HFDataset, out_dir: Path, split: str) -> None:
    split_dir = Path(out_dir) / split
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(split_dir))


def _move_from_pool_to_seed(out_dir: Path, pool_indices: List[int]) -> None:
    # Move acquired data points from pool to already-labeled
    if not pool_indices:
        return
    pool_ds = _load_split(out_dir, "pool")
    seed_ds = _load_split(out_dir, "seed")
    sel = sorted(set(pool_indices))
    keep = [i for i in range(len(pool_ds)) if i not in sel]
    gained = pool_ds.select(sel)
    remain = pool_ds.select(keep) if keep else pool_ds.select([])
    new_seed = concatenate_datasets([seed_ds, gained]) if len(seed_ds) else gained
    _save_split(new_seed, out_dir, "seed")
    _save_split(remain, out_dir, "pool")


@torch.no_grad()
def predict_pool_probs(model, processor, loader, device: str) -> torch.Tensor:
    """Deterministic probs per item; returns [N,2]."""
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout = False
    model.eval()
    batch_probs = []
    for batch in tqdm(loader, desc="pool-scan", leave=True):
        txt_in, img0_in, img1_in, _, _ = prepare_inputs(processor, batch, device)
        t = model.get_text_features(**txt_in)
        i0 = model.get_image_features(**img0_in)
        i1 = model.get_image_features(**img1_in)
        s0, s1 = pairwise_scores(t, i0, i1, model.logit_scale)
        probs = torch.stack([s0, s1], dim=-1).softmax(-1)  # [B,2]
        batch_probs.append(probs.cpu())
    return torch.cat(batch_probs, dim=0) if batch_probs else torch.empty(0, 2)


@torch.no_grad()
def predict_pool_mc_probs(
    model, processor, loader, device: str, num_samples: int
) -> torch.Tensor:
    """MC-Dropout prob stack; returns [T,N,2]."""
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout = True
    else:
        raise ValueError(
            "Model must have attribute `enable_mc_dropout` to run MC inference"
        )
    model.eval()
    samples = [
        predict_pool_probs(model, processor, loader, device) for _ in range(num_samples)
    ]
    return torch.stack(samples, dim=0)  # [T,N,2]


def al_iteration(
    out_dir: Path,
    *,
    model,
    processor,
    device: str,
    optimizer,
    scaler,
    train_epochs: int,
    train_batch_size: int,
    num_workers: int,
    tie_margin: float,
    acquisition_batch_size: int,
    acquisition_strategy: str,
    num_mc_samples: int,
    acq_fit_kwargs: Optional[Dict[str, Any]] = None,  # stateful methods
    acq_score_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # train on current seed
    seed_loader = create_dataloader(
        out_dir, "seed", train_batch_size, num_workers, True
    )
    valid_loader = create_dataloader(
        out_dir, "valid", train_batch_size, num_workers, False
    )
    for epoch in range(1, train_epochs + 1):
        _ = train_one_epoch(
            model, processor, seed_loader, optimizer, scaler, device, epoch
        )

    # validate on fixed valid
    val = validate_epoch(model, processor, valid_loader, device, tie_margin)

    # score pool
    pool_loader = create_dataloader(
        out_dir,
        "pool",
        batch_size=max(64, train_batch_size),
        num_workers=num_workers,
        shuffle=False,
    )
    N_pool = len(pool_loader.dataset)
    if N_pool == 0:
        return {"val": val, "acquired": 0, "pool_before": 0, "pool_after": 0}

    acq: Acquisition = make_acquisition(acquisition_strategy)

    # optional fit step for stateful strategies (no-op for stateless ones)
    acq.fit(**(acq_fit_kwargs or {}))

    if getattr(acq, "requires_mc", False):
        if num_mc_samples <= 1:
            raise ValueError(f"{acquisition_strategy} requires num_mc_samples > 1")
        mc_probs = predict_pool_mc_probs(
            model, processor, pool_loader, device, num_samples=num_mc_samples
        )  # [T,N,2]
        mean_probs = mc_probs.mean(dim=0)  # [N,2]
        scores = acq.score(
            mean_probs=mean_probs, mc_probs=mc_probs, **(acq_score_kwargs or {})
        )  # [N]
    else:
        mean_probs = predict_pool_probs(model, processor, pool_loader, device)  # [N,2]
        scores = acq.score(mean_probs=mean_probs, **(acq_score_kwargs or {}))  # [N]

    k = min(acquisition_batch_size, N_pool)
    selected = torch.topk(
        scores, k=k, largest=True
    ).indices.tolist()  # indices match pool order (shuffle=False)
    _move_from_pool_to_seed(out_dir, selected)

    return {"val": val, "acquired": k, "pool_before": N_pool, "pool_after": N_pool - k}


# full AL driver


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
    seed: int = 42,
) -> None:
    set_seed(seed)
    out_dir = Path(out_dir)
    device = build_device()
    print(f"Running on device {device}")

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
        info = al_iteration(
            out_dir,
            model=model,
            processor=processor,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            train_epochs=train_epochs,
            train_batch_size=train_batch_size,
            num_workers=num_workers,
            tie_margin=tie_margin,
            acquisition_batch_size=acquisition_batch_size,
            acquisition_strategy=acquisition_strategy,
            num_mc_samples=num_mc_samples,
        )
        val = info["val"]
        print(
            f"AL iter {it} | val_loss {val['val_loss']:.4f} | "
            f"pref_acc {val['pref_acc']:.4f} | tie_acc {val['tie_acc']:.4f} | "
            f"overall {val['overall_acc']:.4f} | acquired {info['acquired']}"
        )
        ep_dir = save_epoch(model, out_dir, ckpt_root, it, val)
        if val["overall_acc"] > best_overall:
            best_overall = val["overall_acc"]
            best_path = ep_dir
            save_best_pointer(out_dir, best_path)
        if info["pool_after"] == 0:
            break

    model.save(str(out_dir / "last"))
    if best_path is not None:
        model.save(str(best_path))
