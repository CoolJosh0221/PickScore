from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from tqdm.auto import tqdm
from transformers import CLIPProcessor

from active_learning.data.loaders import create_dataloader
from active_learning.models.base_model import BaseModel
from active_learning.models.model_baseline import CLIPModel
from .losses import pairwise_scores, soft_ce_from_pairs
from .eval import evaluate
from .utils import set_seed, make_run_dir, save_epoch, save_best_pointer


def build_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(model_class, device: str, **kwargs) -> BaseModel:
    """Use same constructor shape as your current CLIPModel"""
    assert issubclass(model_class, BaseModel)
    return model_class(**kwargs).to(device)


def build_processor(pretrained_model_name_or_path: str) -> CLIPProcessor:
    """Use HF processor to tokenize text and preprocess images"""
    return CLIPProcessor.from_pretrained(pretrained_model_name_or_path, use_fast=False)


def build_optimizer(
    model: torch.nn.Module, lr: float, wd: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def build_scaler(device: str) -> torch.amp.GradScaler:
    """AMP scaler; no effects on CPU"""
    return torch.amp.GradScaler(device=device, enabled=torch.cuda.is_available())


def build_loaders(out_dir: Path, batch_size: int, num_workers: int) -> Tuple[Any, Any]:
    train_loader = create_dataloader(out_dir, "train", batch_size, num_workers, True)
    valid_loader = create_dataloader(out_dir, "valid", batch_size, num_workers, False)
    return train_loader, valid_loader


def prepare_inputs(processor: CLIPProcessor, batch: Dict[str, Any], device: str):
    """Select essential features from data samples"""
    caps = batch["caption"]
    imgs0 = batch["image_0"]
    imgs1 = batch["image_1"]
    y0 = batch["label_0"].to(device)
    y1 = batch["label_1"].to(device)

    txt_in = processor(
        text=caps, padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(device)
    img0_in = processor(images=imgs0, return_tensors="pt").to(device)
    img1_in = processor(images=imgs1, return_tensors="pt").to(device)
    return txt_in, img0_in, img1_in, y0, y1


def loss_on_batch(
    model: CLIPModel, processor: CLIPProcessor, batch: Dict[str, Any], device: str
) -> torch.Tensor:
    """Forward one batch and return the soft CE loss"""
    txt_in, img0_in, img1_in, y0, y1 = prepare_inputs(processor, batch, device)
    t = model.get_text_features(**txt_in)
    i0 = model.get_image_features(**img0_in)
    i1 = model.get_image_features(**img1_in)
    s0, s1 = pairwise_scores(t, i0, i1, model.logit_scale)  # uses trainable logit scale
    return soft_ce_from_pairs(s0, s1, y0, y1)


def train_one_epoch(
    model: CLIPModel,
    processor: CLIPProcessor,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    epoch: int,
) -> float:
    # Standard train loop with AMP
    model.train()
    running, steps = 0.0, 0
    for batch in tqdm(loader, desc=f"epoch {epoch} [train]", leave=True):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, enabled=torch.cuda.is_available()):
            loss = loss_on_batch(model, processor, batch, device)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        running += loss.item()
    return running / max(steps, 1)


def validate_epoch(
    model: BaseModel, processor: CLIPProcessor, loader, device: str, tie_margin: float
) -> Dict[str, float]:
    """Delegates to evaluate()"""
    return evaluate(model, processor, loader, device, tie_margin=tie_margin)


def setup_training(
    out_dir: Path,
    *,
    pretrained_model_name_or_path: str,
    train_batch_size: int,
    num_workers: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
):
    """Build all state needed by training"""
    set_seed(seed)
    out_dir = Path(out_dir)
    device = build_device()
    model = build_model(
        CLIPModel, device, pretrained_model_name_or_path=pretrained_model_name_or_path
    )
    processor = build_processor(pretrained_model_name_or_path)
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    scaler = build_scaler(device)
    train_loader, valid_loader = build_loaders(out_dir, train_batch_size, num_workers)
    return {
        "out_dir": out_dir,
        "device": device,
        "model": model,
        "processor": processor,
        "optimizer": optimizer,
        "scaler": scaler,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
    }


def fit(
    out_dir: Path,
    *,
    pretrained_model_name_or_path: str,
    train_batch_size: int,
    num_workers: int,
    train_epochs: int,
    learning_rate: float,
    weight_decay: float,
    tie_margin: float = 0.05,
    seed: int = 42,
) -> None:
    """Full training orchestration; reusable inside active-learning outer loops"""
    state = setup_training(
        out_dir,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        train_batch_size=train_batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )

    device = state["device"]
    model = state["model"]
    processor = state["processor"]
    optimizer = state["optimizer"]
    scaler = state["scaler"]
    train_loader = state["train_loader"]
    valid_loader = state["valid_loader"]

    print(f"Running on device {device}")
    print("Setup finished")

    ckpt_root = make_run_dir(state["out_dir"])
    best_overall = -1.0
    best_path = None

    for epoch in range(1, train_epochs + 1):
        _ = train_one_epoch(
            model, processor, train_loader, optimizer, scaler, device, epoch
        )
        val = validate_epoch(model, processor, valid_loader, device, tie_margin)
        print(
            f"epoch {epoch} | val_loss {val['val_loss']:.4f} | "
            f"pref_acc {val['pref_acc']:.4f} | tie_acc {val['tie_acc']:.4f} | overall {val['overall_acc']:.4f}"
        )
        ep_dir = save_epoch(model, state["out_dir"], ckpt_root, epoch, val)
        if val["overall_acc"] > best_overall:
            best_overall = val["overall_acc"]
            best_path = ep_dir
            save_best_pointer(state["out_dir"], best_path)

    model.save(str(state["out_dir"] / "last"))
    if best_path is not None:
        model.save(str(best_path))
