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
    assert issubclass(model_class, BaseModel)
    return model_class(**kwargs).to(device)


def build_processor(pretrained_model_name_or_path: str) -> CLIPProcessor:
    return CLIPProcessor.from_pretrained(pretrained_model_name_or_path, use_fast=False)


def build_optimizer(
    model: torch.nn.Module, lr: float, wd: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def build_scaler(device: str) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device=device)


def build_loaders(
    out_dir: Path, batch_size: int, num_workers: int, processor
) -> Tuple[Any, Any]:
    train_loader = create_dataloader(
        data_dir=out_dir,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        processor=processor,
        shuffle=True,
    )
    valid_loader = create_dataloader(
        data_dir=out_dir,
        split="valid",
        batch_size=batch_size,
        num_workers=num_workers,
        processor=processor,
        shuffle=False,
    )
    return train_loader, valid_loader


def prepare_inputs(processor: CLIPProcessor, batch: Dict[str, Any], device: str):
    """
    Make model-ready inputs on the right device.
    Images may be tensors (already collated) or PILs; handle both.
    """
    caps = batch["caption"]
    imgs0 = batch["image_0"]
    imgs1 = batch["image_1"]
    y0 = batch["label_0"].to(device, non_blocking=True)
    y1 = batch["label_1"].to(device, non_blocking=True)

    # Text -> token ids on device
    txt = processor(
        text=caps, padding=True, truncation=True, max_length=77, return_tensors="pt"
    )
    txt = {k: v.to(device, non_blocking=True) for k, v in txt.items()}

    # Images -> dicts compatible with HF CLIP .get_image_features(**image_inputs)
    if isinstance(imgs0, torch.Tensor):
        img0_in = {"pixel_values": imgs0.to(device, non_blocking=True)}
    else:
        img0_in = processor(images=imgs0, return_tensors="pt")
        img0_in = {k: v.to(device, non_blocking=True) for k, v in img0_in.items()}

    if isinstance(imgs1, torch.Tensor):
        img1_in = {"pixel_values": imgs1.to(device, non_blocking=True)}
    else:
        img1_in = processor(images=imgs1, return_tensors="pt")
        img1_in = {k: v.to(device, non_blocking=True) for k, v in img1_in.items()}

    return txt, img0_in, img1_in, y0, y1


def loss_on_batch(
    model: CLIPModel, processor: CLIPProcessor, batch: Dict[str, Any], device: str
) -> torch.Tensor:
    txt, img0_in, img1_in, y0, y1 = prepare_inputs(processor, batch, device)
    t = model.get_text_features(**txt)
    i0 = model.get_image_features(**img0_in)
    i1 = model.get_image_features(**img1_in)
    s0, s1 = pairwise_scores(t, i0, i1, model.logit_scale)
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
    model.train()
    running, steps = 0.0, 0

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    for batch in tqdm(loader, desc=f"epoch {epoch} [train]", leave=True):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device):
            loss = loss_on_batch(model, processor, batch, device)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        running += loss.item()

    return running / max(steps, 1)


def validate_epoch(
    model: BaseModel,
    processor: CLIPProcessor,
    loader,
    device: str,
    tie_margin: float,
) -> Dict[str, float]:
    """Delegate to evaluate()."""
    return evaluate(model, processor, loader, device, tie_margin=tie_margin)


def setup_training(
    out_dir: Path,
    checkpoint_dir: Path,
    *,
    pretrained_model_name_or_path: str,
    train_batch_size: int,
    num_workers: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
):
    set_seed(seed)
    out_dir = Path(out_dir)
    device = build_device()
    model = build_model(
        CLIPModel, device, pretrained_model_name_or_path=pretrained_model_name_or_path
    )
    processor = build_processor(pretrained_model_name_or_path)
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    scaler = build_scaler(device)
    train_loader, valid_loader = build_loaders(out_dir, train_batch_size, num_workers, processor)
    return {
        "out_dir": out_dir,
        "checkpoint_dir": checkpoint_dir,
        "device": device,
        "model": model,
        "processor": processor,
        "optimizer": optimizer,
        "scaler": scaler,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
    }
