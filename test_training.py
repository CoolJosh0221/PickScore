import random
from pathlib import Path

import torch
import torch.nn.functional as F

from configs.experiment_configs import ExperimentConfig, FAST_PROTOTYPE
from active_learning.data_utils import create_dataloader
from active_learning.models.model_baseline import CLIPModel
from active_learning.models.model_mcdo import MCDropoutCLIPModel

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calc_loss(predictions0, predictions1, label0, label1):
    y = torch.sign(label0 - label1)  # +1 if img0 preferred
    m = y != 0
    if not m.any():
        return predictions0.sum() * 0.0
    logits = (predictions0[m] - predictions1[m]) * y[m]
    return F.softplus(-logits).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    for batch in loader:
        ...


def train(
    cfg: ExperimentConfig,
    out_dir: Path = Path("./runs"),
):
    """Train model / Seed model with a subset of data before entering AL training loop"""
    out_dir = Path(out_dir)

    train_loader = create_dataloader(
        out_dir, "train", cfg.train_batch_size, 4, True, cfg.image_size
    )
    valid_loader = create_dataloader(
        out_dir, "valid", cfg.train_batch_size, 4, True, cfg.image_size
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel(FAST_PROTOTYPE).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler(device=device, enabled=torch.cuda.is_available())

    for epoch in range(1, cfg.train_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            caption = batch["caption"]
            , x1 = batch["image_0"].cuda(), batch["image_1"].cuda()
            labels0, labels1 = batch["label_0"].cuda(), batch["label_1"].cuda()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=torch.cuda.is_available()):
                predictions0 = model(images0)
                predictions1 = model(images1)
                loss = calc_loss(predictions0, predictions1, labels0, labels1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        val_acc = evaluate(model, valid_loader, device)
