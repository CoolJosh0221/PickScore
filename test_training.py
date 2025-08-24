import random
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import CLIPProcessor

from configs.experiment_configs import ExperimentConfig, FAST_PROTOTYPE
from active_learning.data_utils import create_dataloader
from active_learning.models.model_baseline import CLIPModel
from active_learning.models.model_mcdo import MCDropoutCLIPModel


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _l2n(x):  # [*,D] -> unit vectors
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def calc_loss(
    text_features,  # [B,D]
    image_0_features,  # [B,D]
    image_1_features,  # [B,D]
    label_0,  # [B]  1 if img0 preferred, 0.5 tie, 0 otherwise
    label_1,  # [B]  1 if img1 preferred, 0.5 tie, 0 otherwise
    *,
    logit_scale=None,  # scalar; use model.logit_scale.exp() if you have the model
):
    device = text_features.device
    B = text_features.size(0)

    # 1) normalize features
    t = _l2n(text_features)
    i0 = _l2n(image_0_features)
    i1 = _l2n(image_1_features)

    # 2) CLIP logits for each text against its two images (diagonals only)
    s = float(logit_scale) if logit_scale is not None else 1.0
    all_img = torch.cat([i0, i1], dim=0)  # [2B,D]
    logits_t2i = s * (t @ all_img.t())  # [B,2B]
    t0_logits, t1_logits = logits_t2i.split(B, dim=-1)  # [B,B] each
    idx = torch.arange(B, device=device)
    logits = torch.stack(
        [
            t0_logits[idx, idx],  # [B]
            t1_logits[idx, idx],
        ],
        dim=-1,
    )  # -> [B,2]

    # 3) per-example CE using preference labels (ties handled)
    # CE(logits, class0) and CE(logits, class1)
    ce0 = F.cross_entropy(
        logits, torch.zeros(B, device=device, dtype=torch.long), reduction="none"
    )
    ce1 = F.cross_entropy(
        logits, torch.ones(B, device=device, dtype=torch.long), reduction="none"
    )
    loss = label_0 * ce0 + label_1 * ce1  # [B]
    # tie correction so perfect tie gives zero ideal loss
    tie = (label_0 == label_1).to(logits.dtype)
    loss = loss + tie * torch.log(torch.tensor(0.5, device=device, dtype=logits.dtype))

    loss = loss.mean()

    return loss


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
    processor = CLIPProcessor.from_pretrained(cfg.pretrained_model_name_or_path)

    for epoch in range(1, cfg.train_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            caption = batch["caption"]
            x0, x1 = batch["image_0"], batch["image_1"]
            y0, y1 = batch["label_0"], batch["label_1"]

            image_inputs_0 = processor(
                images=x0,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            image_inputs_1 = processor(
                images=x1,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            text_inputs = processor(
                text=caption,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=torch.cuda.is_available()):
                text_feat = model(text_inputs=text_inputs)
                img_feat_0 = model(image_inputs=image_inputs_0)
                img_feat_1 = model(image_inputs=image_inputs_1)
                s0, s1 = text_feat + img_feat_0, text_feat + img_feat_1
                loss = calc_loss(s0, s1, y0, y1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        val_acc = evaluate(model, valid_loader, device)
