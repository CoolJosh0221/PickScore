"""Data collection utilities for calibration analysis."""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def collect_mc_predictions(model, processor, loader, device: str, num_mc_samples: int = 15):
    """
    Collect MC Dropout predictions.

    Returns:
        mc_probs: [T, N, K] probabilities from T stochastic forward passes
        mean_probs: [N, K] mean probabilities
        labels: [N, K] ground truth labels
    """
    model.enable_mc_dropout = True
    model.eval()

    all_mc_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Collecting MC predictions (T={num_mc_samples})"):
            imgs0 = batch["image_0"].to(device, non_blocking=True)
            imgs1 = batch["image_1"].to(device, non_blocking=True)
            y0 = batch["label_0"]
            y1 = batch["label_1"]

            txt = processor(
                text=batch["caption"],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            txt = {k: v.to(device, non_blocking=True) for k, v in txt.items()}

            batch_probs = []
            for _ in range(num_mc_samples):
                t = model.get_text_features(**txt)
                i0 = model.get_image_features(pixel_values=imgs0)
                i1 = model.get_image_features(pixel_values=imgs1)

                t = F.normalize(t, dim=-1)
                i0 = F.normalize(i0, dim=-1)
                i1 = F.normalize(i1, dim=-1)
                scale = model.logit_scale.exp()
                s0 = (t * i0).sum(-1) * scale
                s1 = (t * i1).sum(-1) * scale

                logits = torch.stack([s0, s1], dim=-1)
                probs = F.softmax(logits, dim=-1)
                batch_probs.append(probs.cpu())

            mc_probs = torch.stack(batch_probs, dim=0)  # [T, B, 2]
            labels = torch.stack([y0, y1], dim=-1)

            all_mc_probs.append(mc_probs)
            all_labels.append(labels)

    model.enable_mc_dropout = False

    mc_probs = torch.cat(all_mc_probs, dim=1)  # [T, N, 2]
    labels = torch.cat(all_labels, dim=0)  # [N, 2]
    mean_probs = mc_probs.mean(dim=0)  # [N, 2]

    return mc_probs, mean_probs, labels


def collect_mc_logits(model, processor, loader, device: str, num_mc_samples: int = 15):
    """
    Collect raw MC Dropout logits (before softmax).

    Returns:
        mc_logits: [T, N, K] logits from T stochastic forward passes
        labels: [N, K] ground truth labels
    """
    model.enable_mc_dropout = True
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Collecting MC logits (T={num_mc_samples})"):
            imgs0 = batch["image_0"].to(device, non_blocking=True)
            imgs1 = batch["image_1"].to(device, non_blocking=True)
            y0 = batch["label_0"]
            y1 = batch["label_1"]

            txt = processor(
                text=batch["caption"],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            txt = {k: v.to(device, non_blocking=True) for k, v in txt.items()}

            batch_logits = []
            for _ in range(num_mc_samples):
                t = model.get_text_features(**txt)
                i0 = model.get_image_features(pixel_values=imgs0)
                i1 = model.get_image_features(pixel_values=imgs1)

                t = F.normalize(t, dim=-1)
                i0 = F.normalize(i0, dim=-1)
                i1 = F.normalize(i1, dim=-1)
                scale = model.logit_scale.exp()
                s0 = (t * i0).sum(-1) * scale
                s1 = (t * i1).sum(-1) * scale

                logits = torch.stack([s0, s1], dim=-1)
                batch_logits.append(logits.cpu())

            mc_logits = torch.stack(batch_logits, dim=0)  # [T, B, 2]
            labels = torch.stack([y0, y1], dim=-1)

            all_logits.append(mc_logits)
            all_labels.append(labels)

    model.enable_mc_dropout = False

    mc_logits = torch.cat(all_logits, dim=1)  # [T, N, 2]
    labels = torch.cat(all_labels, dim=0)  # [N, 2]

    return mc_logits, labels
