from typing import Dict
import torch
from tqdm.auto import tqdm
from transformers import CLIPProcessor
from .losses import pairwise_scores, soft_ce_from_pairs
from .metrics import PrefMetricTracker


@torch.no_grad()
def evaluate(
    model, processor: CLIPProcessor, loader, device: str, tie_margin: float = 0.05
) -> Dict[str, float]:
    model.eval()
    tracker = PrefMetricTracker(tie_margin)
    loss_sum, batches = 0.0, 0
    for b in tqdm(loader, desc="valid", leave=False):
        imgs0 = b["image_0"].to(device, non_blocking=True)
        imgs1 = b["image_1"].to(device, non_blocking=True)
        y0 = b["label_0"].to(device, non_blocking=True)
        y1 = b["label_1"].to(device, non_blocking=True)
        txt = processor(
            text=b["caption"],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        txt = {k: v.to(device, non_blocking=True) for k, v in txt.items()}
        t = model.get_text_features(**txt)
        i0 = model.get_image_features(pixel_values=imgs0)
        i1 = model.get_image_features(pixel_values=imgs1)
        s0, s1 = pairwise_scores(t, i0, i1, model.logit_scale)
        loss = soft_ce_from_pairs(s0, s1, y0, y1)
        loss_sum += loss.item()
        batches += 1
        tracker.update(s0, s1, y0, y1)
    metrics = tracker.compute()
    metrics["val_loss"] = loss_sum / max(batches, 1)
    return metrics
