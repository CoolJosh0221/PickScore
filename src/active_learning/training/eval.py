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
    loss_sum = 0.0
    batches = 0
    for b in tqdm(loader, desc="valid", leave=False):
        caps = b["caption"]
        imgs0 = b["image_0"]
        imgs1 = b["image_1"]
        y0 = b["label_0"].to(device)
        y1 = b["label_1"].to(device)
        txt_in = processor(
            text=caps, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)
        img0_in = processor(images=imgs0, return_tensors="pt").to(device)
        img1_in = processor(images=imgs1, return_tensors="pt").to(device)
        t = model.get_text_features(**txt_in)
        i0 = model.get_image_features(**img0_in)
        i1 = model.get_image_features(**img1_in)
        s0, s1 = pairwise_scores(t, i0, i1, model.logit_scale)
        loss = soft_ce_from_pairs(s0, s1, y0, y1)
        loss_sum += loss.item()
        batches += 1
        tracker.update(s0, s1, y0, y1)
    metrics = tracker.compute()
    metrics["val_loss"] = loss_sum / max(batches, 1)
    return metrics
