import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor

from active_learning.data.loaders import create_dataloader
from active_learning.models.base_model import BaseModel
from active_learning.models.model_baseline import CLIPModel

seed: int = 45510
ds_dir: Path = Path("baselines/dataset/")
result_file: Path = Path("baselines/lower_bound/result.json")
tie_margin = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set seeds
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize results structure
results = {
    "timestamp": datetime.now().isoformat(),
    "seed": seed,
    "tie_margin": tie_margin,
    "device": device,
    "models": [],
}


@torch.no_grad()
def calc_probability_distribution(s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
    logits = torch.stack([s0, s1], dim=1)  # [B,2]
    return F.softmax(logits, dim=1)


@torch.no_grad()
def snap_prediction(probs: torch.Tensor, tie_margin: float) -> torch.Tensor:
    """Snap probs [B,2] to one of [1,0], [0,1], [0.5,0.5]."""
    diff = probs[:, 0] - 0.5
    tie = diff.abs() <= tie_margin
    out = torch.zeros_like(probs)
    out[tie] = 0.5
    out[~tie, 0] = (probs[~tie, 0] > probs[~tie, 1]).float()
    out[~tie, 1] = 1.0 - out[~tie, 0]
    return out


pretrained_models = [
    # "yuvalkirstain/PickScore_v1",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
]

for pretrained_model in pretrained_models:
    print(f"\nEvaluating model: {pretrained_model}")

    # Initialize model results
    model_results = {
        "model_name": pretrained_model,
        "batches": [],
        "total_accuracy": 0,
        "total_samples": 0,
        "total_matches": 0,
    }

    model: BaseModel = CLIPModel(
        pretrained_model_name_or_path=pretrained_model,
    )
    model.eval().to(device)
    processor = CLIPProcessor.from_pretrained(pretrained_model)

    dl = create_dataloader(
        ds_dir,
        split="test",
        batch_size=48,
        num_workers=4,
        processor=processor,
        shuffle=False,
    )

    total_matches, total_samples = 0, 0

    for i, batch in enumerate(dl):
        img0 = batch["image_0"].to(device)  # [B,C,H,W]
        img1 = batch["image_1"].to(device)  # [B,C,H,W]
        captions = batch["caption"]
        imgs = torch.cat([img0, img1], dim=0)  # [2B,C,H,W]

        with torch.no_grad():
            image_feats = model.get_image_features(imgs)  # [2B,d]
            image_feats = F.normalize(image_feats, dim=-1)
            i0, i1 = image_feats.chunk(2, dim=0)  # [B,d] each

            text_inputs = processor(
                text=captions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            text_feats = model.get_text_features(**text_inputs)  # [B,d]
            text_feats = F.normalize(text_feats, dim=-1)

            # Calculate similarities
            s0 = (text_feats * i0).sum(dim=-1)  # [B]
            s1 = (text_feats * i1).sum(dim=-1)  # [B]

            # Apply logit scale if available
            if hasattr(model, "logit_scale"):
                scale = model.logit_scale.exp()
            else:
                scale = 1.0
            s0, s1 = s0 * scale, s1 * scale

            probs = calc_probability_distribution(s0, s1)  # [B,2]
            preds = snap_prediction(probs, tie_margin)

        labels = torch.stack([batch["label_0"], batch["label_1"]], dim=1).to(device)
        matches = (preds == labels).all(dim=1)  # [B]
        batch_matches = matches.sum().item()
        batch_samples = matches.shape[0]
        batch_acc = matches.float().mean().item()

        total_matches += batch_matches
        total_samples += batch_samples

        # Store batch results
        model_results["batches"].append(
            {
                "batch_id": i,
                "accuracy": batch_acc,
                "samples": batch_samples,
                "matches": batch_matches,
            }
        )

        print(f"Batch {i}'s accuracy: {batch_acc:.2%}")

    overall_acc = total_matches / total_samples

    # Store final model results
    model_results["total_accuracy"] = overall_acc
    model_results["total_samples"] = total_samples
    model_results["total_matches"] = total_matches

    results["models"].append(model_results)

    print(f"Overall accuracy for model {pretrained_model}: {overall_acc:.2%}")

print(f"\nSaving results to {result_file}")
with open(result_file, "w") as f:
    json.dump(results, f, indent=4)

print("Evaluation complete!")

# Summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
for model_result in results["models"]:
    print(f"{model_result['model_name']}: {model_result['total_accuracy']:.2%}")
