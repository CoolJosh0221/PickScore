import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor

from active_learning.data.loaders import create_dataloader
from active_learning.models.base_model import BaseModel
from active_learning.models.model_baseline import CLIPModel
from active_learning.training.train import train_one_epoch, validate_epoch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed: int = 45510
data_dir: Path = Path("baselines/dataset/")
result_file: Path = Path("baselines/upper_bound/result.json")
checkpoint_dir = Path("baselines/upper_bound/model_checkpoints/")
tie_margin = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_MODE = True
FORCE_RETRAIN = False

train_config = {
    "train_batch_size": 8,
    "valid_batch_size": 16,
    "num_workers": 4,
    "train_epochs": 6,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "tie_margin": tie_margin,
    "seed": seed,
}

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

structlog.stdlib.recreate_defaults()
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("transformers.utils.hub").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)
logger = structlog.get_logger("logger")

results = {
    "timestamp": datetime.now().isoformat(),
    "seed": seed,
    "tie_margin": tie_margin,
    "device": device,
    "models": [],
}

@torch.no_grad()
def calc_probability_distribution(s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
    logits = torch.stack([s0, s1], dim=1)
    return F.softmax(logits, dim=1)

@torch.no_grad()
def snap_prediction(probs: torch.Tensor, tie_margin: float) -> torch.Tensor:
    diff = probs[:, 0] - 0.5
    tie = diff.abs() <= tie_margin
    out = torch.zeros_like(probs)
    out[tie] = 0.5
    out[~tie, 0] = (probs[~tie, 0] > probs[~tie, 1]).float()
    out[~tie, 1] = 1.0 - out[~tie, 0]
    return out

def checkpoint_exists(model_id: int) -> bool:
    model_checkpoint_dir = checkpoint_dir / str(model_id)
    last_checkpoint = model_checkpoint_dir / "last.pt"
    best_checkpoint = model_checkpoint_dir / "best.pt"
    return last_checkpoint.exists() or best_checkpoint.exists()

def load_checkpoint(model: BaseModel, model_id: int) -> dict:
    model_checkpoint_dir = checkpoint_dir / str(model_id)
    best_checkpoint = model_checkpoint_dir / "best.pt"
    last_checkpoint = model_checkpoint_dir / "last.pt"
    checkpoint_info = {"loaded": False, "checkpoint_path": None}
    if best_checkpoint.exists():
        try:
            model.load(best_checkpoint)
            checkpoint_info = {"loaded": True, "checkpoint_path": str(best_checkpoint)}
            logger.info("checkpoint_loaded", path=str(best_checkpoint), type="best")
        except Exception as e:
            logger.error("checkpoint_load_failed", path=str(best_checkpoint), error=str(e))
    elif last_checkpoint.exists():
        try:
            model.load(last_checkpoint)
            checkpoint_info = {"loaded": True, "checkpoint_path": str(last_checkpoint)}
            logger.info("checkpoint_loaded", path=str(last_checkpoint), type="last")
        except Exception as e:
            logger.error("checkpoint_load_failed", path=str(last_checkpoint), error=str(e))
    return checkpoint_info

def train_model(model, processor, optimizer, scaler, train_loader, valid_loader, model_id):
    model_checkpoint_dir = checkpoint_dir / str(model_id)
    model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    for epoch in range(1, train_config["train_epochs"] + 1):
        logger.info("epoch_started", epoch=epoch + 1, total_epochs=train_config["train_epochs"])
        train_loss = train_one_epoch(
            model=model,
            processor=processor,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )
        train_losses.append(float(train_loss))
        logger.info("epoch_complete", epoch=epoch + 1, total_epochs=train_config["train_epochs"], train_loss=train_loss)
        val = validate_epoch(
            model=model,
            processor=processor,
            loader=valid_loader,
            device=device,
            tie_margin=tie_margin,
        )
        val_loss = float(val["val_loss"])
        val_losses.append(val_loss)
        logger.info("validation_complete", epoch=epoch + 1, total_epochs=train_config["train_epochs"], valid_loss=val_loss)
        logger.info("epoch_complete", epoch=epoch + 1, total_epochs=train_config["train_epochs"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = model_checkpoint_dir / "best.pt"
            model.save(best_checkpoint_path)
            logger.info("best_checkpoint_saved", path=str(best_checkpoint_path), epoch=epoch, val_loss=val_loss, improvement=True)
        if epoch == train_config["train_epochs"]:
            last_checkpoint_path = model_checkpoint_dir / "last.pt"
            model.save(last_checkpoint_path)
            logger.info("last_checkpoint_saved", path=str(last_checkpoint_path), epoch=epoch)
    return {"train_loss": train_losses, "val_loss": val_losses}

pretrained_models = [
    "yuvalkirstain/PickScore_v1",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
]

for model_id, pretrained_model in enumerate(pretrained_models):
    print(f"\nEvaluating model: {pretrained_model}")
    model: BaseModel = CLIPModel(pretrained_model_name_or_path=pretrained_model)
    model.eval().to(device)
    processor = CLIPProcessor.from_pretrained(pretrained_model)

    checkpoint_exists_flag = checkpoint_exists(model_id)

    training_log = {"train_loss": [], "val_loss": []}
    if FORCE_RETRAIN or (TRAIN_MODE and not checkpoint_exists_flag):
        if FORCE_RETRAIN:
            answer = input("Force retrain? y/[n]")
            if answer == "y":
                print("Proceeding to retrain")
            else:
                import sys
                print("Aborting...")
                sys.exit(0)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"],
        )
        scaler = torch.amp.GradScaler(device)
        train_loader = create_dataloader(
            data_dir=data_dir,
            split="train",
            batch_size=train_config["train_batch_size"],
            num_workers=train_config["num_workers"],
            processor=processor,
            shuffle=True,
        )
        valid_loader = create_dataloader(
            data_dir=data_dir,
            split="val",
            batch_size=train_config["valid_batch_size"],
            num_workers=train_config["num_workers"],
            processor=processor,
            shuffle=False,
        )
        logger.info("training_started", model_id=model_id, pretrained_model_name=pretrained_model, **train_config)
        training_log = train_model(model, processor, optimizer, scaler, train_loader, valid_loader, model_id)
    else:
        logger.info("Training skipped; checkpoint already exists", model_id=model_id)

    load_checkpoint(model, model_id)

    model_results = {
        "model_name": pretrained_model,
        "training": training_log,
        "batches": [],
        "total_accuracy": 0,
        "total_samples": 0,
        "total_matches": 0,
    }

    dl = create_dataloader(
        data_dir,
        split="test",
        batch_size=48,
        num_workers=4,
        processor=processor,
        shuffle=False,
    )

    total_matches, total_samples = 0, 0

    for i, batch in enumerate(dl):
        img0 = batch["image_0"].to(device)
        img1 = batch["image_1"].to(device)
        captions = batch["caption"]
        imgs = torch.cat([img0, img1], dim=0)

        with torch.no_grad():
            image_feats = model.get_image_features(imgs)
            image_feats = F.normalize(image_feats, dim=-1)
            i0, i1 = image_feats.chunk(2, dim=0)

            text_inputs = processor(
                text=captions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            text_feats = model.get_text_features(**text_inputs)
            text_feats = F.normalize(text_feats, dim=-1)

            s0 = (text_feats * i0).sum(dim=-1)
            s1 = (text_feats * i1).sum(dim=-1)

            if hasattr(model, "logit_scale"):
                scale = model.logit_scale.exp()
            else:
                scale = 1.0
            s0, s1 = s0 * scale, s1 * scale

            probs = calc_probability_distribution(s0, s1)
            preds = snap_prediction(probs, tie_margin)

        labels = torch.stack([batch["label_0"], batch["label_1"]], dim=1).to(device)
        matches = (preds == labels).all(dim=1)
        batch_matches = matches.sum().item()
        batch_samples = matches.shape[0]
        batch_acc = matches.float().mean().item()

        total_matches += batch_matches
        total_samples += batch_samples

        model_results["batches"].append(
            {
                "batch_id": i,
                "accuracy": batch_acc,
                "samples": batch_samples,
                "matches": batch_matches,
            }
        )

        print(f"Batch {i}'s accuracy: {batch_acc:.2%}")

    overall_acc = total_matches / total_samples if total_samples > 0 else 0.0
    model_results["total_accuracy"] = overall_acc
    model_results["total_samples"] = total_samples
    model_results["total_matches"] = total_matches
    results["models"].append(model_results)
    print(f"Overall accuracy for model {pretrained_model}: {overall_acc:.2%}")

print(f"\nSaving results to {result_file}")
with open(result_file, "w") as f:
    json.dump(results, f, indent=4)

print("Evaluation complete!")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
for model_result in results["models"]:
    print(f"{model_result['model_name']}: {model_result['total_accuracy']:.2%}")
