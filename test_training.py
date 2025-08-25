import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import CLIPProcessor

from active_learning.data_utils import create_dataloader
from active_learning.models.model_baseline import CLIPModel
from configs.experiment_configs import FAST_PROTOTYPE, SMALL_SCALE, ExperimentConfig


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _l2n(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(
        1e-12
    )  # manual L2 with clamp to avoid NaNs instead of F.normalize


def _scores_from_features(
    text_features: torch.Tensor,
    image0_features: torch.Tensor,
    image1_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = _l2n(
        text_features
    )  # normalize here instead of relying on model outputs being normalized
    i0 = _l2n(image0_features)
    i1 = _l2n(image1_features)
    s = float(
        logit_scale
    )  # casts a learnable tensor to Python float; freezes the scale for this call
    B = t.size(0)
    all_img = torch.cat(
        [i0, i1], dim=0
    )  # packs two image batches into 2B to reuse one matmul
    logits_t2i = s * (
        t @ all_img.t()
    )  # builds a [B,2B] matrix only to read the diagonal rows below
    t0_logits, t1_logits = logits_t2i.split(
        B, dim=-1
    )  # splits back to [B,B] blocks instead of computing two dot products
    idx = torch.arange(B, device=t.device)
    s0 = t0_logits[
        idx, idx
    ]  # picks only the diagonal element; off-diagonals are wasted compute
    s1 = t1_logits[idx, idx]
    return s0, s1


def calc_loss(
    text_features: torch.Tensor,
    image_0_features: torch.Tensor,
    image_1_features: torch.Tensor,
    label_0: torch.Tensor,
    label_1: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    device = text_features.device
    B = text_features.size(0)
    s = logit_scale.float()

    t = _l2n(text_features)
    i0 = _l2n(image_0_features)
    i1 = _l2n(image_1_features)
    all_img = torch.cat([i0, i1], dim=0)  # same packed trick to reuse one matmul
    logits_t2i = s * (t @ all_img.t())
    t0_logits, t1_logits = logits_t2i.split(B, dim=-1)
    idx = torch.arange(B, device=device)
    logits = torch.stack(
        [t0_logits[idx, idx], t1_logits[idx, idx]], dim=-1
    )  # builds [B,2] via diagonal reads

    ce0 = F.cross_entropy(
        logits, torch.zeros(B, device=device, dtype=torch.long), reduction="none"
    )  # computes CE to class 0 separately
    ce1 = F.cross_entropy(
        logits, torch.ones(B, device=device, dtype=torch.long), reduction="none"
    )  # computes CE to class 1 separately
    return (
        label_0 * ce0 + label_1 * ce1
    ).mean()  # mixes two CEs with soft labels instead of using a single CE with soft targets


@torch.no_grad()
def pref_metrics(
    s0: torch.Tensor,
    s1: torch.Tensor,
    y0: torch.Tensor,
    y1: torch.Tensor,
    tie_margin: float = 0.05,
) -> dict:
    label_gap = y0 - y1  # 0 encodes ties exactly
    pred_gap = s0 - s1
    is_tie = label_gap == 0
    is_non_tie = ~is_tie

    non_tie_correct = (
        (torch.sign(pred_gap[is_non_tie]) == torch.sign(label_gap[is_non_tie]))
        .sum()
        .item()
    )  # treats zero pred as wrong on non-ties
    non_tie_total = is_non_tie.sum().item()
    tie_correct = (
        (pred_gap[is_tie].abs() <= tie_margin).sum().item()
    )  # fixed absolute margin instead of scale-normalized margin
    tie_total = is_tie.sum().item()

    pref_acc = non_tie_correct / max(
        non_tie_total, 1
    )  # max guards divide-by-zero in a metric rather than raising
    tie_acc = tie_correct / max(tie_total, 1)
    overall_acc = (non_tie_correct + tie_correct) / max(non_tie_total + tie_total, 1)
    return {
        "pref_acc": pref_acc,
        "tie_acc": tie_acc,
        "overall_acc": overall_acc,
        "non_tie_correct": non_tie_correct,
        "non_tie_total": non_tie_total,
        "tie_correct": tie_correct,
        "tie_total": tie_total,
    }


class PrefMetricTracker:
    def __init__(self, tie_margin: float = 0.05):
        self.tie_margin = tie_margin
        self.reset()

    def reset(self) -> None:
        self.non_tie_correct = 0
        self.non_tie_total = 0
        self.tie_correct = 0
        self.tie_total = 0

    @torch.no_grad()
    def update(
        self, s0: torch.Tensor, s1: torch.Tensor, y0: torch.Tensor, y1: torch.Tensor
    ) -> None:
        m = pref_metrics(
            s0, s1, y0, y1, self.tie_margin
        )  # recomputes masks each call instead of carrying them in tracker state
        self.non_tie_correct += m["non_tie_correct"]
        self.non_tie_total += m["non_tie_total"]
        self.tie_correct += m["tie_correct"]
        self.tie_total += m["tie_total"]

    def compute(self) -> dict:
        pref = self.non_tie_correct / max(
            self.non_tie_total, 1
        )  # same defensive max as above
        tie = self.tie_correct / max(self.tie_total, 1)
        overall = (self.non_tie_correct + self.tie_correct) / max(
            self.non_tie_total + self.tie_total, 1
        )
        return {
            "pref_acc": pref,
            "tie_acc": tie,
            "overall_acc": overall,
            "non_tie_total": self.non_tie_total,
            "tie_total": self.tie_total,
        }


@torch.no_grad()
def evaluate(
    model, processor: CLIPProcessor, loader, device, tie_margin: float = 0.05
) -> dict:
    model.eval()
    tracker = PrefMetricTracker(tie_margin)
    loss_sum = 0.0
    batches = 0

    for b in tqdm(
        loader, desc="valid", leave=False
    ):  # tqdm over validation; progress bars in val are uncommon but useful
        captions = b["caption"]
        x0 = b["image_0"]
        x1 = b["image_1"]
        y0 = b["label_0"].to(device)
        y1 = b["label_1"].to(device)

        text_inputs = processor(
            text=captions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)  # builds tokenization per batch instead of caching
        image_inputs0 = processor(images=x0, return_tensors="pt").to(
            device
        )  # uses CLIPProcessor in-loop; slower than precomputed tensors
        image_inputs1 = processor(images=x1, return_tensors="pt").to(device)

        t = model.get_text_features(
            **text_inputs
        )  # assumes model exposes HF-like getters; bypasses a unified forward
        i0 = model.get_image_features(**image_inputs0)
        i1 = model.get_image_features(**image_inputs1)

        loss = calc_loss(
            t, i0, i1, y0, y1, logit_scale=model.logit_scale.exp()
        )  # pulls .exp() every step; could cache outside loop
        loss_sum += loss.item()
        batches += 1

        s0, s1 = _scores_from_features(
            t, i0, i1, model.logit_scale.exp()
        )  # recomputes scaled sims just for metrics
        tracker.update(s0, s1, y0, y1)

    metrics = tracker.compute()
    metrics["val_loss"] = loss_sum / max(
        batches, 1
    )  # averages only over batches that had data
    return metrics


def train(cfg: ExperimentConfig, out_dir: Path = Path("./runs/quick")):
    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)
    out_dir = Path(out_dir)

    train_loader = create_dataloader(
        out_dir, "train", cfg.train_batch_size, 0, True, cfg.image_size
    )
    valid_loader = create_dataloader(
        out_dir, "valid", cfg.train_batch_size, 0, False, cfg.image_size
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device {device}")
    model = CLIPModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler(device=device, enabled=torch.cuda.is_available())
    processor = CLIPProcessor.from_pretrained(
        cfg.pretrained_model_name_or_path, use_fast=False
    )

    print("Setup finished")

    # --- versioned checkpoint root: runs/checkpoints/<YYYYmmdd-HHMMSS> ---
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")  # timestamp for this training run
    ckpt_root = (out_dir / "checkpoints" / run_id).resolve()
    ckpt_root.mkdir(parents=True, exist_ok=True)

    best_overall = -1.0
    best_path = None

    for epoch in range(1, cfg.train_epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch} [train]", leave=False)
        for batch in pbar:
            captions = batch["caption"]
            x0 = batch["image_0"]
            x1 = batch["image_1"]
            y0 = batch["label_0"].to(device)
            y1 = batch["label_1"].to(device)

            text_inputs = processor(
                text=captions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            image_inputs0 = processor(images=x0, return_tensors="pt").to(device)
            image_inputs1 = processor(images=x1, return_tensors="pt").to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device, enabled=torch.cuda.is_available()
            ):
                t = model.get_text_features(**text_inputs)
                i0 = model.get_image_features(**image_inputs0)
                i1 = model.get_image_features(**image_inputs1)
                loss = calc_loss(t, i0, i1, y0, y1, logit_scale=model.logit_scale.exp())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            steps += 1
            running_loss += loss.item()
            pbar.set_postfix(avg_loss=f"{running_loss / steps:.4f}")

        # validate
        val = evaluate(
            model,
            processor,
            valid_loader,
            device,
            tie_margin=getattr(cfg, "tie_margin", 0.05),
        )
        print(
            f"epoch {epoch} | val_loss {val['val_loss']:.4f} | pref_acc {val['pref_acc']:.4f} | tie_acc {val['tie_acc']:.4f} | overall {val['overall_acc']:.4f}"
        )

        # --- versioned save for this epoch ---
        tag = f"epoch-{epoch:04d}_overall-{val['overall_acc']:.4f}"
        ep_dir = ckpt_root / tag
        model.save(str(ep_dir))  # saves full HF-style dir
        (ep_dir / "metrics.json").write_text(
            json.dumps(val, indent=2)
        )  # store metrics with the checkpoint

        # update pointers
        (out_dir / "LAST").write_text(
            str(ep_dir)
        )  # text file pointing to most recent checkpoint
        if val["overall_acc"] > best_overall:
            best_overall = val["overall_acc"]
            best_path = ep_dir
            (out_dir / "BEST").write_text(
                str(ep_dir)
            )  # text file pointing to best checkpoint

    model.save(str(out_dir / "last"))
    if best_path is not None:
        model.save(str(out_dir / "best"))


def main():
    cfg = SMALL_SCALE
    out_dir = Path("./runs/quick")
    train(cfg=cfg, out_dir=out_dir)


if __name__ == "__main__":
    main()
