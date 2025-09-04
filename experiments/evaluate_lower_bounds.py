from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from active_learning.data.loaders import create_dataloader
from active_learning.data.sampling import create_independent_test_set
from active_learning.models.base_model import BaseModel
from active_learning.models.model_mcdo import MCDropoutCLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

out_dir = Path("test_dataset")
out_dir.mkdir(parents=True, exist_ok=True)

create_independent_test_set(
    str(out_dir),
    dataset_name="pickapic-anonymous/pickapic_v1",
    size=500,
    seed=45510,
)

pretrained_models = [
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "openai/clip-vit-base-patch32",
]


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


tie_margin = 0.1

for pretrained_model in pretrained_models:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    model: BaseModel = MCDropoutCLIPModel(
        pretrained_model_name_or_path=pretrained_model
    )
    model.eval().to(device)

    dl = create_dataloader(
        out_dir,
        split="test",
        batch_size=48,
        num_workers=4,
        shuffle=False,
    )

    for batch in dl:
        img0 = batch["image_0"].to(device)  # [B,C,H,W]
        img1 = batch["image_1"].to(device)  # [B,C,H,W]
        captions = batch["caption"]

        imgs = torch.cat([img0, img1], dim=0)  # [2B,C,H,W]

        with torch.no_grad():
            image_feats = model.get_image_features(imgs)  # [2B,d]
            image_feats = F.normalize(image_feats, dim=-1)
            i0, i1 = image_feats.chunk(2, dim=0)  # [B,d] each

            text_inputs = tokenizer(
                text=captions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            text_feats = model.get_text_features(**text_inputs)  # [B,d]
            text_feats = F.normalize(text_feats, dim=-1)

            s0 = (text_feats * i0).sum(dim=-1)  # [B]
            s1 = (text_feats * i1).sum(dim=-1)  # [B]

            if hasattr(model, "logit_scale"):
                scale = model.logit_scale.exp()
            else:
                scale = 1.0

            s0, s1 = s0 * scale, s1 * scale

            probs = calc_probability_distribution(s0, s1)  # [B,2]
            preds = snap_prediction(probs, tie_margin)

        labels = torch.stack([batch["label_0"], batch["label_1"]], dim=1).to(device)

        matches = (preds == labels).all(dim=1)  # [B]
        acc = matches.float().mean().item()

        print(f"Batch accuracy: {acc}")
