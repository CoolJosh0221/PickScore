import torch
from typing import List, Dict, Any
import torchvision.transforms as T

# CLIP normalization
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Replace CLIPProcessor with manual transforms for image inputs
_img_tfms = T.Compose(
    [
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(_CLIP_MEAN, _CLIP_STD),
    ]
)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    caps = [b["caption"] for b in batch]
    img0 = torch.stack([_img_tfms(b["image_0"]) for b in batch], 0)  # [B,3,224,224]
    img1 = torch.stack([_img_tfms(b["image_1"]) for b in batch], 0)
    y0 = torch.tensor([float(b["label_0"]) for b in batch], dtype=torch.float32)
    y1 = torch.tensor([float(b["label_1"]) for b in batch], dtype=torch.float32)
    return {
        "caption": caps,
        "image_0": img0,
        "image_1": img1,
        "label_0": y0,
        "label_1": y1,
    }
