import torch
from typing import List, Dict, Any


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "caption": [b["caption"] for b in batch],
        "image_0": [b["image_0"] for b in batch],
        "image_1": [b["image_1"] for b in batch],
        "label_0": torch.tensor(
            [float(b["label_0"]) for b in batch], dtype=torch.float32
        ),
        "label_1": torch.tensor(
            [float(b["label_1"]) for b in batch], dtype=torch.float32
        ),
    }
