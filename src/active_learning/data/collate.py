import torch
from typing import List, Dict, Any
from transformers import CLIPProcessor


class Collate:
    def __init__(self):
        self.image_processor : CLIPProcessor = None

    def set_processor(self, processor):
        self.image_processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.image_processor is None:
            raise ValueError("image_processor not set. Call set_processor() first.")

        caps = [b["caption"] for b in batch]
        images_0 = [b["image_0"] for b in batch]
        images_1 = [b["image_1"] for b in batch]

        img0 = self.image_processor(
            images=images_0,
            return_tensors="pt",
        ).pixel_values
        img1 = self.image_processor(
            images=images_1,
            return_tensors="pt",
        ).pixel_values

        y0 = torch.tensor([float(b["label_0"]) for b in batch], dtype=torch.float32)
        y1 = torch.tensor([float(b["label_1"]) for b in batch], dtype=torch.float32)

        result = {
            "caption": caps,
            "image_0": img0,
            "image_1": img1,
            "label_0": y0,
            "label_1": y1,
        }

        if batch and "original_index" in batch[0]:
            result["original_indices"] = torch.tensor([b["original_index"] for b in batch])

        return result