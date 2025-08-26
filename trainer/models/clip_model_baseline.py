from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import CLIPModel as HFCLIPModel
from attrs import define, field
from attrs.validators import instance_of

from trainer.models.base_model import BaseModelConfig


@define(slots=True)
class BaselineClipModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.clip_model_baseline.BaselineCLIPModel"
    pretrained_model_name_or_path: str = field(
        default="openai/clip-vit-base-patch32", validator=instance_of(str)
    )


class BaselineCLIPModel(nn.Module):
    def __init__(self, cfg: BaselineClipModelConfig):
        super().__init__()
        self.cfg = cfg
        # Load the pretrained CLIP backbone
        self.model = HFCLIPModel.from_pretrained(cfg.pretrained_model_name_or_path)

    def get_text_features(self, *args, **kwargs) -> torch.Tensor:
        return self.model.get_text_features(*args, **kwargs)

    def get_image_features(self, *args, **kwargs) -> torch.Tensor:
        return self.model.get_image_features(*args, **kwargs)

    def forward(
        self,
        text_inputs: Optional[dict] = None,
        image_inputs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, ...]:
        outs: Tuple[torch.Tensor, ...] = ()
        if text_inputs is not None:
            outs += (self.get_text_features(**text_inputs),)
        if image_inputs is not None:
            outs += (self.get_image_features(**image_inputs),)
        return outs

    @property
    def logit_scale(self) -> nn.Parameter:
        # Expose CLIP logit scale for cosine → logit scoring
        return self.model.logit_scale

    def save(self, path: str) -> None:
        # Preserve HF save format so downstream tools can reload
        self.model.save_pretrained(path)
