from dataclasses import dataclass
from transformers import CLIPModel as HFCLIPModel

from torch import nn

from trainer.models.base_model import BaseModelConfig


@dataclass
class ClipModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.clip_model.CLIPModel"
    pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32"


class CLIPModel(nn.Module):
    def __init__(self, cfg: ClipModelConfig):
        super().__init__()
        self.model = HFCLIPModel.from_pretrained(cfg.pretrained_model_name_or_path)
        # Add dropout layer with 0.1 probability
        self.dropout = nn.Dropout(p=0.1)

    def get_text_features(self, *args, **kwargs):
        features = self.model.get_text_features(*args, **kwargs)
        # Apply dropout to the text features
        if self.training:
            features = self.dropout(features)
        return features

    def get_image_features(self, *args, **kwargs):
        features = self.model.get_image_features(*args, **kwargs)
        # Apply dropout to the image features
        if self.training:
            features = self.dropout(features)
        return features

    def forward(self, text_inputs=None, image_inputs=None):
        outputs = ()
        if text_inputs is not None:
            text_features = self.model.get_text_features(text_inputs)
            # Apply dropout to the text features during training
            if self.training:
                text_features = self.dropout(text_features)
            outputs += text_features,
        if image_inputs is not None:
            image_features = self.model.get_image_features(image_inputs)
            # Apply dropout to the image features during training
            if self.training:
                image_features = self.dropout(image_features)
            outputs += image_features,
        return outputs


    @property
    def logit_scale(self):
        return self.model.logit_scale

    def save(self, path):
        self.model.save_pretrained(path)

