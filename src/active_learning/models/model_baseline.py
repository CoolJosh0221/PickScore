import torch
from transformers import CLIPModel as HFCLIPModel

from .base_model import BaseModel


class CLIPModel(BaseModel):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__()
        self.model = HFCLIPModel.from_pretrained(pretrained_model_name_or_path)

    def get_text_features(self, *args, **kwargs):
        return self.model.get_text_features(*args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)

    def forward(self, text_inputs=None, image_inputs=None):
        outputs = ()
        if text_inputs is not None:
            outputs += (self.model.get_text_features(text_inputs),)
        if image_inputs is not None:
            outputs += (self.model.get_image_features(image_inputs),)
        return outputs

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def load(self, path):
        """Load the entire model parameters."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)

    def save(self, path):
        """Save the entire model parameters"""
        torch.save(self.state_dict(), path)
