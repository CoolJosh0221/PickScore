import torch

from .base_model import BaseModel
from transformers import CLIPModel as HFCLIPModel
from transformers import CLIPConfig as HFCLIPConfig

from torch import nn


class MCDropoutCLIPModel(BaseModel):
    def __init__(self, pretrained_model_name_or_path: str, mc_dropout_p: float):
        super().__init__()

        # Load the pretrained model configuration
        pretrained_model_config = HFCLIPConfig.from_pretrained(
            pretrained_model_name_or_path
        )
        # # Set dropout rates in text encoder
        pretrained_model_config.text_config.attention_dropout = mc_dropout_p
        pretrained_model_config.text_config.dropout = mc_dropout_p

        # # Set dropout rates in vision encoder
        pretrained_model_config.vision_config.attention_dropout = mc_dropout_p
        pretrained_model_config.vision_config.dropout = mc_dropout_p

        self.model = HFCLIPModel.from_pretrained(
            pretrained_model_name_or_path, config=pretrained_model_config
        )

        # Add explicit dropout layers for feature outputs
        self.text_dropout = nn.Dropout(p=mc_dropout_p)
        self.image_dropout = nn.Dropout(p=mc_dropout_p)

        # Disable MCDropout by default
        # Turn on this setting to enable dropout during inference
        # If turned off, model still trains with dropout, but eval() disables it as usual
        self.enable_mc_dropout: bool = False

    def get_text_features(self, *args, **kwargs):
        features = self.model.get_text_features(*args, **kwargs)
        # Apply dropout during training OR if MC dropout is enabled
        if self.training or self.enable_mc_dropout:
            features = self.text_dropout(features)
        return features

    def get_image_features(self, *args, **kwargs):
        features = self.model.get_image_features(*args, **kwargs)
        # Apply dropout during training OR if MC dropout is enabled
        if self.training or self.enable_mc_dropout:
            features = self.image_dropout(features)
        return features

    def forward(self, text_inputs=None, image_inputs=None):
        outputs = ()
        if text_inputs is not None:
            outputs += (self.get_text_features(text_inputs),)
        if image_inputs is not None:
            outputs += (self.get_image_features(image_inputs),)
        return outputs

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def save(self, path):
        """Save the entire model parameters including outer wrapper."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load the entire model parameters."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)

    def eval(self):
        """
        Override eval() to implement MC Dropout
        In standard PyTorch behavior, calling eval() disables dropout
        With MC Dropout, we want to keep dropout active during inference
        """
        if not self.enable_mc_dropout:
            # Standard behavior - disable dropout
            self.model.eval()
            self.training = False
        else:
            # MC Dropout - keep model in eval mode but ensure dropout remains active
            self.model.eval()
            # Ensure dropout layers stay in train mode
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
            # Keep our explicit dropout layers in training mode
            self.text_dropout.train()
            self.image_dropout.train()
            # Technically we're not training, but we want dropout active
            self.training = False
        return self
