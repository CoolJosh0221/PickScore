import torch
import torch.nn as nn
from attrs import define, validators, field
from transformers import CLIPModel
from typing import Dict, Tuple, Optional


# MC Dropout Classes
class DropoutMC(nn.Module):
    def __init__(self, p: float, activate=False):
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.dropout(
            x, self.p, training=self.training or self.activate
        )


class LockedDropoutMC(DropoutMC):
    """Drops same features across sequence positions."""

    def forward(self, x):
        if self.training:
            self.activate = True
        if not self.activate or not self.p:
            return x

        batch_size, seq_len, features = x.size()
        m = x.data.new(batch_size, 1, features).bernoulli_(1 - self.p)
        mask = (m / (1 - self.p)).expand_as(x)
        return mask * x


class WordDropoutMC(DropoutMC):
    """Drops entire words/tokens."""

    def forward(self, x):
        if self.training:
            self.activate = True
        if not self.activate or not self.p:
            return x

        batch_size, seq_len, features = x.size()
        m = x.data.new(batch_size, seq_len, 1).bernoulli_(1 - self.p)
        mask = m.expand_as(x)
        return mask * x


# Configuration
def dropout_validator():
    """Validator for dropout rates (0.0 <= value < 1.0)"""
    return validators.and_(
        validators.instance_of(float), validators.in_(range(0.0, 1.0))
    )


@define
class CLIPDropoutConfig:
    text_attention: float = field(default=0.1, validator=dropout_validator())
    text_feed_forward: float = field(default=0.1, validator=dropout_validator())
    text_word_dropout: float = field(default=0.05, validator=dropout_validator())
    vision_attention: float = field(default=0.1, validator=dropout_validator())
    vision_feed_forward: float = field(default=0.1, validator=dropout_validator())
    text_projection: float = field(default=0.15, validator=dropout_validator())
    vision_projection: float = field(default=0.15, validator=dropout_validator())
    reward_head: float = field(default=0.3, validator=dropout_validator())


# Utilities
def activate_mc_dropout(model: nn.Module, activate: bool, verbose: bool = False):
    """Turn MC dropout on/off throughout model."""
    for name, module in model.named_modules():
        if isinstance(module, DropoutMC):
            if verbose:
                print(f"{name}: {module.activate} -> {activate}")
            module.activate = activate


def wrap_with_dropout(module, dropout_layer):
    """Add dropout after a module's output."""
    original_forward = module.forward

    def new_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        if isinstance(output, tuple):
            return (dropout_layer(output[0]),) + output[1:]
        else:
            return dropout_layer(output)

    module.forward = new_forward
    module._mc_dropout = dropout_layer


def setup_clip_dropout(clip_model, config: CLIPDropoutConfig):
    """Add MC dropout to CLIP model."""

    # 1. Text word dropout
    if hasattr(clip_model, "text_model") and hasattr(
        clip_model.text_model, "embeddings"
    ):
        wrap_with_dropout(
            clip_model.text_model.embeddings.token_embedding,
            WordDropoutMC(config.text_word_dropout),
        )

    # 2. Text transformer layers
    if hasattr(clip_model, "text_model") and hasattr(clip_model.text_model, "encoder"):
        for layer in clip_model.text_model.encoder.layers:
            if hasattr(layer, "self_attn"):
                wrap_with_dropout(
                    layer.self_attn, LockedDropoutMC(config.text_attention)
                )
            if hasattr(layer, "mlp"):
                wrap_with_dropout(layer.mlp, LockedDropoutMC(config.text_feed_forward))

    # 3. Vision transformer layers
    if hasattr(clip_model, "vision_model") and hasattr(
        clip_model.vision_model, "encoder"
    ):
        for layer in clip_model.vision_model.encoder.layers:
            if hasattr(layer, "self_attn"):
                wrap_with_dropout(
                    layer.self_attn, LockedDropoutMC(config.vision_attention)
                )
            if hasattr(layer, "mlp"):
                wrap_with_dropout(
                    layer.mlp, LockedDropoutMC(config.vision_feed_forward)
                )

    # 4. Projections
    if (
        hasattr(clip_model, "text_projection")
        and clip_model.text_projection is not None
    ):
        clip_model.text_projection = nn.Sequential(
            DropoutMC(config.text_projection, activate=False),
            clip_model.text_projection,
        )

    if (
        hasattr(clip_model, "visual_projection")
        and clip_model.visual_projection is not None
    ):
        clip_model.visual_projection = nn.Sequential(
            DropoutMC(config.vision_projection, activate=False),
            clip_model.visual_projection,
        )

    return clip_model


# CLIP Reward Model
class CLIPRewardModel(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        hidden_dim: int = 512,
        dropout_config: Optional[CLIPDropoutConfig] = None,
    ):
        super().__init__()

        # Load and setup CLIP
        self.clip = CLIPModel.from_pretrained(clip_model_name)

        if dropout_config is None:
            dropout_config = CLIPDropoutConfig()

        self.clip = setup_clip_dropout(self.clip, dropout_config)

        # Reward head
        clip_dim = self.clip.config.projection_dim
        self.reward_head = nn.Sequential(
            nn.Linear(clip_dim * 2, hidden_dim),
            nn.ReLU(),
            DropoutMC(dropout_config.reward_head, activate=False),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            DropoutMC(dropout_config.reward_head * 0.5, activate=False),
            nn.Linear(256, 1),
        )

    def forward(self, pixel_values, input_ids, attention_mask=None):
        # Get CLIP features
        image_features = self.clip.get_image_features(pixel_values)
        text_features = self.clip.get_text_features(input_ids, attention_mask)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Predict reward
        combined = torch.cat([image_features, text_features], dim=1)
        return self.reward_head(combined)


# Uncertainty Quantification
class UncertaintyQuantifier:
    @staticmethod
    def get_uncertainty(
        model: CLIPRewardModel,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with uncertainty."""
        model.eval()
        activate_mc_dropout(model, activate=True)

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = model(pixel_values, input_ids, attention_mask)
                predictions.append(pred)

        activate_mc_dropout(model, activate=False)

        predictions = torch.stack(predictions, dim=0)
        mean = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        return mean, uncertainty

    @staticmethod
    def get_bald_scores(
        model: CLIPRewardModel,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_samples: int = 20,
    ) -> torch.Tensor:
        """Get BALD scores for active learning."""
        model.eval()
        activate_mc_dropout(model, activate=True)

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = model(pixel_values, input_ids, attention_mask)
                probs = torch.sigmoid(logits).squeeze(-1)
                class_probs = torch.stack([1 - probs, probs], dim=-1)
                predictions.append(class_probs)

        activate_mc_dropout(model, activate=False)

        predictions = torch.stack(predictions, dim=0)

        # BALD = H(y|x) - E[H(y|x,θ)]
        mean_pred = predictions.mean(dim=0)
        entropy_mean = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=-1)

        entropies = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1)
        mean_entropy = torch.mean(entropies, dim=0)

        return entropy_mean - mean_entropy


# Usage Example
if __name__ == "__main__":
    # Setup
    config = CLIPDropoutConfig(
        text_attention=0.1,
        text_word_dropout=0.05,
        vision_attention=0.1,
        reward_head=0.3,
    )

    model = CLIPRewardModel(dropout_config=config)

    # Example data
    batch_size = 4
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, 77))

    # Get uncertainty
    mean, uncertainty = UncertaintyQuantifier.get_uncertainty(
        model, pixel_values, input_ids, n_samples=20
    )

    # Get BALD scores
    bald_scores = UncertaintyQuantifier.get_bald_scores(
        model, pixel_values, input_ids, n_samples=20
    )

    print(f"Mean rewards: {mean.shape}")
    print(f"Uncertainties: {uncertainty.shape}")
    print(f"BALD scores: {bald_scores.shape}")
    print(f"Sample BALD scores: {bald_scores[:3]}")
