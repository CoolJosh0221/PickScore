from typing import Optional
from attrs import define, field
from attrs.validators import instance_of


@define
class BaseModelConfig:
    """Base config shared by model wrappers."""

    _target_: str = field(
        default="", validator=instance_of(str)
    )  # Hydra instantiate target
    pretrained_model_name_or_path = field(
        default="openai/clip-vit-base-patch32", validator=instance_of(str)
    )  # HF hub ID or local path
