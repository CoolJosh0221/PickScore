from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import CLIPModel as HFCLIPModel
from attrs import define, field
from attrs.validators import ge, lt, in_, instance_of

from trainer.models.base_model import BaseModelConfig
from trainer.models._enums import ApplyTo


@define(slots=True)
class MCDropoutClipModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.clip_model_mcdo.MCDropoutCLIPModel"
    pretrained_model_name_or_path: str = field(
        default="openai/clip-vit-base-patch32", validator=instance_of(str)
    )
    mc_dropout_p: float = field(default=0.1, validator=[ge(0.0), lt(1.0)])  # drop prob
    apply_to: ApplyTo = field(default=ApplyTo.BOTH, converter=ApplyTo, validator=in_({*ApplyTo}))
    active_by_default: bool = False  # keep deterministic unless explicitly enabled


class MCDropoutCLIPModel(nn.Module):
    """
    CLIP wrapper with explicit MC Dropout control.

    Deterministic by default. Enable stochasticity for T-pass sampling via:
        model.eval()
        with model.mc_dropout():
            ...
    """
    def __init__(self, cfg: MCDropoutClipModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = HFCLIPModel.from_pretrained(cfg.pretrained_model_name_or_path)
        self._mc_active = bool(cfg.active_by_default)
        # Set dropout probability on selected towers once at init
        self._set_dropout_p(cfg.apply_to, cfg.mc_dropout_p)

    def enable_mc_dropout(self) -> None:
        self._mc_active = True
        # If in eval(), force only Dropout layers to act as training
        if not self.training:
            self._force_dropout_train_mode()

    def disable_mc_dropout(self) -> None:
        self._mc_active = False
        # Return Dropout layers to eval behavior
        if not self.training:
            for m in self._iter_selected_dropout():
                m.train(False)

    @contextmanager
    def mc_dropout(self):
        """Temporarily enable MC Dropout during eval."""
        prev = self._mc_active
        try:
            self.enable_mc_dropout()
            yield self
        finally:
            self._mc_active = prev
            if not self._mc_active:
                self.disable_mc_dropout()

    def train(self, mode: bool = True):
        # Standard train()
        super().train(mode)
        return self

    def eval(self):
        # Standard eval(), with optional MC override
        super().eval()
        if self._mc_active:
            self._force_dropout_train_mode()
        return self

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
        return self.model.logit_scale

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)

    def _set_dropout_p(self, apply_to: ApplyTo, p: float) -> None:
        # Update dropout probability where requested
        if apply_to in (ApplyTo.TEXT, ApplyTo.BOTH):
            for m in self.model.text_model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = p
        if apply_to in (ApplyTo.IMAGE, ApplyTo.BOTH):
            for m in self.model.vision_model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = p

    def _force_dropout_train_mode(self) -> None:
        # Keep only Dropout layers stochastic; rest of the model stays in eval
        for m in self._iter_selected_dropout():
            m.train(True)

    def _iter_selected_dropout(self):
        # Yield Dropout modules in the selected towers
        if self.cfg.apply_to in (ApplyTo.TEXT, ApplyTo.BOTH):
            for m in self.model.text_model.modules():
                if isinstance(m, nn.Dropout):
                    yield m
        if self.cfg.apply_to in (ApplyTo.IMAGE, ApplyTo.BOTH):
            for m in self.model.vision_model.modules():
                if isinstance(m, nn.Dropout):
                    yield m
