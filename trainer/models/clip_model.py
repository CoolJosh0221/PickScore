import torch

from dataclasses import dataclass
from transformers import CLIPModel as HFCLIPModel
from transformers import CLIPConfig as HFCLIPConfig

from torch import nn

from trainer.models.base_model import BaseModelConfig

from typing import Callable


@dataclass
class ClipModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.clip_model.CLIPModel"
    pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32"
    dropout_rate: float = 0.0
    enable_mc_dropout: bool = False


class CLIPModel(nn.Module):
    def __init__(self, cfg: ClipModelConfig):
        super().__init__()

        # Load the pretrained model configuration
        pretrained_model_config = HFCLIPConfig.from_pretrained(
            cfg.pretrained_model_name_or_path
        )
        # # Set dropout rates in text encoder
        pretrained_model_config.text_config.attention_dropout = cfg.dropout_rate
        pretrained_model_config.text_config.dropout = cfg.dropout_rate

        # # Set dropout rates in vision encoder
        pretrained_model_config.vision_config.attention_dropout = cfg.dropout_rate
        pretrained_model_config.vision_config.dropout = cfg.dropout_rate

        self.model = HFCLIPModel.from_pretrained(
            cfg.pretrained_model_name_or_path, config=pretrained_model_config
        )

        # Add explicit dropout layers for feature outputs
        self.text_dropout = nn.Dropout(p=cfg.dropout_rate)
        self.image_dropout = nn.Dropout(p=cfg.dropout_rate)

        # Store the configuration
        self.dropout_rate = cfg.dropout_rate
        self.enable_mc_dropout = cfg.enable_mc_dropout

        self.acquisition_function: (
            Callable[[torch.Tensor, int], torch.Tensor] | None
        ) = torch.std

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
        self.model.save_pretrained(path)

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

    # def mc_inference(self, text_inputs=None, image_inputs=None, n_samples=10):
    #     """
    #     Perform MC Dropout inference with multiple forward passes to estimate uncertainty
    #     Args:
    #         text_inputs: Text inputs to the model
    #         image_inputs: Image inputs to the model
    #         n_samples: Number of MC samples to take

    #     Returns:
    #         Dictionary with mean and std of features, and samples
    #     """
    #     # Store original MC dropout state
    #     original_mc_state = self.enable_mc_dropout

    #     # Enable MC dropout for inference
    #     self.enable_mc_dropout = True
    #     self.eval()  # This won't disable dropout due to our override

    #     text_features_samples = []
    #     image_features_samples = []

    #     with torch.no_grad():
    #         for _ in range(n_samples):
    #             if text_inputs is not None:
    #                 text_features = self.get_text_features(text_inputs)
    #                 text_features_samples.append(text_features)

    #             if image_inputs is not None:
    #                 image_features = self.get_image_features(image_inputs)
    #                 image_features_samples.append(image_features)

    #     # Restore original MC dropout state
    #     self.enable_mc_dropout = original_mc_state
    #     if not self.enable_mc_dropout:
    #         self.eval()  # Reset to proper eval state

    #     # Process results
    #     results = {}
    #     if text_inputs is not None and text_features_samples:
    #         text_features_stack = torch.stack(text_features_samples)
    #         results["text_features_mean"] = torch.mean(text_features_stack, dim=0)
    #         results["text_features_std"] = torch.std(text_features_stack, dim=0)
    #         results["text_features_samples"] = text_features_stack

    #     if image_inputs is not None and image_features_samples:
    #         image_features_stack = torch.stack(image_features_samples)
    #         results["image_features_mean"] = torch.mean(image_features_stack, dim=0)
    #         results["image_features_std"] = torch.std(image_features_stack, dim=0)
    #         results["image_features_samples"] = image_features_stack

    #     return results

    def calc_probs_with_uncertainty(
        self, prompt, images, processor, n_samples=10, device="cuda"
    ):
        """
        Calculate preference probabilities with uncertainty estimation using MC Dropout

        Args:
            prompt: Text prompt
            images: List of images
            processor: CLIP processor
            n_samples: Number of MC samples
            device: Device to run inference on

        Returns:
            Dictionary with mean probabilities, standard deviations, and all samples
        """
        if self.acquisition_function is None:
            raise ValueError("Acquisition function must be set for this method.")
        elif not callable(self.acquisition_function):
            raise ValueError(
                "Acquisition function must be a callable function or method."
            )

        # Enable MC dropout and switch to evaluation mode
        print(f"Original image dimensions: {[img.size for img in images]}")

        original_mc_state = self.enable_mc_dropout
        self.enable_mc_dropout = True
        self.eval()

        # Process inputs with FIXED SIZE - key change to fix the error
        image_inputs = processor(
            images=images,
            # do_resize=True,
            size={
                "shortest_edge": 224,
                "longest_edge": 224,
            },  # Fixed size for CLIP ViT-H-14
            return_tensors="pt",
        ).to(device)

        text_inputs = processor(
            text=prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        all_probs = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Get features
                image_embs = self.get_image_features(**image_inputs)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

                text_embs = self.get_text_features(**text_inputs)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

                # Calculate scores
                scores = self.logit_scale.exp() * (text_embs @ image_embs.T)[0]
                probs = torch.softmax(scores, dim=-1)
                all_probs.append(probs.cpu())

        # Restore original MC dropout state
        self.enable_mc_dropout = original_mc_state
        if not self.enable_mc_dropout:
            self.eval()  # Reset to proper eval state

        # Calculate statistics
        all_probs_tensor = torch.stack(all_probs)
        mean_probs = torch.mean(all_probs_tensor, dim=0)
        uncertainty_probs = self.acquisition_function(all_probs_tensor, dim=0)

        return {
            "mean_probs": mean_probs.tolist(),
            "uncertainty_probs": uncertainty_probs.tolist(),
            "all_samples": all_probs_tensor.tolist(),
        }

    def calc_score_of_one_image_with_uncertainty(
        self, prompt, image, processor, n_samples=100, device="cuda"
    ):
        """
        Calculate preference probabilities with uncertainty estimation using MC Dropout

        Args:
            prompt: Text prompt
            image: Image (exactly one)
            processor: CLIP processor
            n_samples: Number of MC samples
            device: Device to run inference on

        Returns:
            Dictionary with mean probabilities, standard deviations, and all samples
        """
        # Enable MC dropout and switch to evaluation mode
        if self.acquisition_function is None:
            raise ValueError("Acquisition function must be set for this method.")
        elif not callable(self.acquisition_function):
            raise ValueError(
                "Acquisition function must be a callable function or method."
            )
            

        print(f"Original image dimensions: {image.size}")

        original_mc_state = self.enable_mc_dropout
        self.enable_mc_dropout = True
        self.eval()

        # Process inputs with FIXED SIZE - key change to fix the error
        image_inputs = processor(
            images=image,
            # do_resize=True,
            size={
                "shortest_edge": 224,
                "longest_edge": 224,
            },  # Fixed size for CLIP ViT-H-14
            return_tensors="pt",
        ).to(device)

        text_inputs = processor(
            text=prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        all_scores = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Get features
                image_embs = self.get_image_features(**image_inputs)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

                text_embs = self.get_text_features(**text_inputs)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

                # Calculate scores
                scores = self.logit_scale.exp() * (text_embs @ image_embs.T)[0]
                all_scores.append(scores.cpu())

        # Restore original MC dropout state
        self.enable_mc_dropout = original_mc_state
        if not self.enable_mc_dropout:
            self.eval()  # Reset to proper eval state

        # Calculate statistics
        all_scores_tensor = torch.stack(all_scores)
        mean_score = torch.mean(all_scores_tensor, dim=0)[0]
        std_score = torch.std(all_scores_tensor, dim=0)[0]
        var_score = torch.var(all_scores_tensor, dim=0)[0]
        cv_score = std_score / (mean_score + 1e-8)
        mad_score = torch.median(torch.abs(all_scores_tensor - torch.median(all_scores_tensor, dim=0)[0]), dim=0)[0]
        # different confidence interval width
        iqr_score = torch.quantile(all_scores_tensor, 0.75, dim=0)[0] - torch.quantile(all_scores_tensor, 0.25, dim=0)[0]
        ci90_score = torch.quantile(all_scores_tensor, 0.95, dim=0)[0] - torch.quantile(all_scores_tensor, 0.05, dim=0)[0]
        ci95_score = torch.quantile(all_scores_tensor, 0.975, dim=0)[0] - torch.quantile(all_scores_tensor, 0.25, dim=0)[0]

        return {
            "mean_score": mean_score.item(),
            "std_score": std_score.item(),
            "var_score": var_score.item(),
            "cv_score": cv_score.item(),
            "mad_score": mad_score.item(),
            "iqr_score": iqr_score.item(),
            "ci90_score": ci90_score.item(),
            "ci95_score": ci95_score.item(),
            "all_samples": all_scores_tensor.tolist(),
        }
