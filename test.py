import torch
from torchinfo import summary

from trainer.models.clip_model import ClipModelConfig, CLIPModel

def count_dropouts(model):
    return sum(isinstance(module, torch.nn.Dropout) for module in model.modules())

def main() -> None:
    cfg = ClipModelConfig()
    cfg.pretrained_model_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model = CLIPModel(cfg)
    state_dict = torch.load("outputs/checkpoint-gstep100/pytorch_model.bin", weights_only=False)

    model.load_state_dict(state_dict)
    model.train()
    summary(model)
    print(count_dropouts(model))


if __name__ == "__main__":
    main()