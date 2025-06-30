from pathlib import Path
from PIL import Image

import torch
from transformers import CLIPProcessor

from trainer.models.clip_model import CLIPModel, ClipModelConfig
from evaluation.evaluation_config import EvaluationConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Initialize model configuration with MC dropout
model_config = ClipModelConfig(
    pretrained_model_name_or_path=EvaluationConfig.PRETRAINED_MODEL_NAME,
    dropout_rate=EvaluationConfig.DROPOUT_RATE,
    enable_mc_dropout=False,  # Start with MC dropout disabled
)

# 2. Create the model
model = CLIPModel(model_config)
model = model.to(device)

# 3. Load pretrained weights
print("Loading pretrained weights...")
state_dict = torch.load(
    "outputs/checkpoint-final/pytorch_model.bin", weights_only=False
)
model.model.load_state_dict(state_dict)
print("Weights loaded successfully")

# 5. Load processor for inference
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print(f"Processor name: {processor.__class__.__name__}")
print(f"Processor image size: {processor.image_processor.size}")
print(f"Resize in processor: {processor.image_processor.do_resize}")


real_image_directory = Path("test_images/real_images")

def evaluate_once(image_path: str | Path, prompt) -> None:
    print("\nRunning MC Dropout inference...")

    image = Image.open(image_path)
    # Use the utility method for MC dropout inference
    mc_results = model.calc_score_of_one_image_with_uncertainty(
        prompt=prompt,
        image=image,
        processor=processor,
        n_samples=EvaluationConfig.NUM_SAMPLES,
        device=device,
    )

    # Display results
    # print(f"All scores: {mc_results['all_samples']}")
    print(f"Mean score: {mc_results['mean_score']}")
    print(f"Standard deviation: {mc_results['std_score']}")
    print(f"Variance: {mc_results['std_score']}")
    print(f"Coefficient of variation: {mc_results['cv_score']}")
    print(f"Median absolute deviation: {mc_results['mad_score']}")
    print(f"IQR: {mc_results['iqr_score']}")
    print(f"90% confidence interval width: {mc_results['ci90_score']}")
    print(f"95% confidence interval width: {mc_results['ci95_score']}")

def evaluate_all() -> None:
    raise NotImplementedError("evaluate_all() function is still being implemented")
    for style in EvaluationConfig.prompts.keys():
        for i in range(len(EvaluationConfig.prompts[style])):
            img_path = real_image_directory / style / "{}".format(i)


def main() -> None: ...


if __name__ == "__main__":
    main()
