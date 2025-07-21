import json
from pathlib import Path
from PIL import Image

import torch
from transformers import CLIPProcessor
from icecream import ic

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


def evaluate_once(image: Image.Image, prompt: str) -> dict[str, float]:
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
    # print(f"Mean score: {mc_results['mean_score']}")
    # print(f"Standard deviation: {mc_results['std_score']}")
    # print(f"Variance: {mc_results['std_score']}")
    # print(f"Coefficient of variation: {mc_results['cv_score']}")
    # print(f"Median absolute deviation: {mc_results['mad_score']}")
    # print(f"IQR: {mc_results['iqr_score']}")
    # print(f"90% confidence interval width: {mc_results['ci90_score']}")
    # print(f"95% confidence interval width: {mc_results['ci95_score']}")
    mc_results.pop("all_samples", None)
    return mc_results


def evaluate_for_one_prompt(style: str, prompt_id: int) -> dict[int, dict[str, float]]:
    print(f"Evaluating for style {style} prompt {prompt_id}")
    mc_results: dict[int, dict[str, float]] = dict()
    path_to_directory = Path(f"test_images/real_images/{style}/{prompt_id}")
    for i in range(EvaluationConfig.NUM_IMAGES):
        img_path = path_to_directory / f"{i}.png"
        with Image.open(img_path) as img:
            mc_results[i] = evaluate_once(
                img, EvaluationConfig.prompts[style][prompt_id]
            )

    return mc_results


def evaluate_for_one_style(style: str) -> dict[int, dict[int, dict[str, float]]]:
    mc_results = dict()
    for prompt_id in range(len(EvaluationConfig.prompts[style])):
        mc_results[prompt_id] = evaluate_for_one_prompt(style, prompt_id)
    return mc_results


def evaluate_all() -> dict[str, dict[int, dict[int, dict[str, float]]]]:
    mc_results = dict()
    for style in EvaluationConfig.prompts:
        mc_results[style] = evaluate_for_one_style(style)

    return mc_results


def main() -> None:
    # results = evaluate_all()
    # with open("output.json", "w") as json_file:
    #     json.dump(results, json_file)
    additional_results = evaluate_for_one_style("long")
    with open("output.json", "r") as json_file:
        mc_results = json.load(json_file)

    mc_results["long"] = additional_results
    with open("output_with_long_prompts.json", "w") as json_file:
        mc_results = json.dump(mc_results, json_file)


if __name__ == "__main__":
    main()
