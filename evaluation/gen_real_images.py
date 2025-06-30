import torch

from pathlib import Path

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from evaluation.check_black import is_image_black
from evaluation.evaluation_config import EvaluationConfig

"""
pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", use_safetensors=True
)
pipeline.to("cuda")
"""

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Ensure using the same inference steps as the loaded model and CFG set to 0.

for style in EvaluationConfig.prompts.keys():
    parent_directory_path = Path(f"test_images/real_images/{style}")
    parent_directory_path.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(EvaluationConfig.prompts[style]):
        child_directory_path = parent_directory_path / "{}".format(i)
        child_directory_path.mkdir(parents=True, exist_ok=True)
        for j in range(EvaluationConfig.NUM_IMAGES):
            img = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
            while is_image_black(img):
                print(f"Image {i}-{j} is black, regenerating...")
                img = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]

            img.save(child_directory_path / f"{j}.png")
            print(f"Saved image {style}-{i}-{j}")
