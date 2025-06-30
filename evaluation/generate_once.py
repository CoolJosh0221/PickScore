# import torch
# from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# device = "cuda" if torch.cuda.is_available() else "cpu"
# prompt = "A cat holding a sign that says Hello World"

# pipe = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     torch_dtype=torch.float16,
#     variant="fp16",
#     use_safetensors=True,
# ).to("cuda")

# # Optimal scheduler for 4090
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(
#     pipe.scheduler.config,
#     algorithm_type="dpmsolver++",
#     use_karras_sigmas=True
# )

# # Enable optimizations for 4090
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_vae_slicing()
# pipe.enable_vae_tiling()

# # Generate test images
# image = pipe(prompt, num_inference_steps=50).images[0]

# image = pipe(
#     prompt,
#     num_inference_steps=25,  # Good balance
#     guidance_scale=7.5,
#     height=1024,
#     width=1024,
# ).images[0]

# image.save("tmp.png")

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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
pipe("A girl smiling", num_inference_steps=4, guidance_scale=0).images[0].save("output.png")
