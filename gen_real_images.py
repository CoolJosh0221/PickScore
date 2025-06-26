from pathlib import Path
from diffusers import StableDiffusionPipeline

WIDTH = 224
HEIGHT = 224
NUM_SAMPLES = 50

evaluation_prompts = [
    # Clear, well-defined prompts (should have lower uncertainty)
    "A red apple on a white table, professional photography, sharp focus",
    "Minimalist geometric shapes, black triangles on white background, clean design",
    "Portrait of a golden retriever, outdoor, sunny day, high resolution",
    
    # Complex/artistic prompts (might have higher uncertainty)
    "Dreams melting into reality, surreal abstract art, Salvador Dali style",
    "Cyberpunk city reflected in a rain puddle, neon lights, chaos and order",
    "The feeling of nostalgia visualized as colors and shapes",
    
    # Ambiguous prompts (should have higher uncertainty)
    "Something between a forest and an ocean",
    "Texture of forgotten memories, ethereal and fading",
    
    # Technical/specific prompts (moderate uncertainty)
    "Steampunk mechanical butterfly, brass gears, intricate details, macro shot",
    "Japanese tea ceremony in traditional room, soft lighting, peaceful atmosphere"
]


pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True
)
pipeline.to("cuda")

for i, prompt in enumerate(evaluation_prompts):
    images = pipeline(prompt, width=WIDTH, height=HEIGHT, num_images_per_prompt=NUM_SAMPLES).images
    directory_path = Path(f"test_images/real_images/{i}")
    directory_path.mkdir(parents=True, exist_ok=True)
    for j, image in enumerate(images):
        image.save(f"test_images/real_images/{i}/{j}.png")
        print(f"Saved image {i}-{j}")