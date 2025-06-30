import numpy as np
from PIL import Image

WIDTH = 224
HEIGHT = 224
NUM_SAMPLES = 500

for i in range(NUM_SAMPLES):
    imarray = np.random.rand(HEIGHT, WIDTH, 3) * 255
    img = Image.fromarray(imarray.astype(np.uint8)).convert("RGB")
    img.save(f"test_images/random_images/random_image_{i:03d}.png")