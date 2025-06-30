import random
from PIL import Image

WIDTH = 224
HEIGHT = 224
NUM_SAMPLES = 500

for i in range(NUM_SAMPLES):
    # Generate a random RGB color
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    # Create an image filled with the random color
    img = Image.new("RGB", (WIDTH, HEIGHT), (r, g, b))

    # Save the image
    img.save(f"test_images/solid_colors/solid_color_{i:03d}.png")