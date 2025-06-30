from builtins import isinstance
from PIL import Image

def is_image_black(image_or_image_path) -> bool:
    if isinstance(image_or_image_path, str):
        with Image.open(image_or_image_path) as img:
            img = img.convert("RGB")
            return img.getbbox() is None
    elif isinstance(image_or_image_path, Image.Image):
        img = image_or_image_path.convert("RGB")
        return img.getbbox() is None
    else:
        raise ValueError("Input must be a file path or a PIL Image object.")


if __name__ == "__main__":
    path = "test_images/real_images/0/45.png"
    print(is_image_black(path))