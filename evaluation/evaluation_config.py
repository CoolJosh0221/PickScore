from dataclasses import dataclass
from typing import Dict, List, ClassVar


@dataclass
class EvaluationConfig:
    WIDTH = 224
    HEIGHT = 224
    NUM_IMAGES = 25
    prompts: ClassVar[Dict[str, List[str]]] = {
            "clear": [
                "A red apple on a white table, professional photography, sharp focus",
                "Single origami crane, white paper, minimal shadow, studio lighting",
                "Chess piece (king) in dramatic lighting, black background",
            ],
            # Abstract/artistic prompts
            "abstract": [
                "Dreams melting into reality, surreal abstract art, Salvador Dali style",
                "The sound of jazz visualized, abstract expressionism, vibrant colors",
                "Mathematical beauty, fractals morphing into butterflies",
            ],
            # Ambiguous/vague prompts
            "ambiguous": [
                "Something between a forest and an ocean",
                "The edge of understanding",
                "Almost but not quite a face",
            ],
            # Complex scene prompts
            "complex": [
                "Busy Tokyo street at night, rain, neon reflections, crowds with umbrellas",
                "Steampunk airship battle above Victorian London, detailed, epic scale",
                "Underwater coral reef city with bioluminescent creatures",
            ],
            # Contradiction/impossible prompts
            "contradictory": [
                "A colorless green idea sleeping furiously",
                "Square circle floating in impossible space",
                "Transparent metal organic hybrid creature",
            ],
            # Style mixing prompts
            "style_mixing": [
                "Mona Lisa painted by Picasso in anime style",
                "Van Gogh's Starry Night but it's a photograph",
                "Minecraft landscape with photorealistic textures",
            ],
        }
    
    # MC Dropout Configs
    NUM_SAMPLES = 1000 # Number of MC dropout samples to average over
    PRETRAINED_MODEL_NAME = "openai/clip-vit-base-patch32"
    DROPOUT_RATE = 0.1

    def get_by_category(self, category: str) -> List[str]:
        if category in self.prompts:
            return self.prompts[category]
        else:
            raise KeyError(f"Category {category} not found")
