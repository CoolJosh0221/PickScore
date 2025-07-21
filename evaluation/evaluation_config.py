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
        # Long, detailed prompts
        "long": [
            # 1 – Futuristic megacity sunrise
            "Ultra‑wide cinematic view of cyberpunk megacity at pink‑orange dawn, neon skyscrapers, holographic billboards, maglev trains, wet reflective streets, crowds in glowing smart‑fabric, delivery drones, volumetric fog, hyper‑realistic 8K concept art by Syd Mead & Beeple",
            # 2 – Enchanted forest cathedral
            "Ancient redwood forest forming gothic cathedral nave, sunbeams through stained‑glass leaves, mossy stone altar, bioluminescent fireflies, soft green‑gold palette, high‑detail fantasy matte painting, 32‑bit HDR fog‑rays, magical realism by James Gurney & Studio Ghibli backgrounds",
            # 3 – Steampunk airship duel
            "Epic aerial duel: twin brass‑copper steam airships with gear propellers, patched canvas balloons, coal smoke, lightning from Tesla cannons, crimson sunset over Victorian London, low‑angle motion blur, oil‑paint texture, 4K chiaroscuro illustration by Gregory Manchess & Claire Wendling",
            # 4 – Neo‑Tokyo night market
            # "Near‑future Tokyo night market: glowing lanterns, sizzling yakitori sparks, holographic menus, rain‑slick pavement reflecting magenta‑cyan, crowds in tech‑wear, shallow DOF bokeh, 50 mm f/1.2 photoreal 16‑bit RAW, Blade Runner × Anthony Bourdain vibe",
            # 5 – Cosmic jellyfish ballet
            "Surreal underwater cosmos: planet‑sized translucent jellyfish, luminous tendrils trailing starlight, spiral galaxies in crystalline water, astronaut in antique diving suit, drifting zero‑g bubbles, rich purples & electric blues, ultra‑HD 12K dreamscape by Rafael Araujo & Roger Dean",
            # 6 – Gothic library
            # "Candle‑lit medieval library: towering black‑oak shelves spiraling into darkness, dust‑covered grimoires, wrought‑iron staircases, stained‑glass alchemy windows, dust motes in golden shafts, moody Baroque grading, 3‑point cinematic lighting, 4:5 aspect, intricate Gustave Doré‑style details",
            # 7 – Desert ruins blue hour
            # "Panoramic desert ruins: half‑buried Sumerian ziggurat amid shifting dunes, turquoise twilight with first stars, lone nomad in indigo robe holding lantern, wind‑rippled sand leading lines, subtle long‑exposure blur, photoreal 6K, soft bounce lighting, National Geographic look, Kodak Porta 800 grade",
            # 8 – Cyber‑fae couture
            # "Full‑body fashion portrait: ethereal elven model with iridescent circuit‑board tattoos, color‑shifting translucent nano‑silk gown, biomechanical butterfly wings glowing softly, dark minimalist studio, neon teal‑violet rim light, 85 mm lens look, razor‑sharp eyes, cinematic contrast, Vogue cover × art‑nouveau",
            # 9 – Mimi battle scene
            # "Mimi (IWTLYTYDD) in fierce battle stance, pastel‑pink hair, glowing violet eyes, sleek neon‑aqua combat suit, mid‑slash radiant magenta energy blade and sparks, shadowy wraith foes dissolving in turquoise glow, ruined gothic cathedral, low‑angle cinematic motion blur, dusk through stained glass, cel‑shaded anime, vivid colors, 8K",
        ],
    }

    # MC Dropout Configs
    NUM_SAMPLES = 20  # Number of MC dropout samples to average over
    PRETRAINED_MODEL_NAME = "openai/clip-vit-base-patch32"
    DROPOUT_RATE = 0.1

    def get_by_category(self, category: str) -> List[str]:
        if category in self.prompts:
            return self.prompts[category]
        else:
            raise KeyError(f"Category {category} not found")
