"""
Gudiya & Chintu — Background Removal Pipeline
Uses rembg to strip backgrounds from character images.
Tuned for 3D Pixar-style characters with alpha_matting_erode_size=8
to eliminate white halo artifacts.
"""
import os
from typing import Optional

import numpy as np
from PIL import Image

from config import settings


_session = None


def _get_session():
    """Lazy-init rembg session (one-time model load)."""
    global _session
    if _session is None:
        import rembg
        _session = rembg.new_session(settings.REMBG_MODEL)
    return _session


def remove_background(image: Image.Image) -> Image.Image:
    """
    Remove background from a PIL Image.
    Returns RGBA image with transparent background.
    Uses alpha matting with erosion to kill edge halos.
    """
    import rembg

    session = _get_session()
    img_array = np.array(image)

    result = rembg.remove(
        img_array,
        session=session,
        alpha_matting=settings.REMBG_ALPHA_MATTING,
        alpha_matting_foreground_threshold=settings.REMBG_FOREGROUND_THRESHOLD,
        alpha_matting_background_threshold=settings.REMBG_BACKGROUND_THRESHOLD,
        alpha_matting_erode_size=settings.REMBG_ERODE_SIZE,
    )

    return Image.fromarray(result)


def prepare_character_assets(character_name: str) -> str:
    """
    Process a character's original image:
    1. Remove background
    2. Save as body.png in the character's directory
    
    Returns the path to the processed body.png.
    """
    char_dir = os.path.join(settings.CHARACTERS_DIR, character_name)
    os.makedirs(char_dir, exist_ok=True)

    # Find original image
    original: Optional[Image.Image] = None
    original_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        path = os.path.join(char_dir, f"original{ext}")
        if os.path.exists(path):
            original = Image.open(path).convert('RGBA')
            original_path = path
            break

    if original is None:
        raise FileNotFoundError(
            f"No original image found in {char_dir}. "
            f"Place character1.jpg or character2.png as 'original.jpg/png'."
        )

    # Remove background
    print(f"  [BG Remove] Processing {original_path}...")
    clean = remove_background(original)

    # Save
    body_path = os.path.join(char_dir, "body.png")
    clean.save(body_path, "PNG")
    print(f"  [BG Remove] Saved {body_path}")

    # Also save as 'neutral' if no expression images exist
    neutral_path = os.path.join(char_dir, "neutral.png")
    if not os.path.exists(neutral_path):
        clean.save(neutral_path, "PNG")
        print(f"  [BG Remove] Also saved as neutral: {neutral_path}")

    return body_path


def prepare_all_characters() -> None:
    """Process all characters in the assets directory."""
    for name in os.listdir(settings.CHARACTERS_DIR):
        char_dir = os.path.join(settings.CHARACTERS_DIR, name)
        if os.path.isdir(char_dir):
            # Check if already processed
            body_path = os.path.join(char_dir, "body.png")
            if os.path.exists(body_path):
                print(f"  [BG Remove] {name}/body.png already exists, skipping.")
                continue
            try:
                prepare_character_assets(name)
            except FileNotFoundError as e:
                print(f"  [BG Remove] Warning: {e}")
