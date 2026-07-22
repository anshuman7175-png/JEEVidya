"""
Gudiya & Chintu — Glow Effect Engine
Adds bloom/glow effect behind key elements like formulas and reveal text.
"""
from PIL import Image, ImageDraw, ImageFilter

from config import brand


def apply_glow(base_image: Image.Image, element_image: Image.Image,
               position: tuple, radius: int = brand.GLOW_RADIUS,
               intensity: float = brand.GLOW_INTENSITY) -> Image.Image:
    """
    Add a glow effect behind an element.
    
    1. Takes the element image (RGBA)
    2. Applies GaussianBlur to create a soft bloom
    3. Composites the bloom behind the sharp element
    """
    if base_image.mode != 'RGBA':
        base_image = base_image.convert('RGBA')

    # Create glow layer — same size as base
    glow_layer = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
    glow_layer.paste(element_image, position, element_image)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=radius))

    # Reduce glow intensity by adjusting alpha
    glow_data = glow_layer.split()
    if len(glow_data) == 4:
        alpha = glow_data[3].point(lambda p: int(p * intensity))
        glow_layer = Image.merge('RGBA', (*glow_data[:3], alpha))

    # Composite: base → glow → sharp element
    result = Image.alpha_composite(base_image, glow_layer)

    # Paste sharp element on top
    sharp_layer = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
    sharp_layer.paste(element_image, position, element_image)
    result = Image.alpha_composite(result, sharp_layer)

    return result


def create_text_with_glow(text: str, font, color: tuple,
                          glow_color: tuple = None,
                          glow_radius: int = 12) -> Image.Image:
    """
    Render text with a glow effect behind it.
    Returns an RGBA image of the glowing text.
    """
    if glow_color is None:
        glow_color = color

    # Measure text size
    dummy = Image.new('RGBA', (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + glow_radius * 4
    h = bbox[3] - bbox[1] + glow_radius * 4

    # Create text image
    text_img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_img)
    tx = glow_radius * 2
    ty = glow_radius * 2
    draw.text((tx, ty), text, font=font, fill=glow_color + (200,))

    # Create glow
    glow = text_img.filter(ImageFilter.GaussianBlur(radius=glow_radius))

    # Render sharp text on top
    sharp = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sharp)
    draw.text((tx, ty), text, font=font, fill=color + (255,))

    result = Image.alpha_composite(glow, sharp)
    return result
