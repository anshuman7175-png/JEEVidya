"""
Gudiya & Chintu — Animated Background Generator
Renders the slowly shifting gradient background and chalkboard overlay.
"""
import math
from PIL import Image, ImageDraw

from config import settings, brand


def render_gradient_background(frame_num: int, width: int = settings.WIDTH,
                                height: int = settings.HEIGHT) -> Image.Image:
    """
    Generate an animated gradient background.
    The gradient slowly shifts vertically over time, creating a living feel.
    """
    img = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(img)

    # Slow vertical drift based on frame number
    offset = math.sin(frame_num * 0.015) * 40

    r1, g1, b1 = brand.BG_TOP
    r2, g2, b2 = brand.BG_BOTTOM

    for y in range(height):
        t = min(1.0, max(0.0, (y + offset) / height))
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    return img


def render_chalkboard_overlay(width: int = settings.WIDTH,
                               height: int = settings.HEIGHT) -> Image.Image:
    """
    Generate a chalkboard-style background for explanation scenes.
    Dark green with subtle grid lines.
    """
    img = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(img)

    # Fill with chalkboard color
    draw.rectangle([(0, 0), (width, height)], fill=brand.CHALKBOARD_BG + (255,))

    # Draw subtle grid lines
    grid_spacing = 60
    grid_color = brand.CHALKBOARD_GRID + (30,)  # Very faint

    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

    return img
