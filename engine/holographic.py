"""
JEEVidya V5 — Holographic Formula Panels (Tier 2)
═════════════════════════════════════════════════
Floating LaTeX: MathRenderer output wrapped in a glowing glass panel
that bobs, tilts, and materializes with a scanline sweep. Pure PIL —
the expensive parts (LaTeX raster, glow, panel) are cached per formula;
per-frame work is one composite + offset.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

from engine.math_renderer import MathRenderer

_panel_cache: Dict[Tuple, Image.Image] = {}


def _build_panel(latex: str, color: Tuple[int, int, int],
                 glow_color: Tuple[int, int, int],
                 max_width: int, font_size: int) -> Image.Image:
    """Formula on a glass card with border glow. Cached — built once."""
    formula = MathRenderer.render(latex, color=color, font_size=font_size)
    if formula.width > max_width:
        ratio = max_width / formula.width
        formula = formula.resize(
            (max_width, max(1, int(formula.height * ratio))),
            Image.Resampling.LANCZOS)

    pad_x, pad_y = 54, 38
    w, h = formula.width + pad_x * 2, formula.height + pad_y * 2
    margin = 44                                     # room for the halo
    panel = Image.new("RGBA", (w + margin * 2, h + margin * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(panel)

    # Glass card with rich contrast
    card = (margin, margin, margin + w, margin + h)
    d.rounded_rectangle(card, radius=22, fill=(10, 14, 32, 225))
    d.rounded_rectangle(card, radius=22, outline=glow_color + (220,), width=4)
    # Corner ticks (sci-fi framing)
    for cx, cy, sx, sy in ((card[0], card[1], 1, 1), (card[2], card[1], -1, 1),
                           (card[0], card[3], 1, -1), (card[2], card[3], -1, -1)):
        d.line((cx, cy + sy * 26, cx, cy + sy * 8), fill=glow_color + (255,), width=4)
        d.line((cx + sx * 8, cy, cx + sx * 26, cy), fill=glow_color + (255,), width=4)

    # Halo: blur a copy of the border underneath
    halo = panel.filter(ImageFilter.GaussianBlur(14))
    panel = Image.alpha_composite(halo, panel)
    panel.alpha_composite(formula, (margin + pad_x, margin + pad_y))
    return panel


def formula_panel(latex: str, t_seconds: float, reveal: float,
                  color: Tuple[int, int, int] = (255, 255, 255),
                  glow_color: Tuple[int, int, int] = (0, 212, 255),
                  max_width: int = 960,
                  font_size: int = 46) -> Tuple[Optional[Image.Image], int]:
    """
    The per-frame holographic formula.

    Args:
        t_seconds: local scene time (drives the bob/breathe cycle)
        reveal: 0..1 materialization progress (scanline + fade)

    Returns:
        (RGBA image ready to composite, vertical bob offset in px)
        or (None, 0) while reveal <= 0.
    """
    if reveal <= 0.0 or not latex:
        return None, 0

    key = (latex, color, glow_color, max_width, font_size)
    base = _panel_cache.get(key)
    if base is None:
        base = _build_panel(latex, color, glow_color, max_width, font_size)
        _panel_cache[key] = base

    img = base.copy()

    if reveal < 1.0:
        # Materialize: alpha ramps up + scanline sweeps down the panel
        alpha = img.split()[3].point(lambda p: int(p * reveal))
        img.putalpha(alpha)
        d = ImageDraw.Draw(img)
        y = int(img.height * reveal)
        d.line((0, y, img.width, y), fill=glow_color + (180,), width=3)
        d.line((0, y + 4, img.width, y + 4), fill=(255, 255, 255, 60), width=1)

    # Idle life: slow bob + subtle glow breathing
    bob = int(math.sin(t_seconds * 1.6) * 7)
    breathe = 0.92 + 0.08 * math.sin(t_seconds * 2.3 + 1.0)
    if breathe < 1.0:
        alpha = img.split()[3].point(lambda p: int(p * breathe))
        img.putalpha(alpha)

    return img, bob


def clear_cache() -> None:
    _panel_cache.clear()
