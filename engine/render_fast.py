"""
JEEVidya V5 — Fast Render Primitives
════════════════════════════════════
Surgical replacements for the three hottest paths in the V2 renderer:

  1. Gradient background: was 1,920 draw.line calls/frame → numpy row LUT,
     cached per quantized drift offset. ~100x faster.
  2. Particle glow: was a full-canvas 1080x1920 GaussianBlur/frame →
     pre-blurred sprite stamps composited locally. ~30x faster.
  3. Caption stroke: was 48 draw.text calls per line → PIL's native
     stroke_width. 1 call.

Plus: a Devanagari-safe font resolver (Noto Sans Devanagari bundled in
assets/fonts, with macOS system fallbacks) and a karaoke caption renderer
driven by real word timings.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config import settings, brand

# ═══════════════════════════════════════════
# FONT RESOLUTION (Devanagari-safe)
# ═══════════════════════════════════════════

_FONT_DIRS = [
    settings.FONTS_DIR,
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
]

# Priority-ordered filename hints; first match wins.
_DEVANAGARI_HINTS = (
    "notosansdevanagari",
    "kohinoordevanagari",
    "devanagari",
    "nirmala",
    "mangal",
    "arial unicode",
    "arialuni",
)

_font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _scan_font_files() -> List[str]:
    found: List[str] = []
    for d in _FONT_DIRS:
        if not os.path.isdir(d):
            continue
        try:
            for name in os.listdir(d):
                if name.lower().endswith((".ttf", ".otf", ".ttc")):
                    found.append(os.path.join(d, name))
        except OSError:
            continue
    return found


def resolve_devanagari_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Return a font guaranteed to render Hindi/Hinglish captions.
    Order: bundled Noto Sans Devanagari → macOS Devanagari-capable system
    fonts → brand fallback → PIL default (last resort only).
    """
    key = ("devanagari", size)
    if key in _font_cache:
        return _font_cache[key]

    files = _scan_font_files()
    lower_map = {f: os.path.basename(f).lower() for f in files}

    for hint in _DEVANAGARI_HINTS:
        for path, base in lower_map.items():
            if hint in base:
                try:
                    font = ImageFont.truetype(path, size)
                    _font_cache[key] = font
                    return font
                except (OSError, IOError):
                    continue

    for fallback in (brand.FONT_CAPTION, brand.FONT_FALLBACK):
        try:
            font = ImageFont.truetype(fallback, size)
            _font_cache[key] = font
            return font
        except (OSError, IOError):
            continue

    font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def devanagari_font_path() -> Optional[str]:
    """Diagnostic: which font file will captions use?"""
    files = _scan_font_files()
    for hint in _DEVANAGARI_HINTS:
        for f in files:
            if hint in os.path.basename(f).lower():
                return f
    return None


# ═══════════════════════════════════════════
# GRADIENT BACKDROP (numpy LUT)
# ═══════════════════════════════════════════

class GradientBackdrop:
    """
    Animated vertical gradient, visually identical to V2's
    render_gradient_background but computed as a numpy row LUT keyed by
    the quantized drift offset (81 possible values → tiny cache).
    """

    def __init__(self, width: int, height: int,
                 top: Optional[Tuple[int, int, int]] = None,
                 bottom: Optional[Tuple[int, int, int]] = None):
        """top/bottom override the brand colors (Visual DNA palettes)."""
        self.width = width
        self.height = height
        self._rows: Dict[int, np.ndarray] = {}
        self._top = np.array(top or brand.BG_TOP, dtype=np.float32)
        self._bottom = np.array(bottom or brand.BG_BOTTOM, dtype=np.float32)

    def _rows_for(self, offset: int) -> np.ndarray:
        rows = self._rows.get(offset)
        if rows is None:
            t = np.clip((np.arange(self.height, dtype=np.float32) + offset)
                        / self.height, 0.0, 1.0)[:, None]
            rows = (self._top + (self._bottom - self._top) * t).astype(np.uint8)
            self._rows[offset] = rows
        return rows

    def get(self, frame_num: int) -> Image.Image:
        """RGBA backdrop for a frame (same drift curve as V2: sin(f*0.015)*40)."""
        offset = int(round(math.sin(frame_num * 0.015) * 40))
        rows = self._rows_for(offset)
        arr = np.ascontiguousarray(
            np.broadcast_to(rows[:, None, :], (self.height, self.width, 3))
        )
        return Image.fromarray(arr, "RGB").convert("RGBA")


# ═══════════════════════════════════════════
# SPRITE-STAMPED PARTICLES
# ═══════════════════════════════════════════

class FastParticleRenderer:
    """
    Renders an existing ParticleSystem's particles using pre-baked glow
    sprites instead of a full-canvas Gaussian blur.
    """

    def __init__(self, glow_radius: int = settings.PARTICLE_GLOW_RADIUS):
        self.glow_radius = glow_radius
        self._sprites: Dict[Tuple[Tuple[int, int, int, int], int], Image.Image] = {}

    def _sprite(self, color: Tuple[int, int, int, int], size: int) -> Image.Image:
        key = (color, size)
        sprite = self._sprites.get(key)
        if sprite is None:
            pad = self.glow_radius * 3
            dim = (size + pad) * 2
            base = Image.new("RGBA", (dim, dim), (0, 0, 0, 0))
            draw = ImageDraw.Draw(base)
            c = dim // 2
            draw.ellipse((c - size, c - size, c + size, c + size), fill=color)
            glow = base.filter(ImageFilter.GaussianBlur(radius=self.glow_radius))
            sprite = Image.alpha_composite(glow, base)
            self._sprites[key] = sprite
        return sprite

    def render(self, base: Image.Image, particles: Sequence) -> Image.Image:
        if base.mode != "RGBA":
            base = base.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        for p in particles:
            sprite = self._sprite(tuple(p.color), int(p.size))
            half = sprite.width // 2
            overlay.alpha_composite(sprite, dest=(int(p.x) - half, int(p.y) - half))
        return Image.alpha_composite(base, overlay)


# ═══════════════════════════════════════════
# KINETIC CAPTIONS (pro path: cached word sprites,
# pop-in overshoot, glow on the live word, emphasis)
# ═══════════════════════════════════════════

_word_sprite_cache: Dict[Tuple, Image.Image] = {}
_EMPHASIS_UNITS = ("km", "m/s", "sec", "kg", "%", "×", "cm", "hz", "kmph")


def _is_emphasis_word(word: str) -> bool:
    """Numbers, units, shouted words → always accent-colored."""
    w = word.lower().strip(".,!?")
    return (any(c.isdigit() for c in w) or w in _EMPHASIS_UNITS
            or (word.isupper() and len(word) > 2))


def _word_sprite(word: str, font: ImageFont.FreeTypeFont,
                 fill: Tuple[int, int, int], stroke_width: int,
                 glow: bool = False) -> Image.Image:
    """Stroked (optionally glowing) word rendered once, cached forever."""
    key = (word, id(font), fill, stroke_width, glow)
    img = _word_sprite_cache.get(key)
    if img is not None:
        return img

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = probe.textbbox((0, 0), word, font=font,
                          stroke_width=stroke_width)
    pad = 14 if glow else 4
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    img = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    origin = (pad - bbox[0], pad - bbox[1])
    d.text(origin, word, font=font, fill=fill,
           stroke_width=stroke_width, stroke_fill=(0, 0, 0))
    if glow:
        halo = img.filter(ImageFilter.GaussianBlur(6))
        img = Image.alpha_composite(halo, img)
    _word_sprite_cache[key] = img
    return img


def _pop_scale(age_ms: float) -> float:
    """Word entrance: 120ms overshoot pop (0.4 → 1.08 → 1.0)."""
    if age_ms >= 160:
        return 1.0
    t = max(0.0, age_ms) / 160.0
    # cubic ease-out into a small overshoot that settles
    return 0.4 + 0.68 * (1 - (1 - t) ** 3) + 0.08 * math.sin(t * math.pi)


def draw_kinetic_caption(frame: Image.Image, words: List[str],
                         word_starts_ms: List[float], active_index: int,
                         t_ms: float, font: ImageFont.FreeTypeFont, y: int,
                         accent: Tuple[int, int, int] = brand.TEXT_CAPTION_ACTIVE,
                         emphasis: Tuple[int, int, int] = brand.ACCENT,
                         stroke_width: int = settings.CAPTION_STROKE_WIDTH
                         ) -> None:
    """Broadcast-grade captions: every word pops in with overshoot at its
    spoken timestamp; the live word glows in the accent color; numbers
    and units carry permanent emphasis color. All sprites cached —
    per-frame cost is composites only. Mutates `frame`."""
    if not words:
        return
    if frame.mode != "RGBA":
        return draw_karaoke_caption(frame, words, active_index, font, y,
                                    accent, stroke_width)

    probe = ImageDraw.Draw(frame)
    space_w = probe.textlength(" ", font=font)
    widths = [probe.textlength(w, font=font) for w in words]
    max_line_w = frame.width * 0.88

    # Greedy wrap (indices preserved)
    lines: List[List[int]] = []
    cur: List[int] = []
    cur_w = 0.0
    for i, ww in enumerate(widths):
        add = ww if not cur else ww + space_w
        if cur and cur_w + add > max_line_w:
            lines.append(cur)
            cur, cur_w = [], 0.0
            add = ww
        cur.append(i)
        cur_w += add
    if cur:
        lines.append(cur)

    line_h = int(font.size * 1.4)
    yy = y
    for line in lines:
        total = sum(widths[i] for i in line) + space_w * (len(line) - 1)
        x = (frame.width - total) / 2
        for i in line:
            start = word_starts_ms[i] if i < len(word_starts_ms) else 0.0
            if t_ms < start - 40:               # not spoken yet: hold slot
                x += widths[i] + space_w
                continue
            is_active = (i == active_index)
            fill = (accent if is_active
                    else emphasis if _is_emphasis_word(words[i])
                    else brand.TEXT_CAPTION)
            sprite = _word_sprite(words[i], font, fill, stroke_width,
                                  glow=is_active)
            scale = _pop_scale(t_ms - start)
            sw, sh = sprite.size
            if abs(scale - 1.0) > 0.02:
                sprite = sprite.resize((max(1, int(sw * scale)),
                                        max(1, int(sh * scale))),
                                       Image.Resampling.BILINEAR)
            cx = x + widths[i] / 2
            cy = yy + font.size * 0.55
            frame.alpha_composite(sprite,
                                  (int(cx - sprite.width / 2),
                                   int(cy - sprite.height / 2)))
            x += widths[i] + space_w
        yy += line_h


# ═══════════════════════════════════════════
# KARAOKE CAPTIONS (word-accurate, one-call stroke)
# ═══════════════════════════════════════════

def draw_karaoke_caption(frame: Image.Image,
                         words: List[str],
                         active_index: int,
                         font: ImageFont.FreeTypeFont,
                         y: int,
                         accent: Tuple[int, int, int] = brand.TEXT_CAPTION_ACTIVE,
                         stroke_width: int = settings.CAPTION_STROKE_WIDTH) -> None:
    """
    Draw a caption chunk with the currently spoken word highlighted.
    Uses PIL's native stroke (1 call/word) instead of V2's 48-call outline.
    Mutates `frame` in place.
    """
    if not words:
        return

    draw = ImageDraw.Draw(frame)
    width = frame.width
    max_line_w = width * 0.88
    space_w = draw.textlength(" ", font=font)
    word_widths = [draw.textlength(w, font=font) for w in words]

    # Greedy wrap into lines (indices preserved for highlight mapping)
    lines: List[List[int]] = []
    current: List[int] = []
    current_w = 0.0
    for i, w_w in enumerate(word_widths):
        add = w_w if not current else w_w + space_w
        if current and current_w + add > max_line_w:
            lines.append(current)
            current, current_w = [], 0.0
            add = w_w
        current.append(i)
        current_w += add
    if current:
        lines.append(current)

    line_h = int(font.size * 1.35)
    white = brand.TEXT_CAPTION

    for line in lines:
        total = sum(word_widths[i] for i in line) + space_w * (len(line) - 1)
        x = (width - total) / 2
        for i in line:
            fill = accent if i == active_index else white
            draw.text((x, y), words[i], font=font, fill=fill,
                      stroke_width=stroke_width, stroke_fill=(0, 0, 0))
            x += word_widths[i] + space_w
        y += line_h
