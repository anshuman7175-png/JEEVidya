"""
JEEVidya V5 — Production Caption Renderer (YouTube Shorts)
══════════════════════════════════════════════════════════
Karaoke captions the way the top Shorts channels burn them in:

  • Baloo 2 ExtraBold, shaped by HarfBuzz (correct Devanagari matras,
    conjuncts, reph) and rasterised by FreeType with a TRUE round-joined
    stroke — no dilation squares, no gaps, no tofu.
  • Composited AFTER the lens stack: never bloomed, never chromatically
    fringed, never gate-weaved, never grained. Crisp on every frame.
  • Word-accurate: every word pops in on its own onset (overshoot ease),
    the spoken word is coloured and grows 6 % — a crisp scale, not a
    blurred glow. Layout is computed ONCE per chunk at final scale so
    no neighbour ever shifts when the active word changes.
  • Shorts safe zones: the band is centred in the upper-middle of the
    frame, capped to CAPTION_MAX_WIDTH_FRAC, wrapped to at most
    CAPTION_MAX_LINES lines, and shrinks its font (never overflows) when
    a chunk is too wide. Orphan single words are pulled back onto the
    previous line.
  • Zero dark fringing: word sprites are built in straight (un-premul)
    RGBA from coverage masks; the scale-up for the active word goes
    through premultiplied resampling.

Public API (used by pipeline/compositor_v5.py):

    style = CaptionStyle.for_frame(width, height, font_scale=1.0)
    renderer = CaptionRenderer(style)
    frame = renderer.draw(frame, words, word_start_ms, active_idx, t_ms,
                          accent=(r,g,b), emphasis=(r,g,b), y_frac=0.40)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

from config import brand, settings
from engine.text_shaper import TextShaper, WordRaster, get_shaper

RGB = Tuple[int, int, int]

_NUM_RE = re.compile(r"\d")
_EMPH_WORDS = {
    # Hinglish emphasis vocabulary — numbers are caught by _NUM_RE
    "kyun", "kyon", "kaise", "क्यों", "कैसे", "wow", "arre", "अरे",
    "important", "trick", "formula", "answer", "जवाब", "secret",
}


def is_emphasis_word(word: str) -> bool:
    w = word.strip().strip("?!.,;:।").lower()
    return bool(_NUM_RE.search(w)) or w in _EMPH_WORDS


# ═══════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════

@dataclass(frozen=True)
class CaptionStyle:
    """Everything that is fixed for a render (resolution-scaled)."""
    frame_w: int
    frame_h: int
    font_px: int
    stroke_px: int
    line_height: float = settings.CAPTION_LINE_HEIGHT
    max_width_frac: float = settings.CAPTION_MAX_WIDTH_FRAC
    max_lines: int = settings.CAPTION_MAX_LINES
    word_gap_frac: float = 0.30            # gap between words, × font_px
    fill: RGB = brand.TEXT_CAPTION
    stroke: RGB = (12, 12, 20)             # near-black, not pure black
    shadow_opacity: int = settings.CAPTION_SHADOW_OPACITY
    shadow_blur: int = settings.CAPTION_SHADOW_BLUR
    pop_ms: float = float(settings.CAPTION_POP_MS)
    active_scale: float = settings.CAPTION_ACTIVE_SCALE

    @classmethod
    def for_frame(cls, width: int, height: int,
                  font_scale: float = 1.0) -> "CaptionStyle":
        rs = width / float(settings.WIDTH)
        font_px = max(18, int(round(settings.CAPTION_FONT_SIZE * rs * font_scale)))
        stroke_px = max(2, int(round(settings.CAPTION_STROKE_WIDTH * rs
                                     * math.sqrt(font_scale))))
        return cls(frame_w=int(width), frame_h=int(height),
                   font_px=font_px, stroke_px=stroke_px)

    @property
    def max_width_px(self) -> int:
        return int(self.frame_w * self.max_width_frac)

    def scaled(self, factor: float) -> "CaptionStyle":
        """Same style, smaller type (used when a chunk cannot fit)."""
        return CaptionStyle(
            frame_w=self.frame_w, frame_h=self.frame_h,
            font_px=max(18, int(self.font_px * factor)),
            stroke_px=max(2, int(round(self.stroke_px * factor))),
            line_height=self.line_height, max_width_frac=self.max_width_frac,
            max_lines=self.max_lines, word_gap_frac=self.word_gap_frac,
            fill=self.fill, stroke=self.stroke,
            shadow_opacity=self.shadow_opacity, shadow_blur=self.shadow_blur,
            pop_ms=self.pop_ms, active_scale=self.active_scale)


# ═══════════════════════════════════════════
# WORD SPRITES (coloured, cached)
# ═══════════════════════════════════════════

def _colourise(raster: WordRaster, fill: RGB, stroke: RGB) -> Image.Image:
    """Straight-alpha RGBA sprite: stroke painted UNDER the fill.

    Built from coverage masks so every transparent pixel carries the
    stroke colour (not black) — no dark halo when it is later resampled.
    """
    s = raster.stroke.astype(np.float32) / 255.0
    f = raster.fill.astype(np.float32) / 255.0
    h, w = s.shape
    out = np.empty((h, w, 4), dtype=np.float32)
    # Colour = lerp(stroke, fill, fill_coverage) — everywhere, including
    # fully transparent pixels, so resampling can never pull in black.
    for c in range(3):
        out[..., c] = stroke[c] + (fill[c] - stroke[c]) * f
    out[..., 3] = np.maximum(s, f) * 255.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def _resize_premultiplied(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """LANCZOS resize with premultiplied alpha → no edge darkening."""
    if size == img.size:
        return img
    arr = np.asarray(img, dtype=np.float32)
    a = arr[..., 3:4] / 255.0
    pre = np.concatenate([arr[..., :3] * a, arr[..., 3:4]], axis=2)
    pre_img = Image.fromarray(np.clip(pre, 0, 255).astype(np.uint8), "RGBA")
    small = np.asarray(pre_img.resize(size, Image.Resampling.LANCZOS),
                       dtype=np.float32)
    a2 = small[..., 3:4]
    safe = np.where(a2 > 0.5, a2, 1.0)
    rgb = np.clip(small[..., :3] * 255.0 / safe, 0, 255)
    rgb = np.where(a2 > 0.5, rgb, small[..., :3])
    out = np.concatenate([rgb, a2], axis=2)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


@dataclass
class _Placed:
    text: str
    raster: WordRaster
    x: int            # canvas x of the raster's left edge (band-local)
    y: int            # canvas y of the raster's top edge (band-local)
    cx: float         # word centre (band-local) — scale pivot
    cy: float


@dataclass
class _Layout:
    """One chunk, laid out once at a given style."""
    style: CaptionStyle
    placed: List[_Placed]
    width: int
    height: int
    shadow: Optional[Image.Image] = None
    sprites: Dict[Tuple[str, RGB], Image.Image] = field(default_factory=dict)


# ═══════════════════════════════════════════
# RENDERER
# ═══════════════════════════════════════════

class CaptionRenderer:
    """Layout + karaoke compositor. One instance per render."""

    def __init__(self, style: CaptionStyle):
        self.style = style
        self._shapers: Dict[int, TextShaper] = {}
        self._layouts: Dict[Tuple[Tuple[str, ...], int], _Layout] = {}
        self.font_path = self._shaper(style).path
        self.backend = self._shaper(style).backend

    # ─── shaping helpers ──────────────────────────────────

    def _shaper(self, style: CaptionStyle) -> TextShaper:
        s = self._shapers.get(style.font_px)
        if s is None:
            s = get_shaper(style.font_px, style.stroke_px)
            self._shapers[style.font_px] = s
        return s

    # ─── layout ───────────────────────────────────────────

    def _wrap(self, words: Sequence[str], style: CaptionStyle
              ) -> Optional[List[List[int]]]:
        """Greedy wrap to max_width; returns line → word indices, or
        None if it needs more than max_lines."""
        shaper = self._shaper(style)
        gap = style.font_px * style.word_gap_frac
        max_w = style.max_width_px
        lines: List[List[int]] = [[]]
        cur_w = 0.0
        for i, w in enumerate(words):
            adv = shaper.measure(w)
            add = adv if not lines[-1] else gap + adv
            if lines[-1] and cur_w + add > max_w:
                lines.append([i])
                cur_w = adv
            else:
                lines[-1].append(i)
                cur_w += add
        # Orphan control: a lone last word joins the previous line if
        # the result still fits — otherwise rebalance the two lines.
        if len(lines) >= 2 and len(lines[-1]) == 1 and len(lines[-2]) >= 2:
            prev = lines[-2]
            joined = prev + lines[-1]
            width = sum(shaper.measure(words[i]) for i in joined) \
                + gap * (len(joined) - 1)
            if width <= max_w:
                lines = lines[:-2] + [joined]
            else:
                lines = lines[:-2] + [prev[:-1], [prev[-1]] + lines[-1]]
        if len(lines) > style.max_lines:
            return None
        return lines

    def _layout(self, words: Sequence[str]) -> _Layout:
        key = (tuple(words), self.style.font_px)
        lay = self._layouts.get(key)
        if lay is not None:
            return lay

        style = self.style
        lines = self._wrap(words, style)
        # Shrink-to-fit: never overflow the safe zone, never exceed 2 lines
        factor = 1.0
        while lines is None and factor > 0.55:
            factor -= 0.08
            style = self.style.scaled(factor)
            lines = self._wrap(words, style)
        if lines is None:                       # pathological: hard-split
            lines = [list(range(len(words)))][:style.max_lines]

        shaper = self._shaper(style)
        gap = style.font_px * style.word_gap_frac
        line_h = style.font_px * style.line_height
        ascent = shaper.ascender
        pad = style.stroke_px + style.shadow_blur * 2 + 4

        placed: List[_Placed] = []
        line_widths: List[float] = []
        for li, idxs in enumerate(lines):
            adv = [shaper.measure(words[i]) for i in idxs]
            line_widths.append(sum(adv) + gap * (len(idxs) - 1))
        band_w = int(math.ceil(max(line_widths) if line_widths else 1)) + pad * 2
        band_h = int(math.ceil(line_h * len(lines))) + pad * 2

        for li, idxs in enumerate(lines):
            pen = (band_w - line_widths[li]) / 2.0
            baseline = pad + li * line_h + (line_h - style.font_px) / 2.0 + ascent
            for i in idxs:
                r = shaper.raster(words[i])
                x = int(round(pen - r.origin_x))
                y = int(round(baseline - r.baseline))
                placed.append(_Placed(words[i], r, x, y,
                                      cx=pen + r.advance / 2.0,
                                      cy=baseline - style.font_px * 0.36))
                pen += r.advance + gap

        lay = _Layout(style=style, placed=placed, width=band_w, height=band_h)
        if style.shadow_opacity > 0:
            lay.shadow = self._build_shadow(lay)
        self._layouts[key] = lay
        return lay

    def _build_shadow(self, lay: _Layout) -> Image.Image:
        """Soft, un-offset contact shadow behind the whole chunk. This is
        legibility insurance on bright backgrounds — not a 3D effect."""
        style = lay.style
        mask = np.zeros((lay.height, lay.width), dtype=np.uint8)
        for p in lay.placed:
            s = p.raster.stroke
            h, w = s.shape
            y0, x0 = max(0, p.y), max(0, p.x)
            y1, x1 = min(lay.height, p.y + h), min(lay.width, p.x + w)
            if y1 <= y0 or x1 <= x0:
                continue
            region = mask[y0:y1, x0:x1]
            np.maximum(region, s[y0 - p.y:y1 - p.y, x0 - p.x:x1 - p.x],
                       out=region)
        m = Image.fromarray(mask, "L")
        # Dilate slightly then blur → a soft plate hugging the letters
        m = m.filter(ImageFilter.MaxFilter(3))
        m = m.filter(ImageFilter.GaussianBlur(style.shadow_blur))
        a = (np.asarray(m, dtype=np.float32) * (style.shadow_opacity / 255.0))
        # Shadow is offset 0/+2px: a contact shadow, not a drop shadow
        sh = np.zeros((lay.height, lay.width, 4), dtype=np.uint8)
        sh[..., 3] = np.clip(a, 0, 255).astype(np.uint8)
        img = Image.fromarray(sh, "RGBA")
        dy = max(1, style.font_px // 30)
        return img.transform(img.size, Image.Transform.AFFINE,
                             (1, 0, 0, 0, 1, -dy),
                             resample=Image.Resampling.BILINEAR)

    def _sprite(self, lay: _Layout, p: _Placed, fill: RGB) -> Image.Image:
        key = (p.text, fill)
        img = lay.sprites.get(key)
        if img is None:
            img = _colourise(p.raster, fill, lay.style.stroke)
            lay.sprites[key] = img
        return img

    # ─── timing ───────────────────────────────────────────

    @staticmethod
    def _pop(age_ms: float, pop_ms: float) -> Tuple[float, float]:
        """(scale, alpha) for a word `age_ms` after its onset.
        Back-out overshoot: 0.72 → 1.08 → 1.0, alpha ramps in 40 %."""
        if age_ms <= 0:
            return 0.0, 0.0
        t = min(1.0, age_ms / max(1.0, pop_ms))
        alpha = min(1.0, t / 0.4)
        # ease-out-back (Penner) with a gentle overshoot
        c1, c3 = 1.35, 2.35
        eased = 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2
        scale = 0.72 + 0.28 * eased
        return scale, alpha

    # ─── draw ─────────────────────────────────────────────

    def band_bounds(self, words: Sequence[str], y_frac: float
                    ) -> Tuple[int, int, int, int]:
        """Frame-space (x0, y0, x1, y1) of the caption band for a chunk —
        the compositor uses this to keep heads clear of the text."""
        lay = self._layout(words)
        cy = int(self.style.frame_h * y_frac)
        y0 = cy - lay.height // 2
        x0 = (self.style.frame_w - lay.width) // 2
        return x0, y0, x0 + lay.width, y0 + lay.height

    def draw(self, frame: Image.Image, words: Sequence[str],
             word_start_ms: Sequence[float], active: int, t_ms: float,
             accent: RGB = brand.TEXT_CAPTION_ACTIVE,
             emphasis: RGB = brand.ACCENT,
             y_frac: float = settings.CAPTION_Y_POSITION) -> Image.Image:
        """Composite the karaoke chunk onto `frame` (RGB or RGBA).

        words / word_start_ms: the chunk. active: index of the word
        being spoken (-1 = none yet). Words whose onset is in the future
        are not drawn; words within pop_ms of onset animate in.
        """
        if not words:
            return frame
        lay = self._layout(words)
        style = lay.style

        band = Image.new("RGBA", (lay.width, lay.height), (0, 0, 0, 0))
        if lay.shadow is not None:
            # Only shadow words that are visible (mask by pop alpha ≈ 1)
            visible = all(t_ms - s >= style.pop_ms * 0.4
                          for s in word_start_ms[:max(0, active + 1)]) \
                and active >= len(words) - 1
            if visible:
                band.alpha_composite(lay.shadow)
            else:
                band.alpha_composite(self._partial_shadow(lay, word_start_ms,
                                                          t_ms))

        for i, p in enumerate(lay.placed):
            onset = word_start_ms[i] if i < len(word_start_ms) else 0.0
            age = t_ms - onset
            if age < 0:
                continue
            scale, alpha = self._pop(age, style.pop_ms)
            if alpha <= 0.01:
                continue
            is_active = (i == active)
            colour = accent if is_active else (
                emphasis if is_emphasis_word(p.text) else style.fill)
            sprite = self._sprite(lay, p, colour)
            if is_active:
                scale *= style.active_scale
            if abs(scale - 1.0) > 0.01:
                w = max(1, int(round(sprite.width * scale)))
                h = max(1, int(round(sprite.height * scale)))
                sprite = _resize_premultiplied(sprite, (w, h))
                # Scale about the word's own centre → neighbours never move
                x = int(round(p.cx - (p.cx - p.x) * scale))
                y = int(round(p.cy - (p.cy - p.y) * scale))
            else:
                x, y = p.x, p.y
            if alpha < 0.99:
                a = sprite.getchannel("A").point(lambda v, k=alpha: int(v * k))
                sprite = sprite.copy()
                sprite.putalpha(a)
            band.alpha_composite(sprite, (x, y))

        # Place the band: centred, y_frac is the band centre
        cy = int(style.frame_h * y_frac)
        x0 = (style.frame_w - lay.width) // 2
        y0 = cy - lay.height // 2
        y0 = max(0, min(style.frame_h - lay.height, y0))
        if frame.mode == "RGBA":
            frame.alpha_composite(band, (x0, y0))
            return frame
        rgb = frame if frame.mode == "RGB" else frame.convert("RGB")
        rgb.paste(band, (x0, y0), band)
        return rgb

    def _partial_shadow(self, lay: _Layout, word_start_ms: Sequence[float],
                        t_ms: float) -> Image.Image:
        """Shadow for only the words already on screen."""
        style = lay.style
        mask = np.zeros((lay.height, lay.width), dtype=np.uint8)
        any_word = False
        for i, p in enumerate(lay.placed):
            onset = word_start_ms[i] if i < len(word_start_ms) else 0.0
            if t_ms < onset:
                continue
            any_word = True
            s = p.raster.stroke
            h, w = s.shape
            y0, x0 = max(0, p.y), max(0, p.x)
            y1, x1 = min(lay.height, p.y + h), min(lay.width, p.x + w)
            if y1 <= y0 or x1 <= x0:
                continue
            region = mask[y0:y1, x0:x1]
            np.maximum(region, s[y0 - p.y:y1 - p.y, x0 - p.x:x1 - p.x],
                       out=region)
        out = Image.new("RGBA", (lay.width, lay.height), (0, 0, 0, 0))
        if not any_word:
            return out
        m = Image.fromarray(mask, "L").filter(ImageFilter.MaxFilter(3)) \
            .filter(ImageFilter.GaussianBlur(style.shadow_blur))
        a = np.asarray(m, dtype=np.float32) * (style.shadow_opacity / 255.0)
        arr = np.zeros((lay.height, lay.width, 4), dtype=np.uint8)
        arr[..., 3] = np.clip(a, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGBA")


__all__ = ["CaptionRenderer", "CaptionStyle", "is_emphasis_word"]
