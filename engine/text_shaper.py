"""
JEEVidya V5 — Complex-Script Text Shaper (captions)
═══════════════════════════════════════════════════
Hindi/Hinglish captions need real OpenType shaping. Without it the
i-matra (ि) lands on the WRONG side of its consonant, conjuncts (क्र, त्र)
fall apart into dotted-circle fragments, and reph/anusvara float. PIL's
basic layout engine cannot shape Devanagari; PIL only shapes correctly
when libraqm + fribidi are present on the host — which they are not on
most macOS/Linux render boxes and never on CI.

This module makes captions correct on EVERY machine:

  1. HarfBuzz (uharfbuzz wheel) shapes the string → glyph ids + positions,
     with Devanagari reordering, conjunct ligatures and mark attachment.
  2. FreeType (freetype-py wheel) rasterises each glyph, and its Stroker
     builds a TRUE round-joined outline — no dilation squares, no gaps.
  3. Result: a `WordRaster` — fill mask, stroke mask, metrics — that the
     caption renderer colourises and caches.

If either wheel is missing we degrade to PIL (using raqm when available)
and print ONE loud warning, so a silent regression to broken Hindi is
impossible.

All fonts are bundled in assets/fonts (SIL OFL): Baloo 2 (variable, set
to ExtraBold) as the display face, Mukta ExtraBold as the static fallback.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config import settings

try:  # optional, wheel-only dependencies
    import uharfbuzz as _hb
    import freetype as _ft
    HAS_SHAPER = True
except Exception:  # pragma: no cover - exercised only on hosts without wheels
    _hb = None
    _ft = None
    HAS_SHAPER = False

# Weight requested from variable fonts (Baloo 2 'wght' axis)
CAPTION_WEIGHT = 800

_BUNDLED_FONTS = (
    "Baloo2-Variable.ttf",      # rounded geometric Devanagari + Latin
    "Mukta-ExtraBold.ttf",      # static heavy fallback
)

_warned: Dict[str, bool] = {}


def _warn_once(key: str, msg: str) -> None:
    if not _warned.get(key):
        _warned[key] = True
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        print(f"  [captions] WARNING: {msg}")


# ═══════════════════════════════════════════
# FONT FILE RESOLUTION
# ═══════════════════════════════════════════

def caption_font_path() -> Optional[str]:
    """The font file captions use. Env override → bundled → system scan."""
    override = os.environ.get("JV_CAPTION_FONT")
    if override and os.path.isfile(override):
        return override
    for name in _BUNDLED_FONTS:
        p = os.path.join(settings.FONTS_DIR, name)
        if os.path.isfile(p):
            return p
    # System Devanagari fonts (macOS / Linux) as a last resort
    candidates = (
        "/System/Library/Fonts/Supplemental/Kohinoor Devanagari.ttc",
        "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    )
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def font_is_variable(path: str) -> bool:
    return "variable" in os.path.basename(path).lower() or "[" in path


# ═══════════════════════════════════════════
# WORD RASTER
# ═══════════════════════════════════════════

@dataclass
class WordRaster:
    """A shaped, rasterised word in a tight canvas.

    fill / stroke: uint8 coverage masks of identical shape. `stroke`
    already covers the fill region (outline + interior) so colouring is
    a straight paint-under.
    origin_x: canvas x of the pen origin (can be > 0 for left-hanging
              marks or the stroke overhang). baseline: canvas row of
              the baseline. advance: pen advance in px (layout width).
    """
    fill: np.ndarray
    stroke: np.ndarray
    origin_x: int
    baseline: int
    advance: float

    @property
    def width(self) -> int:
        return int(self.fill.shape[1])

    @property
    def height(self) -> int:
        return int(self.fill.shape[0])


class TextShaper:
    """One shaper per (font file, pixel size, stroke width)."""

    def __init__(self, path: str, size_px: int, stroke_px: int = 0,
                 weight: int = CAPTION_WEIGHT):
        self.path = path
        self.size = int(size_px)
        self.stroke = int(max(0, stroke_px))
        self.weight = weight
        self._cache: Dict[str, WordRaster] = {}
        self.backend = "pil"
        self._pil_font: Optional[ImageFont.FreeTypeFont] = None

        if HAS_SHAPER:
            try:
                self._init_hb_ft()
                self.backend = "harfbuzz"
            except Exception as e:  # pragma: no cover
                _warn_once("hbft", f"HarfBuzz/FreeType init failed ({e}); "
                           "falling back to PIL text layout — Hindi matras "
                           "may be mis-ordered.")
        else:
            _warn_once("nohb", "uharfbuzz/freetype-py not installed — Hindi "
                       "captions will use PIL basic layout (matra order "
                       "may be wrong). `pip install uharfbuzz freetype-py`.")
        if self.backend == "pil":
            self._init_pil()

    # ─── backends ─────────────────────────────────────────

    def _init_hb_ft(self) -> None:
        blob = _hb.Blob.from_file_path(self.path)
        face = _hb.Face(blob)
        self.hb_font = _hb.Font(face)
        self.upem = face.upem or 1000
        # 26.6 fixed point so positions match FreeType exactly
        self.hb_font.scale = (self.size * 64, self.size * 64)
        axes = [a.tag for a in face.axis_infos]
        self.ft_face = _ft.Face(self.path)
        self.ft_face.set_pixel_sizes(0, self.size)
        if "wght" in axes:
            self.hb_font.set_variations({"wght": self.weight})
            try:
                coords = []
                for a in face.axis_infos:
                    coords.append(self.weight if a.tag == "wght"
                                  else a.default_value)
                self.ft_face.set_var_design_coords(coords)
            except Exception:
                pass
        m = self.ft_face.size
        self.ascender = m.ascender / 64.0
        self.descender = m.descender / 64.0      # negative
        self.line_height = m.height / 64.0
        self._load_flags = (_ft.FT_LOAD_NO_HINTING | _ft.FT_LOAD_NO_BITMAP)

    def _init_pil(self) -> None:
        try:
            layout = (ImageFont.Layout.RAQM
                      if ImageFont.core.HAVE_RAQM else ImageFont.Layout.BASIC)
        except Exception:
            layout = None
        try:
            if layout is not None:
                self._pil_font = ImageFont.truetype(self.path, self.size,
                                                    layout_engine=layout)
            else:
                self._pil_font = ImageFont.truetype(self.path, self.size)
            if font_is_variable(self.path):
                try:
                    self._pil_font.set_variation_by_axes([self.weight])
                except Exception:
                    pass
        except Exception:
            self._pil_font = ImageFont.load_default()
        asc, desc = self._pil_font.getmetrics()
        self.ascender = float(asc)
        self.descender = -float(desc)
        self.line_height = float(asc + desc) * 1.05

    # ─── public API ───────────────────────────────────────

    def measure(self, text: str) -> float:
        """Advance width in px (layout width, excludes stroke overhang)."""
        return self.raster(text).advance

    def raster(self, text: str) -> WordRaster:
        r = self._cache.get(text)
        if r is None:
            r = (self._raster_hbft(text) if self.backend == "harfbuzz"
                 else self._raster_pil(text))
            self._cache[text] = r
        return r

    # ─── HarfBuzz + FreeType path ─────────────────────────

    def _shape(self, text: str):
        buf = _hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        _hb.shape(self.hb_font, buf, {"kern": True, "liga": True,
                                      "calt": True})
        return buf.glyph_infos, buf.glyph_positions

    def _raster_hbft(self, text: str) -> WordRaster:
        infos, positions = self._shape(text)
        ft = self.ft_face
        placed: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
        pen_x = 0.0
        advance = 0.0
        stroker = None
        if self.stroke > 0:
            stroker = _ft.Stroker()
            stroker.set(self.stroke * 64, _ft.FT_STROKER_LINECAP_ROUND,
                        _ft.FT_STROKER_LINEJOIN_ROUND, 0)

        for info, pos in zip(infos, positions):
            gid = info.codepoint
            gx = pen_x + pos.x_offset / 64.0
            gy = pos.y_offset / 64.0
            ft.load_glyph(gid, self._load_flags)
            glyph = ft.glyph.get_glyph()
            # Fill bitmap
            fb = glyph.to_bitmap(_ft.FT_RENDER_MODE_NORMAL, _ft.Vector(0, 0),
                                 False)
            fill_arr, fl, ftop = self._bitmap_to_array(fb)
            # Stroke bitmap (outline + interior)
            if stroker is not None:
                sg = ft.glyph.get_glyph()
                sg.stroke(stroker, True)
                sb = sg.to_bitmap(_ft.FT_RENDER_MODE_NORMAL,
                                  _ft.Vector(0, 0), True)
                stroke_arr, sl, stop = self._bitmap_to_array(sb)
            else:
                stroke_arr, sl, stop = fill_arr, fl, ftop
            placed.append((int(round(gx)) + fl, -int(round(gy)) - ftop,
                           fill_arr, None))
            placed.append((int(round(gx)) + sl, -int(round(gy)) - stop,
                           None, stroke_arr))
            pen_x += pos.x_advance / 64.0
        advance = pen_x

        # Canvas bounds (relative to pen origin at (0, baseline=0))
        xs0, ys0, xs1, ys1 = [], [], [], []
        for x, y, f, s in placed:
            a = f if f is not None else s
            if a is None or a.size == 0:
                continue
            xs0.append(x); ys0.append(y)
            xs1.append(x + a.shape[1]); ys1.append(y + a.shape[0])
        if not xs0:  # whitespace-only
            h = int(self.line_height) + 2
            return WordRaster(np.zeros((h, 1), np.uint8),
                              np.zeros((h, 1), np.uint8), 0,
                              int(self.ascender), advance)
        pad = 2
        x0, y0 = min(xs0) - pad, min(ys0) - pad
        x1, y1 = max(xs1) + pad, max(ys1) + pad
        W, H = x1 - x0, y1 - y0
        fill = np.zeros((H, W), np.uint8)
        stroke = np.zeros((H, W), np.uint8)
        for x, y, f, s in placed:
            a = f if f is not None else s
            if a is None or a.size == 0:
                continue
            dst = fill if f is not None else stroke
            yy, xx = y - y0, x - x0
            region = dst[yy:yy + a.shape[0], xx:xx + a.shape[1]]
            np.maximum(region, a, out=region)
        return WordRaster(fill, stroke, origin_x=-x0, baseline=-y0,
                          advance=advance)

    @staticmethod
    def _bitmap_to_array(bitmap_glyph) -> Tuple[np.ndarray, int, int]:
        bm = bitmap_glyph.bitmap
        w, h, pitch = bm.width, bm.rows, bm.pitch
        if w == 0 or h == 0:
            return np.zeros((0, 0), np.uint8), bitmap_glyph.left, bitmap_glyph.top
        buf = np.array(bm.buffer, dtype=np.uint8)
        arr = buf.reshape(h, abs(pitch))[:, :w].copy()
        return arr, bitmap_glyph.left, bitmap_glyph.top

    # ─── PIL fallback ─────────────────────────────────────

    def _raster_pil(self, text: str) -> WordRaster:
        font = self._pil_font
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        try:
            advance = float(probe.textlength(text, font=font))
        except Exception:
            advance = float(len(text) * self.size * 0.5)
        bbox = probe.textbbox((0, 0), text, font=font,
                              stroke_width=self.stroke, anchor="ls")
        pad = 2
        x0, y0 = int(bbox[0]) - pad, int(bbox[1]) - pad
        x1, y1 = int(bbox[2]) + pad, int(bbox[3]) + pad
        W, H = max(1, x1 - x0), max(1, y1 - y0)
        fill_img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(fill_img).text((-x0, -y0), text, font=font, fill=255,
                                      anchor="ls")
        if self.stroke > 0:
            stroke_img = Image.new("L", (W, H), 0)
            ImageDraw.Draw(stroke_img).text((-x0, -y0), text, font=font,
                                            fill=255, anchor="ls",
                                            stroke_width=self.stroke,
                                            stroke_fill=255)
        else:
            stroke_img = fill_img
        return WordRaster(np.asarray(fill_img, dtype=np.uint8),
                          np.asarray(stroke_img, dtype=np.uint8),
                          origin_x=-x0, baseline=-y0, advance=advance)


# ═══════════════════════════════════════════
# SHAPER CACHE
# ═══════════════════════════════════════════

_shapers: Dict[Tuple[str, int, int], TextShaper] = {}


def get_shaper(size_px: int, stroke_px: int = 0,
               path: Optional[str] = None) -> TextShaper:
    p = path or caption_font_path()
    if p is None:
        _warn_once("nofont", "No Devanagari-capable font found. Bundled fonts "
                   "are missing from assets/fonts — captions will render "
                   "with PIL's bitmap default.")
        p = ""
    key = (p, int(size_px), int(stroke_px))
    s = _shapers.get(key)
    if s is None:
        if not p:
            s = _DefaultShaper(size_px, stroke_px)
        else:
            s = TextShaper(p, size_px, stroke_px)
        _shapers[key] = s
    return s


class _DefaultShaper(TextShaper):
    """Last-resort shaper when no font file exists at all."""

    def __init__(self, size_px: int, stroke_px: int):
        self.path = ""
        self.size = int(size_px)
        self.stroke = int(stroke_px)
        self.weight = CAPTION_WEIGHT
        self._cache = {}
        self.backend = "pil"
        self._pil_font = ImageFont.load_default()
        asc, desc = self._pil_font.getmetrics()
        self.ascender = float(asc)
        self.descender = -float(desc)
        self.line_height = float(asc + desc) * 1.05


def devanagari_order_ok(sample: str = "कि") -> bool:
    """Self-check: does the active backend reorder the i-matra before its
    consonant? True means Hindi will look right."""
    if not HAS_SHAPER:
        try:
            return bool(ImageFont.core.HAVE_RAQM)
        except Exception:
            return False
    p = caption_font_path()
    if not p:
        return False
    s = get_shaper(40, 0, p)
    if s.backend != "harfbuzz":
        return False
    infos, _ = s._shape(sample)
    # After reordering, the first glyph belongs to cluster 0 but is NOT the
    # consonant's nominal glyph: HarfBuzz emits the matra first.
    nominal = s.hb_font.get_nominal_glyph(ord(sample[0]))
    return len(infos) >= 2 and infos[0].codepoint != nominal


__all__ = ["TextShaper", "WordRaster", "get_shaper", "caption_font_path",
           "devanagari_order_ok", "HAS_SHAPER", "CAPTION_WEIGHT"]
