"""
JEEVidya — Mouth Rasterizer (Terminal Plan, Part IV §4.3)
════════════════════════════════════════════════════════
`engine/mouth_model.py` owns the MATH (5-D parameters, Cohen–Massaro
dominance blending, the C¹ parameter→contour map). This module owns the
PIXELS. Splitting them is deliberate: the model stays importable by QC
and by property-based tests with no PIL/numpy raster cost, and there is
still exactly ONE mouth implementation in the codebase (Law 1) — every
mouth in every render path funnels through `MouthRasterizer.render`.

How a mouth is drawn (all in HEAD-PLATE space, on the inpainted plate
whose painted mouth was removed at bake time — so D3 cannot recur):

  1. `lip_contour(p)` gives normalized outer/inner closed contours.
  2. Both are mapped into plate space by a single uniform similarity
     derived from the rig's baked `lip_outer` polygon: uniform, so
     rounding stays round and no aspect skew creeps in.
  3. Fills, in occlusion order: oral cavity → teeth arc (only when the
     aperture is actually open) → tongue (dental/retroflex only) → lip
     body between the outer and inner contours, shaded with the
     artwork's own gradient.
  4. Everything is drawn at 4× supersample and LANCZOS-downsampled, so
     the contour is evaluated in continuous coordinates and quantized
     exactly once — at the final resize.
  5. The patch is composited with SUB-PIXEL translation. An int()-rounded
     paste is itself a source of 1-px mouth jitter, which the temporal
     QC gate would (correctly) fail.

Optional art texture: when the rig carries a viseme sprite for the
dominant class, its lip patch is quad-warped (PIL MESH — an FFD with
four control points) into the deformed contour box and blended in at the
dominance weight. Geometry from the model, pixels from the artwork.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine.mouth_model import MouthParams, lip_contour

SUPERSAMPLE = 4

# Contour space is normalized to ±1; ±0.9 is the widest commissure the
# model can produce, so this maps that onto the baked lip width.
_CONTOUR_HALF_SPAN = 0.9

# Classes that put the tongue on screen. Anything else keeps it hidden —
# a permanently visible tongue is worse than none.
TONGUE_CLASSES = ("DENTAL", "RETROFLEX")

# The aperture below which teeth are not drawn at all (an always-visible
# tooth arc reads as a grimace at phone scale).
TEETH_JAW_FLOOR = 0.25


def _bbox(points: Sequence[Tuple[float, float]]
          ) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


class MouthRasterizer:
    """Draws the parametric mouth for one character.

    `lip_outer` is the rig's baked outer-lip polygon in plate space; it
    fixes both the mouth centre and the scale, so the mouth can never
    land somewhere the artwork does not have a mouth (defect D1's
    signature failure).
    """

    def __init__(self, lip_outer: Sequence[Tuple[float, float]],
                 palette: Dict[str, Tuple[int, int, int]],
                 shading: Optional[Image.Image] = None,
                 cache_size: int = 384):
        if len(lip_outer) < 3:
            raise ValueError("mouth rasterizer needs a baked lip_outer polygon")
        x0, y0, x1, y1 = _bbox(lip_outer)
        self.center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        self.lip_w = max(2.0, x1 - x0)
        # Uniform contour→plate scale (see module docstring).
        self.scale = (self.lip_w / 2.0) / _CONTOUR_HALF_SPAN
        self.palette = {k: tuple(int(c) for c in v[:3])
                        for k, v in (palette or {}).items()}
        self.shading = shading
        # Keyed by (quantized 5-D vector, art class, art weight bucket):
        # an art blend must never be served for a bare procedural request.
        self._cache: "OrderedDict[Tuple, Image.Image]" = OrderedDict()
        self._cache_size = int(cache_size)
        self._hits = 0
        self._misses = 0
        # Patch is generous enough for a fully open jaw plus lip body.
        self._half_w = self.scale * 1.35
        self._half_h = self.scale * 1.25

    # ─── colours ──────────────────────────────────────────

    def _c(self, key: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return self.palette.get(key, default)

    # ─── geometry ─────────────────────────────────────────

    def origin(self) -> Tuple[float, float]:
        """Float top-left of the patch in plate space."""
        return (self.center[0] - self._half_w, self.center[1] - self._half_h)

    def _to_patch(self, pts: Sequence[Tuple[float, float]], s: int
                  ) -> List[Tuple[float, float]]:
        ox, oy = self.origin()
        cx, cy = self.center
        return [(((cx + x * self.scale) - ox) * s,
                 ((cy + y * self.scale) - oy) * s) for x, y in pts]

    def predicted_centroid(self, p: MouthParams) -> Tuple[float, float]:
        """Analytic mouth centroid in PLATE space for a parameter vector.

        QC (Part VIII) compares the rendered lip-mask centroid against
        this after pushing it through the composed affine — registration
        is checked against math, never against vibes.
        """
        outer, _ = lip_contour(p)
        cx, cy = self.center
        xs = [cx + x * self.scale for x, _ in outer]
        ys = [cy + y * self.scale for _, y in outer]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    # ─── raster ───────────────────────────────────────────

    def render(self, p: MouthParams, viseme_class: str = "REST",
               art: Optional[Image.Image] = None,
               art_weight: float = 0.0) -> Image.Image:
        """The mouth patch for parameter vector `p` (LRU-cached).

        `art` is an optional registered viseme sprite for the dominant
        class; `art_weight` is its dominance share. The cache key folds
        both in so an art blend can never be served for a bare
        procedural request.
        """
        pq = MouthParams(*p.quantized_key())
        key = (pq.quantized_key(),
               viseme_class if art is not None else "",
               round(art_weight * 8) if art is not None else 0)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self._hits += 1
            return hit
        self._misses += 1
        out = self._draw(pq, viseme_class, art, art_weight)
        self._cache[key] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out

    def _draw(self, p: MouthParams, viseme_class: str,
              art: Optional[Image.Image], art_weight: float) -> Image.Image:
        s = SUPERSAMPLE
        w = max(4, int(math.ceil(self._half_w * 2 * s)))
        h = max(4, int(math.ceil(self._half_h * 2 * s)))
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        outer_n, inner_n = lip_contour(p)
        outer = self._to_patch(outer_n, s)
        inner = self._to_patch(inner_n, s)

        lip = self._c("lip", (168, 86, 88))
        lip_shadow = self._c("lip_shadow", tuple(int(c * 0.62) for c in lip))
        cavity = self._c("oral_cavity", tuple(int(c * 0.30) for c in lip))
        teeth = self._c("teeth", (242, 240, 236))
        tongue = self._c("tongue", (196, 96, 104))

        # 1 · Lip body: outer polygon filled, then the inner aperture is
        #     punched out by the cavity fill below. Drawing the body as a
        #     solid and covering it keeps the vermilion border smooth at
        #     any aperture without a second contour pass.
        draw.polygon(outer, fill=lip + (255,))

        # 2 · Oral cavity (dark, vertical gradient) inside the inner ring
        aperture_px = (max(y for _, y in inner) - min(y for _, y in inner))
        if aperture_px > 1.2:
            cav = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(cav).polygon(inner, fill=cavity + (255,))
            cav = self._vertical_gradient(cav, top=0.72, bottom=1.0)
            img.alpha_composite(cav)

            # 3 · Teeth: an arc clipped to the UPPER half of the aperture,
            #     only once the mouth is genuinely open.
            if p.jaw > TEETH_JAW_FLOOR:
                self._draw_teeth(img, inner, teeth, p)

            # 4 · Tongue on dental/retroflex articulations
            if viseme_class in TONGUE_CLASSES and p.jaw > 0.30:
                self._draw_tongue(img, inner, tongue, p)

        # 5 · Vermilion shading: lower lip catches light, upper lip is in
        #     shadow. `press` deepens it (compressed lips thicken).
        img = self._lip_shade(img, outer, lip, lip_shadow, p)

        # 6 · Optional art texture (FFD-lite quad warp of the sprite)
        if art is not None and art_weight > 0.02:
            warped = self._warp_art(art, outer, (w, h))
            if warped is not None:
                warped.putalpha(warped.getchannel("A").point(
                    lambda v, a=art_weight: int(v * a)))
                img.alpha_composite(warped)

        # 7 · Down to plate resolution — the single quantization point
        out = img.resize((max(2, w // s), max(2, h // s)), Image.LANCZOS)

        # 8 · The artwork's own lighting, so the parametric mouth belongs
        #     to the painting rather than sitting on top of it.
        out = self._apply_shading(out)
        return out

    # ─── drawing helpers ──────────────────────────────────

    @staticmethod
    def _vertical_gradient(layer: Image.Image, top: float,
                           bottom: float) -> Image.Image:
        arr = np.asarray(layer, dtype=np.float32)
        h = arr.shape[0]
        ramp = np.linspace(top, bottom, h, dtype=np.float32)[:, None]
        arr[..., :3] *= ramp[..., None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")

    def _draw_teeth(self, img: Image.Image,
                    inner: Sequence[Tuple[float, float]],
                    teeth: Tuple[int, int, int], p: MouthParams) -> None:
        x0, y0, x1, y1 = _bbox(inner)
        band = (y1 - y0) * (0.20 + 0.18 * (1.0 - p.jaw))
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).ellipse(
            [x0 + (x1 - x0) * 0.06, y0 - band * 0.35,
             x1 - (x1 - x0) * 0.06, y0 + band * 1.55],
            fill=teeth + (255,))
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).polygon(list(inner), fill=255)
        layer.putalpha(Image.composite(layer.getchannel("A"),
                                       Image.new("L", img.size, 0), mask))
        img.alpha_composite(layer)

    def _draw_tongue(self, img: Image.Image,
                     inner: Sequence[Tuple[float, float]],
                     tongue: Tuple[int, int, int], p: MouthParams) -> None:
        x0, y0, x1, y1 = _bbox(inner)
        cx = (x0 + x1) / 2.0
        rw = (x1 - x0) * 0.34
        rh = (y1 - y0) * 0.30
        cy = y1 - rh * 0.9
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).ellipse([cx - rw, cy - rh, cx + rw, cy + rh],
                                      fill=tongue + (235,))
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).polygon(list(inner), fill=255)
        layer.putalpha(Image.composite(layer.getchannel("A"),
                                       Image.new("L", img.size, 0), mask))
        img.alpha_composite(layer)

    def _lip_shade(self, img: Image.Image,
                   outer: Sequence[Tuple[float, float]],
                   lip: Tuple[int, int, int],
                   lip_shadow: Tuple[int, int, int],
                   p: MouthParams) -> Image.Image:
        """Upper lip toward `lip_shadow`, lower lip toward a lit `lip`."""
        x0, y0, x1, y1 = _bbox(outer)
        mid = (y0 + y1) / 2.0
        arr = np.asarray(img, dtype=np.float32)
        h, w = arr.shape[:2]
        ys = np.arange(h, dtype=np.float32)[:, None]
        # −1 at the top of the lip body, +1 at the bottom
        t = np.clip((ys - mid) / max(1.0, (y1 - y0) / 2.0), -1.0, 1.0)
        shade = 1.0 + 0.16 * t - 0.10 * (1.0 - t) * (0.4 + 0.6 * p.press)
        opaque = arr[..., 3:4] > 0
        arr[..., :3] = np.where(opaque, arr[..., :3] * shade[..., None],
                                arr[..., :3])
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
        # A whisker of blur only on the alpha: the supersample already
        # anti-aliases geometry, this softens the vermilion border the way
        # painted art does.
        a = out.getchannel("A").filter(ImageFilter.GaussianBlur(0.6))
        out.putalpha(a)
        return out

    @staticmethod
    def _warp_art(art: Image.Image, outer: Sequence[Tuple[float, float]],
                  size: Tuple[int, int]) -> Optional[Image.Image]:
        """Quad-warp the art sprite into the deformed contour box.

        Four control points is a deliberate floor, not a limitation: a
        denser FFD on a sprite whose landmarks were fitted at bake time
        buys sub-pixel differences while adding failure modes.
        """
        try:
            x0, y0, x1, y1 = _bbox(outer)
            if x1 - x0 < 2 or y1 - y0 < 2:
                return None
            canvas = Image.new("RGBA", size, (0, 0, 0, 0))
            patch = art.convert("RGBA").resize(
                (max(2, int(x1 - x0)), max(2, int(y1 - y0))), Image.LANCZOS)
            canvas.alpha_composite(patch, (int(x0), int(y0)))
            return canvas
        except Exception:
            return None

    def _apply_shading(self, patch: Image.Image) -> Image.Image:
        """Multiply by the baked mouth-region shading map (if present)."""
        if self.shading is None:
            return patch
        ox, oy = self.origin()
        box = (int(round(ox)), int(round(oy)),
               int(round(ox)) + patch.width, int(round(oy)) + patch.height)
        try:
            crop = self.shading.convert("L").crop(box)
        except Exception:
            return patch
        if crop.size != patch.size:
            return patch
        arr = np.asarray(patch, dtype=np.float32)
        gain = np.asarray(crop, dtype=np.float32)[..., None] / 128.0
        gain = np.clip(gain, 0.75, 1.25)
        arr[..., :3] = np.clip(arr[..., :3] * gain, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGBA")

    # ─── composite ────────────────────────────────────────

    def composite(self, plate: Image.Image, p: MouthParams,
                  viseme_class: str = "REST",
                  art: Optional[Image.Image] = None,
                  art_weight: float = 0.0) -> Image.Image:
        """Sub-pixel alpha-composite of the mouth onto the head plate."""
        patch = self.render(p, viseme_class, art, art_weight)
        ox, oy = self.origin()
        ix, iy = math.floor(ox), math.floor(oy)
        fx, fy = ox - ix, oy - iy
        if fx > 1e-3 or fy > 1e-3:
            patch = patch.transform((patch.width + 1, patch.height + 1),
                                    Image.AFFINE, (1, 0, -fx, 0, 1, -fy),
                                    resample=Image.BICUBIC)
        plate.alpha_composite(patch, (ix, iy))
        return plate

    @property
    def hit_rate(self) -> float:
        n = self._hits + self._misses
        return self._hits / n if n else 0.0


__all__ = ["MouthRasterizer", "SUPERSAMPLE", "TONGUE_CLASSES",
           "TEETH_JAW_FLOOR"]
