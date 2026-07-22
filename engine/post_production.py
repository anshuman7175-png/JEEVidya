"""
JEEVidya V5 — Post-Production Grade (Tier 2)
════════════════════════════════════════════
Film-style finishing as pure numpy LUTs — split-tone, S-curve film
emulation, vignette, and grain — parameterized by Visual DNA so every
topic carries its own "film stock". Costs ~10-20 ms/frame.

Deterministic: grain tiles are seeded from the DNA, so graded segments
stay bit-identical (Tier 0 cache-safe).
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from engine.visual_dna import VisualDNA

_N_GRAIN_TILES = 8


def _film_curve(x: np.ndarray, strength: float) -> np.ndarray:
    """Gentle S-curve: lifted shadows crushed slightly, highlights rolled."""
    t = x / 255.0
    s = t + strength * (t * t * (3 - 2 * t) - t)   # blend toward smoothstep
    return np.clip(s * 255.0, 0, 255)


class ColorGrade:
    """Precomputed per-channel LUTs + vignette mask + grain tiles."""

    def __init__(self, dna: Optional["VisualDNA"] = None,
                 width: int = 1080, height: int = 1920):
        self.width, self.height = width, height

        if dna is not None:
            warmth = float(dna.genes.get("grade_warmth", 0.0))
            self.grain_amt = float(dna.genes.get("grain", 0.12))
            self.vignette_amt = float(dna.genes.get("vignette", 0.35))
            curve = 0.10 + 0.10 * float(dna.genes.get("energy", 0.6))
            seed = dna.seed
        else:
            warmth, self.grain_amt, self.vignette_amt, curve, seed = \
                0.0, 0.10, 0.30, 0.12, 1234

        # ── Split-tone: cool shadows / warm highlights (or inverted) ──
        x = np.arange(256, dtype=np.float32)
        shadow_w = np.clip(1.0 - x / 128.0, 0, 1)       # weight in shadows
        high_w = np.clip((x - 128.0) / 127.0, 0, 1)     # weight in highlights
        # warmth > 0: highlights toward amber, shadows toward teal
        r_shift = 10.0 * warmth * high_w - 6.0 * warmth * shadow_w
        b_shift = -10.0 * warmth * high_w + 8.0 * warmth * shadow_w
        g_shift = 2.0 * warmth * high_w

        base = _film_curve(x, curve)
        self.lut = np.stack([
            np.clip(base + r_shift, 0, 255),
            np.clip(base + g_shift, 0, 255),
            np.clip(base + b_shift, 0, 255),
        ], axis=1).astype(np.uint8)                      # (256, 3)

        # ── Vignette (radial falloff, precomputed float mask) ──
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        nx = (xx / width - 0.5) * 2.0
        ny = (yy / height - 0.5) * 2.0
        r2 = nx * nx + ny * ny * 0.8                     # portrait-friendly
        self.vignette = (1.0 - self.vignette_amt *
                         np.clip(r2 - 0.25, 0, 1.4) / 1.4)[..., None]

        # ── Grain: pre-baked seeded noise tiles cycled per frame ──
        rng = np.random.default_rng(seed & 0x7FFFFFFF)
        amp = self.grain_amt * 14.0
        self._grain = [
            rng.normal(0.0, amp, size=(height, width, 1)).astype(np.float32)
            for _ in range(_N_GRAIN_TILES)
        ] if amp > 0.5 else []

    def apply(self, frame: Image.Image, frame_num: int = 0) -> Image.Image:
        """Grade one RGB frame. LUT → vignette → grain, all vectorized."""
        rgb = frame.convert("RGB") if frame.mode != "RGB" else frame
        arr = np.asarray(rgb)

        out = np.empty_like(arr)
        out[..., 0] = self.lut[arr[..., 0], 0]
        out[..., 1] = self.lut[arr[..., 1], 1]
        out[..., 2] = self.lut[arr[..., 2], 2]

        graded = out.astype(np.float32) * self.vignette
        if self._grain:
            graded += self._grain[frame_num % _N_GRAIN_TILES]

        return Image.fromarray(
            np.clip(graded, 0, 255).astype(np.uint8), "RGB")
