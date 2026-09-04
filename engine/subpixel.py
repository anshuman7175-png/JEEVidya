"""
JEEVidya V5 — Sub-Pixel Compositing & Sampled Motion Blur
═════════════════════════════════════════════════════════
Integer-pixel pastes are why 30fps 2D motion JUDDERS: a sprite gliding
at 1.4 px/frame visibly snaps 1-2-1-2. Film pipelines place everything
on a continuous raster. This module brings that here:

  paste_subpixel   bilinear fractional placement — the sprite's energy
                   is distributed across the 4 neighbouring pixel
                   phases. Motion becomes continuous. THE difference
                   between "slideshow" and "footage".
  motion_ghosts    2-sample motion blur along an object's ACTUAL
                   velocity vector (from its spring state) — fast moves
                   streak like a camera saw them, not teleport.
  gate_weave       0.3 px seeded full-frame wobble, the mechanical
                   fingerprint of a film gate. Subliminal, organic.

All numpy, all deterministic, all cache-friendly.
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
from PIL import Image


def _fractional_shift(sprite: Image.Image, fx: float, fy: float
                      ) -> Image.Image:
    """Shift an RGBA sprite by a sub-pixel amount via bilinear weights.
    Returns a sprite 1px larger in each axis containing the shifted image.

    Done in PREMULTIPLIED alpha. In straight alpha the four taps average
    real colour with the (0,0,0,0) of the transparent neighbour, and the
    silhouette picks up a dark 1-px fringe on every frame — the most
    persistent "shadow outline" a cut-out character can have.
    """
    arr = np.asarray(sprite, dtype=np.float32)
    h, w = arr.shape[:2]
    has_alpha = arr.shape[2] == 4
    if has_alpha:
        a = arr[..., 3:4] / 255.0
        arr = np.concatenate([arr[..., :3] * a, arr[..., 3:4]], axis=2)
    out = np.zeros((h + 1, w + 1, arr.shape[2]), dtype=np.float32)

    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    out[:h, :w] += arr * w00
    out[:h, 1:] += arr * w10
    out[1:, :w] += arr * w01
    out[1:, 1:] += arr * w11

    if has_alpha:
        oa = out[..., 3:4]
        safe = np.where(oa > 0.5, oa, 1.0)
        rgb = np.clip(out[..., :3] * 255.0 / safe, 0, 255)
        out = np.concatenate([np.where(oa > 0.5, rgb, out[..., :3]), oa],
                             axis=2)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8),
                           sprite.mode)


def paste_subpixel(frame: Image.Image, sprite: Image.Image,
                   x: float, y: float) -> Image.Image:
    """Alpha-composite `sprite` centered-x/bottom-anchored at the FLOAT
    position (x, y). Mutates and returns `frame` (RGBA)."""
    if frame.mode != "RGBA":
        frame = frame.convert("RGBA")
    if sprite.mode != "RGBA":
        sprite = sprite.convert("RGBA")

    px = x - sprite.width / 2.0
    py = y - sprite.height
    ix, iy = math.floor(px), math.floor(py)
    fx, fy = px - ix, py - iy

    if fx > 0.02 or fy > 0.02:
        sprite = _fractional_shift(sprite, fx, fy)

    # Clip-safe composite
    if ix >= frame.width or iy >= frame.height or \
       ix + sprite.width <= 0 or iy + sprite.height <= 0:
        return frame
    sx0, sy0 = max(0, -ix), max(0, -iy)
    dx0, dy0 = max(0, ix), max(0, iy)
    cw = min(sprite.width - sx0, frame.width - dx0)
    ch = min(sprite.height - sy0, frame.height - dy0)
    if cw <= 0 or ch <= 0:
        return frame
    region = sprite.crop((sx0, sy0, sx0 + cw, sy0 + ch))
    frame.alpha_composite(region, (dx0, dy0))
    return frame


def motion_ghosts(frame: Image.Image, sprite: Image.Image,
                  x: float, y: float, vx: float, vy: float,
                  threshold: float = 2.5) -> Image.Image:
    """Motion ghosts disabled: multi-sample sprite stamping creates
    ghost trails / shadow artifacts behind moving characters."""
    return frame


def gate_weave(frame: Image.Image, frame_num: int, seed: int,
               amp: float = 0.35) -> Image.Image:
    """Sub-pixel full-frame wobble — the film-gate signature.
    Uses one bilinear affine resample (~10 ms at 1080p)."""
    p = (seed % 613) * 0.37
    dx = (math.sin(frame_num * 0.61 + p) * 0.6
          + math.sin(frame_num * 1.13 + p * 2) * 0.4) * amp
    dy = (math.sin(frame_num * 0.53 + p * 3) * 0.7
          + math.sin(frame_num * 1.31 + p * 4) * 0.3) * amp
    return frame.transform(frame.size, Image.Transform.AFFINE,
                           (1, 0, dx, 0, 1, dy),
                           resample=Image.Resampling.BILINEAR)
