"""
JEEVidya V5 — Alpha Hygiene
═══════════════════════════
Every dark halo, grey fringe and "shadow form" around a character comes
from ONE mistake: resampling straight (un-premultiplied) RGBA whose
transparent pixels are (0,0,0,0). Bilinear / LANCZOS / rotation then
averages real colour with black, and the edge goes dark. This module is
the single place that mistake is prevented.

  bleed_edges(img)        Push the colour of opaque pixels outward into
                          the transparent margin (alpha unchanged). Do
                          this ONCE at load time and every later resample
                          has valid colour to sample. Idempotent.
  premultiply / unpremultiply
  resize_premultiplied    LANCZOS/BILINEAR resize with no edge darkening.
  transform_premultiplied Any PIL transform (AFFINE/MESH/rotate) done in
                          premultiplied space.
  blend_premultiplied     Cross-fade two RGBA plates in premultiplied
                          space (straight-space lerps produce a ghost
                          because colour is weighted where alpha is 0).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


def _to_f32(img: Image.Image) -> np.ndarray:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return np.asarray(img, dtype=np.float32)


def premultiply(img: Image.Image) -> Image.Image:
    arr = _to_f32(img)
    a = arr[..., 3:4] / 255.0
    out = np.concatenate([arr[..., :3] * a, arr[..., 3:4]], axis=2)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGBA")


def unpremultiply(img: Image.Image) -> Image.Image:
    arr = _to_f32(img)
    a = arr[..., 3:4]
    safe = np.where(a > 0.5, a, 1.0)
    rgb = np.clip(arr[..., :3] * 255.0 / safe, 0, 255)
    rgb = np.where(a > 0.5, rgb, arr[..., :3])
    out = np.concatenate([rgb, a], axis=2)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGBA")


def bleed_edges(img: Image.Image, passes: int = 3) -> Image.Image:
    """Fill transparent pixels with the colour of the nearest opaque
    pixels; alpha is untouched. Cheap iterative dilation — `passes`
    controls how many pixels outward the colour reaches (3 is plenty
    for any downscale ≤ 4×)."""
    arr = _to_f32(img)
    a = arr[..., 3]
    if a.min() > 250:            # no transparent region at all
        return img if img.mode == "RGBA" else img.convert("RGBA")
    rgb = arr[..., :3].copy()
    opaque = a > 8
    if not opaque.any():
        return img
    # Zero out colour under transparent pixels so they never contaminate
    rgb[~opaque] = 0.0
    weight = opaque.astype(np.float32)
    for _ in range(passes):
        # 3x3 box sum of colour and weight, then normalise where weight>0
        wsum = _box3(weight)
        csum = _box3(rgb)
        fill_mask = (~opaque) & (wsum > 0)
        if not fill_mask.any():
            break
        rgb[fill_mask] = csum[fill_mask] / wsum[fill_mask][:, None]
        opaque = opaque | fill_mask
        weight = opaque.astype(np.float32)
    # Whatever is still empty far from the silhouette: fill with the
    # silhouette's mean colour (no black anywhere in the plate).
    if (~opaque).any():
        mean = rgb[a > 8].mean(axis=0)
        rgb[~opaque] = mean
    out = np.concatenate([rgb, a[..., None]], axis=2)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGBA")


def _box3(x: np.ndarray) -> np.ndarray:
    """3×3 box sum with zero padding (works for HxW and HxWxC)."""
    p = np.pad(x, ((1, 1), (1, 1)) + ((0, 0),) * (x.ndim - 2))
    h, w = x.shape[:2]
    out = np.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            out += p[dy:dy + h, dx:dx + w]
    return out


def resize_premultiplied(img: Image.Image, size: Tuple[int, int],
                         resample=Image.Resampling.LANCZOS) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    if tuple(size) == img.size:
        return img
    return unpremultiply(premultiply(img).resize(size, resample))


def transform_premultiplied(img: Image.Image, size: Tuple[int, int],
                            method, data,
                            resample=Image.Resampling.BILINEAR) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    pre = premultiply(img).transform(size, method, data, resample=resample)
    return unpremultiply(pre)


def rotate_premultiplied(img: Image.Image, angle: float, **kw) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    kw.setdefault("resample", Image.Resampling.BILINEAR)
    return unpremultiply(premultiply(img).rotate(angle, **kw))


def blend_premultiplied(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
    """Cross-fade a→b at t in premultiplied space. Where the two
    silhouettes disagree the result is a clean partial-alpha edge, not a
    dark ghost carrying (0,0,0) colour."""
    t = max(0.0, min(1.0, float(t)))
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    pa = _to_f32(premultiply(a))
    pb = _to_f32(premultiply(b))
    if pa.shape != pb.shape:
        return b if t >= 0.5 else a
    mix = pa * (1.0 - t) + pb * t
    return unpremultiply(Image.fromarray(np.clip(mix + 0.5, 0, 255)
                                         .astype(np.uint8), "RGBA"))


def scale_alpha(img: Image.Image, factor: float) -> Image.Image:
    """Multiply alpha by `factor` without touching colour."""
    if factor >= 0.995:
        return img
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    a = img.getchannel("A").point(lambda p, f=factor: int(p * f))
    out = img.copy()
    out.putalpha(a)
    return out


__all__ = ["premultiply", "unpremultiply", "bleed_edges",
           "resize_premultiplied", "transform_premultiplied",
           "rotate_premultiplied", "blend_premultiplied", "scale_alpha"]
