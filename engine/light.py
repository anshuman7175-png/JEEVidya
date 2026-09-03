"""
JEEVidya V5 — Light & Depth Pass
════════════════════════════════
What makes flat 2D read as a LIT 3D SPACE:

  rim_light        edge glow on the character's key side, derived from
                   the alpha channel (dilate − erode on the light side)
  ambient_wrap     low-frequency scene-color tint so cutout characters
                   sit IN the palette instead of pasted ON it
  contact_shadow   soft grounding ellipse under every character — the
                   single cheapest realism win in compositing
  rack_focus       blur + darken the background plane on close-ups
                   (a fake depth-of-field pull that reads as intent)
  bloom            threshold → downsample → blur → add. Real bloom on
                   real highlights (formulas, glows, sparks), not a
                   uniform haze
  chromatic_aberration  sub-pixel R/B splay scaled toward the edges —
                   the lens fingerprint that whispers "shot, not drawn"
  god_rays         cached radial light shafts behind reveals

Heavy work is cached per (source, params); per-frame cost is composites
and one numpy pass. All deterministic.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

_cache: Dict[Tuple, Image.Image] = {}


# ═══════════════════════════════════════════
# CHARACTER LIGHTING (cached per expression image)
# ═══════════════════════════════════════════

def rim_light(char_img: Image.Image, side: float = 1.0,
              color: Tuple[int, int, int] = (140, 200, 255),
              strength: float = 0.75) -> Image.Image:
    """Key-side edge light from the alpha silhouette.
    side: +1 light from the right, −1 from the left."""
    if char_img.mode != "RGBA":
        char_img = char_img.convert("RGBA")
    alpha = np.asarray(char_img.split()[3], dtype=np.float32) / 255.0

    # Edge on the lit side: alpha minus alpha shifted AWAY from the light
    shift = max(2, char_img.width // 90)
    moved = np.roll(alpha, int(-side * shift), axis=1)
    edge = np.clip(alpha - moved, 0.0, 1.0)
    # Soften vertically too (top rim from the sky light)
    top = np.clip(alpha - np.roll(alpha, shift, axis=0), 0.0, 1.0) * 0.5
    edge = np.clip(edge + top, 0.0, 1.0)

    rim = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rim[..., 0], rim[..., 1], rim[..., 2] = color
    rim[..., 3] = (edge * 255 * strength).astype(np.uint8)
    rim_img = Image.fromarray(rim).filter(
        ImageFilter.GaussianBlur(max(1, shift // 2)))

    return Image.alpha_composite(char_img, rim_img)


def ambient_wrap(char_img: Image.Image,
                 scene_color: Tuple[int, int, int],
                 amount: float = 0.18) -> Image.Image:
    """Tint the character toward the scene palette (bounce light).
    Shadow side (screen-left half, feathered) gets slightly more."""
    if char_img.mode != "RGBA":
        char_img = char_img.convert("RGBA")
    arr = np.asarray(char_img, dtype=np.float32)
    tint = np.array(scene_color, dtype=np.float32)

    # Horizontal feather: more wrap on the left (fill side)
    w = arr.shape[1]
    grad = np.linspace(1.25, 0.75, w, dtype=np.float32)[None, :, None]
    a = amount * grad
    rgb = arr[..., :3] * (1 - a) + tint[None, None, :] * a * \
        (arr[..., :3].mean(axis=2, keepdims=True) / 255.0 + 0.35)
    result = arr.copy()
    result[..., :3] = np.clip(rgb, 0, 255)
    return Image.fromarray(result.astype(np.uint8))


def contact_shadow(width: int, softness: int = 0,
                   opacity: int = 80) -> Image.Image:
    """Soft grounding ellipse, cached per size."""
    softness = softness or max(6, width // 10)
    key = ("shadow", width, softness, opacity)
    out = _cache.get(key)
    if out is not None:
        return out
    h = max(10, width // 4)
    img = Image.new("RGBA", (width + softness * 4, h + softness * 4),
                    (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((softness * 2, softness * 2, softness * 2 + width,
               softness * 2 + h), fill=(0, 0, 10, opacity))
    out = img.filter(ImageFilter.GaussianBlur(softness))
    _cache[key] = out
    return out


# ═══════════════════════════════════════════
# DEPTH & LENS (per-frame numpy passes)
# ═══════════════════════════════════════════

def rack_focus(bg: Image.Image, amount: float) -> Image.Image:
    """Fake DoF pull on the background plane: blur + gentle darken.
    amount 0..1; 0 returns the input untouched."""
    if amount <= 0.02:
        return bg
    blurred = bg.filter(ImageFilter.GaussianBlur(1 + amount * 5))
    if amount > 0.15:
        arr = np.asarray(blurred.convert(bg.mode), dtype=np.float32)
        arr[..., :3] *= (1.0 - 0.22 * amount)
        blurred = Image.fromarray(arr.astype(np.uint8), bg.mode)
    return blurred


def bloom(frame: Image.Image, threshold: int = 190,
          strength: float = 0.55, downsample: int = 4) -> Image.Image:
    """Physically-motivated bloom: only true highlights halo.
    threshold → /4 downsample → blur → upsample → screen-add."""
    rgb = frame if frame.mode == "RGB" else frame.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)

    bright = np.clip(arr - threshold, 0, 255) * (255.0 / max(1, 255 - threshold))
    small = Image.fromarray(bright.astype(np.uint8)).resize(
        (rgb.width // downsample, rgb.height // downsample),
        Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(6))
    halo = np.asarray(small.resize(rgb.size, Image.Resampling.BILINEAR),
                      dtype=np.float32)

    out = 255.0 - (255.0 - arr) * (255.0 - halo * strength) / 255.0  # screen
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def halation(frame: Image.Image, threshold: int = 215,
             strength: float = 0.45) -> Image.Image:
    """Film halation: highlights bleed red-orange through the emulsion
    and reflect off the film base. The warm ring around bright objects
    is one of the strongest 'shot on film' cues that digital lacks."""
    rgb = frame if frame.mode == "RGB" else frame.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)

    bright = np.clip(arr.max(axis=2) - threshold, 0, 255) \
        * (255.0 / max(1, 255 - threshold))
    small = Image.fromarray(bright.astype(np.uint8)).resize(
        (rgb.width // 4, rgb.height // 4),
        Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(9))
    glow = np.asarray(small.resize(rgb.size, Image.Resampling.BILINEAR),
                      dtype=np.float32) * strength

    out = arr.copy()
    out[..., 0] += glow            # full red bleed
    out[..., 1] += glow * 0.32     # a little orange
    out[..., 2] += glow * 0.06
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def chromatic_aberration(frame: Image.Image,
                         strength: float = 1.0) -> Image.Image:
    """Edge-weighted R/B channel splay (px shift grows toward edges).
    strength ≈ 1 → max 2px at the frame border: felt, not seen."""
    if strength <= 0.05:
        return frame
    rgb = frame if frame.mode == "RGB" else frame.convert("RGB")
    arr = np.asarray(rgb)
    shift = max(1, int(round(2 * strength)))
    h, w = arr.shape[:2]

    out = arr.copy()
    # Horizontal splay, masked to the outer thirds so faces stay clean
    edge = w // 3
    out[:, :edge, 0] = arr[:, :edge, 0]                      # keep
    out[:, shift:edge, 0] = arr[:, :edge - shift, 0]         # R inward L
    out[:, w - edge:w - shift, 2] = arr[:, w - edge + shift:, 2]  # B inward R
    return Image.fromarray(out)


def god_rays(size: Tuple[int, int],
             color: Tuple[int, int, int] = (255, 235, 180),
             n_rays: int = 7, seed: int = 5) -> Image.Image:
    """Cached radial light shafts (composited behind reveals, screen mode)."""
    key = ("rays", size, color, n_rays, seed)
    out = _cache.get(key)
    if out is not None:
        return out
    w, h = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = w // 2, int(h * 0.30)
    import random as _r
    rng = _r.Random(seed)
    for i in range(n_rays):
        a = math.pi / 2 + (i - n_rays / 2) * 0.22 + rng.uniform(-0.05, 0.05)
        length = h * rng.uniform(0.7, 1.05)
        spread = rng.uniform(0.03, 0.07)
        p1 = (cx + math.cos(a - spread) * length,
              cy + math.sin(a - spread) * length)
        p2 = (cx + math.cos(a + spread) * length,
              cy + math.sin(a + spread) * length)
        d.polygon([(cx, cy), p1, p2],
                  fill=color + (rng.randint(10, 26),))
    out = img.filter(ImageFilter.GaussianBlur(8))
    _cache[key] = out
    return out


def clear_cache() -> None:
    _cache.clear()
