"""
JEEVidya — Rig v3 Bakes (Terminal Plan, Part III)
═════════════════════════════════════════════════
The one-time, per-character construction that makes defect classes
D1–D3 UNREPRESENTABLE instead of merely fixed.

  §3.1  Landmark EVERY pose at full resolution — all 478 points, not 5
        boxes. Detection failure on any pose is a hard error; there is
        no heuristic fallback in v3, because the heuristic is exactly
        what historically put the mouth on the eyes.
  §3.2  Register canonical→pose with Umeyama+IRLS on the rigid skull
        subset (engine/registration.py). θ recovers each pose's head
        ROLL, which is why the head sits correctly instead of being
        merely translated.
  §3.3  Bake a HEADLESS body per pose. A cross-fade between two headless
        bodies cannot produce two faces (D2) — the single head plate is
        composited once, on top, afterwards.
  §3.4  Bake OCCLUDERS: head-region pixels that differ from canonical
        (a hand crossing the face). Drawn AFTER the head so the hand
        still passes in front.
  §3.5  Bake the canonical HEAD PLATE: crop the head, then inpaint the
        painted mouth and both eyes OUT. D3 dies here — there is no
        painted mouth left for a feathered backing to hide.
  §3.6  Fit the 5-D mouth targets FROM the character's viseme art, so
        the parametric mouth reproduces the artist's shapes rather than
        a programmer's guess.

Seam energy conservation: the head plate's alpha ramp and the headless
body's alpha ramp are COMPLEMENTARY across the neck band (α_head +
α_body = 1 everywhere). That is an identity, not a tuning value, so the
seam cannot show a bright or dark line — asserted by `seam_error()`.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from config import settings
from engine.registration import (RegistrationError, SimilarityTransform,
                                 register_pose)
from engine.rig import HeadGeometry, PoseEntry, Rig, N_LANDMARKS, rig_dir
from tools.art_eyes import (EyeMeasureError, eyeball_sprite, lid_sprite,
                            measure_pair, socket_backdrop)

# ═══════════════════════════════════════════
# MediaPipe canonical landmark index sets
# ═══════════════════════════════════════════

FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)

LIP_OUTER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321,
             405, 314, 17, 84, 181, 91, 146)
LIP_INNER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318,
             402, 317, 14, 87, 178, 88, 95)

# Subject's left eye (screen right) and right eye, upper then lower lid
LID_UPPER_L = (362, 398, 384, 385, 386, 387, 388, 466, 263)
LID_LOWER_L = (362, 382, 381, 380, 374, 373, 390, 249, 263)
LID_UPPER_R = (133, 173, 157, 158, 159, 160, 161, 246, 33)
LID_LOWER_R = (133, 155, 154, 153, 145, 144, 163, 7, 33)

# Refined-landmark iris rings (present only with refine_landmarks/478)
IRIS_L = (473, 474, 475, 476, 477)
IRIS_R = (468, 469, 470, 471, 472)

BROW_L = (336, 296, 334, 293, 300, 285, 295, 282, 283, 276)
BROW_R = (70, 63, 105, 66, 107, 46, 53, 52, 65, 55)

CHIN = 152
FOREHEAD = 10

# Bake tuning — all proportional to face height, never literal pixels.
HEAD_DILATE = 0.02          # ×face_h, mask growth so hair is not clipped
SEAM_BAND = 0.04            # ×face_h, neck collar feather band
HEAD_NECK_OVERLAP = 0.12    # ×face_h, solid neck extension overlap below seam so head never detaches
# ×face_h, transparent breathing room kept around the head plate.
#
# HEAD_DILATE grows the MASK so hair is not clipped, but the crop used to
# tighten straight back onto that grown mask's bbox — so hair landed on
# row 0 with zero bleed room and engine/head_transform.py's single BICUBIC
# resample (head bob, roll, 2.5D squash) sheared it off flat. The plate
# must therefore be cropped LOOSER than the mask: the dilation itself,
# plus room for the largest excursion the composed affine can produce.
#
# NB: distinct from the viseme-plate PLATE_MARGIN further down — one
# module-level name for both would silently shadow this one.
HEAD_PLATE_MARGIN = 0.35    # ×face_h, ample margin for rotation and hair shear
INPAINT_DILATE = 0.035      # ×face_h, feature-mask growth before inpaint
OCCLUDER_RGB_DELTA = 26.0   # mean |ΔRGB| that counts as "different art"
# Phase 1 orphan gate: alpha above this on a headless pose ABOVE the
# neck seam band means head mass the flood failed to claim. 8 matches
# the `solid` threshold in head_mask(), so the gate and the mask agree
# on what counts as opaque — a stricter gate than mask would fail on
# the art's own antialias fringe, a looser one would ship orphan hair.
ORPHAN_ALPHA_THRESH = 8


class BakeError(RuntimeError):
    """A bake that cannot be completed correctly fails loudly (Law 1)."""


# ═══════════════════════════════════════════
# Geometry helpers (pure, unit-testable)
# ═══════════════════════════════════════════

def _pick(lms: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    return np.asarray([lms[i] for i in idx], dtype=np.float64)


def face_height(lms: np.ndarray) -> float:
    """Chin→forehead distance: the scale every threshold is derived from,
    so gates are resolution-independent (Part VIII)."""
    return float(max(8.0, abs(lms[CHIN][1] - lms[FOREHEAD][1])))


def iris_mp(lms: np.ndarray, idx: Sequence[int],
            fh: float) -> Tuple[float, float, float]:
    """MediaPipe's own iris ring as (cx, cy, r).

    Retained ONLY as the seed that locates each eye and for reporting how
    far off it was; the geometry that renders comes from the pixel
    measurement in tools/art_eyes.py, because this ring carries the
    proportions of a photographed human eye.
    """
    pts = _pick(lms, idx)
    c = pts.mean(axis=0)
    r = float(np.linalg.norm(pts - c, axis=1).mean())
    # A refined-iris ring collapses to its centre on some exports;
    # fall back to an anatomical radius so the eye is never a dot.
    if r < 0.5:
        r = 0.055 * fh
    return (float(c[0]), float(c[1]), r)


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull. Deterministic (Law 4): the output
    ordering depends only on coordinate sort order, never on hashing."""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    if len(pts) <= 2:
        return pts

    def half(seq):
        out: List[np.ndarray] = []
        for p in seq:
            while len(out) >= 2:
                (ox, oy), (bx, by) = out[-2], out[-1]
                if (bx - ox) * (p[1] - oy) - (by - oy) * (p[0] - ox) > 0:
                    break
                out.pop()
            out.append(p)
        return out

    lower = half(pts)
    upper = half(pts[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def polygon_mask(poly: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon to a float32 0/1 mask of shape (h, w)."""
    from PIL import ImageDraw
    w, h = size
    img = Image.new("L", (w, h), 0)
    if len(poly) >= 3:
        ImageDraw.Draw(img).polygon([(float(x), float(y)) for x, y in poly],
                                    fill=255)
    return np.asarray(img, dtype=np.float32) / 255.0


def _distance_blur(mask: np.ndarray, radius: float) -> np.ndarray:
    """Grow + soften a mask by `radius` px using a signed-distance ramp.

    Uses scipy's exact EDT when available; otherwise a separable box
    cascade, which converges to a Gaussian and is deterministic. Either
    way the result is monotone in distance, which is all the seam
    identity requires.
    """
    if radius <= 0:
        return mask.astype(np.float32)
    try:
        from scipy.ndimage import distance_transform_edt
        inside = mask > 0.5
        if not inside.any():
            return mask.astype(np.float32)
        # distance OUTSIDE the mask → linear ramp to zero over `radius`
        d_out = distance_transform_edt(~inside)
        ramp = np.clip(1.0 - d_out / max(1e-6, radius), 0.0, 1.0)
        return np.maximum(mask.astype(np.float32), ramp.astype(np.float32))
    except Exception:
        out = mask.astype(np.float32)
        k = max(1, int(radius))
        for _ in range(3):
            out = _box_blur(out, k)
        return np.clip(out * 1.6, 0.0, 1.0)


def _box_blur(a: np.ndarray, k: int) -> np.ndarray:
    """Separable box blur with edge replication (no wrap-around bleed)."""
    if k < 1:
        return a
    pad = k
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(p, axis=0)
    a1 = (c[2 * k:, :] - c[:-2 * k, :]) / (2 * k)
    c = np.cumsum(a1, axis=1)
    a2 = (c[:, 2 * k:] - c[:, :-2 * k]) / (2 * k)
    return a2[:a.shape[0], :a.shape[1]]


def _flood_from_seed(region: np.ndarray, seed: np.ndarray,
                     max_iter: int = 256) -> np.ndarray:
    """Boolean flood fill of `region` from `seed` pixels.

    Uses scipy connected-component labelling when available (exact,
    fast); otherwise alternates full horizontal/vertical run-fills until
    stable — the same deterministic, dependency-free scheme
    tools/pose_stager.py uses for its border flood.
    """
    seed = seed & region
    if not seed.any():
        return seed
    try:
        from scipy.ndimage import label
        lbl, n = label(region)
        if n == 0:
            return seed
        hit = np.unique(lbl[seed])
        hit = hit[hit > 0]
        return np.isin(lbl, hit)
    except Exception:
        mask = seed.copy()

        def _runs(m: np.ndarray, close: np.ndarray) -> np.ndarray:
            h = close.shape[0]
            run_id = np.cumsum(~close, axis=1)
            out = m.copy()
            for y in range(h):
                rc = close[y]
                if not rc.any():
                    continue
                rm = m[y] & rc
                if not rm.any():
                    continue
                ids = run_id[y]
                seeded = np.zeros(int(ids.max()) + 1, dtype=bool)
                seeded[ids[rm]] = True
                out[y] |= rc & seeded[ids]
            return out

        for _ in range(max_iter):
            before = int(mask.sum())
            mask = _runs(mask, region)
            mask = _runs(mask.T, region.T).T
            if int(mask.sum()) == before:
                break
        return mask


def head_mask(lms: np.ndarray, alpha: np.ndarray,
              seam_y: Optional[float] = None,
              overlap: float = 0.0,
              character: Optional[str] = None) -> np.ndarray:
    """§3.3/§3.4 — binary claim on every pixel belonging to the head.

    Grows the landmark convex hull upwards above the forehead to seed the
    hair cap and ponytail, then floods connected opaque pixels.
    Bounded below by seam_y + overlap: the neck flood extends into the collar
    so head tilts and rotations maintain a 100% solid neck connection.
    Restricted laterally by the anatomical head envelope and neck column so
    raised arms/hands/fists are never claimed as head.
    """
    h, w = alpha.shape
    fh = face_height(lms)
    hull = convex_hull(_pick(lms, FACE_OVAL))

    chin_y = float(hull[:, 1].max())
    neck_x = float(lms[152, 0]) if len(lms) > 152 else float(hull[:, 0].mean())
    neck_hw = 0.32 * fh
    forehead_y = float(hull[:, 1].min())
    forehead_x = float(lms[10, 0]) if len(lms) > 10 else float(hull[:, 0].mean())

    hair_cap_y = max(0.0, forehead_y - 0.55 * fh)
    x0, x1 = forehead_x - 0.45 * fh, forehead_x + 0.45 * fh
    extended = np.vstack([hull, np.array([[x0, hair_cap_y], [x1, hair_cap_y]])])
    seed = polygon_mask(convex_hull(extended), (w, h)) > 0.5

    solid = alpha > 8
    region = solid.copy()
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    if seam_y is not None:
        region &= yy < int(math.ceil(seam_y + overlap))
        region &= ((yy < chin_y) | (np.abs(xx - neck_x) <= neck_hw))

    return _flood_from_seed(region, seed).astype(np.float32)


def crop_padded(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int
                ) -> np.ndarray:
    """Crop [y0:y1, x0:x1] from an RGBA array, allowing the box to lie
    partly OUTSIDE the source; anything outside is fully transparent.

    Outside the source art there is nothing but transparency anyway, so a
    box that overhangs is not an approximation — it is the exact answer,
    and it lets the plate keep a real margin even when the character's
    hair reaches the very top row of body.png.
    """
    h, w = arr.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0, arr.shape[2]), dtype=arr.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def border_opaque_counts(arr: np.ndarray, thresh: int = 8
                         ) -> Dict[str, int]:
    """Opaque pixels touching each border of an RGBA plate."""
    a = arr[..., 3]
    return {"top": int((a[0, :] > thresh).sum()),
            "bottom": int((a[-1, :] > thresh).sum()),
            "left": int((a[:, 0] > thresh).sum()),
            "right": int((a[:, -1] > thresh).sum())}


def complementary_ramps(mask: np.ndarray, seam_y: float, band_px: float,
                        overlap_px: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """§3.3 — solid overlap seam.

    Returns (head_alpha_factor, body_alpha_factor).
    The head plate includes a solid neck stump extending past seam_y into the collar.
    The headless body maintains solid torso/collar coverage below seam_y, while
    preserving 100% of non-head art (e.g. raised hands, pencil) above the seam.
    Combined opacity across the neck connection is >= 1.0 everywhere, guaranteeing
    100% solid opacity with zero semi-transparency or detachment under head rotation.
    """
    h, w = mask.shape
    ys = np.arange(h, dtype=np.float32)[:, None]
    band = max(1.0, float(band_px))

    overlap_y = seam_y + overlap_px
    ramp_head = np.clip((overlap_y - ys) / band, 0.0, 1.0)
    head = np.clip(mask * ramp_head, 0.0, 1.0)

    ramp_body = np.clip((ys - (seam_y - band)) / band, 0.0, 1.0)
    body = np.clip((1.0 - mask) + mask * ramp_body, 0.0, 1.0)

    return head.astype(np.float32), body.astype(np.float32)


def seam_error(head_a: np.ndarray, body_a: np.ndarray) -> float:
    """Max opacity deficit from 1.0 across the combined layers."""
    deficit = np.maximum(0.0, 1.0 - (head_a + body_a))
    return float(np.max(deficit))


# ═══════���═══════════════════════════════════
# §3.5 — head plate: crop + inpaint features out
# ═══════════════════════════════════════════

def feature_mask(lms: np.ndarray, size: Tuple[int, int], fh: float,
                 eye_apertures: Optional[Sequence[Sequence[
                     Tuple[float, float]]]] = None,
                 img_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """Binary mask of the painted features to remove before inpainting.

    The outer lip ring always comes out: leaving its ink outline behind
    is what produced ghost lips under the parametric mouth.
    Chin shading, cheek blush, and philtrum are 100% preserved.
    """
    w, h = size
    m = np.zeros((h, w), dtype=np.float32)
    lip_pts = _pick(lms, LIP_OUTER)
    m = np.maximum(m, polygon_mask(convex_hull(lip_pts), (w, h)))

    # Erase resting mouth corners/dimples so smaller visemes don't leave dark holes on cheeks
    # Left corner is 61, Right corner is 291
    if len(lms) > 291:
        pt_l = lms[61]
        pt_r = lms[291]
        r_corner = max(4.0, 0.055 * fh)
        yy, xx = np.ogrid[:h, :w]
        circ_l = ((xx - pt_l[0])**2 + (yy - pt_l[1])**2 <= r_corner**2).astype(np.float32)
        circ_r = ((xx - pt_r[0])**2 + (yy - pt_r[1])**2 <= r_corner**2).astype(np.float32)
        m = np.maximum(m, np.maximum(circ_l, circ_r))

    if not eye_apertures:
        for idx in (LID_UPPER_L + LID_LOWER_L, LID_UPPER_R + LID_LOWER_R):
            m = np.maximum(m, polygon_mask(convex_hull(_pick(lms, idx)),
                                           (w, h)))
    # Tight dilation strictly over the lip boundary (0.015*fh) so chin crease & cheeks are 100% preserved
    grown = _distance_blur(m, max(4.0, 0.015 * fh))
    return (grown > 0.25).astype(np.uint8)


def inpaint(rgba: Image.Image, mask: np.ndarray) -> Image.Image:
    """Remove the masked features, filling from surrounding skin.

    Prefers `cv2.inpaint(INPAINT_NS)` (Navier–Stokes: it continues the
    artwork's shading gradient into the hole, which is precisely why the
    plate keeps its painted 3D form). Falls back to normalized
    convolution — iterative alpha-weighted diffusion — when cv2 is
    unavailable, so a rig can still be built on a bare install.
    """
    arr = np.asarray(rgba.convert("RGBA")).copy()
    if mask.sum() == 0:
        return Image.fromarray(arr)
    try:
        import cv2
        rgb = np.ascontiguousarray(arr[..., :3])
        radius = max(3, int(0.02 * max(arr.shape[:2])))
        filled = cv2.inpaint(rgb, mask, radius, cv2.INPAINT_NS)
        arr[..., :3] = filled
        return Image.fromarray(arr)
    except Exception:
        return Image.fromarray(_normalized_convolution(arr, mask))


def _normalized_convolution(arr: np.ndarray, mask: np.ndarray,
                            iterations: int = 24) -> np.ndarray:
    """Alpha-weighted diffusion inpaint: repeatedly blur the known
    pixels and re-substitute them, so colour flows inward from the hole
    boundary. Deterministic and dependency-free."""
    out = arr.astype(np.float32)
    hole = mask.astype(bool)
    known = (~hole) & (arr[..., 3] > 0)
    w = known.astype(np.float32)
    rgb = out[..., :3] * w[..., None]
    for _ in range(iterations):
        wb = _box_blur(w, 2)
        rgbb = np.stack([_box_blur(rgb[..., c], 2) for c in range(3)], axis=-1)
        safe = np.maximum(wb, 1e-6)
        filled = rgbb / safe[..., None]
        # keep the originals, accept the diffused values in the hole
        rgb = np.where(known[..., None], out[..., :3], filled) * \
            np.maximum(w, (wb > 1e-4).astype(np.float32))[..., None]
        w = np.maximum(w, (wb > 1e-4).astype(np.float32))
    out[..., :3] = np.where(hole[..., None],
                            np.clip(rgb / np.maximum(w, 1e-6)[..., None],
                                    0, 255),
                            out[..., :3])
    return np.clip(out, 0, 255).astype(np.uint8)


def shading_map(plate: Image.Image, lms_plate: np.ndarray,
                fh: float) -> Image.Image:
    """A luminance map of the mouth region on the INPAINTED plate.

    The parametric mouth multiplies its procedural fills by this map, so
    lips drawn by code inherit the artwork's own key light instead of
    reading as a flat decal pasted on a shaded face.
    """
    arr = np.asarray(plate.convert("RGBA"), dtype=np.float32)
    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] +
           0.114 * arr[..., 2])
    box = _pick(lms_plate, LIP_OUTER)
    cx, cy = float(box[:, 0].mean()), float(box[:, 1].mean())
    r = 0.30 * fh
    h, w = lum.shape
    x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
    x1, y1 = min(w, int(cx + r)), min(h, int(cy + r))
    region = lum[y0:y1, x0:x1]
    if region.size == 0:
        return Image.new("L", (1, 1), 128)
    # Normalize against the region mean: the map encodes RELATIVE
    # shading, so it is exposure-independent across characters.
    mean = float(region.mean()) or 1.0
    norm = np.clip(128.0 * region / mean, 0, 255).astype(np.uint8)
    out = Image.new("L", plate.size, 128)
    out.paste(Image.fromarray(norm), (x0, y0))
    return out


# ═══════════════════════════════════════════
# Palette extraction — regions, not pinprick samples
# ═══════════════════════════════════════════
# A palette entry is not decoration: the parametric mouth and eyes are
# PAINTED with these colours, and QC then DETECTS the rendered features
# by them. A wrong entry therefore breaks the render and the measurement
# at once.
#
# The old form averaged a small box at one landmark. Landmark 13 is the
# inner upper lip: on stylised art that box straddles the ink lip line
# and the skin above it, so "lip" came out as muddy skin — invisible
# lips that no colour search can find. The iris entry sampled the iris
# CENTRE, which on cartoon eyes is the specular highlight, so "iris"
# came out near-white and matched the sclera, the page, and the whites
# of the eyes at every tolerance.
#
# Region statistics fix both: take every pixel of the anatomical region,
# trim the extremes that are known contaminants (ink outline, specular
# highlight), and use the MEDIAN, which a few stray pixels cannot move.
# Then enforce the separations the renderer and the gates depend on.

PALETTE_MIN_PX = 24         # fewer pixels than this is not a measurement
LIP_SKIN_SEP = 40.0         # lips must be findable against the face
IRIS_LASH_SEP = 56.0        # a closed lid must not read as an iris
SEP_MAX_STEPS = 32          # bounded ⇒ deterministic


def _disc_mask(size: Tuple[int, int], cx: float, cy: float,
               r: float) -> np.ndarray:
    w, h = size
    yy, xx = np.ogrid[:h, :w]
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= max(1.0, r) ** 2
            ).astype(np.float32)


def _region_px(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Opaque RGB pixels of a region, as (N, 3) float32."""
    sel = (mask > 0.5) & (arr[..., 3] > 60)
    if not sel.any():
        return np.zeros((0, 3), dtype=np.float32)
    return arr[sel][:, :3].astype(np.float32)


def _luma(px: np.ndarray) -> np.ndarray:
    return 0.299 * px[..., 0] + 0.587 * px[..., 1] + 0.114 * px[..., 2]


def _trimmed_median(px: np.ndarray, drop_dark: float = 0.20,
                    drop_light: float = 0.10
                    ) -> Optional[Tuple[int, int, int]]:
    """Median of a region after dropping its darkest and lightest tails.

    The tails are the contaminants: the artwork's ink outline at the dark
    end, the key-light specular at the bright end. The median of what
    remains is the colour a human would call "the lip" or "the iris".
    """
    if len(px) < 1:
        return None
    lum = _luma(px)
    order = np.argsort(lum, kind="stable")
    n = len(px)
    a = int(n * drop_dark)
    b = max(a + 1, n - int(n * drop_light))
    sel = px[order[a:b]] if b > a else px
    return tuple(int(round(v)) for v in np.median(sel, axis=0))


def _luma_band(px: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Pixels whose luma falls between the `lo` and `hi` percentiles."""
    if len(px) == 0:
        return px
    lum = _luma(px)
    a, b = np.percentile(lum, lo), np.percentile(lum, hi)
    return px[(lum >= a) & (lum <= b)]


def _cheb(a, b) -> float:
    """Chebyshev RGB distance — the same metric the QC colour masks use,
    so "separated enough" here means "separable there"."""
    if a is None or b is None:
        return 0.0
    return float(max(abs(int(x) - int(y)) for x, y in zip(a, b, strict=True)))


def _redness(px: np.ndarray) -> np.ndarray:
    return px[..., 0] - 0.5 * (px[..., 1] + px[..., 2])


def _push_warm(color, ref, min_sep: float):
    """Deepen a colour toward vermilion until it is `min_sep` from `ref`.

    Only reached when the artwork genuinely gives lips the same value as
    skin (flat cel shading with a line-art mouth). Some separation is
    then required, not optional: a mouth that cannot be distinguished
    from the cheek is invisible on a phone and unmeasurable by QC.
    """
    r, g, b = (float(c) for c in color)
    for _ in range(SEP_MAX_STEPS):
        if _cheb((r, g, b), ref) >= min_sep:
            break
        r = min(255.0, r * 1.02 + 2.0)
        g *= 0.90
        b *= 0.90
    return tuple(int(round(max(0.0, min(255.0, c)))) for c in (r, g, b))


def _push_dark(color, ref, min_sep: float):
    """Darken a colour (lash ink) until it clears `ref` (the iris)."""
    r, g, b = (float(c) for c in color)
    for _ in range(SEP_MAX_STEPS):
        if _cheb((r, g, b), ref) >= min_sep:
            break
        r, g, b = r * 0.85, g * 0.85, b * 0.85
        if max(r, g, b) < 1.0:
            break
    return tuple(int(round(max(0.0, min(255.0, c)))) for c in (r, g, b))


def extract_palette(arr: np.ndarray, lms: np.ndarray, fh: float,
                    iris_l: Tuple[float, float, float],
                    iris_r: Tuple[float, float, float]
                    ) -> Dict[str, Tuple[int, int, int]]:
    """Robust, region-based palette for one character's head plate.

    `arr` is the RGBA head crop BEFORE inpainting (the features must
    still be painted to be measured), `lms` its landmarks in plate space,
    and the two iris tuples are (cx, cy, r) in the same space.
    """
    h, w = arr.shape[:2]
    size = (w, h)

    def hull_mask(idx: Sequence[int]) -> np.ndarray:
        return polygon_mask(convex_hull(_pick(lms, idx)), size)

    # ── skin: cheeks and mid-face, away from lips, eyes and brows ──
    skin_mask = np.zeros((h, w), dtype=np.float32)
    for lm in (50, 280, 205, 425):
        if lm < len(lms):
            skin_mask = np.maximum(
                skin_mask, _disc_mask(size, lms[lm][0], lms[lm][1],
                                      0.045 * fh))
    skin = _trimmed_median(_region_px(arr, skin_mask), 0.20, 0.20) \
        or (214, 176, 152)

    # ── lips: the vermilion BAND between the outer and inner rings ──
    outer, inner = hull_mask(LIP_OUTER), hull_mask(LIP_INNER)
    band_px = _region_px(arr, np.clip(outer - inner, 0.0, 1.0))
    if len(band_px) < PALETTE_MIN_PX:
        band_px = _region_px(arr, outer)     # closed mouth: rings coincide
    lip = _trimmed_median(band_px, 0.25, 0.10) or _push_warm(skin, skin,
                                                             LIP_SKIN_SEP)
    if _cheb(lip, skin) < LIP_SKIN_SEP and len(band_px) >= PALETTE_MIN_PX:
        # The band's median is skin-like because the band also covers the
        # skin margin the landmark hull always includes. The reddest
        # quartile of the same pixels IS the vermilion.
        red = _redness(band_px)
        sub = band_px[red >= np.percentile(red, 75.0)]
        cand = _trimmed_median(sub, 0.10, 0.10)
        if cand is not None and _cheb(cand, skin) > _cheb(lip, skin):
            lip = cand
    lip = _push_warm(lip, skin, LIP_SKIN_SEP)

    lip_shadow = _trimmed_median(_luma_band(band_px, 0.0, 30.0), 0.10, 0.10)
    if lip_shadow is None or _luma(np.array([lip_shadow], dtype=np.float32))[0] \
            > _luma(np.array([lip], dtype=np.float32))[0] - 8.0:
        lip_shadow = _darken(lip, 0.72)

    # ── eyes: pool both sockets, then separate by luma role ──
    eye_hull = np.maximum(hull_mask(LID_UPPER_L + LID_LOWER_L),
                          hull_mask(LID_UPPER_R + LID_LOWER_R))
    iris_disc = np.maximum(
        _disc_mask(size, iris_l[0], iris_l[1], iris_l[2] * 0.80),
        _disc_mask(size, iris_r[0], iris_r[1], iris_r[2] * 0.80))
    eye_px = _region_px(arr, eye_hull)
    iris_px = _region_px(arr, np.minimum(iris_disc, np.maximum(eye_hull, 0.6)))

    # Iris: drop the specular highlight (bright tail) and the pupil
    # (dark tail) — what is left is the coloured ring itself.
    iris = _trimmed_median(iris_px, 0.15, 0.30) \
        if len(iris_px) >= PALETTE_MIN_PX else None
    iris_lum = _luma(np.array([iris], dtype=np.float32))[0] if iris else 255.0
    if iris is None or iris_lum > 210.0:
        # The iris ring collapsed (unrefined landmarks) or the sample
        # landed on the sclera: fall back to the socket's dark half,
        # which is the eyeball, never the eye white.
        iris = _trimmed_median(_luma_band(eye_px, 5.0, 40.0), 0.05, 0.05) \
            or (92, 58, 38)

    sclera = _trimmed_median(_luma_band(eye_px, 75.0, 100.0), 0.05, 0.05) \
        or (248, 247, 245)
    if _luma(np.array([sclera], dtype=np.float32))[0] < \
            _luma(np.array([skin], dtype=np.float32))[0] + 8.0:
        sclera = (248, 247, 245)     # no eye white in the art: use paper

    lash = _trimmed_median(_luma_band(eye_px, 0.0, 8.0), 0.0, 0.0) \
        or _darken(skin, 0.25)
    # A lid that reads as an iris makes blink closure unverifiable.
    lash = _push_dark(lash, iris, IRIS_LASH_SEP)

    return {
        "skin": skin,
        "lip": lip,
        "lip_shadow": lip_shadow,
        "oral_cavity": _darken(lip_shadow, 0.55),
        "teeth": _mix(sclera, (242, 240, 236), 0.5),
        "tongue": _mix(lip, (196, 96, 104), 0.5),
        "sclera": sclera,
        "iris": iris,
        "lash": lash,
    }


# ═══════════════════════════════════════════
# §3.6 — fit the 5-D mouth targets from the art
# ═══════════════════════════════���═══════════

def normalize_contour(pts: np.ndarray, fh: float) -> np.ndarray:
    """Map a lip ring into the model's normalized mouth space: centred at
    the ring centroid, scaled so one unit is 0.25·face_h. This is the
    same space `mouth_model.lip_contour` emits, which is what makes the
    fit comparable across characters and resolutions."""
    c = pts.mean(axis=0)
    scale = max(1e-6, 0.25 * fh)
    return (pts - c) / scale


FIT_CONTOUR_SAMPLES = 64     # polar samples per ring in the 5-D fit


def _resample_polar(poly: np.ndarray, n: int,
                    center: Optional[np.ndarray] = None) -> np.ndarray:
    """Resample a closed ring at `n` UNIFORM POLAR ANGLES about `center`,
    returning points RELATIVE to that centre.

    This is what establishes CORRESPONDENCE between a measured lip ring
    and a model contour. Arc-length resampling does not: it preserves
    each ring's own arbitrary starting vertex and winding direction, so
    point *k* of the observation and point *k* of the model describe
    different parts of the mouth. MediaPipe's LIP_OUTER ring starts at
    the left commissure and `mouth_model.lip_contour` starts at the right,
    a half-ring phase error — which made the least-squares fit minimize
    a meaningless quantity and collapse every viseme onto the same
    degenerate corner of the parameter space (jaw=0 ⇒ a mouth that never
    opens). Sampling both rings by angle about a shared centre is
    phase- and winding-invariant, so shape is compared against shape.

    Valid because a lip ring is star-convex about its centroid.
    """
    p = np.asarray(poly, dtype=np.float64)
    c = p.mean(axis=0) if center is None else np.asarray(center,
                                                        dtype=np.float64)
    d = p - c
    ang = np.arctan2(d[:, 1], d[:, 0])
    rad = np.hypot(d[:, 0], d[:, 1])
    order = np.argsort(ang)
    ang, rad = ang[order], rad[order]
    # Tile ±2π so np.interp wraps continuously across the seam at ±π.
    ang_p = np.concatenate([ang - 2.0 * math.pi, ang, ang + 2.0 * math.pi])
    rad_p = np.concatenate([rad, rad, rad])
    t = np.linspace(-math.pi, math.pi, n, endpoint=False)
    r = np.interp(t, ang_p, rad_p)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _contour_residual(params, observed_outer: np.ndarray,
                      observed_inner: np.ndarray) -> float:
    """Mean squared distance between the model's lip rings and the
    measured ones, in correspondence (see `_resample_polar`).

    Both rings of a pair are sampled about their OUTER ring's centre, so
    the inner ring's offset relative to the outer one — which is what
    distinguishes an open jaw from a closed one — is preserved rather
    than normalized away.
    """
    from engine.mouth_model import MouthParams, lip_contour
    p = MouthParams(*params).clamped()
    n = FIT_CONTOUR_SAMPLES
    outer, inner = lip_contour(p, n=n)
    m_outer = np.asarray(outer, dtype=np.float64)
    m_inner = np.asarray(inner, dtype=np.float64)
    obs_c = np.asarray(observed_outer, dtype=np.float64).mean(axis=0)
    mod_c = m_outer.mean(axis=0)
    err = 0.0
    for obs, model in ((observed_outer, m_outer), (observed_inner, m_inner)):
        if len(obs) == 0:
            continue
        a = _resample_polar(obs, n, center=obs_c)
        b = _resample_polar(model, n, center=mod_c)
        err += float(np.mean(np.sum((a - b) ** 2, axis=1)))
    return err


def fit_mouth_target(observed_outer: np.ndarray, observed_inner: np.ndarray,
                     seed: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Least-squares fit of the 5 mouth parameters that best reproduce an
    art viseme's lip contours (§3.6).

    Art-derived targets beat hand-guessed ones because the artist already
    decided what this character's "oo" looks like. Uses scipy when
    present, and a deterministic coordinate-descent refinement otherwise
    so the result never depends on which optional packages are installed.
    """
    from engine.mouth_model import PARAM_NAMES
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0)]
    x0 = list(seed) if seed is not None else [0.3, 0.5, 0.2, 0.2, 0.0]

    def err_of(x) -> float:
        return _contour_residual(x, observed_outer, observed_inner)

    # ── Multi-start, because this residual is genuinely multi-modal ──
    # `pull` (corner raise) and `jaw` (opening) trade off against each
    # other: raising the corners and dropping the jaw both lengthen the
    # ring vertically, so a lone descent from one seed slides into a
    # nearby basin and pins `pull` against its bound. Coarse-scanning the
    # space first and polishing the most promising starts finds the true
    # optimum — e.g. chintu REST goes from (jaw .34, pull −1.0) at
    # residual .117 to a correct closed smile at residual .026.
    # The grid is fixed, so the bake stays reproducible.
    grid = (
        (0.0, 0.25, 0.5, 0.75, 1.0),      # jaw
        (0.2, 0.5, 0.8),                  # width
        (0.1, 0.5, 0.9),                  # round
        (0.0, 0.4, 0.8),                  # press
        (-0.6, 0.0, 0.6),                 # pull
    )
    starts = [(err_of(x0), list(x0))]
    for jaw in grid[0]:
        for width in grid[1]:
            for rnd in grid[2]:
                for press in grid[3]:
                    for pull in grid[4]:
                        c = [jaw, width, rnd, press, pull]
                        starts.append((err_of(c), c))
    starts.sort(key=lambda t: t[0])
    # Always polish the seed (index 0 pre-sort is kept by value) plus the
    # best few grid points; more than this buys nothing measurable.
    candidates = [list(x0)] + [c for _, c in starts[:6]]

    best, best_err = list(x0), err_of(x0)
    for start in candidates:
        cur, cur_err = list(start), err_of(start)
        try:
            from scipy.optimize import minimize
            res = minimize(_contour_residual, cur,
                           args=(observed_outer, observed_inner),
                           method="L-BFGS-B", bounds=bounds)
            if res.success and float(res.fun) < cur_err:
                cur, cur_err = list(res.x), float(res.fun)
        except Exception:
            pass

        # Deterministic polish (also the sole optimizer without scipy):
        # shrinking coordinate descent, fixed schedule ⇒ same fit always.
        step = 0.25
        for _ in range(8):
            improved = False
            for k in range(len(cur)):
                for direction in (+1.0, -1.0):
                    cand = list(cur)
                    lo, hi = bounds[k]
                    cand[k] = min(hi, max(lo, cand[k] + direction * step))
                    e = err_of(cand)
                    if e < cur_err - 1e-12:
                        cur, cur_err, improved = cand, e, True
            if not improved:
                step *= 0.5
        if cur_err < best_err - 1e-12:
            best, best_err = cur, cur_err

    return {name: float(v) for name, v in zip(PARAM_NAMES, best, strict=True)}


# ═══════════════════════════════════════════
# The bake
# ═══════════════════════════════════════════

@dataclass
class BakeReport:
    poses: int = 0
    occluders: int = 0
    targets: int = 0
    plates: int = 0
    worst_rms: float = 0.0
    seam_err: float = 0.0
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def _poses_dir(character: str) -> str:
    return os.path.join(settings.CHARACTERS_DIR, character, "poses")


def _visemes_src_dir(character: str) -> str:
    return os.path.join(settings.CHARACTERS_DIR, character, "visemes_src")


def bake(rig: Rig, body: Image.Image, detect, canonical_pose: str = "neutral"
         ) -> BakeReport:
    """Produce every v3 artifact for `rig`, mutating it in place.

    `detect(Image) -> Optional[List[(x, y)]]` is injected rather than
    imported so the bake is testable with synthetic landmarks and so the
    MediaPipe dependency stays confined to the builder.
    """
    report = BakeReport()
    d = rig_dir(rig.character)
    os.makedirs(d, exist_ok=True)
    for sub in ("headless", "headmask", "occluder"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)

    # ─── canonical landmarks ──────────────────────────────
    canon = detect(body)
    if canon is None or len(canon) < N_LANDMARKS:
        raise BakeError(
            f"character '{rig.character}': canonical face detection "
            f"returned {0 if canon is None else len(canon)} landmarks, need "
            f"{N_LANDMARKS}. Rig v3 has NO heuristic fallback — install "
            f"mediapipe and confirm body.png shows a detectable face.")
    canon_lms = np.asarray(canon, dtype=np.float64)
    fh = face_height(canon_lms)
    body_arr = np.asarray(body.convert("RGBA"))
    alpha = body_arr[..., 3]

    # ─── §3.5 head plate ──────────────────────────────────
    # Perfection Plan Phase 1: the neck seam bounds the silhouette flood
    # from below, so the head can never claim the torso through the neck.
    # The joint exists before the mask (rig_builder derived it from the
    # silhouette); only a rig with no neck joint at all falls back to an
    # unbounded flood, where the vertical seam ramp remains the split.
    _neck = rig.joints.get("neck")
    hmask = head_mask(canon_lms, alpha,
                      seam_y=float(_neck[1]) if _neck else None,
                      overlap=HEAD_NECK_OVERLAP * fh,
                      character=rig.character)
    ys, xs = np.nonzero(hmask > 0.01)
    if len(ys) == 0:
        raise BakeError(f"character '{rig.character}': empty head mask")
    # The crop is deliberately LOOSER than the mask by PLATE_MARGIN·face_h
    # so the head transform has pixels to read past the art on every side.
    # The box may overhang body.png; crop_padded() fills that with the
    # transparency that is actually there.
    margin = int(math.ceil(HEAD_PLATE_MARGIN * fh))
    x0, x1 = int(xs.min()) - margin, int(xs.max()) + 1 + margin
    y0, y1 = int(ys.min()) - margin, int(ys.max()) + 1 + margin

    seam_y = float(_neck[1]) if _neck else float(y1)
    head_ramp, body_ramp = complementary_ramps(
        hmask, seam_y, SEAM_BAND * fh, overlap_px=HEAD_NECK_OVERLAP * fh)
    report.seam_err = seam_error(head_ramp, body_ramp)

    head_full = body_arr.copy().astype(np.float32)
    head_full[..., 3] = head_full[..., 3] * head_ramp
    head_arr = crop_padded(
        np.clip(head_full, 0, 255).astype(np.uint8), x0, y0, x1, y1)
    touch = border_opaque_counts(head_arr)
    if any(touch.values()):
        raise BakeError(
            f"character '{rig.character}': head plate touches its own "
            f"border ({touch}) even with a {margin}px margin — the head "
            f"transform would shear that art off. The head mask is "
            f"claiming pixels beyond the head (check clothing/limbs) or "
            f"PLATE_MARGIN is too small for this artwork.")
    head_crop = Image.fromarray(head_arr)
    head_crop.save(os.path.join(d, "head_canonical.png"))

    # landmarks in plate space
    plate_lms = canon_lms - np.array([x0, y0], dtype=np.float64)

    # ─── §3.5b measure the eyes the ARTIST drew ───────────
    # Must run on head_crop, BEFORE inpainting: the eyes have to still be
    # painted to be measured. MediaPipe's lid centroid is a reliable
    # SEED (Δ ≤ 14px) even though its scale is 2.2–2.9× too small, so we
    # keep the centre and re-derive the size from pixels.
    crop_arr = np.asarray(head_crop.convert("RGBA"))
    seed_l = _pick(plate_lms, LID_UPPER_L + LID_LOWER_L).mean(axis=0)
    seed_r = _pick(plate_lms, LID_UPPER_R + LID_LOWER_R).mean(axis=0)
    try:
        art_l, art_r = measure_pair(crop_arr, (seed_l[0], seed_l[1]),
                                    (seed_r[0], seed_r[1]), fh)
    except EyeMeasureError as exc:
        raise BakeError(
            f"character '{rig.character}': could not measure the drawn "
            f"eyes — {exc}") from exc
    art_eye_l, art_eye_r = art_l.to_dict(), art_r.to_dict()

    # Cut the DRAWN eyeball out as a sprite the renderer can translate.
    # Synthesizing the eye from flat colour every frame threw away the
    # artist's shading, lash overlap and highlight, and painted eye-white
    # the art never had. Moving real pixels keeps all of it.
    # Three art assets per eye, cut from the SAME plate crop so they
    # register with each other by construction: the eyeball gaze moves,
    # the socket it uncovers, and the lid that closes over it.
    for _eye, _key in ((art_l, "art_eye_l"), (art_r, "art_eye_r")):
        _s = "l" if _key.endswith("_l") else "r"
        _dst = art_eye_l if _s == "l" else art_eye_r
        for _fn, _payload, _kimg, _korg in (
                (f"eyeball_{_s}.png", eyeball_sprite(crop_arr, _eye),
                 "eyeball", "eyeball_origin"),
                (f"socket_{_s}.png", socket_backdrop(crop_arr, _eye),
                 "socket_img", "socket_origin"),
                (f"lid_{_s}.png", lid_sprite(crop_arr, _eye),
                 "lid_img", "lid_origin")):
            _arr, _org = _payload
            Image.fromarray(_arr, "RGBA").save(os.path.join(d, _fn))
            _dst.update({_kimg: _fn,
                         _korg: [int(_org[0]), int(_org[1])]})
    print(f"  [RigV3] {rig.character}: art eyes measured — "
          f"iris r={art_l.iris_r:.1f}/{art_r.iris_r:.1f}px "
          f"(MediaPipe said {iris_mp(plate_lms, IRIS_L, fh)[2]:.1f}/"
          f"{iris_mp(plate_lms, IRIS_R, fh)[2]:.1f}px)")

    # Eyes stay in the plate: the artwork's eye IS the resting eye.
    fmask = feature_mask(plate_lms, head_crop.size, fh,
                         eye_apertures=(art_l.aperture, art_r.aperture),
                         img_arr=crop_arr)
    plate = inpaint(head_crop, fmask)
    plate.save(os.path.join(d, "head_plate.png"))

    shading = shading_map(plate, plate_lms, fh)
    shading.save(os.path.join(d, "mouth_shading.png"))

    # ─── §3.5 feature geometry + palette (plate space) ────
    def poly(idx) -> List[Tuple[float, float]]:
        return [(float(p[0]), float(p[1])) for p in _pick(plate_lms, idx)]

    # Measured on the head crop BEFORE inpainting: the features have to
    # still be painted to be sampled. Region statistics, not pinpricks.
    # The eye entries come from the pixel measurement (§3.5b), which is
    # the only source that cannot confuse an iris with a lash.
    iris_geo_l = (art_l.iris_c[0], art_l.iris_c[1], art_l.iris_r)
    iris_geo_r = (art_r.iris_c[0], art_r.iris_c[1], art_r.iris_r)
    palette = extract_palette(crop_arr, plate_lms, fh,
                              iris_geo_l, iris_geo_r)
    # Measured eye colours win over the landmark-sampled ones: they were
    # taken from inside the segmented iris/sclera, so they cannot pick up
    # hair, a glasses frame or the lash line.
    #
    # The two eyes are measured independently, so prefer the better
    # sample per role rather than always trusting the left eye: chintu's
    # right eye sits behind a glasses lens, which pulls its "sclera" to a
    # grey (156,147,146) that is not eye-white. Eye white is the BRIGHTER
    # of the two measurements; the inks are the DARKER.
    def _lum(c) -> float:
        return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]

    for _k, _prefer in (("sclera", max), ("iris", None),
                        ("pupil", min), ("lash", min)):
        _cands = [c for c in (art_l.colors.get(_k), art_r.colors.get(_k))
                  if c]
        if not _cands:
            continue
        if _prefer is None or len(_cands) == 1:
            _v = _cands[0]
        else:
            _v = _prefer(_cands, key=_lum)
        palette[_k] = tuple(int(c) for c in _v)

    # Re-enforce the separations the renderer and the gates depend on.
    # extract_palette() established them, but the measured values above
    # replaced the very entries it had corrected, silently discarding the
    # guarantee. A closed lid is painted with `skin` plus a `lash` line
    # and blink closure is verified by the absence of iris-coloured
    # pixels, so BOTH must stay clear of the iris or a fully closed eye
    # is indistinguishable from an open one.
    palette["lash"] = _push_dark(palette["lash"], palette["iris"],
                                 IRIS_LASH_SEP)
    if _cheb(palette["skin"], palette["iris"]) < IRIS_LASH_SEP:
        # Vanishingly rare (an iris the same value as the face), but if it
        # happens the iris is what must move: skin is the face's identity.
        palette["iris"] = _push_dark(palette["iris"], palette["skin"],
                                     IRIS_LASH_SEP)
        palette["lash"] = _push_dark(palette["lash"], palette["iris"],
                                     IRIS_LASH_SEP)

    rig.head = HeadGeometry(
        plate="head_plate.png",
        landmarks=[(float(p[0]), float(p[1])) for p in plate_lms],
        lip_outer=poly(LIP_OUTER), lip_inner=poly(LIP_INNER),
        lid_upper_l=poly(LID_UPPER_L), lid_lower_l=poly(LID_LOWER_L),
        lid_upper_r=poly(LID_UPPER_R), lid_lower_r=poly(LID_LOWER_R),
        brow_l=poly(BROW_L), brow_r=poly(BROW_R),
        iris_l=iris_geo_l, iris_r=iris_geo_r,
        art_eye_l=art_eye_l, art_eye_r=art_eye_r,
        palette=palette, shading="mouth_shading.png",
        offset=(float(x0), float(y0)), face_height=fh)

    # ─── §3.1–3.4 per-pose registration + bakes ───────────
    pose_files: Dict[str, str] = {canonical_pose: ""}
    pdir = _poses_dir(rig.character)
    if os.path.isdir(pdir):
        for fname in sorted(os.listdir(pdir)):
            if fname.lower().endswith(".png"):
                pose_files[os.path.splitext(fname)[0]] = \
                    os.path.join(pdir, fname)

    rig.canonical_pose = canonical_pose
    rig.poses = {}
    for name, path in pose_files.items():
        img = body if not path else Image.open(path).convert("RGBA")
        if img.size != body.size:
            # A pose exported at the wrong canvas must not destroy the
            # whole bake (BakeError here demoted the rig to v2 and the
            # render path refused it). Same aspect ratio ⇒ it is just a
            # different-resolution export of the same canvas: rescale it.
            # Different aspect ⇒ registration is meaningless: EXCLUDE the
            # pose and keep the rest of the rig v3.
            src_aspect = img.size[0] / img.size[1]
            dst_aspect = body.size[0] / body.size[1]
            if abs(src_aspect - dst_aspect) <= 0.01 * dst_aspect:
                note = (f"pose '{name}' is {img.size}, canonical is "
                        f"{body.size} — same aspect, rescaled to the "
                        f"canonical canvas.")
                print(f"  [RigV3] {rig.character}: ⚠ {note}")
                report.notes.append(note)
                img = img.resize(body.size, Image.LANCZOS)
            else:
                note = (f"pose '{name}' is {img.size}, canonical is "
                        f"{body.size} (different aspect) — pose EXCLUDED "
                        f"from the library; the rest of the rig stays v3. "
                        f"Re-export the art on the canonical canvas.")
                print(f"  [RigV3] {rig.character}: ⚠ {note}")
                report.notes.append(note)
                continue
        if path:
            # A single unregistrable pose must not destroy the whole v3
            # bake (that demotes the rig to v2, the render path refuses
            # it, and the compositor falls back to whole-image expression
            # swaps — the exact D1 chaos v3 exists to prevent). The
            # anchoring guarantee is per-pose: a pose that cannot be
            # registered within budget is EXCLUDED from the library, so
            # only verified poses ever render.
            lms = detect(img)
            if lms is None or len(lms) < N_LANDMARKS:
                note = (f"pose '{name}': face detection failed — pose "
                        f"EXCLUDED from the library (§3.1: no heuristic "
                        f"boxes). Re-export the art to restore it.")
                print(f"  [RigV3] {rig.character}: ⚠ {note}")
                report.notes.append(note)
                continue
            pose_lms = np.asarray(lms, dtype=np.float64)
            try:
                xform = register_pose(canon_lms, pose_lms, pose_name=name,
                                      face_height_px=fh)
            except RegistrationError as e:
                note = (f"{e} — pose EXCLUDED from the library; the rest "
                        f"of the rig stays v3.")
                print(f"  [RigV3] {rig.character}: ⚠ {note}")
                report.notes.append(note)
                continue
        else:
            pose_lms = canon_lms
            xform = SimilarityTransform.identity()

        entry = _bake_pose(rig, name, img, pose_lms, xform, fh,
                           body_arr, seam_y, d)
        rig.poses[name] = entry
        report.poses += 1
        if entry.occluded:
            report.occluders += 1
        report.worst_rms = max(report.worst_rms, xform.rms)

    # ─── §3.6 viseme plates + mouth targets from the art ──
    # ONE landmark measurement per source frame feeds BOTH the sprite
    # cut and the 5-D parameter fit, so geometry and pixels can never
    # disagree (independent coordinate passes are exactly what made the
    # mouth "fly" historically).
    report.plates, report.targets = _bake_viseme_plates(rig, detect, fh, d)

    rig.version = 3
    return report


def _bake_pose(rig: Rig, name: str, img: Image.Image, pose_lms: np.ndarray,
               xform: SimilarityTransform, fh: float, canon_arr: np.ndarray,
               canon_seam_y: float, d: str) -> PoseEntry:
    """§3.3/§3.4 — headless body, head mask, and occluder for one pose."""
    arr = np.asarray(img.convert("RGBA"))
    _neck = rig.joints.get("neck")
    canon_neck_x = float(_neck[0]) if _neck else 0.0
    seam_y = xform.apply_point(canon_neck_x, canon_seam_y)[1]
    pmask = head_mask(pose_lms, arr[..., 3], seam_y=seam_y,
                      overlap=HEAD_NECK_OVERLAP * fh,
                      character=rig.character)
    head_ramp, body_ramp = complementary_ramps(
        pmask, seam_y, SEAM_BAND * fh, overlap_px=HEAD_NECK_OVERLAP * fh)

    headless = arr.copy().astype(np.float32)
    headless[..., 3] = headless[..., 3] * body_ramp

    # Phase 1 gate — zero orphan head pixels. Above the seam band the
    # body factor is exactly (1 − mask), so an opaque headless pixel up
    # there is either:
    #   • BODY mass that legitimately rises past the neck — a raised
    #     hand, a pencil held beside the face (gudiya's neutral pose) —
    #     which is CONNECTED down through the seam to the torso and
    #     must stay on the headless body so it moves with the body, or
    #   • ORPHAN head mass the silhouette flood failed to claim
    #     (severed hair, a spike, a bun) — DISCONNECTED from everything
    #     below the seam. That art would freeze in place while the head
    #     moves, which is precisely the "orphan hair" defect.
    # The two are separated by the same deterministic flood the mask
    # uses: claim everything connected to the below-seam body; whatever
    # opaque mass remains above the gate line is a true orphan. Refuse
    # to write the asset rather than ship it.
    hl_a = np.clip(headless[..., 3], 0, 255)
    gate_top = int(math.floor(seam_y - SEAM_BAND * fh))
    if gate_top > 0:
        opaque = hl_a > ORPHAN_ALPHA_THRESH
        yy = np.arange(opaque.shape[0])[:, None]
        body_seed = opaque & (yy >= gate_top)
        body_connected = _flood_from_seed(opaque, body_seed)
        orphan_mask = opaque & (yy < gate_top) & ~body_connected
        orphan = int(orphan_mask.sum())
        if orphan:
            oys, oxs = np.nonzero(orphan_mask)
            raise BakeError(
                f"character '{rig.character}', pose '{name}': {orphan} "
                f"opaque headless pixels above the neck seam "
                f"(y<{gate_top}, bbox x[{oxs.min()},{oxs.max()}] "
                f"y[{oys.min()},{oys.max()}]) — head mass the silhouette "
                f"flood did not claim, disconnected from the body below "
                f"the seam, would render as orphan hair. The art above "
                f"the seam is disconnected from the face silhouette "
                f"(check for fully transparent gaps).")

    hl_rel = os.path.join("headless", f"{name}.png")
    Image.fromarray(np.clip(headless, 0, 255).astype(np.uint8)) \
        .save(os.path.join(d, hl_rel))

    hm_rel = os.path.join("headmask", f"{name}.png")
    Image.fromarray((np.clip(head_ramp, 0, 1) * 255).astype(np.uint8)) \
        .save(os.path.join(d, hm_rel))

    # Delta occluders disabled: RGB delta between AI poses and canonical face
    # erroneously captured whole faces/ears/necks as static ghost overlays.
    occ_rel: Optional[str] = None
    occluded = False

    return PoseEntry(name=name,
                     landmarks=[(float(p[0]), float(p[1])) for p in pose_lms],
                     xform=xform.to_dict(),
                     headless=hl_rel, headmask=hm_rel, occluder=occ_rel,
                     seam_y=float(seam_y), occluded=occluded)


# Plate-cut tuning. The feather is at CANONICAL scale (the plate is
# emitted after normalisation), so every character's sprite edge is
# equally soft regardless of source resolution.
PLATE_FEATHER_PX = 1.2
PLATE_MARGIN = 0.05          # ×face_h, crop margin around the lip bbox
PLATE_DILATE = 0.012         # ×face_h, mask growth past the outer ring so
                             # the artwork's vermilion border is not clipped


def _bake_viseme_plates(rig: Rig, detect, canon_fh: float,
                        d: str) -> Tuple[int, int]:
    """§3.6 — bake art viseme PLATES and fit 5-D mouth targets from
    `visemes_src/<VISEME>.png`, using ONE landmark measurement per frame.

    Per source frame: detect landmarks (fail LOUDLY on a miss — a
    heuristic fallback box is exactly what historically put the mouth on
    the eyes), take the outer-lip ring, normalise by the face-height
    ratio to canonical scale, alpha-cut the lip polygon with a
    ~1.2 px Gaussian feather, write `rig/visemes/<VISEME>.png` and
    register it in `rig.visemes`. The SAME measured contours drive the
    `fit_mouth_target` fit, so sprite and geometry cannot disagree.

    Returns (plates_written, targets_fitted). Shapes the artist did not
    draw keep the built-in articulatory defaults
    (engine/mouth_model.DEFAULT_TARGETS) — partial art is never worse
    than no art. LID_* sprite entries are preserved untouched.
    """
    from PIL import ImageFilter
    from engine.mouth_model import DEFAULT_TARGETS
    from engine.rig import VISEME_NAMES

    src = _visemes_src_dir(rig.character)
    # The v3 plate bake OWNS the mouth-class sprite registry: clear any
    # legacy v1/v2 bakes so two coordinate systems cannot fight. Lid
    # sprites are a different subsystem and survive.
    rig.visemes = {k: v for k, v in rig.visemes.items()
                   if k.startswith("LID_")}
    rig.mouth_targets = {}
    if not os.path.isdir(src):
        return 0, 0

    os.makedirs(os.path.join(d, "visemes"), exist_ok=True)
    plates = 0
    targets = 0
    for fname in sorted(os.listdir(src)):
        if not fname.lower().endswith(".png"):
            continue
        vis = os.path.splitext(fname)[0].upper()
        if vis not in VISEME_NAMES:
            print(f"  [RigV3] {rig.character}: visemes_src/{fname} is not "
                  f"a viseme class — skipped")
            continue
        path = os.path.join(src, fname)
        img = Image.open(path).convert("RGBA")
        iw, ih = img.size
        # Head crop detection first: prevents MediaPipe from mistaking puckered lips for the nose
        hc_x0, hc_x1 = int(0.12 * iw), int(0.88 * iw)
        hc_y0, hc_y1 = int(0.0), int(0.68 * ih)
        head_crop = img.crop((hc_x0, hc_y0, hc_x1, hc_y1))
        lms_hc = detect(head_crop)
        if lms_hc is not None and len(lms_hc) >= N_LANDMARKS:
            lms = [(p[0] + hc_x0, p[1] + hc_y0) for p in lms_hc]
        else:
            lms = detect(img)

        if lms is None or len(lms) < N_LANDMARKS:
            raise BakeError(
                f"character '{rig.character}': face detection FAILED on "
                f"viseme source '{path}' "
                f"({0 if lms is None else len(lms)}/{N_LANDMARKS} "
                f"landmarks). Rig v3 has NO heuristic fallback box — fix "
                f"or remove the file.")
        lm = np.asarray(lms, dtype=np.float64)
        local_fh = face_height(lm)
        outer_px = _pick(lm, LIP_OUTER)
        inner_px = _pick(lm, LIP_INNER)

        # ── the 5-D fit, from the SAME measurement ────────
        outer_n = normalize_contour(outer_px, local_fh)
        inner_n = normalize_contour(inner_px, local_fh)
        seed = None
        for v, p in DEFAULT_TARGETS.items():
            if getattr(v, "value", str(v)) == vis:
                seed = p.as_tuple()
                break
        rig.mouth_targets[vis] = fit_mouth_target(outer_n, inner_n, seed)
        targets += 1

        # ── the plate cut: anatomical super-ellipse crop with smoothstep feather ──
        ratio = canon_fh / local_fh          # source px → canonical px

        # Outer lip boundaries → centre
        min_x, max_x = float(outer_px[:, 0].min()), float(outer_px[:, 0].max())
        min_y, max_y = float(outer_px[:, 1].min()), float(outer_px[:, 1].max())

        # Puckered cartoon visemes have tall upper lips: shift center up and use generous top padding
        if "ROUNDED" in vis:
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0 - 0.02 * local_fh
            pad_top = 0.075 * local_fh
            pad_bot = 0.040 * local_fh
            pad_side = 0.065 * local_fh
        else:
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            pad_top = 0.050 * local_fh
            pad_bot = 0.035 * local_fh
            pad_side = 0.060 * local_fh

        x0 = int(cx - (max_x - min_x) / 2.0 - pad_side)
        x1 = int(cx + (max_x - min_x) / 2.0 + pad_side)
        y0 = int(cy - (max_y - min_y) / 2.0 - pad_top)
        y1 = int(cy + (max_y - min_y) / 2.0 + pad_bot)

        crop = img.crop((x0, y0, x1, y1))
        cw, ch = crop.size

        # Soft anatomical super-ellipse mask (order 3.5: solid inner 75%, smoothstep falloff in outer 25%)
        yy, xx = np.ogrid[:ch, :cw]
        rx = cw / 2.0
        ry = ch / 2.0
        norm_dist = ((np.abs(xx - rx) / rx)**3.5 + (np.abs(yy - ry) / ry)**3.5)**(1.0 / 3.5)

        mask = np.clip((1.0 - norm_dist) / 0.25, 0.0, 1.0)
        mask = mask * mask * (3 - 2 * mask)  # smoothstep

        crop_arr = np.array(crop).copy()
        crop_arr[..., 3] = (mask * 255).astype(np.uint8)

        out_w = max(2, int(round(cw * ratio)))
        out_h = max(2, int(round(ch * ratio)))
        plate = Image.fromarray(crop_arr).resize((out_w, out_h), Image.LANCZOS)

        rel = os.path.join("visemes", f"{vis}.png")
        plate.save(os.path.join(d, rel))
        rig.visemes[vis] = rel
        plates += 1
    return plates, targets


# ═══════════════════════════════════════════
# colour helpers
# ═══════════════════════════════════════════

def _darken(rgb, f: float) -> Tuple[int, int, int]:
    rgb = rgb or (140, 90, 90)
    return tuple(max(0, min(255, int(c * f))) for c in rgb)


def _mix(a, b, t: float) -> Tuple[int, int, int]:
    a = a or b
    return tuple(int(av + (bv - av) * t) for av, bv in zip(a, b, strict=True))
