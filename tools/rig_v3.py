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

# Palette sample sites: (key, landmark index, radius factor)
_PALETTE_SITES = (
    ("skin", 50, 0.03),
    ("lip", 13, 0.012),
    ("lip_shadow", 17, 0.012),
)

# Bake tuning — all proportional to face height, never literal pixels.
HEAD_DILATE = 0.02          # ×face_h, mask growth so hair is not clipped
SEAM_BAND = 0.06            # ×face_h, complementary feather width
INPAINT_DILATE = 0.035      # ×face_h, feature-mask growth before inpaint
OCCLUDER_RGB_DELTA = 26.0   # mean |ΔRGB| that counts as "different art"


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


def head_mask(lms: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """§3.3 — head mask = convex hull of the face landmarks, extended UP
    to the alpha silhouette top (so hair is included), then dilated by
    0.02·face_h and softened with a distance ramp.

    Extending to the silhouette top rather than to the face oval is what
    keeps a fringe or a topknot travelling with the head instead of being
    left behind on the body.
    """
    h, w = alpha.shape
    fh = face_height(lms)
    hull = convex_hull(_pick(lms, FACE_OVAL))

    # Extend the hull upward to the silhouette top over the head's x-span
    solid = alpha > 40
    rows = np.nonzero(solid.any(axis=1))[0]
    top_y = float(rows[0]) if len(rows) else 0.0
    x0, x1 = float(hull[:, 0].min()), float(hull[:, 0].max())
    pad_x = 0.10 * fh
    extended = np.vstack([hull,
                          np.array([[x0 - pad_x, top_y],
                                    [x1 + pad_x, top_y]])])
    hull = convex_hull(extended)

    mask = polygon_mask(hull, (w, h))
    mask = _distance_blur(mask, HEAD_DILATE * fh)
    # A head mask may never claim transparent pixels.
    return (mask * (alpha > 0).astype(np.float32)).astype(np.float32)


def complementary_ramps(mask: np.ndarray, seam_y: float, band_px: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """§3.3 — the energy-preserving seam.

    Returns (head_alpha_factor, body_alpha_factor) with the identity
    `head + body == 1` at every pixel, by construction: the body factor
    is literally `1 - head`. The head factor is the head mask crossed
    with a vertical ramp that falls from 1 to 0 across the neck band, so
    neither layer can contribute extra energy and no seam line exists.
    """
    h, w = mask.shape
    ys = np.arange(h, dtype=np.float32)[:, None]
    band = max(1.0, float(band_px))
    ramp = np.clip(((seam_y + band) - ys) / (2.0 * band), 0.0, 1.0)
    head = np.clip(mask * ramp, 0.0, 1.0)
    return head.astype(np.float32), (1.0 - head).astype(np.float32)


def seam_error(head_a: np.ndarray, body_a: np.ndarray) -> float:
    """Max deviation from the α_head + α_body = 1 identity. The QC seam
    gate asserts this is ~0; it is 0 by construction, so a non-zero
    value means someone reintroduced an independent ramp."""
    return float(np.max(np.abs(head_a + body_a - 1.0)))


# ═══════════════════════════════════════════
# §3.5 — head plate: crop + inpaint features out
# ═══════════════════════════════════════════

def feature_mask(lms: np.ndarray, size: Tuple[int, int],
                 fh: float) -> np.ndarray:
    """Binary mask of the painted features to remove: the outer lip ring
    and both lid rings, dilated so the artwork's ink outline goes too.
    Leaving the outline behind is what produced ghost lips under the
    parametric mouth."""
    w, h = size
    m = np.zeros((h, w), dtype=np.float32)
    for idx in (LIP_OUTER, LID_UPPER_L + LID_LOWER_L,
                LID_UPPER_R + LID_LOWER_R):
        m = np.maximum(m, polygon_mask(convex_hull(_pick(lms, idx)), (w, h)))
    grown = _distance_blur(m, INPAINT_DILATE * fh)
    return (grown > 0.35).astype(np.uint8)


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
        return Image.fromarray(arr, "RGBA")
    try:
        import cv2
        rgb = np.ascontiguousarray(arr[..., :3])
        radius = max(3, int(0.02 * max(arr.shape[:2])))
        filled = cv2.inpaint(rgb, mask, radius, cv2.INPAINT_NS)
        arr[..., :3] = filled
        return Image.fromarray(arr, "RGBA")
    except Exception:
        return Image.fromarray(_normalized_convolution(arr, mask), "RGBA")


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
    out.paste(Image.fromarray(norm, "L"), (x0, y0))
    return out


# ═══════════════════════════════════════════
# §3.6 — fit the 5-D mouth targets from the art
# ═══════════════════════════════════════════

def normalize_contour(pts: np.ndarray, fh: float) -> np.ndarray:
    """Map a lip ring into the model's normalized mouth space: centred at
    the ring centroid, scaled so one unit is 0.25·face_h. This is the
    same space `mouth_model.lip_contour` emits, which is what makes the
    fit comparable across characters and resolutions."""
    c = pts.mean(axis=0)
    scale = max(1e-6, 0.25 * fh)
    return (pts - c) / scale


def _contour_residual(params, observed_outer: np.ndarray,
                      observed_inner: np.ndarray) -> float:
    from engine.mouth_model import MouthParams, lip_contour
    p = MouthParams(*params).clamped()
    n = max(len(observed_outer), len(observed_inner))
    outer, inner = lip_contour(p, n=n)
    err = 0.0
    for obs, model in ((observed_outer, outer), (observed_inner, inner)):
        if len(obs) == 0:
            continue
        m = _resample_closed(np.asarray(model, dtype=np.float64), len(obs))
        err += float(np.mean(np.sum((m - obs) ** 2, axis=1)))
    return err


def _resample_closed(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polyline to n points by arc length, so the fit
    compares shapes rather than accidental vertex counts."""
    if len(poly) == n:
        return poly
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1] or 1.0
    t = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(t, cum, closed[:, 0])
    y = np.interp(t, cum, closed[:, 1])
    return np.stack([x, y], axis=1)


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

    best = list(x0)
    best_err = _contour_residual(best, observed_outer, observed_inner)
    try:
        from scipy.optimize import minimize
        res = minimize(_contour_residual, x0,
                       args=(observed_outer, observed_inner),
                       method="L-BFGS-B", bounds=bounds)
        if res.success and res.fun < best_err:
            best, best_err = list(res.x), float(res.fun)
    except Exception:
        pass

    # Deterministic polish (also the sole optimizer without scipy):
    # shrinking coordinate descent, fixed schedule ⇒ same input, same fit.
    step = 0.25
    for _ in range(6):
        improved = False
        for k in range(len(best)):
            for direction in (+1.0, -1.0):
                cand = list(best)
                lo, hi = bounds[k]
                cand[k] = min(hi, max(lo, cand[k] + direction * step))
                err = _contour_residual(cand, observed_outer, observed_inner)
                if err < best_err - 1e-12:
                    best, best_err, improved = cand, err, True
        if not improved:
            step *= 0.5
    return {name: float(v) for name, v in zip(PARAM_NAMES, best)}


# ═══════════════════════════════════════════
# The bake
# ═══════════════════════════════════════════

@dataclass
class BakeReport:
    poses: int = 0
    occluders: int = 0
    targets: int = 0
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
    hmask = head_mask(canon_lms, alpha)
    ys, xs = np.nonzero(hmask > 0.01)
    if len(ys) == 0:
        raise BakeError(f"character '{rig.character}': empty head mask")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    seam_y = float(rig.joints.get("neck", (0.0, y1))[1])
    head_ramp, body_ramp = complementary_ramps(hmask, seam_y, SEAM_BAND * fh)
    report.seam_err = seam_error(head_ramp, body_ramp)

    head_full = body_arr.copy().astype(np.float32)
    head_full[..., 3] = head_full[..., 3] * head_ramp
    head_crop = Image.fromarray(
        np.clip(head_full[y0:y1, x0:x1], 0, 255).astype(np.uint8), "RGBA")
    head_crop.save(os.path.join(d, "head_canonical.png"))

    # landmarks in plate space
    plate_lms = canon_lms - np.array([x0, y0], dtype=np.float64)
    fmask = feature_mask(plate_lms, head_crop.size, fh)
    plate = inpaint(head_crop, fmask)
    plate.save(os.path.join(d, "head_plate.png"))

    shading = shading_map(plate, plate_lms, fh)
    shading.save(os.path.join(d, "mouth_shading.png"))

    # ─── §3.5 feature geometry + palette (plate space) ────
    def poly(idx) -> List[Tuple[float, float]]:
        return [(float(p[0]), float(p[1])) for p in _pick(plate_lms, idx)]

    def iris(idx) -> Tuple[float, float, float]:
        pts = _pick(plate_lms, idx)
        c = pts.mean(axis=0)
        r = float(np.linalg.norm(pts - c, axis=1).mean())
        # A refined-iris ring collapses to its centre on some exports;
        # fall back to an anatomical radius so the eye is never a dot.
        if r < 0.5:
            r = 0.055 * fh
        return (float(c[0]), float(c[1]), r)

    palette: Dict[str, Tuple[int, int, int]] = {}
    plate_arr = np.asarray(head_crop.convert("RGBA"))
    for key, lm_idx, rad in _PALETTE_SITES:
        px, py = plate_lms[lm_idx]
        palette[key] = _sample(plate_arr, px, py, max(1.0, rad * fh))
    palette.setdefault("oral_cavity", _darken(palette.get("lip"), 0.35))
    palette.setdefault("teeth", (242, 240, 236))
    palette.setdefault("tongue", _mix(palette.get("lip"), (196, 96, 104), 0.5))
    palette.setdefault("sclera", (248, 247, 245))
    palette.setdefault("iris", _sample(plate_arr, *plate_lms[IRIS_L[0]],
                                       radius=max(1.0, 0.02 * fh)))
    palette.setdefault("lash", _darken(palette.get("skin"), 0.25))

    rig.head = HeadGeometry(
        plate="head_plate.png",
        landmarks=[(float(p[0]), float(p[1])) for p in plate_lms],
        lip_outer=poly(LIP_OUTER), lip_inner=poly(LIP_INNER),
        lid_upper_l=poly(LID_UPPER_L), lid_lower_l=poly(LID_LOWER_L),
        lid_upper_r=poly(LID_UPPER_R), lid_lower_r=poly(LID_LOWER_R),
        brow_l=poly(BROW_L), brow_r=poly(BROW_R),
        iris_l=iris(IRIS_L), iris_r=iris(IRIS_R),
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
            raise BakeError(
                f"pose '{name}' is {img.size}, canonical is {body.size}. "
                f"Poses must share the canonical canvas or every "
                f"registration is meaningless.")
        if path:
            lms = detect(img)
            if lms is None or len(lms) < N_LANDMARKS:
                raise BakeError(
                    f"pose '{name}': face detection failed. Rig v3 "
                    f"landmarks every pose (§3.1) — a pose that inherits "
                    f"body.png's boxes is defect D1.")
            pose_lms = np.asarray(lms, dtype=np.float64)
            xform = register_pose(canon_lms, pose_lms, pose_name=name,
                                  face_height_px=fh)
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

    # ─── §3.6 mouth targets from the art ──────────────────
    report.targets = _bake_mouth_targets(rig, detect, fh)

    rig.version = 3
    return report


def _bake_pose(rig: Rig, name: str, img: Image.Image, pose_lms: np.ndarray,
               xform: SimilarityTransform, fh: float, canon_arr: np.ndarray,
               canon_seam_y: float, d: str) -> PoseEntry:
    """§3.3/§3.4 — headless body, head mask, and occluder for one pose."""
    arr = np.asarray(img.convert("RGBA"))
    pmask = head_mask(pose_lms, arr[..., 3])
    # The seam follows the pose's own head: transforming the canonical
    # seam keeps the band on the neck even when the pose leans.
    seam_y = xform.apply_point(0.0, canon_seam_y)[1]
    head_ramp, body_ramp = complementary_ramps(pmask, seam_y, SEAM_BAND * fh)

    headless = arr.copy().astype(np.float32)
    headless[..., 3] = headless[..., 3] * body_ramp
    hl_rel = os.path.join("headless", f"{name}.png")
    Image.fromarray(np.clip(headless, 0, 255).astype(np.uint8), "RGBA") \
        .save(os.path.join(d, hl_rel))

    hm_rel = os.path.join("headmask", f"{name}.png")
    Image.fromarray((np.clip(head_ramp, 0, 1) * 255).astype(np.uint8), "L") \
        .save(os.path.join(d, hm_rel))

    # Occluder: head-region pixels whose ART differs from canonical —
    # a hand or a prop crossing the face. Composited AFTER the head so
    # it still passes in front of it.
    occ_rel: Optional[str] = None
    occluded = False
    if arr.shape == canon_arr.shape:
        delta = np.abs(arr[..., :3].astype(np.float32) -
                       canon_arr[..., :3].astype(np.float32)).mean(axis=-1)
        inside = (head_ramp > 0.25) & (arr[..., 3] > 50)
        occ = inside & (delta > OCCLUDER_RGB_DELTA)
        if occ.sum() > (0.004 * fh * fh):     # ignore antialias speckle
            out = np.zeros_like(arr)
            out[occ] = arr[occ]
            occ_rel = os.path.join("occluder", f"{name}.png")
            Image.fromarray(out, "RGBA").save(os.path.join(d, occ_rel))
            occluded = True

    return PoseEntry(name=name,
                     landmarks=[(float(p[0]), float(p[1])) for p in pose_lms],
                     xform=xform.to_dict(),
                     headless=hl_rel, headmask=hm_rel, occluder=occ_rel,
                     seam_y=float(seam_y), occluded=occluded)


def _bake_mouth_targets(rig: Rig, detect, fh: float) -> int:
    """§3.6 — fit 5-D targets from `visemes_src/<VISEME>.png`.

    Shapes the artist did not draw keep the built-in articulatory
    defaults (engine/mouth_model.DEFAULT_TARGETS), so a character with
    two hand-drawn visemes is strictly better off than one with none —
    partial art is never worse than no art.
    """
    from engine.mouth_model import DEFAULT_TARGETS, MouthParams
    src = _visemes_src_dir(rig.character)
    rig.mouth_targets = {}
    if not os.path.isdir(src):
        return 0

    fitted = 0
    for fname in sorted(os.listdir(src)):
        if not fname.lower().endswith(".png"):
            continue
        vis = os.path.splitext(fname)[0].upper()
        try:
            img = Image.open(os.path.join(src, fname)).convert("RGBA")
            lms = detect(img)
            if lms is None or len(lms) < N_LANDMARKS:
                continue
            lm = np.asarray(lms, dtype=np.float64)
            local_fh = face_height(lm)
            outer = normalize_contour(_pick(lm, LIP_OUTER), local_fh)
            inner = normalize_contour(_pick(lm, LIP_INNER), local_fh)
            seed = None
            for v, p in DEFAULT_TARGETS.items():
                if getattr(v, "value", str(v)) == vis:
                    seed = p.as_tuple()
                    break
            rig.mouth_targets[vis] = fit_mouth_target(outer, inner, seed)
            fitted += 1
        except Exception as e:
            print(f"  [RigV3] viseme '{vis}' fit skipped: {e}")
    return fitted


# ═══════════════════════════════════════════
# colour helpers
# ═══════════════════════════════════════════

def _sample(arr: np.ndarray, x: float, y: float,
            radius: float = 3.0) -> Tuple[int, int, int]:
    h, w = arr.shape[:2]
    r = max(1, int(radius))
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    patch = arr[y0:y1, x0:x1]
    if patch.size == 0:
        return (200, 170, 150)
    opaque = patch[..., 3] > 60
    if not opaque.any():
        return (200, 170, 150)
    return tuple(int(v) for v in patch[opaque][:, :3].mean(axis=0))


def _darken(rgb, f: float) -> Tuple[int, int, int]:
    rgb = rgb or (140, 90, 90)
    return tuple(max(0, min(255, int(c * f))) for c in rgb)


def _mix(a, b, t: float) -> Tuple[int, int, int]:
    a = a or b
    return tuple(int(av + (bv - av) * t) for av, bv in zip(a, b))
