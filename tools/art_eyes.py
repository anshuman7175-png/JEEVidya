"""
JEEVidya — Art-Measured Eye Geometry (rig v3 §3.5b)
══════════════════════════════════════════════════════════════════════
Measures the EYE THE ARTIST DREW, from pixels, instead of trusting
MediaPipe's lid ring.

Why this module exists
──────────────────────
MediaPipe FaceMesh is trained on photographs of human faces. Its lid
polylines and iris ring therefore carry HUMAN proportions. Our art is
stylised chibi/Pixar: the drawn eye is 2.2–2.9× larger than the ring
MediaPipe reports, and the ring lands on the eyeball's upper third.

Consequences of believing MediaPipe on this art (all observed):

  • The parametric eye rendered as a small patch floating inside a much
    larger painted eye — the drawn iris stayed visible around it.
  • The lid's "cap" fill (skin from the lid margin upward) was clipped
    by the patch rectangle, not by the eye, so a blink painted a
    HARD-EDGED SKIN RECTANGLE over the brow.
  • `feature_mask` inpainted only that small hull, so the artwork's eye
    survived the plate and no amount of drawing on top could hide it.
  • blink=1 could not occlude an iris it was 2.5× too small to cover,
    so the QC blink-closure gate was unpassable by construction.

So the geometry is measured here, and the renderer is art-first: the
artwork's own eye IS the resting eye. See `engine/eye_model.py`.

What is measured (all in head-plate pixels)
───────────────────────────────────────────
  aperture : polygon of the drawn eye opening — sclera + iris + the
             painted lash line, i.e. everything that is "the eye" and
             not skin. This is the CLIP for every eye pixel we draw:
             nothing the renderer paints can leave it, which makes the
             skin-rectangle defect unrepresentable.
  iris     : centre + semi-axes + angle of the drawn eyeball, by
             ellipse fit (a circle cannot fit chintu's squashed eye:
             axes 28.7 × 39.4 px).
  colours  : sclera / iris / pupil / lash sampled from INSIDE their own
             measured regions, so they cannot pick up shirt or hair.

Method — and why each step is the way it is
───────────────────────────────────────────
1 ROI: a generous box about the MediaPipe eye CENTRE. The centre is the
  one thing MediaPipe gets right on this art (Δ ≤ 14 px); only its
  SCALE is wrong. So we keep the centre as a seed and re-derive size.

2 Skin reference from the ROI border ring, never a global palette: the
  ROI border is guaranteed to be face, and a local reference absorbs
  the artwork's own shading gradient across the face.

3 "Eye-like" = far from skin AND (bright-desaturated | dark).
     bright-desaturated → sclera (eye white)
     dark               → iris, pupil, lash
  Hue/saturation, not raw distance, is what separates an eye from skin
  on stylised art where the lip/skin/shirt tones are all warm.

4 Largest component nearest the seed, then CLOSE + hole-fill. Filling
  holes is what puts the specular CATCHLIGHT inside the aperture — it
  is a bright saturated dot that fails the "eye-like" test on its own.

5 Iris = aperture minus sclera, ellipse-fitted. Fitting the boundary
  (not the pixel spread) keeps a lash that overlaps the eyeball's top
  from dragging the centre upward.

Every step is deterministic: same art in, same numbers out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ═══════════════════════════════════════════
# Tuning — proportional to face height, never literal pixels, so the
# same constants hold at any export resolution (Part VIII).
# ═══════════════════════════════════════════

ROI_SPAN = 0.32          # ×face_h, half-width of the search box
SKIN_BORDER = 6          # px ring of the ROI used as the skin reference
SKIN_DELTA = 40.0        # max|ΔRGB| from skin that counts as "not skin"
SCLERA_V_MIN = 150.0     # HSV V above this is a candidate eye white
SCLERA_S_MAX = 70.0      # …and S below this (bright AND desaturated)
DARK_V_MAX = 110.0       # HSV V below this is iris / pupil / lash
MIN_COMPONENT = 40       # px, smaller blobs are noise
MIN_APERTURE = 0.010     # ×ROI area, below this the measurement failed
MAX_APERTURE = 0.85      # ×ROI area, above this it leaked into the face
IRIS_MIN_RATIO = 0.06    # iris r_eq ≥ this ×face_h, else implausible
IRIS_MAX_RATIO = 0.30    # iris r_eq ≤ this ×face_h
PUPIL_PCTL = 12.0        # darkest luma percentile inside the iris = pupil
SAMPLE_MIN_PX = 24       # fewer pixels than this is not a measurement

# Iris body selection, in COLOUR space rather than by brightness (§3).
# The pupil and the sclera are both achromatic, so a saturation floor
# excludes them by construction; the luma window then drops the black
# lash below and the specular catchlight above. Verified on all four
# eyes of the two characters: 1000–1400 px survive per eye and the
# medians agree to within a few units — (100,40,21), (94,42,25),
# (104,44,9), (113,52,13) — i.e. a stable warm brown, not ink.
IRIS_SAT_MIN = 25.0      # max(RGB)−min(RGB) at or above this is chromatic
IRIS_LUM_MIN = 24.0      # below this is lash/pupil ink, not iris colour
IRIS_LUM_MAX = 190.0     # above this is sclera or the catchlight

# Lash / lid-line ink, by the same colour-space separation: ink is dark
# AND near-achromatic, while shaded eyelid skin stays warm and saturated.
LASH_LUM_MAX = 90.0      # ink is no brighter than this
LASH_SAT_MAX = 45.0      # …and no more saturated than this
# A closed lid is painted with `skin` + a `lash` line, and the blink gate
# passes only when no iris colour survives. Both must therefore be
# separable from the iris in the SAME metric the gate uses (Chebyshev).
# The eyeball sprite's alpha ellipse is the iris grown by this much on
# every side, and `socket_backdrop` MUST inpaint exactly that ellipse.
# The two were independent (sprite 1.0× + feather, socket 1.30×), which
# left a ring of inpainted smear around the iris on every resting frame —
# visible as a pale halo the artwork does not have. One constant makes
# "the sprite exactly covers the hole" true by construction.
SPRITE_FEATHER = 1.2

LASH_IRIS_SEP = 56.0
SKIN_IRIS_SEP = 56.0
SEP_MAX_STEPS = 48       # bounded ⇒ deterministic

# Upper-lid strip extraction. The strip is the band of plate between the
# aperture's top and the BROW, and its height is measured per character
# rather than assumed: the previous bake took a fixed 0.42 × eye-height,
# but on chibi art the brow sits closer than that, so the band it cut was
# mostly eyebrow ink — chintu's strips came out with mean RGB (120,76,60)
# and (69,38,29), i.e. black brow, and a closed eye filled with ink.
# Walking outward from the aperture and stopping AT the brow cannot make
# that mistake on any face, because the stop is the brow itself.
LID_INK_LUM_MAX = 105.0  # a row's pixel is ink if it is this dark…
LID_INK_SAT_MAX = 60.0   # …and this close to achromatic
LID_LASH_FRAC = 0.55     # ≥ this share of a row is ink ⇒ ink row
LID_LASH_MAX = 0.30      # ×eye-height, cap on the lash zone kept at bottom
LID_SKIN_MIN = 0.09      # ×eye-height, less brow-free skin than this fails


class EyeMeasureError(RuntimeError):
    """Measurement that cannot be trusted fails loudly (Law 1).

    Never silently degrade to the MediaPipe ring: that ring is exactly
    the defect this module exists to remove, so falling back to it
    would reintroduce the skin rectangle on the very art that most
    needs measuring.
    """


@dataclass(frozen=True)
class ArtEye:
    """One eye as the ARTIST drew it, in head-plate pixels.

    aperture   : (N,2) closed polygon of the drawn eye opening
    iris_c     : (cx, cy) centre of the drawn eyeball
    iris_axes  : (semi_major, semi_minor) of the fitted ellipse
    iris_angle : ellipse rotation, degrees
    iris_r     : sqrt(a·b) — the single scale gaze/pupil maths uses
    colors     : sclera / iris / pupil / lash, each sampled in-region
    """
    aperture: Tuple[Tuple[float, float], ...]
    iris_c: Tuple[float, float]
    iris_axes: Tuple[float, float]
    iris_angle: float
    iris_r: float
    colors: Dict[str, Tuple[int, int, int]]

    def to_dict(self) -> dict:
        return {
            "aperture": [list(p) for p in self.aperture],
            "iris_c": list(self.iris_c),
            "iris_axes": list(self.iris_axes),
            "iris_angle": self.iris_angle,
            "iris_r": self.iris_r,
            "colors": {k: list(v) for k, v in self.colors.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "ArtEye":
        return ArtEye(
            aperture=tuple(tuple(map(float, p)) for p in d.get("aperture", ())),
            iris_c=tuple(map(float, d.get("iris_c", (0.0, 0.0)))),
            iris_axes=tuple(map(float, d.get("iris_axes", (0.0, 0.0)))),
            iris_angle=float(d.get("iris_angle", 0.0)),
            iris_r=float(d.get("iris_r", 0.0)),
            colors={k: tuple(int(c) for c in v)
                    for k, v in d.get("colors", {}).items()},
        )


# ═══════════════════════════════════════════
# Pixel helpers (pure)
# ═══════════════════════════════════════════

def _require_cv2():
    try:
        import cv2
        return cv2
    except Exception as exc:                       # pragma: no cover
        raise EyeMeasureError(
            "measuring the artwork's eyes needs OpenCV (connected "
            "components, morphology and ellipse fitting). Install "
            "opencv-python-headless.") from exc


def _hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(V, S) planes of an RGB uint8-ranged float array."""
    cv2 = _require_cv2()
    h = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8),
                     cv2.COLOR_RGB2HSV).astype(np.float32)
    return h[..., 2], h[..., 1]


def _border_median(roi: np.ndarray, b: int = SKIN_BORDER) -> np.ndarray:
    """Median colour of the ROI's border ring — the local skin tone.

    The ROI is centred on the eye and spans 0.64·face_h, so its border
    is face, never eye. A local reference (not a global palette entry)
    is what makes the same thresholds work on a shaded cheek and a
    lit brow.
    """
    b = max(1, min(b, min(roi.shape[:2]) // 3))
    ring = np.concatenate([
        roi[:b].reshape(-1, 3), roi[-b:].reshape(-1, 3),
        roi[:, :b].reshape(-1, 3), roi[:, -b:].reshape(-1, 3)])
    return np.median(ring, axis=0)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Close interior holes: catchlights and highlights belong to the eye.

    A specular catchlight is bright AND saturated, so it fails the
    eye-like test and punches a hole in the mask. Left open, the
    renderer would be allowed to paint through it.
    """
    cv2 = _require_cv2()
    ff = mask.copy()
    pad = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1)
    return np.maximum(mask, (ff == 0).astype(np.uint8))


def _largest_near(mask: np.ndarray, seed: Tuple[float, float]) -> np.ndarray:
    """Component nearest `seed`, ignoring specks.

    Nearest-to-seed rather than largest-overall: on chintu the glasses
    frame is a bigger dark blob than his eye, and "largest" would
    select the frame.
    """
    cv2 = _require_cv2()
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    best, best_d = 0, float("inf")
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_COMPONENT:
            continue
        d = (cent[i][0] - seed[0]) ** 2 + (cent[i][1] - seed[1]) ** 2
        if d < best_d:
            best_d, best = d, i
    if best == 0:
        return np.zeros_like(mask)
    return (lab == best).astype(np.uint8)


def _trimmed_median(px: np.ndarray, lo: float, hi: float
                    ) -> Optional[Tuple[int, int, int]]:
    """Median of the pixels between the `lo` and `hi` luma quantiles.

    Trimming removes the antialiased boundary pixels that blend a
    region into its neighbour — the pixels that made "lip" read as
    shirt and "iris" read as lash.
    """
    if len(px) < SAMPLE_MIN_PX:
        return None
    lum = px @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    order = np.argsort(lum)
    a = int(len(order) * max(0.0, lo))
    b = int(len(order) * min(1.0, 1.0 - hi))
    keep = px[order[a:max(b, a + 1)]]
    if len(keep) == 0:
        return None
    med = np.median(keep, axis=0)
    return (int(round(med[0])), int(round(med[1])), int(round(med[2])))


def _cheb(a: Sequence[float], b: Sequence[float]) -> float:
    """Chebyshev RGB distance — the metric the QC colour masks use, so
    "separated here" means "separable there"."""
    return float(max(abs(int(x) - int(y)) for x, y in zip(a, b, strict=True)))


def _darken_rgb(c: Sequence[float], k: float) -> Tuple[int, int, int]:
    return tuple(int(round(max(0.0, min(255.0, float(v) * k))))
                 for v in c)  # type: ignore[return-value]


def _push_from(color: Sequence[float], ref: Sequence[float],
               min_sep: float) -> Tuple[int, int, int]:
    """Darken `color` until it is at least `min_sep` from `ref`.

    Only does anything when the artwork genuinely gives the lid line and
    the iris the same value. Some separation is then required, not
    optional: the renderer paints a closed lid with this colour and the
    blink gate verifies closure by looking for surviving iris pixels, so
    a lid that reads as an iris makes closure unverifiable — and a blink
    that never registers is exactly the broken-looking eye being fixed.
    """
    r, g, b = (float(v) for v in color)
    for _ in range(SEP_MAX_STEPS):
        if _cheb((r, g, b), ref) >= min_sep:
            break
        r, g, b = r * 0.86, g * 0.86, b * 0.86
        if max(r, g, b) < 1.0:
            break
    return _darken_rgb((r, g, b), 1.0)


def _contour_poly(mask: np.ndarray, offset: Tuple[int, int],
                  max_pts: int = 72) -> Tuple[Tuple[float, float], ...]:
    """External contour of `mask` as a polygon in plate space."""
    cv2 = _require_cv2()
    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cont:
        return ()
    c = max(cont, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(c) > max_pts:                     # uniform decimation, keeps shape
        idx = np.linspace(0, len(c) - 1, max_pts).round().astype(int)
        c = c[idx]
    return tuple((float(x + offset[0]), float(y + offset[1])) for x, y in c)


# ═══════════════════════════════════════════
# The measurement
# ═══════════════════════════════════════════

def measure_eye(art: np.ndarray, seed: Tuple[float, float],
                face_h: float, label: str = "eye") -> ArtEye:
    """Measure one drawn eye on the head crop BEFORE inpainting.

    `art`  : RGB(A) head crop, plate space — features must still be painted.
    `seed` : approximate eye centre (the MediaPipe lid-ring centroid).
    """
    cv2 = _require_cv2()
    rgb = np.asarray(art)[..., :3].astype(np.float32)
    H, W = rgb.shape[:2]

    span = max(12, int(ROI_SPAN * face_h))
    x0 = max(0, int(round(seed[0])) - span)
    y0 = max(0, int(round(seed[1])) - span)
    x1 = min(W, int(round(seed[0])) + span)
    y1 = min(H, int(round(seed[1])) + span)
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise EyeMeasureError(
            f"{label}: eye ROI degenerate ({x1 - x0}x{y1 - y0}px) — the "
            f"landmark seed {seed} lies outside the head plate.")

    roi = rgb[y0:y1, x0:x1]
    V, S = _hsv(roi)
    skin = _border_median(roi)
    not_skin = np.abs(roi - skin).max(axis=2) > SKIN_DELTA
    sclera_like = (V > SCLERA_V_MIN) & (S < SCLERA_S_MAX)
    dark_like = V < DARK_V_MAX

    # ── 1 · aperture: the drawn eye opening ──
    eye = (not_skin & (sclera_like | dark_like)).astype(np.uint8)
    eye = cv2.morphologyEx(eye, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    eye = cv2.morphologyEx(eye, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ap = _largest_near(eye, (seed[0] - x0, seed[1] - y0))
    if ap.sum() == 0:
        raise EyeMeasureError(
            f"{label}: found no eye-like region near {seed}. The art may "
            f"not show an open eye there, or the eye is drawn in skin "
            f"tones (no sclera and no dark iris) — such art cannot drive "
            f"gaze or blink and must be re-exported.")
    ap = cv2.morphologyEx(ap, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    ap = _fill_holes(ap)

    roi_area = float(ap.shape[0] * ap.shape[1])
    frac = ap.sum() / roi_area
    if frac < MIN_APERTURE:
        raise EyeMeasureError(
            f"{label}: measured aperture is {frac * 100:.2f}% of the search "
            f"box — too small to be the drawn eye (expected ≳"
            f"{MIN_APERTURE * 100:.0f}%). Segmentation failed.")
    if frac > MAX_APERTURE:
        raise EyeMeasureError(
            f"{label}: measured aperture is {frac * 100:.0f}% of the search "
            f"box — the mask leaked out of the eye into the face "
            f"(expected ≲{MAX_APERTURE * 100:.0f}%).")

    # ── 2 · iris: aperture minus eye white, ellipse-fitted ──
    inner = cv2.erode(ap, np.ones((3, 3), np.uint8))
    iris_m = ((inner > 0) & (~sclera_like)).astype(np.uint8)
    iris_m = cv2.morphologyEx(iris_m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    iris_m = cv2.morphologyEx(iris_m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    iris_m = _largest_near(iris_m, (seed[0] - x0, seed[1] - y0))
    if iris_m.sum() < MIN_COMPONENT:
        raise EyeMeasureError(
            f"{label}: no iris inside the measured aperture — the whole "
            f"opening reads as eye white. Gaze would have nothing to move.")
    iris_m = _fill_holes(iris_m)

    cont, _ = cv2.findContours(iris_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cpts = max(cont, key=cv2.contourArea)
    if len(cpts) >= 5:
        (ecx, ecy), (ew, eh), ang = cv2.fitEllipse(cpts)
        a, b_ = ew / 2.0, eh / 2.0
    else:                       # degenerate contour: fall back to the disc
        ys, xs = np.nonzero(iris_m)
        ecx, ecy = float(xs.mean()), float(ys.mean())
        a = b_ = float(np.sqrt(iris_m.sum() / np.pi))
        ang = 0.0
    r_eq = float(np.sqrt(max(a * b_, 1e-6)))
    if not (IRIS_MIN_RATIO * face_h <= r_eq <= IRIS_MAX_RATIO * face_h):
        raise EyeMeasureError(
            f"{label}: measured iris radius {r_eq:.1f}px is "
            f"{r_eq / face_h:.3f}×face_height, outside the plausible "
            f"[{IRIS_MIN_RATIO}, {IRIS_MAX_RATIO}] band. The fit latched "
            f"onto a lash, a glasses frame or a shadow, not the eyeball.")

    # ── 3 · colours, each from inside its OWN measured region ──
    sclera_px = roi[(ap > 0) & sclera_like]
    iris_px = roi[iris_m > 0]
    iris_lum = (iris_px @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
                ) if len(iris_px) else np.zeros(0, dtype=np.float32)

    colors: Dict[str, Tuple[int, int, int]] = {}
    sc = _trimmed_median(sclera_px, 0.10, 0.10)
    colors["sclera"] = sc if sc is not None else (246, 245, 242)

    # Iris body: the CHROMATIC pixels of the eyeball.
    #
    # A luma percentile cannot find this. On this artwork the pupil is a
    # large black disc — ~30% of the fitted ellipse, with 25% of iris
    # pixels at luma ≤ 5 — so any "drop the darkest N%" rule with N below
    # the pupil's share returns the pupil, and `iris` came out (37,9,5):
    # visually black, indistinguishable from every ink outline in the
    # frame, which made the QC iris mask match 17–25% of the whole body
    # and left blink-closure unmeasurable.
    #
    # Colour space separates the three parts cleanly instead, because
    # they differ in KIND, not merely in brightness:
    #     pupil  → dark   AND achromatic (saturation ≈ 0)
    #     sclera → bright AND achromatic
    #     iris   → the coloured ring in between
    # Selecting on saturation therefore excludes pupil, sclera, catchlight
    # and the black lash by construction. Measured across all four eyes
    # this yields a stable warm brown (94–113, 40–52, 9–28) from
    # 1000–1400 px per eye — the colour a human calls "her eye colour".
    if len(iris_px) >= SAMPLE_MIN_PX:
        iris_sat = iris_px.max(axis=1) - iris_px.min(axis=1)
        chromatic = ((iris_sat >= IRIS_SAT_MIN)
                     & (iris_lum >= IRIS_LUM_MIN)
                     & (iris_lum <= IRIS_LUM_MAX))
        body = iris_px[chromatic]
        ib = (_trimmed_median(body, 0.10, 0.10) if len(body) >= SAMPLE_MIN_PX
              else None)
        if ib is None:
            # A genuinely greyscale eye (monochrome art). Fall back to the
            # mid-luma band, which is the best available answer, rather
            # than failing a bake over a stylistic choice.
            lo_q, hi_q = np.percentile(iris_lum, [55.0, 92.0])
            band = iris_px[(iris_lum >= lo_q) & (iris_lum <= hi_q)]
            ib = _trimmed_median(band, 0.10, 0.10)
        colors["iris"] = ib if ib is not None else (92, 62, 44)

        # Pupil: the achromatic dark core, by the same separation.
        pupil_sel = iris_px[(iris_sat < IRIS_SAT_MIN)
                            & (iris_lum <= max(IRIS_LUM_MIN, 60.0))]
        pp = _trimmed_median(pupil_sel, 0.0, 0.25) if len(pupil_sel) else None
        if pp is None:
            pp = _trimmed_median(
                iris_px[iris_lum <= np.percentile(iris_lum, PUPIL_PCTL)],
                0.0, 0.25)
        colors["pupil"] = pp if pp is not None else (22, 16, 14)
    else:
        colors["iris"], colors["pupil"] = (92, 62, 44), (22, 16, 14)

    # Lash: the painted INK line on the aperture's rim.
    #
    # "Darkest 20% of a ring around the aperture" is not enough. That ring
    # is mostly eyelid SKIN, and where the segmented aperture already
    # excludes the lash (or a glasses frame sits nearby, as on chintu) its
    # dark tail is merely shaded skin — measured (123,59,33) and
    # (127,60,16), a warm mid-brown only Δ=34 from the iris. The renderer
    # paints a closed lid with `skin` + a `lash` line and the blink gate
    # then asks "is any iris colour still visible", so a lash that reads
    # as an iris makes a fully closed eye indistinguishable from an open
    # one — the exact failure being chased.
    #
    # Ink is separable the same way the iris was: it is dark AND
    # (near-)achromatic, whereas shaded skin stays warm and saturated.
    # Search a wider band on both sides of the rim so the line is found
    # whether it falls just inside or just outside the aperture.
    band = (cv2.dilate(ap, np.ones((9, 9), np.uint8))
            - cv2.erode(ap, np.ones((3, 3), np.uint8))) > 0
    band_px = roi[band]
    ls = None
    if len(band_px) >= SAMPLE_MIN_PX:
        b_lum = band_px @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        b_sat = band_px.max(axis=1) - band_px.min(axis=1)
        ink = band_px[(b_lum <= LASH_LUM_MAX) & (b_sat <= LASH_SAT_MAX)]
        if len(ink) >= SAMPLE_MIN_PX:
            ls = _trimmed_median(ink, 0.0, 0.10)
    if ls is None:
        # No ink line in this art (soft-shaded eye). Derive the lid line
        # from the pupil, which IS this artwork's darkest ink, so the
        # separation the blink gate needs holds by construction.
        ls = _darken_rgb(colors["pupil"], 1.0)
    colors["lash"] = _push_from(ls, colors["iris"], LASH_IRIS_SEP)

    return ArtEye(
        aperture=_contour_poly(ap, (x0, y0)),
        iris_c=(float(ecx + x0), float(ecy + y0)),
        iris_axes=(float(a), float(b_)),
        iris_angle=float(ang),
        iris_r=r_eq,
        colors=colors,
    )


def _sprite_axes(eye: "ArtEye", feather: float = SPRITE_FEATHER
                 ) -> Tuple[float, float]:
    """The eyeball sprite's alpha semi-axes. `socket_backdrop` inpaints
    exactly this ellipse, so the sprite covers the hole with nothing left
    over — see SPRITE_FEATHER."""
    ax, ay = eye.iris_axes
    return (max(ax, 2.0) + feather * 2.0, max(ay, 2.0) + feather * 2.0)


def eyeball_sprite(art: np.ndarray, eye: "ArtEye",
                   feather: float = SPRITE_FEATHER
                   ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Cut the EYEBALL THE ARTIST DREW out of the plate as an RGBA sprite.

    This is what makes the eye look hand-drawn instead of vector. The
    renderer used to synthesize the eye every frame — flat sclera fill,
    flat iris oval, limbal ring, catchlight — which discards the artist's
    soft shading, the lash overlap and the highlight, and reads as
    harder, darker and duller than the surrounding art. Worse, filling
    the whole aperture with `sclera` paints eye-white the artist never
    drew: on this art the eye is almost entirely iris, so the fill showed
    as bright crescents either side of a too-small iris.

    Cutting the drawn eyeball out and MOVING it keeps every one of those
    painted details, so a resting frame is pixel-identical to the art and
    gaze becomes a translation of real artwork.

    Returns (RGBA sprite, (x0, y0) origin in plate space). The alpha is
    the iris ellipse, feathered by `feather` px so the sprite's rim
    blends into the sclera behind it rather than showing a cut edge.
    """
    cv2 = _require_cv2()
    h, w = art.shape[:2]
    ax, ay = _sprite_axes(eye, feather)
    cx, cy = eye.iris_c
    pad = int(math.ceil(max(ax, ay))) + 2
    x0 = max(0, int(math.floor(cx)) - pad)
    y0 = max(0, int(math.floor(cy)) - pad)
    x1 = min(w, int(math.ceil(cx)) + pad)
    y1 = min(h, int(math.ceil(cy)) + pad)
    if x1 - x0 < 3 or y1 - y0 < 3:
        raise EyeMeasureError("eyeball sprite box collapsed")

    crop = art[y0:y1, x0:x1, :3].astype(np.uint8)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.ellipse(mask, (int(round(cx - x0)), int(round(cy - y0))),
                (int(round(ax)), int(round(ay))),
                float(eye.iris_angle), 0, 360, 255, -1)
    if feather > 0:
        k = max(3, int(feather * 2) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), feather)
    return np.dstack([crop, mask]), (x0, y0)


def socket_backdrop(art: np.ndarray, eye: "ArtEye"
                    ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """The eye with its EYEBALL PAINTED OUT — what gaze uncovers.

    When the eyeball sprite translates, something has to be behind it.
    Filling that with the `sclera` palette colour is wrong twice over: it
    is flat where the artist painted a gradient, and the sample can be a
    shadow (chintu measured a grey 156,147,146 on one eye and a white
    251,241,236 on the other, so his eyes did not match each other).

    Inpainting the iris out of the artwork instead gives a backdrop with
    the artist's own shading and per-eye tone, and needs no colour
    decision at all.

    The hole is EXACTLY the eyeball sprite's alpha ellipse (`_sprite_axes`),
    which is the only size that is right in both directions:

      • larger (it used to be 1.30×) and the sprite cannot cover it, so
        every resting frame shows a ring of inpainted smear around the
        iris — a pale halo the artist never painted;
      • smaller and a gaze shift drags the sprite off a surviving crescent
        of the ORIGINAL iris, leaving two irises in one eye.

    Because the sprite's footprint and the hole are the same ellipse, the
    region the sprite can ever vacate is exactly the region that was
    inpainted, and a resting frame reconstructs the artwork.

    Returns (RGBA patch, (x0, y0) origin in plate space); alpha is the
    aperture, so the backdrop can never paint onto the cheek.
    """
    cv2 = _require_cv2()
    h, w = art.shape[:2]
    ap = np.asarray(eye.aperture, dtype=np.float64)
    if len(ap) < 3:
        raise EyeMeasureError("socket backdrop needs a measured aperture")
    pad = 3
    x0 = max(0, int(math.floor(ap[:, 0].min())) - pad)
    y0 = max(0, int(math.floor(ap[:, 1].min())) - pad)
    x1 = min(w, int(math.ceil(ap[:, 0].max())) + pad)
    y1 = min(h, int(math.ceil(ap[:, 1].max())) + pad)
    if x1 - x0 < 3 or y1 - y0 < 3:
        raise EyeMeasureError("socket backdrop box collapsed")

    crop = np.ascontiguousarray(art[y0:y1, x0:x1, :3].astype(np.uint8))
    hole = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    ax, ay = _sprite_axes(eye)
    cv2.ellipse(hole, (int(round(eye.iris_c[0] - x0)),
                       int(round(eye.iris_c[1] - y0))),
                (max(2, int(round(ax))), max(2, int(round(ay)))),
                float(eye.iris_angle), 0, 360, 255, -1)
    filled = cv2.inpaint(crop, hole,
                         max(3, int(0.35 * max(ax, ay))), cv2.INPAINT_NS)

    alpha = np.zeros_like(hole)
    cv2.fillPoly(alpha, [np.round(ap - [x0, y0]).astype(np.int32)], 255)
    return np.dstack([filled, alpha]), (x0, y0)


def lid_sprite(art: np.ndarray, eye: "ArtEye"
               ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """The artist's own upper eyelid, as a strip that slides down to blink.

    A blink used to be a flat ellipse of `skin` dropped over the eye,
    which reads as a hole punched in the face rather than a closed lid —
    no crease, no lash, no shading, and the palette's skin tone is a
    single sample of a face that is painted with a gradient.

    The lid the artist DREW is the band of plate between the aperture's
    top and the eyebrow. Sliding those pixels down covers the eye with
    real skin that already carries the lid crease and lash edge.

    WHERE THE BAND ENDS IS MEASURED, NOT ASSUMED. The previous bake took a
    fixed 0.42 × eye-height above the aperture, on the reasoning that a
    full eye-height "would drag the eyebrow down into the eye". On chibi
    art the brow is closer than 0.42 eye-heights, so 0.42 dragged it in
    anyway: chintu's baked strips had mean RGB (120,76,60) and (69,38,29)
    — eyebrow ink, not skin — and a closed eye filled with a dark smear.
    Any fixed fraction has this failure mode on some face. So this walks
    up from the aperture and STOPS AT THE BROW, which cannot be wrong
    about the brow's position because the brow is what halts it:

      1. the LASH zone — ink rows contiguous with the aperture's top edge,
         which are the lid line the artist drew and belong in the strip,
         at its bottom, as the lid's leading edge;
      2. the SKIN rows above them, taken while rows stay non-ink;
      3. the BROW — the next ink row after that skin — which ends the band.

    A face whose brow leaves less than `LID_SKIN_MIN` of an eye-height of
    skin gives no lid to sample, and this raises rather than shipping a
    strip of eyebrow (Law 1).

    Returns (RGBA strip, (x0, y0) origin in plate space); y0 is the top of
    the sampled rows and the strip's BOTTOM row is the aperture's top.
    """
    h, w = art.shape[:2]
    ap = np.asarray(eye.aperture, dtype=np.float64)
    if len(ap) < 3:
        raise EyeMeasureError("lid sprite needs a measured aperture")
    x0 = max(0, int(math.floor(ap[:, 0].min())) - 3)
    x1 = min(w, int(math.ceil(ap[:, 0].max())) + 3)
    top = int(math.floor(ap[:, 1].min()))
    bot = int(math.ceil(ap[:, 1].max()))
    if x1 - x0 < 3:
        raise EyeMeasureError("lid sprite box collapsed")
    eye_h = max(3, bot - top)

    # Per-row ink share over the eye's own width, in the same colour-space
    # separation the lash colour uses: ink is dark AND near-achromatic,
    # while shaded eyelid skin stays warm and saturated.
    col = art[:top, x0:x1, :3].astype(np.float32)
    if col.shape[0] == 0:
        raise EyeMeasureError(f"{eye.name}: aperture touches the plate top")
    lum = col @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sat = col.max(axis=2) - col.min(axis=2)
    ink_row = (((lum <= LID_INK_LUM_MAX) & (sat <= LID_INK_SAT_MAX))
               .mean(axis=1) >= LID_LASH_FRAC)

    r = top - 1                                  # walk upward from the eye
    lash_cap = int(round(eye_h * LID_LASH_MAX))
    lash_n = 0
    while r >= 0 and ink_row[r] and lash_n < lash_cap:
        r -= 1
        lash_n += 1
    skin_n = 0
    while r >= 0 and not ink_row[r]:             # stops on the brow
        r -= 1
        skin_n += 1

    if skin_n < max(2, int(round(eye_h * LID_SKIN_MIN))):
        raise EyeMeasureError(
            f"{eye.name}: only {skin_n}px of brow-free eyelid skin above a "
            f"{eye_h}px eye — no lid to sample")

    src_y0 = top - (lash_n + skin_n)
    strip = np.zeros((lash_n + skin_n, x1 - x0, 4), dtype=np.uint8)
    strip[..., :3] = art[src_y0:top, x0:x1, :3]
    strip[..., 3] = 255
    return strip, (x0, src_y0)


def measure_pair(art: np.ndarray, seed_l: Tuple[float, float],
                 seed_r: Tuple[float, float], face_h: float
                 ) -> Tuple[ArtEye, ArtEye]:
    """Measure both eyes and cross-check them for symmetry.

    A face has two eyes of near-equal size. If the two measurements
    disagree by more than 45%, one of them latched onto something that
    is not an eye, and we cannot tell which — so the bake fails rather
    than shipping one correct eye and one wrong one.
    """
    left = measure_eye(art, seed_l, face_h, "eye_l")
    right = measure_eye(art, seed_r, face_h, "eye_r")
    big, small = max(left.iris_r, right.iris_r), min(left.iris_r, right.iris_r)
    if small > 0 and big / small > 1.45:
        raise EyeMeasureError(
            f"the two eyes measured {left.iris_r:.1f}px and "
            f"{right.iris_r:.1f}px ({big / small:.2f}× apart). Eyes on one "
            f"face are near-equal, so one fit is wrong — refusing to bake "
            f"a rig with one good eye and one bad one.")
    return left, right


__all__ = ["ArtEye", "EyeMeasureError", "measure_eye", "measure_pair"]
