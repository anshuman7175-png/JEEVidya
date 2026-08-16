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

    # Lash: the darkest ring just OUTSIDE the aperture (the painted lash
    # line sits on the aperture's rim). Sampled outside so it cannot be
    # confused with the pupil.
    rim = (cv2.dilate(ap, np.ones((5, 5), np.uint8)) - ap) > 0
    rim_px = roi[rim]
    ls = _trimmed_median(rim_px, 0.0, 0.80)
    colors["lash"] = ls if ls is not None else (34, 24, 22)

    return ArtEye(
        aperture=_contour_poly(ap, (x0, y0)),
        iris_c=(float(ecx + x0), float(ecy + y0)),
        iris_axes=(float(a), float(b_)),
        iris_angle=float(ang),
        iris_r=r_eq,
        colors=colors,
    )


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
