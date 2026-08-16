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
# Containment invariant (§2b): the fitted eyeball must sit inside the eye
# opening. Expressed as a fraction of the fitted ellipse's own area so it
# holds at any export resolution. This is a rasterization tolerance, not a
# licence to overflow — a correct fit measures 0.0–0.3% here, while the
# lash-contaminated fit this guards against measured 18–34%.
IRIS_SPILL_MAX = 0.01

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

# ── The aperture is a CLIP, so its rim must be outside the drawn eye ──
#
# Segmentation puts the boundary partway through the antialiased lash,
# which leaves a 1–3 px ring of the ORIGINAL painted eye outside the
# clip. Every eye pixel is masked to the aperture, so that ring survives
# whatever is drawn: filling the eye with lid skin left the artwork's own
# lash and sclera showing around the fill as a hard, jagged outline — the
# "cracked eggshell" on gudiya's blink.
#
# Two properties fix it, and both are needed:
#   grow   : the clip must SWALLOW the antialiased rim, so there is no
#            original-eye pixel left outside it to show through.
#   smooth : the raw contour is a pixel staircase, and decimating it by
#            index keeps the steps. A staircase clip reads as a ragged
#            edge at any zoom, so the boundary is low-passed into the
#            smooth curve the artist actually drew.
APERTURE_GROW = 0.009    # ×face_h, dilation of the clip past the lash
APERTURE_SMOOTH = 5      # circular moving-average window on the contour

# ── The eye-like test has a hole in tone space, so the rim is repaired ──
#
# "Eye-like" is `not skin AND (bright-desaturated OR dark)`. Between those
# two clauses lies a MID-LUMA BAND — HSV V in (DARK_V_MAX, SCLERA_V_MIN),
# i.e. 110…150 — that neither claims, and on this art the iris passes
# straight through it. Measured in the excluded pixels: chintu's left iris
# shades to (125,55,29) at its foot, his right eye's lens-shadowed white to
# (134,126,128), gudiya's lower iris to (144,62,12). Every one of them is
# ≥99% "not skin" and 0% ink — unambiguously eye, and unambiguously dropped.
#
# The consequence was a BITE out of the aperture's lower rim, 5–9 px deep,
# and it broke the chain: the iris ellipse fitted to the drawn eyeball then
# spilled 7% of its area through that bite, the containment invariant
# failed, and the whole v3 bake was refused for both characters (rigs fell
# back to v2, which the render path rightly rejects).
#
# Widening the tone test is not the fix — it is what leaks. `not_skin`
# compares against the ROI border median, and that ring holds hair on
# chintu (median 112,54,46), so admitting the mid band globally made the
# aperture swallow the entire ROI (measured: frac 0.83 → 1.00).
#
# The repair is GEOMETRIC and local instead. A drawn eye opening is a
# smooth, near-convex almond; a shallow bite into its rim is a threshold
# artifact, while a DEEP concavity is real shape (or a leak) and must be
# left alone. So the aperture's convex-hull deficiencies are filled only
# where their own maximum depth is under this bound — measured depths are
# 1–9 px against a 12–15 px bound, and every filled component is non-ink,
# ≥99%-not-skin eye pixels.
APERTURE_CONCAVE_MAX = 0.05   # ×face_h, deepest rim bite treated as artifact

# ── Gaze travel is bounded by the artwork, not by a fixed fraction ─�������������
#
# Gaze used to translate the eyeball by ±0.55·iris_r (±18 px on chintu),
# but this art draws an iris that nearly fills the opening — the real
# sclera margin is a few pixels. The consequences were both visible:
# the iris rode onto the lash, and `socket_backdrop` had to inpaint the
# WHOLE iris ellipse to hide the artwork's own iris behind it, which on
# an eye that is almost all iris has no clean pixels to reconstruct from
# and produced the radial brown smear seen behind every moving eye.
#
# Measuring the margin instead makes the excursion exactly what the
# drawing affords, so the inpaint shrinks to the thin crescent the
# eyeball can actually uncover — a region completely surrounded by real
# sclera, which is the case inpainting handles well.
GAZE_MARGIN_SAFETY = 1.0     # px kept between the iris rim and the lash
GAZE_MAX_FRAC = 0.45         # ×iris_r, hard cap on measured travel

# ── The lid strip is SAMPLED ART, so what it samples must be validated ──
#
# The lid was "the rows immediately above the aperture, 0.42 of an eye
# height". On this art that band reaches the eyebrow on gudiya and the
# GLASSES RIM on chintu (both visible in the baked strips), and the
# renderer then slid that ink down over the eye: a blink drew a dark bar
# across the eyeball, and a stretched copy of it smeared over the socket.
#
# So the band is MEASURED: rows are accepted upward from the aperture only
# while they are still eyelid skin. The test that matters is which SKIN the
# row is compared against, and getting that wrong is what produced the
# flat-disc blinks:
#
#   A reference taken over a block as tall as the eye reaches past the lid
#   — the forehead on chintu, the hair fringe on gudiya. It then reads as
#   bright face skin (252,190,151) on one eye and as fringe brown
#   (100,50,35) on the other, so the two eyes disagreed about their own
#   face, EVERY genuine lid row (a shaded 160,90,63 inside the glasses
#   lens) was rejected as "not skin", all four strips collapsed to the
#   4px flat fallback — and a blink became a flat disc of whichever tone
#   the contaminated block happened to hold.
#
# The lid is therefore judged against ITSELF: the reference is the first
# clean rows found above the lash, and it then tracks the accepted rows as
# an exponential mean. That is what lets a lid darken smoothly into its
# crease (a gradient the artist painted) while still ending the walk at the
# STEP in tone that a brow, a glasses rim or a hairline always is.
#
# But a tracking reference alone is not enough, and gudiya's right eye shows
# why. Its lid is a real 30-row gradient (240,158,104 → 232,145,90) and then
# STEPS into the fringe at (198,110,60), (178,90,49), (172,85,44). Measured
# row-to-row those steps are 38, 45, 35 — each below LID_SKIN_DELTA, so the
# EMA followed them, and the walk ratcheted 7 rows up into the hair it was
# written to stop at. A reference that chases the edge cannot detect it.
#
# So the band is bounded twice: per row against the running reference, which
# catches a sharp edge, AND cumulatively against the SEED, which catches a
# slow ratchet no single step betrays. The two thresholds are far apart in
# the art — the genuine gradient drifts 14 from its seed over 30 rows, while
# the fringe is 42 away the moment it starts — so the cumulative bound
# separates lid from hair cleanly instead of by a hair's breadth.
LID_BAND_FRAC = 0.55      # ×aperture height, cap on the sampled band
LID_LASH_SKIP_FRAC = 0.30  # ×aperture height, how far up the lash may reach
LID_BAND_MIN = 4          # px, a band thinner than this is not a lid
LID_ROW_INK_MAX = 0.34    # a row with more ink than this is lash/brow/frame
LID_ROW_DIRT_MAX = 0.10   # ≤10% of a row may be ink or non-skin
LID_SKIN_DELTA = 46.0     # max|ΔRGB| a row's median may sit from the running
                          # lid reference before the walk calls it a new
                          # feature rather than more of the same eyelid
LID_SEED_DRIFT_MAX = 30.0  # max|ΔRGB| the band may drift from its seed in
                           # total, so no sequence of small steps can walk
                           # the strip into a brow, fringe or glasses rim
LID_REF_ROWS = 3          # rows above the lash that seed the reference
LID_REF_EMA = 0.35        # weight of a newly accepted row in the reference


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
    gaze_range : (dx, dy) px the eyeball may travel before its rim
                 reaches the drawn opening. This is the artwork's own
                 sclera margin, so a gaze of ±1 is the largest look the
                 drawing can hold rather than a guessed fraction.
    """
    aperture: Tuple[Tuple[float, float], ...]
    iris_c: Tuple[float, float]
    iris_axes: Tuple[float, float]
    iris_angle: float
    iris_r: float
    colors: Dict[str, Tuple[int, int, int]]
    gaze_range: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "aperture": [list(p) for p in self.aperture],
            "iris_c": list(self.iris_c),
            "iris_axes": list(self.iris_axes),
            "iris_angle": self.iris_angle,
            "iris_r": self.iris_r,
            "colors": {k: list(v) for k, v in self.colors.items()},
            "gaze_range": list(self.gaze_range),
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
                  max_pts: int = 72, smooth: int = 0
                  ) -> Tuple[Tuple[float, float], ...]:
    """External contour of `mask` as a polygon in plate space.

    `smooth` low-passes the boundary with a CIRCULAR moving average before
    decimation. Contours run along pixel edges, so the raw polygon is a
    staircase; decimating it by index preserves those steps and the clip
    reads as a ragged edge. The average must wrap around the seam or the
    join between the last and first vertex stays a visible corner.
    """
    cv2 = _require_cv2()
    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cont:
        return ()
    c = max(cont, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if smooth > 1 and len(c) >= smooth * 2:
        k = int(smooth) | 1                  # odd ⇒ symmetric, no drift
        pad = k // 2
        ring = np.concatenate([c[-pad:], c, c[:pad]], axis=0)
        ker = np.ones(k, dtype=np.float64) / k
        c = np.stack([np.convolve(ring[:, 0], ker, mode="valid"),
                      np.convolve(ring[:, 1], ker, mode="valid")], axis=1)
    if len(c) > max_pts:                     # uniform decimation, keeps shape
        idx = np.linspace(0, len(c) - 1, max_pts).round().astype(int)
        c = c[idx]
    return tuple((float(x + offset[0]), float(y + offset[1])) for x, y in c)


def _ellipse_mask(shape: Tuple[int, int], c: Tuple[float, float],
                  axes: Tuple[float, float], angle: float,
                  grow: float = 0.0) -> np.ndarray:
    """Filled ellipse mask, drawn with sub-pixel centre rounding."""
    cv2 = _require_cv2()
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, (int(round(c[0])), int(round(c[1]))),
                (max(1, int(round(axes[0] + grow))),
                 max(1, int(round(axes[1] + grow))),
                 ), angle, 0, 360, 1, -1)
    return m


def _gaze_margin(ap: np.ndarray, iris_c: Tuple[float, float],
                 iris_axes: Tuple[float, float], iris_angle: float,
                 cap: float) -> Tuple[float, float]:
    """Largest (dx, dy) the eyeball may travel and stay inside the opening.

    Translating the iris ellipse and testing containment answers this
    directly from the art, and does so for the real, non-elliptical
    aperture — a fixed fraction of `iris_r` cannot, because how much
    sclera a drawing affords is a property of the drawing. Both signs are
    tested and the smaller kept, so gaze stays symmetric around rest.
    """
    safety = GAZE_MARGIN_SAFETY
    axes = (iris_axes[0] + safety, iris_axes[1] + safety)
    inside = ap > 0

    def travel(ux: float, uy: float) -> float:
        best = 0.0
        d = 0.5
        while d <= cap:
            m = _ellipse_mask(ap.shape,
                              (iris_c[0] + ux * d, iris_c[1] + uy * d),
                              axes, iris_angle)
            if np.any((m > 0) & ~inside):
                break
            best = d
            d += 0.5
        return best

    dx = min(travel(-1.0, 0.0), travel(1.0, 0.0))
    dy = min(travel(0.0, -1.0), travel(0.0, 1.0))
    return float(dx), float(dy)


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

    # ── 2 · iris: the CHROMATIC eyeball, ellipse-fitted ──
    #
    # `ap` deliberately INCLUDES the painted lash (see "What is measured"
    # above), so "aperture minus eye white" is iris + pupil + LASH. The
    # lash is a dark arc that touches the eyeball's top rim, so it survives
    # `_largest_near` as ONE component and the ellipse is then fitted to
    # the UNION of eyeball and lash. Measured on this art that returned an
    # iris TALLER than the opening it has to sit inside — chintu L 57×79 px
    # in a 92×73 px aperture — which is geometrically impossible, and it
    # starved every consumer downstream: `_gaze_margin` found the iris out
    # of bounds at its very first probe and returned 0.0 travel, and the
    # lid band was measured against an opening its own "iris" overflowed.
    #
    # Ink and iris differ in KIND, not in brightness — the separation this
    # module already trusts for the iris body colour (IRIS_SAT_MIN et al)
    # and for the lash colour (`_ink`). So fit to the CHROMATIC pixels of
    # the opening: that drops the achromatic lash AND the achromatic pupil,
    # and the CONVEX HULL of what remains is the eyeball's outer boundary —
    # the pupil is interior to it, while a lash overlapping the eyeball's
    # top is simply absent instead of dragging the fit upward. Hulling
    # rather than contour-fitting also stops a ring broken by the pupil or
    # the catchlight from fitting a lens shape.
    inner = cv2.erode(ap, np.ones((3, 3), np.uint8))
    roi_sat = roi.max(axis=2) - roi.min(axis=2)
    roi_lum = _luma(roi)
    chroma = ((roi_sat >= IRIS_SAT_MIN)
              & (roi_lum >= IRIS_LUM_MIN)
              & (roi_lum <= IRIS_LUM_MAX))
    iris_m = ((inner > 0) & (~sclera_like) & chroma & (~_ink(roi))
              ).astype(np.uint8)
    iris_m = cv2.morphologyEx(iris_m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    iris_m = cv2.morphologyEx(iris_m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if iris_m.sum() < MIN_COMPONENT:
        raise EyeMeasureError(
            f"{label}: no chromatic iris inside the measured aperture — the "
            f"opening reads as eye white and ink only. Gaze would have "
            f"nothing to move.")

    # Every surviving pixel is already inside the opening, so the hull over
    # ALL of them is the eyeball. No nearest-component pick is needed, and
    # a ring split in two by a large pupil cannot bias the fit.
    ys, xs = np.nonzero(iris_m)
    hull = cv2.convexHull(
        np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2))
    if len(hull) >= 5:
        (ecx, ecy), (ew, eh), ang = cv2.fitEllipse(hull)
        a, b_ = ew / 2.0, eh / 2.0
    else:                       # degenerate hull: fall back to the disc
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

    # ── 2b · hard invariant: the eyeball fits inside its own opening ──
    #
    # This is the check whose absence let an iris taller than its aperture
    # ship silently, and with it a rig whose gaze could not move and whose
    # blink painted a blob. A ratio test would not catch it (a wide-but-too
    # -tall iris passes any single ratio); containment is the real property,
    # so test containment. The tolerance covers only ellipse RASTERIZATION
    # against an irregular hand-drawn rim, not real overflow: the hull came
    # from pixels strictly inside `erode(ap)`, so a correct fit spills a
    # sub-percent sliver at worst.
    ell = _ellipse_mask(ap.shape, (ecx, ecy), (a, b_), float(ang))
    ell_area = max(1, int((ell > 0).sum()))
    spill = int(((ell > 0) & (ap == 0)).sum())
    if spill > IRIS_SPILL_MAX * ell_area:
        ah = int((ap > 0).any(axis=1).sum())
        aw = int((ap > 0).any(axis=0).sum())
        raise EyeMeasureError(
            f"{label}: fitted iris {2 * a:.0f}×{2 * b_:.0f}px spills "
            f"{spill}px ({spill / ell_area * 100:.1f}%) outside the "
            f"{aw}×{ah}px aperture it must sit inside. The eyeball cannot "
            f"be larger than the eye opening — the fit has latched onto "
            f"the lash, the brow or a glasses rim, and a rig baked from it "
            f"would have zero gaze travel and a stretched lid.")

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

    # ── 4 · the exported clip: grown past the lash, then smoothed ──
    #
    # Growth is what removes the cracked-eggshell ring. `ap` ends at the
    # midpoint of the antialiased lash, so any pixel of the ORIGINAL eye
    # outside it survives clipping and outlines whatever is drawn. Dilating
    # the clip swallows that rim; the lid then paints over a region strictly
    # larger than the painted eye, and nothing of the old eye remains to
    # show through.
    grow_px = max(1, int(round(APERTURE_GROW * face_h)))
    ap_clip = cv2.dilate(ap, np.ones((grow_px * 2 + 1,) * 2, np.uint8))
    ap_clip = _fill_holes(ap_clip)

    # Gaze is bounded by the UNGROWN opening: travel must respect where the
    # artist drew the lash, not the padded clip.
    gx, gy = _gaze_margin(ap, (ecx, ecy), (a, b_), float(ang),
                          cap=GAZE_MAX_FRAC * r_eq)

    return ArtEye(
        aperture=_contour_poly(ap_clip, (x0, y0), smooth=APERTURE_SMOOTH),
        iris_c=(float(ecx + x0), float(ecy + y0)),
        iris_axes=(float(a), float(b_)),
        iris_angle=float(ang),
        iris_r=r_eq,
        colors=colors,
        gaze_range=(gx, gy),
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

    # The INK on the rim must be part of the hole, not part of the source.
    #
    # An inpaint reconstructs the hole from its boundary, and this hole's
    # boundary is the drawn eye's rim: mostly the black lash line, with only
    # a sliver of sclera between it and the iris. Feeding that boundary in
    # produced the radial brown/black STARBURST visible in every baked
    # socket — and since gaze uncovers exactly the crescent at the rim, that
    # smear is the part that shows.
    #
    # Adding the ink to the hole leaves the sclera as the only source, so
    # the reconstruction is eye-white shaded the way the artist shaded it.
    # TELEA (not NS) because it propagates along the boundary normal, which
    # keeps that shading a gradient rather than streaks.
    band = cv2.dilate(hole, np.ones((5, 5), np.uint8)) > 0
    px = crop.astype(np.float32)
    lum, sat = _luma(px), px.max(axis=2) - px.min(axis=2)
    ink = (lum <= LASH_LUM_MAX) & (sat <= LASH_SAT_MAX) & band
    hole = np.maximum(hole, (ink * 255).astype(np.uint8))

    filled = cv2.inpaint(crop, hole,
                         max(3, int(0.18 * max(ax, ay))), cv2.INPAINT_TELEA)
    # A light blur INSIDE the hole only: the reconstruction is smooth art,
    # never a texture, and this removes the last inpaint filaments without
    # touching one pixel the artist drew.
    soft = cv2.GaussianBlur(filled, (0, 0), 1.2)
    m = (hole > 0)[..., None]
    filled = np.where(m, soft, filled)

    alpha = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(alpha, [np.round(ap - [x0, y0]).astype(np.int32)], 255)
    return np.dstack([filled, alpha]), (x0, y0)


def _luma(px: np.ndarray) -> np.ndarray:
    return px @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _ink(px: np.ndarray) -> np.ndarray:
    """Dark AND near-achromatic pixels: a lash, a brow, a glasses rim.

    Reference-free by design. It is the one lid test that can be trusted
    before any tone has been established, which is why the walk uses it to
    find the lash before it decides what the eyelid's skin even looks like.
    """
    lum = _luma(px)
    sat = px.max(axis=-1) - px.min(axis=-1)
    return (lum <= LASH_LUM_MAX) & (sat <= LASH_SAT_MAX)


def _row_tone(row: np.ndarray) -> Optional[np.ndarray]:
    """A row's own skin tone: the median of its NON-INK pixels.

    Excluding the ink is what makes the tone comparable from row to row —
    include it and a row holding the lash's tip reads as a tone step, so the
    walk stops one row into a lid it should have kept walking up.
    """
    keep = row[~_ink(row)]
    if len(keep) < max(3, row.shape[0] // 4):
        return None
    return np.median(keep, axis=0)


def _dirty(px: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Pixels that are not clean eyelid skin: ink, or far from `skin`.

    Ink (brow, lash, a glasses rim) is dark AND near-achromatic — the same
    separation the lash colour uses, so "ink" means one thing everywhere in
    this module. Everything else that is simply the wrong colour (a hair
    strand, a bright frame highlight, the sclera) is caught by distance
    from the local reference.
    """
    lum = _luma(px)
    sat = px.max(axis=-1) - px.min(axis=-1)
    ink = (lum <= LASH_LUM_MAX) & (sat <= LASH_SAT_MAX)
    far = np.abs(px - skin).max(axis=-1) > LID_SKIN_DELTA
    return ink | far


def lid_sprite(art: np.ndarray, eye: "ArtEye"
               ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """The artist's own upper eyelid, as a strip that slides down to blink.

    A blink used to be a flat ellipse of `skin` dropped over the eye, which
    reads as a hole punched in the face rather than a closed lid — no
    crease, no shading, and the palette's skin tone is a single sample of a
    face painted with a gradient.

    The lid the artist DREW is the band of plate immediately above the
    aperture, so this cuts that band out and the renderer slides it down at
    NATURAL SCALE (see `EyeRasterizer._lid_shear`). Two things about the
    band are measured rather than assumed, and both were defects:

      • HOW TALL it may be. A fixed fraction of the eye height reached the
        eyebrow on gudiya and the glasses rim on chintu, and the renderer
        duly slid that ink over the eyeball. So the walk climbs only while
        each row is still this eyelid — little ink, and its tone within
        `LID_SKIN_DELTA` of the rows already accepted — and never exceeds
        `LID_BAND_FRAC` of the opening.

      • WHAT IS IN IT. Even an accepted row can hold a few stray dark
        pixels — the tip of a lash, one strand of hair. Each is replaced by
        its own row's clean median, so the vertical skin gradient survives
        while no ink pixel remains to slide over the eye.

    The reference tone is SEEDED from the first clean rows above the lash
    and then tracks the band as it is accepted, so it is this eyelid's own
    skin under this eyelid's own shading. Anything wider — a block as tall
    as the eye — reaches the forehead or the hair fringe, and comparing a
    shaded in-lens eyelid against bright forehead skin is what rejected
    every real lid row and left all four blinks a flat disc.

    Returns (RGBA strip, (x0, y0) origin in plate space). The strip's
    BOTTOM row is the row directly above the aperture, which is the lid's
    leading edge at rest; the renderer aligns that row with the closing
    edge, so a resting frame leaves the artwork untouched.
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
    if top <= 0:
        raise EyeMeasureError(
            "the measured eye touches the top of the head plate, so there "
            "is no eyelid above it to sample — the head crop is too tight.")

    rgb = art[..., :3].astype(np.float32)
    ap_h = max(3, bot - top)
    cap = max(LID_BAND_MIN, int(round(ap_h * LID_BAND_FRAC)))

    # 1 · Walk past the lash.
    #
    # The lash is not a failure to stop for — it is the rim the aperture was
    # grown over, and it sits between the opening and the lid. It is found by
    # its INK alone, which needs no reference tone and so cannot be fooled by
    # one: a row is lash while it is mostly dark and achromatic.
    skip_cap = max(2, int(round(ap_h * LID_LASH_SKIP_FRAC)))
    lid_bot = top
    while lid_bot > 0 and (top - lid_bot) < skip_cap:
        row = rgb[lid_bot - 1, x0:x1]
        if float(np.mean(_ink(row))) <= LID_ROW_INK_MAX:
            break
        lid_bot -= 1

    # 2 · Seed the reference from the lid's own first clean rows.
    #
    # `_row_tone` ignores each row's ink, so a few surviving lash pixels
    # cannot drag the seed dark, and the seed is the EYELID's tone — shaded
    # by whatever the artist put over it (a glasses lens, a fringe shadow) —
    # rather than the bright face skin further up.
    seed: List[np.ndarray] = []
    for i in range(LID_REF_ROWS):
        yy = lid_bot - 1 - i
        if yy < 0:
            break
        t = _row_tone(rgb[yy, x0:x1])
        if t is not None:
            seed.append(t)
    if seed:
        skin = np.median(np.stack(seed), axis=0)
    else:                                   # the lid is entirely ink
        blk = rgb[max(0, top - ap_h):top, x0:x1].reshape(-1, 3)
        keep = blk[~_ink(blk)]
        skin = (np.median(keep, axis=0) if len(keep) >= SAMPLE_MIN_PX
                else np.median(blk, axis=0))

    # 3 · Collect the band, the reference tracking what it accepts.
    #
    # Tracking is the whole point: eyelid skin darkens gradually toward the
    # crease, so a fixed reference either rejects the far end of a real
    # gradient or admits a brow that is merely one step darker. An EMA
    # follows the gradient and still sees the step.
    ref = skin.copy()
    y = lid_bot
    while y > 0 and (lid_bot - y) < cap:
        row = rgb[y - 1, x0:x1]
        tone = _row_tone(row)
        if (tone is None                                  # all ink
                or float(np.mean(_ink(row))) > LID_ROW_INK_MAX
                or float(np.abs(tone - ref).max()) > LID_SKIN_DELTA
                # …and against the seed, so small steps cannot add up
                or float(np.abs(tone - skin).max()) > LID_SEED_DRIFT_MAX):
            break
        ref = (1.0 - LID_REF_EMA) * ref + LID_REF_EMA * tone
        y -= 1
    if lid_bot - y < LID_BAND_MIN:
        # The brow, the frame or the hairline sits directly on the eye, so
        # the artist drew no lid band to sample. The measured reference
        # tone is still this eyelid's own skin, so a flat band of it is the
        # honest answer — and it is bounded to LID_BAND_MIN rows, which the
        # renderer edge-extends rather than stretching.
        band = np.repeat(skin[None, None, :], LID_BAND_MIN, axis=0)
        band = np.repeat(band, x1 - x0, axis=1)
        src_y0 = max(0, lid_bot - LID_BAND_MIN)
    else:
        band = rgb[y:lid_bot, x0:x1].copy()
        src_y0 = y
        # De-ink each row against ITS OWN tone, not the band's.
        #
        # The band is allowed to be a gradient, so its top row can legitimately
        # sit further from the bottom row's tone than LID_SKIN_DELTA. Judging
        # every row against one reference would then condemn whole rows at the
        # far end of a real gradient and flatten them; judging a row against
        # itself condemns only the specks that differ from their own
        # neighbourhood, which is exactly what a stray lash tip or hair strand
        # is.
        for r in range(band.shape[0]):
            tone = _row_tone(band[r])
            if tone is None:
                band[r] = skin
                continue
            bad = _dirty(band[r], tone)
            if not bad.any():
                continue
            good = band[r][~bad]
            fill = (np.median(good, axis=0) if len(good) >= 3 else tone)
            band[r][bad] = fill

    strip = np.zeros((band.shape[0], x1 - x0, 4), dtype=np.uint8)
    strip[..., :3] = np.clip(band, 0, 255).astype(np.uint8)
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
