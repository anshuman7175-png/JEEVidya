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

# The containment test is taken against the opening dilated by this much,
# because that is what the tolerance above is *for*: a smooth ellipse
# rasterized against a hand-drawn, antialiased rim disagrees along a
# one-pixel seam. Measured on the corrected fits, spill against the raw
# aperture is 0.5–2.3% (a 1px rim on a ~50px disc is ~2% of its area, i.e.
# the seam alone can exceed the budget) while spill against the 1px-dilated
# aperture collapses to 0.00–0.14%. So the seam is charged to rasterization,
# where it belongs, and IRIS_SPILL_MAX stays at 1% for REAL overflow — the
# lash-contaminated fits still measure 18–34% and still fail.
IRIS_RASTER_TOL_PX = 1

# ── eyeball CORE extraction (§2) ───────────────────────────────────────
#
# The eyeball is separated from the painted lash by SHAPE, not by tone.
# Measured on this art (.tmp/shape.py): inside the opening, "not sclera"
# is one region = a thick round iris disc plus a thin full-width lash band
# welded to its top rim (gudiya eye_l: 12 lash rows above a disc of
# inscribed diameter 53). A morphological opening with a disc kernel
# deletes anything thinner than the kernel and — by the definition of
# opening — cannot erode a region that contains it, so the iris survives
# untouched while the lash is removed. Verified: the core's inscribed
# radius equals the full region's to 1.000 on all four eyes.
#
# The radius is stated as a fraction of the region's own inscribed radius
# so it scales with the art. A SEARCHED radius ("stop when area stops
# changing") was implemented and REJECTED by measurement: the area curve
# is locally flat well before the lash detaches (gudiya eye_l 3488→3446→
# 3423 at r=1,2,3 versus the real cliff at r=7), so the search halted at
# r=3 and kept the lash. This fraction lands past the cliff on all four
# eyes, and IRIS_CORE_IOU_MIN below proves it did on any new art.
IRIS_CORE_THIN = 0.30

# The fitted ellipse must actually describe the core it was fitted to.
# This is what makes the estimator self-checking: a lash-contaminated or
# crescent region cannot be summarised by an ellipse, so the agreement
# collapses. Measured 0.937–0.972 on the correct fits; the crescent-hull
# fit that shipped the broken rig scores far below this floor.
IRIS_CORE_IOU_MIN = 0.90

IRIS_SAT_MIN = 25.0      # max(RGB)−min(RGB) at or above this is chromatic
IRIS_LUM_MIN = 24.0      # below this is lash/pupil ink, not iris colour
IRIS_LUM_MAX = 190.0     # above this is sclera or the catchlight

# ── per-eye TONE GAIN ──────────────────────────────────────────────────
#
# Every threshold above is an ABSOLUTE level on 0…255, calibrated against
# an eye drawn in open light. An eye drawn behind a TINTED GLASSES LENS is
# the same eye multiplied by a constant < 1, so those levels land in the
# wrong place on it — and chintu wears glasses over his right eye only,
# which makes the two eyes of one face measure differently.
#
# Measured on chintu, with gudiya as the no-glasses control:
#
#     eye            white level   sat p90   sclera px   chroma px
#     chintu  left       254          80        1408        1384
#     chintu  right      155          51         265         830
#     gudiya  left       245          96         997        1558
#     gudiya  right      238          94         754        1289
#
# chintu's right eye is a UNIFORMLY DARKER copy of his left: brightness
# ×0.61 and saturation ×0.64, i.e. one gain, not a different drawing.
# gudiya's two eyes sit at ×0.97 — so this is the lens, not the artist.
#
# Under those absolute gates the tinted eye lost 81% of its sclera and
# read as "not sclera" almost everywhere, while the chroma test kept only
# a thin lit crescent of the iris (hull 892px vs 2139px on the open eye,
# bbox 40×36 vs 57×50). The hull of a crescent is not the hull of a disc,
# so the fitted eyeball came out 1.53× smaller on the right than the left
# and fell out of the plausible-radius band entirely.
#
# The fix is to make the measurement INVARIANT to that multiplication:
# read each eye's own white level and scale the absolute gates by it.
# Then the same drawing measures the same behind glass as in open air.
#
# Only true LEVEL gates are scaled. `SKIN_DELTA` is a distance from a
# local reference rather than a level, and scaling it was measured to
# change nothing on any of the four eyes, so it is left alone.
# `SCLERA_S_MAX` is an upper bound on "how grey is it", and a tint scales
# saturation DOWN, so a desaturated white stays desaturated without help.
TONE_REF_WHITE = 255.0   # the white level the absolute gates assume
TONE_WHITE_PCTL = 97.0   # percentile of V inside the opening = its "white"
# Never exceed 1.0: a gain above one would LOOSEN the calibrated gates.
# The floor stops a near-black ROI from collapsing every gate to zero.
TONE_GAIN_MIN = 0.35

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
GAZE_STEP = 0.5              # px per probe; the instrument must resolve this

# ── Gaze travel is ASYMMETRIC, because the drawing is ────────────────────
#
# The margin used to be reduced to one number per axis with
# `min(travel(-1), travel(+1))`, i.e. "how far can the eyeball move if it
# must be able to move the same distance both ways". That assumes the
# artist drew the eyeball CENTRED in its opening. Measured on this art, in
# every one of the four eyes, they did not:
#
#     eye              slack left   slack right   (px of opening beyond
#     chintu  eye_l        +3          +28         the fitted eyeball)
#     chintu  eye_r       +29           +4
#     gudiya  eye_l        +8          +19
#     gudiya  eye_r       +27           +1
#
# Both characters are drawn LOOKING OFF TO ONE SIDE — the eyeball sits hard
# against one corner with the whole sclera on the other side, which is
# plainly visible in the artwork and is exactly how eyes are drawn. The
# slack is on the outer canthus of each eye, mirrored between the pair.
#
# So the symmetric minimum is ~0 by construction: gudiya's eye_r can travel
# 11.7 px toward the sclera and 0.0 px toward the corner it already touches,
# and `min` reports 0.0. That single number then failed the bake for one
# character and, before the guard existed, was laundered into a guessed
# 0.55·iris_r excursion that drove the iris straight through the lash.
#
# The honest model is a per-direction budget: the eyeball may move as far as
# the drawing affords in EACH direction, which for an eye already looking
# right means a lot of room left and none right. Nothing is loosened — each
# of the four numbers is the same containment walk that produced the pair.
#
# The bake still refuses a horizontally FROZEN eye, because that is a real
# defect rather than a stylistic choice. Measured total horizontal span
# (left+right) is 0.45–0.60×iris_r on all four eyes, so this floor sits ~3×
# below every real measurement while a true zero still fails.
GAZE_H_SPAN_MIN = 0.15       # ×iris_r, min (left+right) travel to be usable

# Fractional bits for fixed-point ellipse rasterisation (see _ellipse_mask).
# 3 bits = 1/8 px, comfortably finer than GAZE_STEP, which is the whole point:
# an integer-rounded ellipse cannot measure a margin of a few px.
_SUBPIX_BITS = 3

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

# ── Is the band SKIN at all? ─────────────────────────────────────────────
#
# Every bound above is RELATIVE — each row is compared to the rows already
# accepted. That is what lets a real lid gradient through, but it is also
# blind in one direction: if the walk STARTS on something that is not skin,
# every later row agrees with it and the whole band is admitted. chintu is
# exactly that case. His glasses rim crosses the face where the upper lid
# would be, so the rows immediately above his opening are frame ink, the
# seed is taken from that ink, and a band of it is accepted unanimously.
#
# The renderer then does the rest of the damage — and not by stretching, as
# was assumed: `_lid_shear` runs at natural scale and CLAMPS, repeating the
# strip's top row for the rest of the closure. Clamping a 10px ink strip
# paints the whole lens with rim brown, which is the flat blob filling
# chintu's glasses.
#
# So the walk needs one ABSOLUTE anchor, and the artwork already contains
# the right one: the skin just BELOW the opening. It is the same face under
# the same light and, for a character in glasses, behind the same lens — so
# it carries the lens shading a forehead sample would miss, while being
# unambiguously skin. Measured on the plate:
#
#            above the opening      below the opening
#   chintu L (157, 88, 62)          (247,169,132)   ← lum ratio 0.56
#   chintu R (105, 52, 36)          (157, 88, 62)   ← rim, then skin
#   gudiya L (248,170,118)          (245,167,115)   ← lum ratio ~1.00
#
# gudiya's lid matches her cheek because it IS her lid; chintu's does not
# because it is his frame. A row is therefore lid only while it stays
# within these ratios of the below-eye reference, and the walk stops at the
# first row that is not — so chintu keeps whatever genuine lens-shaded skin
# lies between his lash and his frame, and stops at the frame.
LID_SKIN_LUM_RATIO = 0.72     # a row this much darker than the cheek is not
                              # eyelid skin — it is frame, brow or hair
LID_SKIN_WARMTH_RATIO = 0.55  # nor is one this much less warm (R−B); ink is
                              # dark AND desaturated, skin is neither
LID_BELOW_SKIP_FRAC = 0.06    # ×ap_h, rows below the opening that are lower lash
LID_BELOW_FRAC = 0.55         # ×ap_h, how far down the cheek reference reaches


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
    gaze_box   : (left, right, up, down) px the eyeball may travel before
                 its rim reaches the drawn opening. This is the artwork's
                 own sclera margin, so a gaze of ±1 is the largest look the
                 drawing can hold rather than a guessed fraction. Four
                 numbers rather than two because the eyeball is drawn hard
                 against one corner of its opening on every eye of this art
                 (see GAZE_H_SPAN_MIN) — a symmetric budget is ~0 there.
    tone_gain  : this eye's own brightness scale relative to the level the
                 absolute tone gates are calibrated against (1.0 = drawn in
                 open light, 0.61 = measured behind chintu's tinted lens).
                 Carried on the record because the lid band is sampled in a
                 SECOND pass (`lid_sprite`) that must judge ink and skin by
                 the same scaled bounds the opening was measured with.
    """
    aperture: Tuple[Tuple[float, float], ...]
    iris_c: Tuple[float, float]
    iris_axes: Tuple[float, float]
    iris_angle: float
    iris_r: float
    colors: Dict[str, Tuple[int, int, int]]
    gaze_box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    tone_gain: float = 1.0

    def to_dict(self) -> dict:
        return {
            "aperture": [list(p) for p in self.aperture],
            "iris_c": list(self.iris_c),
            "iris_axes": list(self.iris_axes),
            "iris_angle": self.iris_angle,
            "iris_r": self.iris_r,
            "colors": {k: list(v) for k, v in self.colors.items()},
            "gaze_box": list(self.gaze_box),
            "tone_gain": self.tone_gain,
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
            # Both of these were omitted, so a round-tripped record silently
            # lost its measured gaze budget and its tone gain and reverted to
            # the "unmeasured" defaults.
            gaze_box=tuple(map(float, d.get("gaze_box",
                                            (0.0, 0.0, 0.0, 0.0))))[:4],
            tone_gain=float(d.get("tone_gain", 1.0)),
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
    """Filled ellipse mask, rasterised at TRUE sub-pixel precision.

    `cv2.ellipse` takes integer coordinates, so the obvious call has to round
    the centre to a whole pixel. That rounding silently destroyed the gaze
    measurement: `_gaze_margin` walks the eyeball outward in 0.5 px steps,
    but a rounded centre only moves when the accumulated offset crosses 0.5,
    so the walk rasterised 1 px jumps and reported the SAME spill for d=0.5
    and d=1.0 (measured 1.42% at both on gudiya's left eye). A margin of a
    few px — which is all this art affords, because the iris nearly fills the
    opening — cannot be resolved by an instrument quantised to 1 px.

    OpenCV's `shift` argument is the fix: it interprets the centre and axes
    as fixed-point with `shift` fractional bits, so the ellipse is rasterised
    against the sub-pixel geometry it was actually fitted to. `_SUBPIX_BITS`
    of 3 gives eighth-pixel resolution — well under the 0.5 px step — while
    staying far from any integer overflow.

    Strictly more accurate for every caller, including the containment
    invariant, which previously judged a rounded ellipse against the rim.
    """
    cv2 = _require_cv2()
    m = np.zeros(shape, np.uint8)
    s = _SUBPIX_BITS
    k = float(1 << s)
    cv2.ellipse(m,
                (int(round(c[0] * k)), int(round(c[1] * k))),
                (max(1, int(round((axes[0] + grow) * k))),
                 max(1, int(round((axes[1] + grow) * k)))),
                angle, 0, 360, 1, -1, shift=s)
    return m


def _repair_rim(ap: np.ndarray, max_depth: float) -> np.ndarray:
    """Fill SHALLOW bites in the aperture's rim; leave deep concavities alone.

    The eye-like test is `not_skin AND (bright-desaturated OR dark)`, which
    leaves a mid-luma band (HSV V in 110…150) that neither clause claims. On
    this art the iris shades straight through it, so the segmented opening
    comes out with a chunk missing from its lower rim — measured here as
    185–679 px at (125,55,29) / (134,126,128) / (144,62,12), each ≥99%
    "not skin" and 0% ink, i.e. unambiguously eye and unambiguously dropped.
    That bite is what the fitted eyeball then spills through, failing the
    containment invariant and refusing the whole v3 bake.

    Widening the tone test is not the fix — `not_skin` compares against the
    ROI border median, which holds hair on chintu, so admitting the mid band
    globally makes the aperture swallow the entire ROI.

    So the repair is GEOMETRIC and local. A drawn eye opening is a smooth,
    near-convex almond, so a shallow dent in its rim is a threshold artifact,
    while a deep concavity is real shape (an eye corner) or a leak and must
    survive untouched. Each convex-hull deficiency is therefore filled only
    when its OWN maximum depth is under `max_depth`.

    Depth is the deficiency's distance-to-the-aperture maximum: a bite is
    enclosed by real eye on three sides, so this measures how far the missing
    region reaches away from the rim that surrounds it. Measured 7.0–11.6 px
    against a 13.8–15.4 px bound on this art, so the genuine bites are
    repaired with margin and nothing else in the frame qualifies.

    Deterministic: same mask in, same mask out.
    """
    cv2 = _require_cv2()
    cont, _ = cv2.findContours(ap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cont:
        return ap
    c = max(cont, key=cv2.contourArea)
    hull = np.zeros_like(ap)
    cv2.fillPoly(hull, [cv2.convexHull(c)], 1)
    deficiency = ((hull > 0) & (ap == 0)).astype(np.uint8)
    if deficiency.sum() == 0:
        return ap

    # Distance from every non-aperture pixel to the nearest aperture pixel.
    # Computed once on the whole ROI so each component is measured in the
    # same field, never relative to its own bounding box.
    dist = cv2.distanceTransform((ap == 0).astype(np.uint8), cv2.DIST_L2, 5)
    n, lab = cv2.connectedComponents(deficiency, 8)[:2]
    out = ap.copy()
    for i in range(1, n):
        comp = lab == i
        if float(dist[comp].max()) <= max_depth:
            out[comp] = 1
    return _fill_holes(out)


def _gaze_margin(ap: np.ndarray, iris_c: Tuple[float, float],
                 iris_axes: Tuple[float, float], iris_angle: float,
                 cap: float) -> Tuple[float, float, float, float]:
    """Travel the eyeball may make in each direction: (left, right, up, down).

    Translating the iris ellipse and testing containment answers this
    directly from the art, and does so for the real, non-elliptical
    aperture — a fixed fraction of `iris_r` cannot, because how much
    sclera a drawing affords is a property of the drawing.

    Each direction is reported SEPARATELY. Collapsing the four numbers into
    a symmetric pair with `min` was a modelling error, not a safety margin:
    this art draws both characters looking off to one side, so one direction
    on each axis is legitimately ~0 and the minimum is therefore ~0 even
    though 19–29 px of sclera sits on the other side. See GAZE_H_SPAN_MIN.

    REFERENCE: containment is judged against the aperture UNION THE IRIS AT
    REST, not the aperture alone, and the reason is measurable. The iris is
    an ellipse fitted to a hand-drawn disc and then rasterised, so even at
    rest — where the artist unambiguously drew it inside the opening — it
    reports 2…16 px outside the segmented rim (0.10…0.43% of its area).
    A test that asks "any pixel outside the aperture" therefore fails at the
    very first step for every eye on this art, breaks the loop, and returns
    0.0 travel. That is exactly the zero this function used to emit, and it
    is why `EyeGeometry.gaze_offset` was silently falling back to a guessed
    fraction of `iris_r`. That fallback now raises for a measured rig, so a
    zero emitted here is a loud bake failure rather than a wrong animation.

    Taking rest as the reference removes that fitting artifact WITHOUT
    loosening anything: the iris is permitted only where it already
    legitimately sits, and every pixel of NEW overhang is still counted.
    At d=0 the spill is 0 by construction, so the walk starts from a true
    zero and each half-pixel step is judged on its own merit.

    The surviving tolerance is `IRIS_SPILL_MAX`, deliberately the SAME
    budget the bake's containment invariant enforces. Two predicates that
    disagree about what "inside" means is the defect being repaired here;
    one rounded boundary pixel must not mean two different verdicts in two
    different places.

    `GAZE_MARGIN_SAFETY` still inflates the travelling ellipse, so the
    eyeball is held a real margin clear of the painted lash rather than
    being allowed to graze it.
    """
    safety = GAZE_MARGIN_SAFETY
    axes = (iris_axes[0] + safety, iris_axes[1] + safety)
    rest = _ellipse_mask(ap.shape, iris_c, axes, iris_angle)
    allowed = (ap > 0) | (rest > 0)
    budget = IRIS_SPILL_MAX

    def travel(ux: float, uy: float) -> float:
        best = 0.0
        d = GAZE_STEP
        while d <= cap:
            m = _ellipse_mask(ap.shape,
                              (iris_c[0] + ux * d, iris_c[1] + uy * d),
                              axes, iris_angle)
            area = max(1, int((m > 0).sum()))
            spill = int(((m > 0) & ~allowed).sum())
            if spill / area > budget:
                break
            best = d
            d += GAZE_STEP
        return best

    return (float(travel(-1.0, 0.0)), float(travel(1.0, 0.0)),
            float(travel(0.0, -1.0)), float(travel(0.0, 1.0)))


def _tone_gain(V: np.ndarray, ap: np.ndarray, label: str) -> float:
    """This eye's own brightness scale, read off the eye itself.

    The absolute tone gates (`SCLERA_V_MIN`, `DARK_V_MAX`, `IRIS_LUM_*`,
    `_ink`) are levels on 0…255 calibrated against an eye drawn in open
    light. A tinted lens multiplies the whole eye by a constant, which moves
    every one of those levels to the wrong place on that eye — see the TONE
    GAIN note above for the measurement that proves chintu's right eye is
    his left eye ×0.61, not a different drawing.

    The eye's white level is the honest scale factor: an eye opening always
    contains its brightest paint (sclera, or a catchlight), so a high
    percentile of V inside the opening tracks the multiplication directly. A
    percentile rather than the max because a single blown pixel is noise.

    Clamped on both sides, and each bound is load-bearing:
      • never above 1.0 — a gain over one would LOOSEN calibrated gates and
        let skin, hair or a frame into the aperture.
      • never below TONE_GAIN_MIN — a near-black ROI (a bad seed landing off
        the face) would otherwise collapse every gate to zero and "succeed"
        by matching everything.
    """
    vals = V[ap > 0]
    if vals.size == 0:
        raise EyeMeasureError(
            f"{label}: cannot read a tone level — the provisional eye "
            f"opening is empty.")
    white = float(np.percentile(vals, TONE_WHITE_PCTL))
    return float(min(1.0, max(TONE_GAIN_MIN, white / TONE_REF_WHITE)))


def _segment_opening(roi: np.ndarray, V: np.ndarray, S: np.ndarray,
                     skin: np.ndarray, seed_local: Tuple[float, float],
                     face_h: float, gain: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """The drawn eye opening at a given tone gain.

    Shared by both measurement passes so the provisional opening that
    ESTABLISHES the gain and the final opening MEASURED at that gain cannot
    drift apart. Returns (aperture mask, sclera_like mask); the caller owns
    every invariant, because pass 1 must tolerate a rough answer while pass
    2 must not.

    `SCLERA_S_MAX` and `SKIN_DELTA` are deliberately NOT scaled: the first
    is an upper bound on "how grey is it" and a tint only desaturates, the
    second is a distance from a local reference rather than a level, and
    scaling it was measured to change nothing on any of the four eyes.
    """
    cv2 = _require_cv2()
    not_skin = np.abs(roi - skin).max(axis=2) > SKIN_DELTA
    sclera_like = (V > SCLERA_V_MIN * gain) & (S < SCLERA_S_MAX)
    dark_like = V < DARK_V_MAX * gain

    eye = (not_skin & (sclera_like | dark_like)).astype(np.uint8)
    eye = cv2.morphologyEx(eye, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    eye = cv2.morphologyEx(eye, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ap = _largest_near(eye, seed_local)
    if ap.sum() == 0:
        return ap, sclera_like
    ap = cv2.morphologyEx(ap, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    ap = _fill_holes(ap)
    # Repair the mid-luma bites the tone test leaves in the rim, BEFORE any
    # area check, so `frac` and every consumer see one final aperture.
    ap = _repair_rim(ap, APERTURE_CONCAVE_MAX * face_h)
    return ap, sclera_like


def _eyeball_core(ap: np.ndarray, sclera_like: np.ndarray,
                  seed_local: Tuple[float, float]
                  ) -> Tuple[np.ndarray, int, float]:
    """The drawn eyeball inside `ap`, with the painted lash removed by shape.

    Returns (core, radius_used, inscribed_radius). See IRIS_CORE_THIN for
    why an opening is the right operator and why its radius is derived from
    the region's own inscribed radius rather than searched.

    This is deliberately a STRUCTURAL measurement — aperture minus eye
    white — with no tone thresholds in it. The previous version fitted the
    eyeball to a chroma-gated mask, which made the rig's GEOMETRY depend on
    absolute brightness/saturation levels; on a uniformly tinted eye
    (chintu's right, gain 0.608) the gates carved the disc into a broken
    ring, and the convex hull of a ring-with-gaps is a wedge. Geometry now
    comes from geometry; tone is used only to pick COLOURS further down.
    """
    cv2 = _require_cv2()
    region = _fill_holes(((ap > 0) & (~sclera_like)).astype(np.uint8))
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    region = _fill_holes(_largest_near(region, seed_local))
    if region.sum() == 0:
        return region, 0, 0.0

    dist = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    r_ins = float(dist.max())
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    r = max(2, int(round(IRIS_CORE_THIN * r_ins)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    core = cv2.morphologyEx(region, cv2.MORPH_OPEN, k)
    if core.sum() == 0:            # nothing thick enough survived
        return core, r, r_ins
    # Seed from the thickest point of the region, not the landmark: the
    # landmark centroid can sit on the lash, and the disc is by definition
    # where the inscribed circle is largest.
    core = _fill_holes(_largest_near(core, (float(cx), float(cy))))
    return core, r, r_ins


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
    seed_local = (seed[0] - x0, seed[1] - y0)

    # ── 1 · aperture: the drawn eye opening, measured at its OWN tone ──
    #
    # Two passes, because the gain and the opening define each other: the
    # gain is the eye's white level, which can only be read INSIDE the
    # opening, and the opening is found with gates that need the gain.
    #
    # Pass 1 runs at unit gain purely to locate the eye. It is allowed to be
    # wrong about the eye's EXTENT — on the tinted eye it recovers only the
    # lit crescent — because all it has to do is land inside the opening well
    # enough for a high percentile of V to see that eye's brightest paint.
    # No invariant is enforced on it; every one of them belongs to pass 2.
    ap0, sclera0 = _segment_opening(roi, V, S, skin, seed_local, face_h, 1.0)
    if ap0.sum() == 0:
        raise EyeMeasureError(
            f"{label}: found no eye-like region near {seed}. The art may "
            f"not show an open eye there, or the eye is drawn in skin "
            f"tones (no sclera and no dark iris) — such art cannot drive "
            f"gaze or blink and must be re-exported.")

    gain = _tone_gain(V, ap0, label)

    # Pass 2 is the measurement. At gain 1.0 it is bit-identical to pass 1,
    # so an eye drawn in open light is unaffected by any of this.
    if gain >= 1.0:
        ap, sclera_like = ap0, sclera0
    else:
        ap, sclera_like = _segment_opening(
            roi, V, S, skin, seed_local, face_h, gain)
    if ap.sum() == 0:
        raise EyeMeasureError(
            f"{label}: the eye opening vanished when re-measured at its own "
            f"tone gain {gain:.3f}. The ROI is too dark to segment.")

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
    # The eyeball is separated from the lash by SHAPE (`_eyeball_core`): the
    # lash is thin, the eyeball is thick, and a morphological opening tells
    # them apart without consulting a single brightness threshold. Fitting
    # to a chroma-gated mask was tried and is what shipped the broken rig —
    # on a tinted eye the gates punched the disc into a broken ring, and a
    # ring-with-gaps hulls to a wedge (chintu eye_r: hull 892px, bbox 40×36
    # for a real disc of 1733px, 51×50).
    iris_m, core_r, core_r_ins = _eyeball_core(ap, sclera_like, seed_local)
    if iris_m.sum() < MIN_COMPONENT:
        raise EyeMeasureError(
            f"{label}: no eyeball inside the measured aperture — the opening "
            f"reads as eye white and a thin ink line only (largest round "
            f"structure survived an r={core_r}px opening with "
            f"{int(iris_m.sum())}px). Gaze would have nothing to move.")

    # Fit the ellipse to the core's OUTLINE. Contour-fitting beats hulling
    # here by measurement (IoU 0.937–0.972 vs 0.925–0.964, spill 0.00–0.14%
    # vs 0.20–0.84%, on all four eyes): the core is already a solid,
    # hole-filled disc, so there are no gaps for a hull to bridge, and the
    # hull's straight bridging edges only inflate the fit. The moment
    # estimator was also measured and is worse than both.
    cs, _ = cv2.findContours(iris_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c for c in cs if len(c) >= 5]
    if cs:
        (ecx, ecy), (ew, eh), ang = cv2.fitEllipse(max(cs, key=cv2.contourArea))
        a, b_ = ew / 2.0, eh / 2.0
    else:                       # degenerate outline: fall back to the disc
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

    # 2b-i · the ellipse must DESCRIBE the core it was fitted to. This is
    # the estimator checking its own premise: an ellipse is only a valid
    # summary of a round region, so a crescent, a wedge or a lash-welded
    # blob cannot score well here no matter how the axes come out. The
    # radius-band and containment tests below both passed for fits that were
    # visibly wrong, because "plausible size" and "inside the opening" are
    # satisfied by a small wedge; agreement with the measured shape is not.
    iou = (int(((ell > 0) & (iris_m > 0)).sum())
           / max(1, int(((ell > 0) | (iris_m > 0)).sum())))
    if iou < IRIS_CORE_IOU_MIN:
        raise EyeMeasureError(
            f"{label}: fitted eyeball {2 * a:.0f}×{2 * b_:.0f}px agrees with "
            f"the measured eyeball region only IoU={iou:.3f} (needs "
            f"≥{IRIS_CORE_IOU_MIN}). The region is not ellipse-shaped, so it "
            f"is not the drawn eyeball — it is a crescent or a lash-welded "
            f"blob, and every gaze/lid number derived from it would be "
            f"fiction. Region {int(iris_m.sum())}px, opening radius "
            f"r={core_r}px of inscribed {core_r_ins:.1f}px, tone gain "
            f"{gain:.3f}.")

    # 2b-ii · containment, measured against the opening plus a ONE-PIXEL
    # rasterization seam (IRIS_RASTER_TOL_PX). The budget itself is
    # unchanged; what is corrected is where it is charged. A smooth ellipse
    # and a hand-drawn antialiased rim disagree along their shared edge, and
    # on a ~50px disc that seam alone is ~2% of the area — so charging it to
    # "overflow" would either fail a correct fit or force the budget up.
    ap_tol = cv2.dilate(
        ap, np.ones((IRIS_RASTER_TOL_PX * 2 + 1,) * 2, np.uint8))
    spill = int(((ell > 0) & (ap_tol == 0)).sum())
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
        chromatic = ((iris_sat >= IRIS_SAT_MIN * gain)
                     & (iris_lum >= IRIS_LUM_MIN * gain)
                     & (iris_lum <= IRIS_LUM_MAX * gain))
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
        pupil_sel = iris_px[(iris_sat < IRIS_SAT_MIN * gain)
                            & (iris_lum <= max(IRIS_LUM_MIN, 60.0) * gain)]
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
        ink = band_px[_ink(band_px, gain)]
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
    g_l, g_r, g_u, g_d = _gaze_margin(ap, (ecx, ecy), (a, b_), float(ang),
                                      cap=GAZE_MAX_FRAC * r_eq)
    # A TOTALLY frozen eye is a measurement failure, never a usable answer,
    # and it must not be laundered into a guessed fraction of iris_r — that
    # is what drove the iris through the painted lash.
    #
    # But the test is on the SPAN, not on each direction. An eyeball drawn
    # hard against one corner of its opening — which is how every eye in this
    # art is drawn — genuinely affords 0 px toward that corner, and demanding
    # travel in both directions would refuse correct artwork. What no drawn
    # open eye can be is frozen on an axis: it must have sclera SOMEWHERE.
    h_span, v_span = g_l + g_r, g_u + g_d
    if h_span < GAZE_H_SPAN_MIN * r_eq or v_span <= 0.0:
        raise EyeMeasureError(
            f"{label}: measured gaze travel is left={g_l:.1f} right={g_r:.1f} "
            f"up={g_u:.1f} down={g_d:.1f} px — horizontal span {h_span:.1f}px "
            f"(needs ≥{GAZE_H_SPAN_MIN * r_eq:.1f}px = "
            f"{GAZE_H_SPAN_MIN}×iris_r), vertical span {v_span:.1f}px (needs "
            f">0). The fitted eyeball cannot move inside the drawn opening, "
            f"so the iris/aperture measurement is wrong, not the artwork; "
            f"check the tone gain ({gain:.3f}) and the fitted axes "
            f"({a:.1f}, {b_:.1f}) against an aperture of "
            f"{int(ap.sum())} px.")

    return ArtEye(
        aperture=_contour_poly(ap_clip, (x0, y0), smooth=APERTURE_SMOOTH),
        iris_c=(float(ecx + x0), float(ecy + y0)),
        iris_axes=(float(a), float(b_)),
        iris_angle=float(ang),
        iris_r=r_eq,
        colors=colors,
        gaze_box=(g_l, g_r, g_u, g_d),
        tone_gain=gain,
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


def _ink(px: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Dark AND near-achromatic pixels: a lash, a brow, a glasses rim.

    Reference-free by design. It is the one lid test that can be trusted
    before any tone has been established, which is why the walk uses it to
    find the lash before it decides what the eyelid's skin even looks like.

    `gain` is the eye's own tone gain (see TONE_REF_WHITE). At the default
    1.0 this is exactly the historical test. Below 1.0 the bounds scale
    with the signal, which keeps ink separable from a merely TINTED iris:
    under a lens the iris darkens toward the unscaled ink bound and starts
    reading as ink, which is what erased the shaded rim of chintu's right
    eyeball and left a crescent for the hull to fit.
    """
    lum = _luma(px)
    sat = px.max(axis=-1) - px.min(axis=-1)
    return (lum <= LASH_LUM_MAX * gain) & (sat <= LASH_SAT_MAX * gain)


def _row_tone(row: np.ndarray, gain: float = 1.0) -> Optional[np.ndarray]:
    """A row's own skin tone: the median of its NON-INK pixels.

    Excluding the ink is what makes the tone comparable from row to row —
    include it and a row holding the lash's tip reads as a tone step, so the
    walk stops one row into a lid it should have kept walking up.

    `gain` scales that ink test to the eye's own tone: behind a tinted lens
    the eyelid itself is darkened, and an unscaled bound calls the whole row
    ink, returns None, and ends the walk at the first genuine lid row.
    """
    keep = row[~_ink(row, gain)]
    if len(keep) < max(3, row.shape[0] // 4):
        return None
    return np.median(keep, axis=0)


def _dirty(px: np.ndarray, skin: np.ndarray,
           gain: float = 1.0) -> np.ndarray:
    """Pixels that are not clean eyelid skin: ink, or far from `skin`.

    Ink (brow, lash, a glasses rim) is dark AND near-achromatic — the same
    separation the lash colour uses, so "ink" means one thing everywhere in
    this module. Everything else that is simply the wrong colour (a hair
    strand, a bright frame highlight, the sclera) is caught by distance
    from the local reference.
    """
    far = np.abs(px - skin).max(axis=-1) > LID_SKIN_DELTA
    return _ink(px, gain) | far


def _lum1(t: Sequence[float]) -> float:
    """Rec.601 luma of one RGB triple."""
    return 0.299 * float(t[0]) + 0.587 * float(t[1]) + 0.114 * float(t[2])


def _warmth(t: Sequence[float]) -> float:
    """R−B. Skin is warm; frame ink, hair and shadow are not.

    Paired with luma this separates lid from frame on chintu, where a
    brightness test alone is marginal: his lens-shaded lid is dark but
    stays warm, while the rim is dark AND neutral.
    """
    return float(t[0]) - float(t[2])


def _lid_skin_below(rgb: np.ndarray, x0: int, x1: int, bot: int,
                    ap_h: int, gain: float, label: str
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Real skin sampled BELOW the opening — the lower lid and cheek.

    This is the absolute anchor the relative row bounds lack. It is taken
    below the eye rather than above it because that is the one place on
    this art guaranteed to be face skin: above the eye may be a brow, a
    fringe or — on chintu — a glasses rim, and a reference taken there is
    what let a band of frame ink be accepted as an eyelid.

    Returns (rows, median) where `rows` are the clean sampled rows ordered
    NEAREST-the-eye first, so a synthesized lid can keep the artist's
    gradient by simply reversing them.
    """
    h = rgb.shape[0]
    y0 = min(h, bot + max(1, int(round(ap_h * LID_BELOW_SKIP_FRAC))))
    y1 = min(h, y0 + max(LID_BAND_MIN, int(round(ap_h * LID_BELOW_FRAC))))
    rows: List[np.ndarray] = []
    for yy in range(y0, y1):
        row = rgb[yy, x0:x1]
        if float(np.mean(_ink(row, gain))) > LID_ROW_INK_MAX:
            continue
        t = _row_tone(row, gain)
        if t is not None:
            rows.append(t)
    if len(rows) < LID_BAND_MIN:
        raise EyeMeasureError(
            f"{label}: no clean skin below the eye opening (rows "
            f"{y0}..{y1} are ink or transparent), so the lid has no "
            f"reference tone to be judged against. The head crop is too "
            f"tight below the eye, or the opening was mis-measured.")
    return np.stack(rows), np.median(np.stack(rows), axis=0)


def _lid_skin_like(tone: np.ndarray, ref: np.ndarray) -> bool:
    """Is this row eyelid SKIN, judged against the below-eye reference?"""
    ref_l, ref_w = _lum1(ref), _warmth(ref)
    if ref_l <= 1.0:
        return True                       # no usable reference; other bounds rule
    if _lum1(tone) < LID_SKIN_LUM_RATIO * ref_l:
        return False
    if ref_w > 8.0 and _warmth(tone) < LID_SKIN_WARMTH_RATIO * ref_w:
        return False
    return True


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
    # The lid is sampled from the same paint the opening was measured in, so
    # it is judged at the same tone. Without this the lash walk and every row
    # test run open-light bounds over a lens-shaded lid.
    gain = float(eye.tone_gain)

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
        if float(np.mean(_ink(row, gain))) <= LID_ROW_INK_MAX:
            break
        lid_bot -= 1

    # 2 · Seed the reference from the lid's own first clean rows.
    #
    # `_row_tone` ignores each row's ink, so a few surviving lash pixels
    # cannot drag the seed dark, and the seed is the EYELID's tone — shaded
    # by whatever the artist put over it (a glasses lens, a fringe shadow) —
    # rather than the bright face skin further up.
    # The below-eye skin is measured FIRST, because the seed itself has to be
    # checked against it: on chintu the rows that would seed the reference
    # are frame ink, and a seed taken from ink admits a band of ink.
    below_rows, below_ref = _lid_skin_below(rgb, x0, x1, bot, ap_h, gain,
                                           "lid sprite")
    seed: List[np.ndarray] = []
    for i in range(LID_REF_ROWS):
        yy = lid_bot - 1 - i
        if yy < 0:
            break
        t = _row_tone(rgb[yy, x0:x1], gain)
        if t is not None and _lid_skin_like(t, below_ref):
            seed.append(t)
    if seed:
        skin = np.median(np.stack(seed), axis=0)
    else:
        # Nothing above the lash is skin — the frame or brow sits straight on
        # the eye. The honest reference is then the cheek's own tone, and the
        # band below is synthesized from real sampled skin rather than ink.
        skin = below_ref.copy()

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
        tone = _row_tone(row, gain)
        if (tone is None                                  # all ink
                or float(np.mean(_ink(row, gain))) > LID_ROW_INK_MAX
                or float(np.abs(tone - ref).max()) > LID_SKIN_DELTA
                # …and against the seed, so small steps cannot add up
                or float(np.abs(tone - skin).max()) > LID_SEED_DRIFT_MAX
                # …and absolutely, against skin measured below the eye, so a
                # walk that begins on a glasses rim cannot ratify itself
                or not _lid_skin_like(tone, below_ref)):
            break
        ref = (1.0 - LID_REF_EMA) * ref + LID_REF_EMA * tone
        y -= 1
    if lid_bot - y < LID_BAND_MIN:
        # The brow, the frame or the hairline sits directly on the eye, so
        # the artist drew no upper-lid band to sample. This is chintu: his
        # glasses rim occupies the rows an eyelid would be cut from.
        #
        # The lid is therefore built from the skin BELOW the opening — his
        # own lower lid and cheek, real sampled pixels carrying the real
        # lens shading and the artist's own gradient, not a palette fill.
        # The rows are reversed so the row that was nearest the eye ends up
        # at the strip's BOTTOM, which is the leading edge: the skin that
        # arrives at the closing edge is the skin that genuinely adjoins
        # the opening, and the tone recedes away from it exactly as the
        # artist painted it receding down the cheek.
        #
        # A flat fill was the old answer here, and it is why a closed eye
        # read as a hole punched in the face: no gradient, no shading, one
        # sampled tone stretched over the whole lens.
        n = below_rows.shape[0]
        band = np.repeat(below_rows[::-1][:, None, :], x1 - x0, axis=1)
        src_y0 = max(0, lid_bot - n)
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
            tone = _row_tone(band[r], gain)
            if tone is None:
                band[r] = skin
                continue
            bad = _dirty(band[r], tone, gain)
            if not bad.any():
                continue
            good = band[r][~bad]
            fill = (np.median(good, axis=0) if len(good) >= 3 else tone)
            band[r][bad] = fill

    # ── Invariant: the strip is SKIN ────────────────────────────────────
    #
    # The renderer clamps vertically, so the strip's TOP row is what fills a
    # deep closure — one ink row there paints the entire eye with it. That
    # is precisely how a 10px band of glasses rim became a flat brown blob
    # over chintu's lenses while every gate still reported a pass. Both the
    # top row and the band as a whole are therefore checked against the
    # below-eye skin, and a failure stops the bake instead of shipping.
    top_tone = _row_tone(band[0], gain)
    if top_tone is None or not _lid_skin_like(top_tone, below_ref):
        raise EyeMeasureError(
            f"lid sprite top row is not skin (tone={None if top_tone is None else tuple(int(v) for v in top_tone)}, "
            f"cheek reference={tuple(int(v) for v in below_ref)}). The "
            f"renderer repeats this row for the whole closure, so it would "
            f"paint the eye with it.")
    ink_frac = float(np.mean(_ink(band.reshape(-1, 3), gain)))
    if ink_frac > LID_ROW_DIRT_MAX:
        raise EyeMeasureError(
            f"lid sprite is {ink_frac * 100:.1f}% ink after de-inking "
            f"(limit {LID_ROW_DIRT_MAX * 100:.0f}%) — the band is lash, "
            f"brow or glasses frame, not eyelid skin.")

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
