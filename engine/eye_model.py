"""
JEEVidya — Parametric Eyes, Lids & Living Gaze (Terminal Plan, Part V)
══════════════════════════════════════════════════════════════════════
Renders eyeball + iris + pupil + lids ON the inpainted socket of the
head plate, in head space. Replaces the slid-rectangle blink (D9) and
the decoupled lids (D10) with geometry that GUARANTEES correctness:

  • Iris/pupil are clipped to the socket contour — a saccade can never
    slide the iris onto the cheek.
  • At blink=1 the upper-lid path IS the lower-lid contour — full iris
    occlusion is a geometric identity, not a tuned offset. QC then
    asserts it on rendered pixels (zero iris-color pixels at blink=1).
  • Lids track vertical gaze (~0.6 gain), brow raise lifts the upper
    lid (~0.15), blink pulls the brow down (~0.08): the D10 couplings.

Blink dynamics use the MEASURED human profile (close ≈85 ms cubic
ease-in, hold ≈25 ms, open ≈180 ms ease-out, ±8% jitter, independent
1-frame L/R offset) — not a symmetric sin^0.7.

Living gaze: micro-saccades (0.1–0.3° ≈ 1.5–4.5% iris radius, Poisson
1–2 Hz) during holds so eyes never freeze; pupil dilation +5–8% on
emphasis; anticipatory glances scheduled by the acting layer 200–350 ms
before a reveal. All procedural, all seeded, all deterministic.

Rendering: 4× supersample → LANCZOS down, sub-pixel composite. Same
discipline as engine/mouth_model.py.
"""
from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

_log = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# Coupling gains (Part V, D10)
# ═══════════════════════════════════════════

LID_GAZE_GAIN = 0.6      # lids track eye_dy at this gain
BROW_LID_LIFT = 0.15     # brow raise lifts the upper lid
BLINK_BROW_PULL = 0.08   # blink pulls the brow down

# Blink profile (measured human, milliseconds)
BLINK_CLOSE_MS = 85.0
BLINK_HOLD_MS = 25.0
BLINK_OPEN_MS = 180.0
BLINK_JITTER = 0.08      # ±8% duration jitter per blink
BLINK_TOTAL_MS = BLINK_CLOSE_MS + BLINK_HOLD_MS + BLINK_OPEN_MS

# Micro-saccade statistics (Part V "living gaze")
MICROSACCADE_RATE_HZ = 1.5          # Poisson mean rate during holds
MICROSACCADE_AMP = (0.015, 0.045)   # amplitude as fraction of iris radius ×3
                                    # (0.1–0.3° of a ~10° iris span)
PUPIL_EMPHASIS_GAIN = 0.08          # up to +8% dilation on emphasis

# Gaze travel for LEGACY rigs only (no measured sclera margin baked).
# A v3 rig carries `gaze_box` measured from the artwork, and that always
# wins — see EyeGeometry.gaze_offset. This fraction is retained purely so a rig
# baked before the measurement still moves its eyes; applying it to this
# art is what slid the iris onto the painted lash.
LEGACY_GAZE_FRAC = 0.55

SUPERSAMPLE = 4


# ═══════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════

def _pt2(v) -> Tuple[float, float]:
    """A 2-vector from a possibly-absent rig field. A legacy rig has no
    art-eye assets at all, so every read of them must tolerate absence."""
    try:
        return (float(v[0]), float(v[1]))
    except Exception:
        return (0.0, 0.0)


@dataclass(frozen=True)
class EyeGeometry:
    """One eye's baked geometry in HEAD-PLATE space (from rig v3 §3.5).

    socket : (N,2) polygon of the eye opening (sclera boundary)
    lid_upper / lid_lower : polylines left→right along the lid margins
    iris_c : iris center;  iris_r : iris radius (px, head space)

    §3.5b — when the rig carries a MEASURED eye (tools/art_eyes.py) the
    fields below are populated and are AUTHORITATIVE:

    aperture   : the outline of the eye the artist actually drew. It is
                 the clip for every pixel this module paints, which is
                 what makes the blink-skin-rectangle unrepresentable
                 rather than merely unlikely.
    iris_axes  : semi-axes of the drawn eyeball. A circle cannot fit
                 stylised art (chintu measures 28.7 × 39.4 px), and a
                 circular iris inside an oval eye leaves crescents of
                 the artwork's own iris showing on either side.
    iris_angle : that ellipse's rotation, degrees.
    colors     : sclera/iris/pupil/lash sampled inside their own
                 measured regions, so they cannot pick up shirt or hair.

    `measured` is the single predicate the renderer branches on. Legacy
    v1/v2 rigs leave it False and keep the old socket behaviour.
    """
    socket: Tuple[Tuple[float, float], ...]
    lid_upper: Tuple[Tuple[float, float], ...]
    lid_lower: Tuple[Tuple[float, float], ...]
    iris_c: Tuple[float, float]
    iris_r: float
    aperture: Tuple[Tuple[float, float], ...] = ()
    iris_axes: Tuple[float, float] = (0.0, 0.0)
    iris_angle: float = 0.0
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    # §3.5b — the artist's own eyeball, cut from the plate as a sprite.
    # When present the renderer MOVES these pixels instead of synthesizing
    # an eye from flat colour, so a resting frame equals the artwork.
    eyeball: str = ""
    eyeball_origin: Tuple[float, float] = (0.0, 0.0)
    # The socket the eyeball uncovers when gaze moves (the artwork's eye
    # with its iris inpainted out), and the artist's own eyelid skin. Both
    # exist so no flat palette colour is ever painted inside the eye.
    socket_img: str = ""
    socket_origin: Tuple[float, float] = (0.0, 0.0)
    lid_img: str = ""
    lid_origin: Tuple[float, float] = (0.0, 0.0)
    # (left, right, up, down) px the eyeball may travel before its rim
    # reaches the drawn opening — the artwork's OWN sclera margin, measured
    # in art_eyes. On this art the iris nearly fills the eye, so the true
    # margin is a few px; the generic 0.55·iris_r excursion used before was
    # ~18 px, which drove the iris onto the lash and forced
    # `socket_backdrop` to inpaint the whole ellipse, producing the brown
    # radial smear behind the eye.
    #
    # FOUR values, not a symmetric pair: measured on this art, every eye is
    # drawn with the eyeball hard against one corner of its opening (both
    # characters look off to one side), so the sclera margin is 19–29 px on
    # one side and 1–4 px on the other. A symmetric budget is min() of those
    # and therefore ~0, which is why this used to measure as "frozen".
    gaze_box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def gaze_offset(self, gx: float, gy: float) -> Tuple[float, float]:
        """Px the eyeball moves for a gaze of (gx, gy), each in [-1, 1].

        Resolves the gaze against the MEASURED per-direction margin, picking
        the budget for the direction actually being looked in. The legacy
        fraction survives only for rigs baked before that margin existed
        (`measured` is False), so an old rig still animates instead of
        freezing its eyes open.

        Three things here were load-bearing bugs and are deliberately not
        restored:

        • The budget was symmetric — `min(left, right)` per axis — which on
          this art is ~0, because every eye is drawn with its eyeball hard
          against one corner of the opening. Correct artwork measured as a
          frozen eye. Each direction now carries its own budget.

        • The test was `dx > 0 or dy > 0`, so a half-measured range — a real
          horizontal margin with a zero vertical one — was returned verbatim
          and the eye simply could not look up or down, silently. Both axes
          must be present for the measurement to be believed.

        • A MEASURED rig reporting zero is a bake defect, not an old rig.
          Quietly substituting 0.55·iris_r (~18 px on this art, against a
          true margin of a few px) is what drove the iris onto the painted
          lash and forced the socket inpaint that smeared brown behind the
          eye. A measured rig with no margin now raises, because the rig is
          wrong and must be re-baked rather than animated wrongly.
        """
        left, right, up, down = self.gaze_box
        # Horizontal and vertical spans, not individual directions: a drawn
        # eye legitimately affords zero travel toward the corner its eyeball
        # already touches, so requiring every direction to be non-zero would
        # reject correct art. What it cannot be is frozen on an AXIS.
        if (left + right) > 0.0 and (up + down) > 0.0:
            dx = float(gx) * (right if gx >= 0.0 else left)
            dy = float(gy) * (down if gy >= 0.0 else up)
            return (dx, dy)
        if self.measured:
            raise ValueError(
                f"eye was measured from artwork but carries gaze_box "
                f"{self.gaze_box} — re-bake the rig; refusing to guess an "
                f"excursion that would paint the iris over the lash.")
        _log.warning(
            "[eye_model] legacy rig: no measured gaze margin, falling back "
            "to %.2f x iris_r (%.1f px). Re-bake for art-accurate gaze.",
            LEGACY_GAZE_FRAC, self.iris_r * LEGACY_GAZE_FRAC)
        r = self.iris_r * LEGACY_GAZE_FRAC
        return (float(gx) * r, float(gy) * r)

    @property
    def measured(self) -> bool:
        """True when this eye was measured from the artwork's pixels."""
        return len(self.aperture) >= 3

    @property
    def clip(self) -> Tuple[Tuple[float, float], ...]:
        """The polygon no painted eye pixel may leave."""
        return self.aperture if self.measured else self.socket

    @property
    def axes(self) -> Tuple[float, float]:
        """Iris semi-axes, falling back to a circle of radius `iris_r`."""
        a, b = self.iris_axes
        if a > 0.0 and b > 0.0:
            return (float(a), float(b))
        return (self.iris_r, self.iris_r)

    @staticmethod
    def from_rig_dict(d: dict) -> "EyeGeometry":
        return EyeGeometry(
            socket=tuple(map(tuple, d["socket"])),
            lid_upper=tuple(map(tuple, d["lid_upper"])),
            lid_lower=tuple(map(tuple, d["lid_lower"])),
            iris_c=tuple(d["iris"][:2]),
            iris_r=float(d["iris"][2]),
            aperture=tuple(map(tuple, d.get("aperture") or ())),
            iris_axes=tuple(float(v) for v in
                            (d.get("iris_axes") or (0.0, 0.0)))[:2],
            iris_angle=float(d.get("iris_angle") or 0.0),
            colors={k: tuple(int(c) for c in v)
                    for k, v in (d.get("colors") or {}).items()},
            eyeball=str(d.get("eyeball") or ""),
            eyeball_origin=_pt2(d.get("eyeball_origin")),
            socket_img=str(d.get("socket_img") or ""),
            socket_origin=_pt2(d.get("socket_origin")),
            lid_img=str(d.get("lid_img") or ""),
            lid_origin=_pt2(d.get("lid_origin")),
            gaze_box=tuple(float(v) for v in
                           (d.get("gaze_box") or (0.0, 0.0, 0.0, 0.0)))[:4],
        )

    @staticmethod
    def synth_from_box(box: Tuple[int, int, int, int]) -> "EyeGeometry":
        """Legacy-rig fallback: synthesize plausible geometry from an eye
        Box. Used only for v1/v2 rigs; v3 bakes real polylines."""
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        w, h = (x1 - x0), (y1 - y0)
        rx, ry = w / 2.0, h / 2.0
        n = 12
        upper, lower = [], []
        for i in range(n + 1):
            t = i / n
            x = x0 + w * t
            arc = math.sin(math.pi * t)          # 0→1→0 across the eye
            upper.append((x, cy - ry * arc))
            lower.append((x, cy + ry * 0.72 * arc))
        socket = tuple(upper) + tuple(reversed(lower))
        return EyeGeometry(socket=socket,
                           lid_upper=tuple(upper),
                           lid_lower=tuple(lower),
                           iris_c=(cx, cy),
                           iris_r=min(rx, ry) * 0.62)


@dataclass(frozen=True)
class EyeState:
    """Per-frame pose of both eyes. All values dimensionless.

    blink   : 0 open … 1 fully closed (per eye after L/R offset)
    eye_dx / eye_dy : gaze offset in iris radii (≈ [-1, 1])
    brow    : brow raise −1…+1
    squint  : lower-lid rise 0…1 (also coupled to mouth `pull` upstream)
    pupil   : pupil dilation multiplier around 1.0
    """
    blink_l: float = 0.0
    blink_r: float = 0.0
    eye_dx: float = 0.0
    eye_dy: float = 0.0
    brow: float = 0.0
    squint: float = 0.0
    pupil: float = 1.0

    def quantized_key(self) -> Tuple[float, ...]:
        q = 1 / 64
        vals = (self.blink_l, self.blink_r, self.eye_dx, self.eye_dy,
                self.brow, self.squint, self.pupil)
        return tuple(round(v / q) * q for v in vals)


# ═══════════════════════════════════════════
# Blink dynamics — measured human profile
# ═══════════════════════════════════════════

def blink_amount(dt_ms: float, jitter: float = 0.0) -> float:
    """Blink openness→closure at `dt_ms` since blink onset ∈ [0, 1].

    close ~85 ms cubic ease-in → hold ~25 ms → open ~180 ms ease-out.
    `jitter` ∈ [-1, 1] scales all three durations by (1 + 0.08·jitter).
    """
    k = 1.0 + BLINK_JITTER * max(-1.0, min(1.0, jitter))
    close_ms, hold_ms, open_ms = (BLINK_CLOSE_MS * k, BLINK_HOLD_MS * k,
                                  BLINK_OPEN_MS * k)
    if dt_ms < 0:
        return 0.0
    if dt_ms < close_ms:
        t = dt_ms / close_ms
        return t * t * t                       # cubic ease-in: accelerating slam
    dt_ms -= close_ms
    if dt_ms < hold_ms:
        return 1.0                             # lids sealed
    dt_ms -= hold_ms
    if dt_ms < open_ms:
        t = dt_ms / open_ms
        return 1.0 - (1.0 - (1.0 - t) ** 3)    # ease-out: slow release
    return 0.0


class BlinkScheduler:
    """Deterministic blink onsets: log-normal intervals, 10% double-blink,
    independent 1-frame L/R offset. Seeded per character."""

    def __init__(self, seed: str, fps: int):
        h = int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16)
        self._rng = np.random.default_rng(h)
        self.fps = int(fps)
        self._frame_ms = 1000.0 / self.fps
        self._onset_ms = float(self._rng.uniform(600, 2200))
        self._jitter = float(self._rng.uniform(-1, 1))
        # Which eye leads by one frame this blink (0=L leads, 1=R leads)
        self._lead = int(self._rng.integers(0, 2))
        self._double_pending = False

    def _interval_ms(self) -> float:
        # Log-normal: median ≈ 2.4 s, human resting blink statistics
        return float(np.exp(self._rng.normal(np.log(2400.0), 0.45)))

    def sample(self, t_ms: float) -> Tuple[float, float]:
        """(blink_l, blink_r) at time t_ms. Advances internal schedule."""
        if t_ms >= self._onset_ms + BLINK_TOTAL_MS * 1.2:
            if self._double_pending:
                self._double_pending = False
                self._onset_ms = t_ms + 150.0
            else:
                self._double_pending = bool(self._rng.uniform() < 0.10)
                self._onset_ms = t_ms + self._interval_ms()
            self._jitter = float(self._rng.uniform(-1, 1))
            self._lead = int(self._rng.integers(0, 2))
        dt = t_ms - self._onset_ms
        off = self._frame_ms  # 1-frame independent offset
        b_lead = blink_amount(dt, self._jitter)
        b_lag = blink_amount(dt - off, self._jitter)
        return (b_lead, b_lag) if self._lead == 0 else (b_lag, b_lead)

    def force_blink(self, t_ms: float) -> None:
        """Pull the next blink to `t_ms` (cut-masking, phrase boundaries)."""
        self._onset_ms = min(self._onset_ms, t_ms)


class GazeEngine:
    """Living gaze: micro-saccades + emphasis pupil dilation.

    Poisson micro-saccades (1–2 Hz) with 0.1–0.3° amplitude keep held
    eyes alive; each saccade is a ~30 ms step with a tiny overshoot.
    Deterministic per seed.
    """

    def __init__(self, seed: str):
        h = int(hashlib.sha256((seed + ":gaze").encode()).hexdigest()[:12], 16)
        self._rng = np.random.default_rng(h)
        self._next_ms = float(self._rng.exponential(1000.0 / MICROSACCADE_RATE_HZ))
        self._cur = (0.0, 0.0)     # current micro-offset (iris-radius units)
        self._prev = (0.0, 0.0)
        self._step_start = -1e9

    def sample(self, t_ms: float) -> Tuple[float, float]:
        if t_ms >= self._next_ms:
            amp = float(self._rng.uniform(*MICROSACCADE_AMP)) * 3.0
            ang = float(self._rng.uniform(0, 2 * math.pi))
            self._prev = self._cur
            self._cur = (amp * math.cos(ang), amp * math.sin(ang))
            self._step_start = t_ms
            self._next_ms = t_ms + float(
                self._rng.exponential(1000.0 / MICROSACCADE_RATE_HZ))
        # ~30 ms step with 8% overshoot then settle
        t = (t_ms - self._step_start) / 30.0
        if t >= 1.0:
            return self._cur
        e = 1.0 - (1.0 - t) ** 2
        e *= 1.0 + 0.08 * math.sin(math.pi * min(1.0, t))
        return (self._prev[0] + (self._cur[0] - self._prev[0]) * e,
                self._prev[1] + (self._cur[1] - self._prev[1]) * e)

    @staticmethod
    def pupil_for_emphasis(emphasis: float) -> float:
        """Pupil scale from prosody emphasis ∈ [0,1] → 1.0 … 1.08."""
        return 1.0 + PUPIL_EMPHASIS_GAIN * max(0.0, min(1.0, emphasis))


# ═══════════════════════════════════════��═══
# Coupled state — applies the D10 gains
# ═══════════════════════════════════════════

def couple(state: EyeState) -> EyeState:
    """Apply lid↔gaze↔brow couplings. Idempotence is NOT expected —
    call exactly once per frame, on the raw acting channels."""
    blink_mean = (state.blink_l + state.blink_r) / 2.0
    brow = state.brow - BLINK_BROW_PULL * blink_mean
    # Brow raise opens the lids a little; gaze-down drops them
    lid_bias = (-BROW_LID_LIFT * max(0.0, brow)
                + LID_GAZE_GAIN * max(0.0, state.eye_dy) * 0.25)
    return replace(
        state,
        brow=max(-1.0, min(1.0, brow)),
        blink_l=max(0.0, min(1.0, state.blink_l + lid_bias)),
        blink_r=max(0.0, min(1.0, state.blink_r + lid_bias)),
    )


# ═══════════════════════════════════════════
# Rasterizer
# ═══════════════════════════════════════════

def _resample(poly: Sequence[Tuple[float, float]], n: int) -> np.ndarray:
    """Arc-length resample a polyline to n points."""
    p = np.asarray(poly, dtype=np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1] if cum[-1] > 0 else 1.0
    ts = np.linspace(0.0, total, n)
    out = np.empty((n, 2))
    out[:, 0] = np.interp(ts, cum, p[:, 0])
    out[:, 1] = np.interp(ts, cum, p[:, 1])
    return out


def _clip_polygon(geo: EyeGeometry) -> np.ndarray:
    """The closed contour no painted eye pixel may leave.

    A measured aperture is the drawn eye's own outline and is sampled
    finely; a legacy rig only has the synthesized socket.
    """
    src = geo.clip if geo.clip else geo.socket
    clip = np.asarray(src, dtype=np.float64)
    return _resample(np.vstack([clip, clip[:1]]),
                     96 if geo.measured else 48)


# ── Why the lid is a SCANLINE and not a polygon ────────────────────────
#
# The lid margin used to be a 24-point polyline interpolated from the
# aperture's upper rim to its lower rim in POLAR coordinates about the
# eyeball centre, and the covered region was the polygon between that
# margin and a cap above it. Both halves of that scheme leaked:
#
#   • Polar interpolation lerps each point's ANGLE about the centre, so a
#     point does not travel straight down — it swings sideways. Points
#     near the corners swing furthest and overtake their neighbours, so
#     the margin stops being x-ordered (measured: up to 5 reversals at
#     closure 0.5). A polyline that doubles back makes the lid polygon
#     self-intersect, and PIL fills a self-intersecting polygon by the
#     even-odd rule — which leaves the crossed lobe EMPTY. That hole is
#     the ragged crescent of iris that survived a 0.7–1.0 blink.
#
#   • Even x-ordered, 24 chords only approximate a curved rim, so the
#     closure-1 margin missed the true rim by a sliver (measured: 4
#     transparent aperture pixels on chintu's right eye).
#
# Coverage is therefore computed per PIXEL COLUMN of the rasterized
# aperture instead. For every column the aperture's own first and last
# rows are known exactly, and the lid's leading edge in that column is
#
#     row(c) = top + (bot - top + 1) · c
#
# with pixels above it covered. At c=0 that covers nothing; at c=1 it
# covers every aperture pixel in every column — full occlusion as an
# identity at pixel resolution, with no tuned offset. The leading edge
# is one row per column, so it cannot double back and the even-odd hole
# is not merely unlikely but unrepresentable. Curvature is inherited
# from the artwork's two rims, so the closing edge still bends the way
# the drawn eye does.


class EyeRasterizer:
    """Draws one eye (socket-clipped eyeball + lids) at 4× supersample.

    palette keys used: sclera, iris, lash, skin, lip_shadow (crease).
    Returns an RGBA patch + its float paste origin in head space.
    """

    def __init__(self, geo: EyeGeometry, palette: Dict[str, Tuple[int, int, int]],
                 cache_size: int = 512,
                 sprite: Optional[Image.Image] = None,
                 socket: Optional[Image.Image] = None,
                 lid: Optional[Image.Image] = None):
        self.geo = geo
        # §3.5b — the artist's eyeball. When present, gaze TRANSLATES these
        # pixels; the synthetic sclera/iris/catchlight path below is the
        # fallback for rigs baked before the measurement existed.
        self.sprite = sprite
        # §3.5b — the two layers that let the eye animate with NO flat
        # palette fill anywhere inside it:
        #   socket : the drawn eye with its iris inpainted out, so a gaze
        #            shift uncovers the artist's own eye white and shading
        #            instead of one sampled `sclera` colour (which measured
        #            a grey shadow on one of chintu's eyes and a white on
        #            the other, making his two eyes different colours).
        #   lid    : the strip of face the artist painted just above the
        #            aperture. Sliding it down closes the eye with real
        #            skin carrying the real crease and lash, replacing the
        #            flat `skin` ellipse that read as a punched hole.
        self.socket = socket
        self.lid = lid
        # Measured, in-region eye colours beat the landmark-sampled
        # palette: they were taken from inside the segmented iris and
        # sclera, so they cannot be the lash line or a glasses frame.
        self.palette = dict(palette)
        self.palette.update(geo.colors)
        self._cache: "OrderedDict[Tuple, Tuple[Image.Image, Tuple[float, float]]]" \
            = OrderedDict()
        self._cache_size = cache_size
        # The patch bounds the CLIP polygon (the measured aperture where
        # one exists). Padding is cosmetic only — since every stroke is
        # masked to the aperture, the patch border can no longer become a
        # visible edge the way the old lid "cap" did.
        clip = geo.clip or geo.socket
        xs = [p[0] for p in clip]
        ys = [p[1] for p in clip]
        pad = max(2.0, geo.iris_r * 0.35)
        self._x0, self._y0 = min(xs) - pad, min(ys) - pad
        self._x1, self._y1 = max(xs) + pad, max(ys) + pad

        # ── The supersampled patch grid, rasterized aperture and its
        # per-column extents. All three depend only on geometry, so they
        # are built once here; `render` reads them. Caching them is also
        # what lets the lid be a scanline: the aperture's true first and
        # last row per column are needed every frame.
        S = SUPERSAMPLE
        self._w = max(int(math.ceil((self._x1 - self._x0) * S)), 2)
        self._h = max(int(math.ceil((self._y1 - self._y0) * S)), 2)
        self._clip_poly = _clip_polygon(geo)
        mask = Image.new("L", (self._w, self._h), 0)
        ImageDraw.Draw(mask).polygon(
            [((x - self._x0) * S, (y - self._y0) * S)
             for x, y in self._clip_poly], fill=255)
        self._clip_mask = mask
        ap = np.asarray(mask) > 127
        self._ap = ap
        col = ap.any(axis=0)
        self._col_any = col
        top = ap.argmax(axis=0).astype(np.float64)
        bot = (self._h - 1 - ap[::-1].argmax(axis=0)).astype(np.float64)
        # A column with no aperture gets an empty span, so no closure and
        # no squint can ever paint in it.
        top[~col] = 0.0
        bot[~col] = -1.0
        self._col_top, self._col_bot = top, bot

    # ── Scanline lid coverage ──────────────────────────────────────────

    def _margin_rows(self, closure: float, lower: bool = False) -> np.ndarray:
        """Per-column row of a lid's leading edge, in patch pixels.

        `lower=False` is the upper lid sweeping DOWN from the aperture's
        top rim: at closure 0 it has covered nothing, at closure 1 it has
        reached past the bottom rim. `lower=True` is the lower lid rising
        UP from the bottom rim. The `+1` makes closure 1 inclusive of the
        final aperture row, so full closure covers the eye exactly.
        """
        c = max(0.0, min(1.0, closure))
        span = self._col_bot - self._col_top + 1.0
        return (self._col_bot - span * c) if lower else (self._col_top + span * c)

    def _cover(self, rows: np.ndarray, lower: bool = False) -> np.ndarray:
        """Aperture pixels a lid at `rows` occludes."""
        ys = np.arange(self._h, dtype=np.float64)[:, None]
        side = (ys > rows[None, :]) if lower else (ys < rows[None, :])
        return side & self._ap

    def _edge_points(self, rows: np.ndarray) -> List[Tuple[float, float]]:
        """The leading edge as a polyline, one point per aperture column.
        X-ordered by construction, so it can never self-intersect."""
        idx = np.nonzero(self._col_any)[0]
        return [(float(x), float(rows[x])) for x in idx]

    def _lid_shear(self, src: Image.Image, origin: Tuple[float, float],
                   rows: np.ndarray) -> np.ndarray:
        """The lid strip at NATURAL SCALE, sheared per column so its bottom
        row rides the leading edge. Returns (H, W, 3) over the whole patch.

        The previous version stretched the strip to whatever height the
        closure needed. On this art that is a 28px band over a 95px opening
        — a 3.4× smear, which is why a blink looked like a bar of mush
        drawn across the eye. A lid does not stretch as it closes; it
        TRANSLATES over the eyeball. So here the pixels keep their aspect
        and each column is merely shifted down by its own amount:

            shift[x] = rows[x] − (rest bottom row)

        Per column, not one shift for the patch, because the leading edge
        is a curve — the eye closes deepest at its middle. A single shift
        would either tear the lid away from the rim at the corners or crush
        it at the centre; the shear keeps the strip continuous and lets its
        bottom row follow the curve exactly. At closure 0 the shift is the
        one pixel that returns the strip where it was cut from, so a
        resting frame is untouched artwork.

        Sampling is CLAMPED vertically, which is the guarantee that
        matters: the lid's alpha is the coverage mask, so every covered
        pixel must have a colour, and a covered pixel above the shifted
        strip would otherwise be a transparent hole in the lid. Clamping
        repeats the strip's topmost row — measured clean eyelid skin — so
        the deep part of a closure continues that tone instead of
        distorting the crease to reach it.
        """
        S = SUPERSAMPLE
        sw = max(1, int(round(src.width * S)))
        sh = max(1, int(round(src.height * S)))          # natural scale
        arr = np.asarray(src.convert("RGB").resize((sw, sh), Image.LANCZOS))
        oy = int(round((origin[1] - self._y0) * S))
        ox = int(round((origin[0] - self._x0) * S))

        # Per-column shift that puts the strip's bottom row on the edge.
        shift = np.rint(rows - float(oy + sh - 1))
        shift[~self._col_any] = 0.0
        # Columns outside the aperture are never painted (the coverage mask
        # is empty there), but they still index the array, so keep them in
        # range rather than relying on the mask.
        ys = np.arange(self._h)[:, None]
        sy = np.clip(ys - shift[None, :] - oy, 0, sh - 1).astype(np.intp)
        sx = np.clip(np.arange(self._w) - ox, 0, sw - 1).astype(np.intp)
        return arr[sy, sx[None, :].repeat(self._h, axis=0)]

    @staticmethod
    def _as_mask(m: np.ndarray) -> Image.Image:
        return Image.fromarray((m * 255).astype(np.uint8), "L")

    def _color(self, key: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
        v = self.palette.get(key, default)
        return tuple(int(c) for c in v[:3])

    def render(self, state: EyeState, left: bool) -> Tuple[Image.Image, Tuple[float, float]]:
        blink = state.blink_l if left else state.blink_r
        key = (round(blink * 64), round(state.eye_dx * 64),
               round(state.eye_dy * 64), round(state.brow * 32),
               round(state.squint * 32), round(state.pupil * 64), left)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit

        S = SUPERSAMPLE
        w, h = self._w, self._h
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        def T(pts: np.ndarray) -> List[Tuple[float, float]]:
            return [((x - self._x0) * S, (y - self._y0) * S) for x, y in pts]

        geo = self.geo
        clip_poly = self._clip_poly

        # 0 · The aperture clip, rasterized once in __init__. EVERY pixel
        # this method paints is masked to it at the end, so no stroke can
        # reach the patch border. This is the structural fix for the
        # hard-edged skin rectangle: the lid's fill used to be bounded
        # only by the patch rectangle, so a blink painted a rectangle of
        # skin over the brow. A stroke that cannot leave the drawn eye
        # cannot draw a rectangle.
        clip_mask = self._clip_mask

        # 1 · The eyeball, offset by gaze and clipped to the aperture.
        #
        # The excursion is the MEASURED per-direction sclera margin. This
        # line used to read `iris_r * 0.55` — the legacy guess, hardcoded
        # here — which made the whole measured budget dead code: the bake
        # computed it, the rig stored it, and the renderer ignored it and
        # moved the iris ~18 px on art that affords a few. That is the smear
        # behind the eye. `gaze_offset` resolves the sign per axis, because
        # the margin is asymmetric on every eye of this art.
        gaze = np.asarray(geo.gaze_offset(state.eye_dx, state.eye_dy))
        ic = np.asarray(geo.iris_c) + gaze

        ball = Image.new("RGBA", img.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(ball)
        # The drawn eyeball is not a circle on stylised art. A circle
        # inside an oval eye left crescents of the artwork's own iris
        # uncovered at the sides.
        ax, ay = geo.axes
        ra, rb = ax * S, ay * S
        rest = ((geo.iris_c[0] - self._x0) * S, (geo.iris_c[1] - self._y0) * S)
        moved = ((ic[0] - self._x0) * S, (ic[1] - self._y0) * S)
        iris_rgb = self._color("iris", (74, 52, 38))
        sclera_rgb = self._color("sclera", (245, 243, 238))

        def _oval(img_draw, sx: float, sy: float, fill=None, outline=None,
                  width: int = 0, center=None) -> None:
            """Ellipse at (sx, sy)× the iris semi-axes, rotated by the
            measured angle, centred on the gaze-shifted iris unless
            `center` says otherwise. Rotation matters: an unrotated
            ellipse on a tilted eye shows the same crescent gap it was
            meant to close."""
            cx, cy = moved if center is None else center
            box = [cx - ra * sx, cy - rb * sy, cx + ra * sx, cy + rb * sy]
            if abs(geo.iris_angle) < 0.5:
                img_draw.ellipse(box, fill=fill, outline=outline, width=width)
                return
            th = math.radians(geo.iris_angle)
            pts = []
            for i in range(64):
                t = 2 * math.pi * i / 64
                px, py = ra * sx * math.cos(t), rb * sy * math.sin(t)
                pts.append((cx + px * math.cos(th) - py * math.sin(th),
                            cy + px * math.sin(th) + py * math.cos(th)))
            if fill is not None:
                img_draw.polygon(pts, fill=fill)
            if outline is not None:
                img_draw.line(pts + pts[:1], fill=outline, width=max(1, width))

        def art_layer(src: Image.Image, origin: Tuple[float, float],
                      shift: Tuple[float, float] = (0.0, 0.0)) -> Image.Image:
            """`src` upscaled to the supersampled grid and placed at its
            baked plate-space origin plus `shift`, on a transparent layer
            the size of the patch. Placing every art layer through one
            function is what keeps the eyeball and the socket behind it in
            register at any render scale. (The lid is handled by
            `_lid_shear`, which shears per column and clamps.)
            """
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            up = src.resize((max(1, int(round(src.width * S))),
                            max(1, int(round(src.height * S)))),
                           Image.LANCZOS)
            # paste (not alpha_composite) because it clips out-of-bounds
            # boxes instead of raising; `layer` is empty, so the two agree.
            layer.paste(up, (int(round((origin[0] + shift[0] - self._x0) * S)),
                             int(round((origin[1] + shift[1] - self._y0) * S))),
                        up)
            return layer

        if self.socket is not None:
            # ART BACKDROP — the drawn eye with its iris inpainted out.
            # Its own alpha is the aperture, so it cannot reach the cheek.
            img.alpha_composite(art_layer(self.socket, geo.socket_origin))
        elif self.sprite is not None:
            # No baked socket: repaint only the FOOTPRINT the eyeball
            # vacates, never the whole aperture — this art is almost all
            # iris, so flooding the aperture with `sclera` paints eye-white
            # the artist never drew. 1.12× covers the largest excursion.
            _oval(draw, 1.12, 1.12, fill=sclera_rgb + (255,), center=rest)

        if self.sprite is not None:
            # ART PATH — move the artist's pixels. At gaze 0 the sprite
            # lands exactly where it was cut from, so a resting frame is
            # the artwork: its shading, lash overlap and highlight all
            # survive, which synthesis threw away.
            ball.alpha_composite(art_layer(self.sprite, geo.eyeball_origin,
                                           (gaze[0], gaze[1])))
            # Pupil dilation is deliberately NOT applied here: the pupil
            # is part of the drawing, and redrawing it flat would undo the
            # very shading this path exists to keep.
        else:
            # SYNTHETIC FALLBACK — legacy rigs with no baked sprite.
            draw.polygon(T(clip_poly), fill=sclera_rgb + (255,))
            _oval(bd, 1.0, 1.0, fill=iris_rgb + (255,))
            # limbal ring — darker iris edge, reads as depth at phone scale
            ring = tuple(int(c * 0.55) for c in iris_rgb)
            _oval(bd, 1.0, 1.0, outline=ring + (255,),
                  width=max(1, int(min(ra, rb) * 0.10)))
            pupil_rgb = self._color("pupil", (18, 12, 10))
            ps = 0.42 * state.pupil
            _oval(bd, ps, ps, fill=pupil_rgb + (255,))
            # catchlight — upper-left, the most "alive" pixel in the face
            clr = min(ra, rb) * 0.22
            cx, cy = moved
            bd.ellipse([cx - ra * 0.45 - clr, cy - rb * 0.45 - clr,
                        cx - ra * 0.45 + clr, cy - rb * 0.45 + clr],
                       fill=(255, 255, 255, 230))
        ball.putalpha(Image.composite(ball.split()[3], Image.new("L", img.size, 0),
                                      clip_mask))
        img.alpha_composite(ball)

        # 3 · Lower lid (squint rise), then upper lid at blink closure.
        # Both are SCANLINE coverage masks over the rasterized aperture
        # (see the note above `_clip_polygon`), so neither can leave the
        # drawn eye and neither can leave a hole inside it.
        skin = self._color("skin", (232, 190, 160))
        if state.squint > 0.01:
            lo_cover = self._cover(
                self._margin_rows(0.18 * state.squint, lower=True), lower=True)
            img.paste(Image.new("RGBA", img.size, skin + (255,)), (0, 0),
                      self._as_mask(lo_cover))

        closure = max(0.0, min(1.0, blink))
        rows = self._margin_rows(closure)
        cover = self._cover(rows)
        edge = self._edge_points(rows - 0.5)

        if self.lid is not None:
            # ART LID — the artist's own eyelid skin, slid down over the
            # eye. Colour and extent come from different places, and both
            # matter:
            #
            #   pixels : the baked strip at NATURAL SCALE, sheared per
            #            column so its bottom row rides the leading edge
            #            (see `_lid_shear`). That bottom row is the row
            #            directly above the aperture — the artist's own
            #            crease and lash — so the skin arriving at the
            #            closing edge is drawn skin, undistorted, and a
            #            resting frame is untouched artwork.
            #   alpha  : the coverage mask itself. Because the shear samples
            #            with vertical clamping, every covered pixel has
            #            real skin behind it; the alpha alone decides the
            #            shape. The old code took the opposite approach —
            #            paste the strip, then intersect with a polygon —
            #            which meant any disagreement between the two
            #            became a transparent hole in the lid.
            rgb = self._lid_shear(self.lid, geo.lid_origin, rows)
            alpha = (cover * 255).astype(np.uint8)
            img.alpha_composite(Image.fromarray(
                np.dstack([rgb, alpha]), "RGBA"))
        else:
            # SYNTHETIC FALLBACK — flat skin, for rigs with no baked lid.
            img.paste(Image.new("RGBA", img.size, skin + (255,)), (0, 0),
                      self._as_mask(cover))
            if edge and closure > 0.02:
                # Crease + lash on the leading edge, which a flat fill has
                # none of. One point per column, so this polyline is
                # monotonic and cannot cross itself. The ART path needs
                # none of this — the strip's own bottom row IS the lash the
                # artist drew, and painting a palette line over it would
                # put flat colour back inside the eye.
                crease = self._color("lip_shadow", (150, 100, 84))
                draw.line(edge, fill=crease + (160,),
                          width=max(1, int(S * 0.9)))
                lash = self._color("lash", (38, 26, 24))
                draw.line(edge, fill=lash + (255,), width=max(1, int(S * 1.6)))

        # 4 · Clip to the aperture, then downsample.
        # Applied at supersampled resolution so the boundary is
        # antialiased by the LANCZOS step and the eye's rim blends into
        # the artwork's lash line instead of showing a stair-stepped edge.
        img.putalpha(Image.composite(img.split()[3],
                                     Image.new("L", img.size, 0), clip_mask))
        img = img.resize((max(w // S, 1), max(h // S, 1)), Image.LANCZOS)
        img = img.filter(ImageFilter.GaussianBlur(0.4))
        out = (img, (self._x0, self._y0))
        self._cache[key] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out


# ═══════════════════════════════════════════
# Facade — one call per frame from the head assembler
# ═══════════════════════════════════════════

class EyePair:
    """Both eyes of one character. Owns scheduler, gaze engine and
    rasterizers; the head assembler calls `composite(plate, t_ms, ...)`."""

    def __init__(self, geo_l: EyeGeometry, geo_r: EyeGeometry,
                 palette: Dict[str, Tuple[int, int, int]],
                 seed: str, fps: int,
                 sprite_l: Optional[Image.Image] = None,
                 sprite_r: Optional[Image.Image] = None,
                 socket_l: Optional[Image.Image] = None,
                 socket_r: Optional[Image.Image] = None,
                 lid_l: Optional[Image.Image] = None,
                 lid_r: Optional[Image.Image] = None):
        self.left = EyeRasterizer(geo_l, palette, sprite=sprite_l,
                                  socket=socket_l, lid=lid_l)
        self.right = EyeRasterizer(geo_r, palette, sprite=sprite_r,
                                   socket=socket_r, lid=lid_r)
        self.blinks = BlinkScheduler(seed, fps)
        self.gaze = GazeEngine(seed)

    def state_at(self, t_ms: float, eye_dx: float = 0.0, eye_dy: float = 0.0,
                 brow: float = 0.0, squint: float = 0.0,
                 emphasis: float = 0.0,
                 blink_override: Optional[float] = None) -> EyeState:
        if blink_override is not None:
            bl = br = max(0.0, min(1.0, blink_override))
        else:
            bl, br = self.blinks.sample(t_ms)
        mdx, mdy = self.gaze.sample(t_ms)
        raw = EyeState(blink_l=bl, blink_r=br,
                       eye_dx=eye_dx + mdx, eye_dy=eye_dy + mdy,
                       brow=brow, squint=squint,
                       pupil=GazeEngine.pupil_for_emphasis(emphasis))
        return couple(raw)

    def composite(self, plate: Image.Image, state: EyeState) -> Image.Image:
        """Alpha-composite both eyes onto the head plate (sub-pixel)."""
        for rast, is_left in ((self.left, True), (self.right, False)):
            patch, (ox, oy) = rast.render(state, is_left)
            ix, iy = math.floor(ox), math.floor(oy)
            fx, fy = ox - ix, oy - iy
            if fx > 1e-3 or fy > 1e-3:
                patch = patch.transform(
                    (patch.width + 1, patch.height + 1), Image.AFFINE,
                    (1, 0, -fx, 0, 1, -fy), resample=Image.BICUBIC)
            plate.alpha_composite(patch, (ix, iy))
        return plate


__all__ = ["EyeGeometry", "EyeState", "EyePair", "EyeRasterizer",
           "BlinkScheduler", "GazeEngine", "blink_amount", "couple",
           "LID_GAZE_GAIN", "BROW_LID_LIFT", "BLINK_BROW_PULL"]
