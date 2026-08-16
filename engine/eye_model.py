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
import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

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


# ═══════════════════════════════════════════
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


def _aperture_margins(geo: EyeGeometry, n: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Upper and lower rim of the MEASURED aperture, as x-ordered arcs.

    Deriving the two margins from the aperture itself (rather than from
    MediaPipe's polylines, even rescaled) guarantees the sweep starts on
    the drawn eye's true top edge and ends on its true bottom edge. That
    is what makes blink=1 fully occlude the iris: the closing path
    becomes the aperture's own lower rim, so there is nothing left of
    the opening for an iris pixel to survive in.

    For each of `n` columns spanning the aperture, take the topmost and
    bottommost boundary point. A closed contour always yields both, so
    the two arcs together re-describe the aperture exactly.
    """
    ap = np.asarray(geo.clip, dtype=np.float64)
    x0, x1 = float(ap[:, 0].min()), float(ap[:, 0].max())
    if x1 - x0 < 1e-6:
        up = _resample(geo.lid_upper, n)
        return up, _resample(geo.lid_lower, n)

    # Densify the contour so every column contains samples.
    closed = np.vstack([ap, ap[:1]])
    dense = _resample(closed, max(4 * n, 256))
    xs = np.linspace(x0, x1, n)
    half = (x1 - x0) / max(n - 1, 1)
    upper = np.empty((n, 2))
    lower = np.empty((n, 2))
    for i, x in enumerate(xs):
        band = dense[np.abs(dense[:, 0] - x) <= half]
        if len(band) == 0:                       # column fell in a notch
            j = int(np.argmin(np.abs(dense[:, 0] - x)))
            band = dense[j:j + 1]
        upper[i] = (x, float(band[:, 1].min()))
        lower[i] = (x, float(band[:, 1].max()))
    return upper, lower


def _lid_path(geo: EyeGeometry, closure: float, n: int = 24) -> np.ndarray:
    """Upper-lid margin at `closure` ∈ [0,1].

    Interpolates the upper margin toward the LOWER margin along an arc
    concentric with the eyeball, so at closure=1 the path IS the lower
    margin — full occlusion by identity (D9).

    On a measured eye both margins come from the aperture, so "the lower
    margin" is the drawn eye's own bottom rim. The MediaPipe polylines
    described a band across the middle third of the drawn eye, which is
    why a fully-closed blink used to leave the iris plainly visible.
    """
    if geo.measured:
        up, lo = _aperture_margins(geo, n)
    else:
        up = _resample(geo.lid_upper, n)
        lo = _resample(geo.lid_lower, n)
    c = np.asarray(geo.iris_c)
    # Concentric-arc interpolation: lerp radius + angle about the eyeball
    # center rather than straight lines, so the lid margin stays curved.
    vu, vl = up - c, lo - c
    ru, rl = np.linalg.norm(vu, axis=1), np.linalg.norm(vl, axis=1)
    au, al = np.arctan2(vu[:, 1], vu[:, 0]), np.arctan2(vl[:, 1], vl[:, 0])
    # unwrap angle difference to the short way
    da = (al - au + math.pi) % (2 * math.pi) - math.pi
    t = max(0.0, min(1.0, closure))
    r = ru + (rl - ru) * t
    a = au + da * t
    return c + np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


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
        w = int(math.ceil((self._x1 - self._x0) * S))
        h = int(math.ceil((self._y1 - self._y0) * S))
        img = Image.new("RGBA", (max(w, 2), max(h, 2)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        def T(pts: np.ndarray) -> List[Tuple[float, float]]:
            return [((x - self._x0) * S, (y - self._y0) * S) for x, y in pts]

        geo = self.geo
        clip_poly = _resample(np.vstack([np.asarray(geo.clip, dtype=np.float64),
                                        np.asarray(geo.clip[:1],
                                                   dtype=np.float64)]), 96) \
            if geo.measured else _resample(geo.socket, 48)

        # 0 · The aperture clip. EVERY pixel this method paints is masked
        # to it at the end, so no stroke can reach the patch border. This
        # is the structural fix for the hard-edged skin rectangle: the
        # lid's fill used to be bounded only by the patch rectangle, so a
        # blink painted a rectangle of skin over the brow. A stroke that
        # cannot leave the drawn eye cannot draw a rectangle.
        clip_mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(clip_mask).polygon(T(clip_poly), fill=255)

        # 1 · The eyeball, offset by gaze and clipped to the aperture.
        # Gaze travels in iris radii, so it scales with the DRAWN eye.
        gaze = np.array([state.eye_dx, state.eye_dy]) * geo.iris_r * 0.55
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
            function is what keeps the eyeball, the socket behind it and
            the lid over it in register at any render scale."""
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
        # Both are painted with a generous fill and then masked to the
        # aperture in step 4, so their extent is bounded by the drawn eye
        # rather than by the polygon we happen to hand PIL.
        skin = self._color("skin", (232, 190, 160))
        _, ap_lower = (_aperture_margins(geo, 24) if geo.measured
                       else (None, _resample(geo.lid_lower, 24)))
        if state.squint > 0.01:
            lo_path = _lid_path(geo, 1.0 - 0.18 * state.squint)
            floor = ap_lower.copy()
            floor[:, 1] += max(geo.iris_r, 2.0) * 2.0
            draw.polygon(T(np.vstack([lo_path, floor[::-1]])),
                         fill=skin + (255,))

        closure = max(0.0, min(1.0, blink))
        lid = _lid_path(geo, closure)
        # Lid polygon: the margin plus a cap well above the aperture top.
        # The cap only needs to be tall enough to cover the whole opening;
        # the aperture mask decides where it actually lands.
        cap = lid.copy()
        cap[:, 1] -= (self._y1 - self._y0) + max(geo.iris_r, 2.0)

        if self.lid is not None:
            # ART LID — the artist's own eyelid skin, slid down over the
            # eye. Two things are combined, and both are needed:
            #
            #   pixels : the baked strip, shifted down by `closure` × the
            #            aperture's height. It was cut from just above the
            #            aperture, so at closure 1 it lands exactly on the
            #            opening, and the skin arriving at the leading edge
            #            is the crease and lash the artist actually drew.
            #   shape  : the region above the CURVED `_lid_path` margin.
            #            The strip's own bottom edge is a straight row, and
            #            a straight-edged lid at half closure is the single
            #            most obvious tell that a blink is fake. Masking by
            #            the margin gives the closing edge the drawn eye's
            #            own curvature, and at closure 1 that margin IS the
            #            aperture's lower rim — so the eye is fully covered
            #            by geometric identity, not by a tuned offset.
            ap_np = np.asarray(geo.clip, dtype=np.float64)
            travel = float(ap_np[:, 1].max() - ap_np[:, 1].min()) * closure
            lid_layer = art_layer(self.lid, geo.lid_origin, (0.0, travel))
            shape = Image.new("L", img.size, 0)
            ImageDraw.Draw(shape).polygon(T(np.vstack([lid, cap[::-1]])),
                                          fill=255)
            lid_layer.putalpha(ImageChops.multiply(lid_layer.split()[3], shape))
            img.alpha_composite(lid_layer)
        else:
            # SYNTHETIC FALLBACK — flat skin, for rigs with no baked lid.
            draw.polygon(T(np.vstack([lid, cap[::-1]])), fill=skin + (255,))
            # Crease + lash line on the leading edge
            crease = self._color("lip_shadow", (150, 100, 84))
            draw.line(T(lid), fill=crease + (160,), width=max(1, int(S * 0.9)))
            lash = self._color("lash", (38, 26, 24))
            draw.line(T(lid), fill=lash + (255,), width=max(1, int(S * 1.6)))

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
