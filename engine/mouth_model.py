"""
JEEVidya — Parametric Mouth with True Coarticulation (Terminal Plan, Part IV)
═════════════════════════════════════════════════════════════════════════════
Replaces sprite-over-sprite viseme blending (D5) with a continuous 5-D
parameter space and Cohen–Massaro DOMINANCE blending.

Why dominance beats linear mixing: naive `p = Σ wᵢ·target(vᵢ)` produces
AVERAGE mouths, not COARTICULATED mouths. Real speech anticipates — lips
round for "u" DURING the preceding "s" ("suno") — and a bilabial closure
resists being averaged away by its vowel neighbours. Dominance gives each
viseme class a per-parameter raised-cosine envelope with class-specific
width and peak: bilabials get near-total dominance on jaw/press (a
closure cannot be diluted); vowels get wide soft dominance on round/width
(they spread into neighbours — that IS anticipatory rounding).

Continuity guarantee: dominance envelopes are C¹ and the parameter→
contour map is C¹ ⇒ the mouth trajectory is provably continuous. The
"popping" bug class is eliminated by construction, then asserted by the
slew-rate clamp and the QC jerk metric.

Both blenders are shipped (Law 5): `blend_linear` and `blend_dominance`.
The QC viseme-discriminability metric decides which is default; the
expectation from phonetics is a decisive dominance win on closures.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from engine.visemes import V

PARAM_NAMES = ("jaw", "width", "round", "press", "pull")
N_PARAMS = len(PARAM_NAMES)

# Max articulator slew: ~12 cm/s lip velocity scaled to a normalized
# face ⇒ full jaw excursion floor-to-ceiling in no less than ~80 ms.
MAX_SLEW_PER_MS = 1.0 / 80.0

# Cache quantization (Part IV §4.3): jaw 1/64, others 1/32
_Q = (1 / 64, 1 / 32, 1 / 32, 1 / 32, 1 / 32)


@dataclass(frozen=True)
class MouthParams:
    """One point in the 5-D mouth space. All in [0,1] except pull ∈ [-1,1]."""
    jaw: float = 0.0     # inner aperture height, 0 = sealed
    width: float = 0.5   # commissure separation
    round: float = 0.0   # pucker/protrusion, aperture → circular
    press: float = 0.0   # lip compression (bilabial thickening)
    pull: float = 0.0    # commissure vertical pull (smile + / frown −)

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.jaw, self.width, self.round, self.press, self.pull)

    def quantized_key(self) -> Tuple[float, ...]:
        # _Q must track as_tuple()'s arity; strict=True makes adding a sixth
        # articulatory parameter without a quantum an immediate failure.
        return tuple(round(v / q) * q
                     for v, q in zip(self.as_tuple(), _Q, strict=True))

    def clamped(self) -> "MouthParams":
        c = lambda v: max(0.0, min(1.0, v))
        return MouthParams(c(self.jaw), c(self.width), c(self.round),
                           c(self.press), max(-1.0, min(1.0, self.pull)))

    @staticmethod
    def lerp(a: "MouthParams", b: "MouthParams", t: float) -> "MouthParams":
        return MouthParams(*(av + (bv - av) * t for av, bv
                             in zip(a.as_tuple(), b.as_tuple(), strict=True)))

    def distance(self, other: "MouthParams") -> float:
        return math.sqrt(sum((a - b) ** 2
                             for a, b in zip(self.as_tuple(),
                                             other.as_tuple(), strict=True)))


# ═══════════════════════════════════════════
# Articulatory target table (built-in defaults; rig v3 overrides these
# with targets least-squares-fitted from the character's viseme art)
# ═══════════════════════════════════════════

DEFAULT_TARGETS: Dict[V, MouthParams] = {
    V.REST:          MouthParams(0.00, 0.50, 0.00, 0.05, 0.00),
    V.BILABIAL:      MouthParams(0.00, 0.46, 0.05, 0.85, 0.00),
    V.LABIODENTAL:   MouthParams(0.08, 0.52, 0.00, 0.45, 0.02),
    V.DENTAL:        MouthParams(0.20, 0.56, 0.00, 0.10, 0.02),
    V.RETROFLEX:     MouthParams(0.30, 0.52, 0.08, 0.05, 0.00),
    V.OPEN_A:        MouthParams(1.00, 0.62, 0.05, 0.00, 0.00),
    V.MID_E:         MouthParams(0.55, 0.72, 0.00, 0.00, 0.12),
    V.CLOSED_I:      MouthParams(0.22, 0.85, 0.00, 0.05, 0.18),
    V.ROUNDED_TENSE: MouthParams(0.30, 0.28, 0.95, 0.15, -0.05),
    V.ROUNDED_LAX:   MouthParams(0.55, 0.36, 0.70, 0.05, -0.02),
}


# ═══════════════════════════════════════════
# Dominance table: per class, per parameter →
# (peak, forward_width_ms, backward_width_ms)
# ═══════════════════════════════════════════
#
# peak ∈ (0, 1]: how insistently this class claims the parameter.
# widths: how far (ms) its influence spreads into neighbours.
# Values from phonetics-literature defaults (Cohen & Massaro 1993 and
# articulatory-synthesis practice); tunable, shipped conservative.

_D = Dict[str, Tuple[float, float, float]]

DOMINANCE: Dict[V, _D] = {
    V.REST: {
        "jaw": (0.5, 60, 60), "width": (0.4, 60, 60), "round": (0.3, 60, 60),
        "press": (0.4, 60, 60), "pull": (0.4, 60, 60)},
    V.BILABIAL: {                       # closure is near-absolute on jaw/press
        "jaw": (1.0, 45, 45), "width": (0.35, 60, 60), "round": (0.2, 70, 70),
        "press": (1.0, 45, 45), "pull": (0.3, 60, 60)},
    V.LABIODENTAL: {
        "jaw": (0.9, 50, 50), "width": (0.4, 60, 60), "round": (0.25, 70, 70),
        "press": (0.9, 50, 50), "pull": (0.35, 60, 60)},
    V.DENTAL: {
        "jaw": (0.75, 55, 55), "width": (0.5, 65, 65), "round": (0.2, 80, 80),
        "press": (0.5, 55, 55), "pull": (0.4, 65, 65)},
    V.RETROFLEX: {
        "jaw": (0.7, 55, 55), "width": (0.45, 65, 65), "round": (0.35, 80, 80),
        "press": (0.4, 55, 55), "pull": (0.4, 65, 65)},
    V.OPEN_A: {                         # vowels: wide soft spread
        "jaw": (0.85, 90, 110), "width": (0.6, 100, 120), "round": (0.5, 110, 130),
        "press": (0.6, 90, 90), "pull": (0.5, 100, 100)},
    V.MID_E: {
        "jaw": (0.7, 90, 110), "width": (0.7, 100, 120), "round": (0.45, 110, 130),
        "press": (0.5, 90, 90), "pull": (0.6, 100, 100)},
    V.CLOSED_I: {
        "jaw": (0.65, 85, 100), "width": (0.85, 95, 115), "round": (0.5, 105, 125),
        "press": (0.5, 85, 85), "pull": (0.7, 95, 95)},
    V.ROUNDED_TENSE: {                  # rounding spreads FAR backward
        "jaw": (0.65, 85, 100), "width": (0.8, 100, 140), "round": (0.95, 110, 160),
        "press": (0.6, 85, 85), "pull": (0.6, 95, 95)},
    V.ROUNDED_LAX: {
        "jaw": (0.7, 85, 100), "width": (0.7, 100, 130), "round": (0.8, 110, 150),
        "press": (0.5, 85, 85), "pull": (0.5, 95, 95)},
}


def _envelope(t_ms: float, center_ms: float, peak: float,
              wf: float, wb: float) -> float:
    """Raised-cosine dominance envelope, C¹ everywhere (zero-derivative
    at center and at the ±width edges). Asymmetric: backward width `wb`
    (anticipation, t < center) may exceed forward width `wf` (carryover).
    """
    d = t_ms - center_ms
    w = wf if d >= 0 else wb
    if w <= 0:
        return peak if abs(d) < 1e-9 else 0.0
    x = abs(d) / w
    if x >= 1.0:
        return 0.0
    return peak * (0.5 + 0.5 * math.cos(math.pi * x))


# ═══════════════════════════════════════════
# Blenders
# ═══════════════════════════════════════════

@dataclass(frozen=True)
class Segment:
    """A timed viseme segment from the aligner (engine/align.py)."""
    viseme: V
    start_ms: float
    end_ms: float

    @property
    def center_ms(self) -> float:
        return 0.5 * (self.start_ms + self.end_ms)


class MouthTrack:
    """Timed segments → continuous 5-D parameter trajectory.

    mode='dominance' (default, Cohen–Massaro) or 'linear' (Law 5 gate
    comparison baseline).
    """

    def __init__(self, segments: Sequence[Segment],
                 targets: Optional[Dict[V, MouthParams]] = None,
                 mode: str = "dominance"):
        self.segments = sorted(segments, key=lambda s: s.start_ms)
        self.targets = dict(DEFAULT_TARGETS)
        if targets:
            self.targets.update(targets)
        if mode not in ("dominance", "linear"):
            raise ValueError(f"unknown blend mode: {mode}")
        self.mode = mode
        self._last: Optional[Tuple[float, MouthParams]] = None  # (t, params)

    # ─── core evaluation ──────────────────────────────────

    def params_at(self, t_ms: float, jaw_gate: float = 1.0) -> MouthParams:
        """The mouth at time t. `jaw_gate` (0..1, from the amplitude
        envelope) scales jaw only — silence closes the mouth even when
        the aligner claims speech (Tier-2 gating, Part VII §7.2)."""
        if self.mode == "dominance":
            p = self._blend_dominance(t_ms)
        else:
            p = self._blend_linear(t_ms)
        p = MouthParams(p.jaw * max(0.0, min(1.0, jaw_gate)),
                        p.width, p.round, p.press, p.pull).clamped()
        return self._slew_clamp(t_ms, p)

    def _blend_dominance(self, t_ms: float) -> MouthParams:
        """p_k = Σᵢ dᵢₖ(t)·targetᵢₖ / Σᵢ dᵢₖ(t) per parameter k."""
        num = [0.0] * N_PARAMS
        den = [0.0] * N_PARAMS
        # Only segments whose envelope can reach t matter (max width 160ms)
        for seg in self._near(t_ms, reach_ms=200.0):
            tgt = self.targets[seg.viseme].as_tuple()
            dom = DOMINANCE[seg.viseme]
            # envelope holds at peak across the segment interior; the
            # raised cosine shapes only the flanks outside [start, end]
            if seg.start_ms <= t_ms <= seg.end_ms:
                ref = t_ms  # inside: full peak
            elif t_ms < seg.start_ms:
                ref = seg.start_ms
            else:
                ref = seg.end_ms
            for k, name in enumerate(PARAM_NAMES):
                peak, wf, wb = dom[name]
                d = _envelope(t_ms, ref, peak, wf, wb)
                num[k] += d * tgt[k]
                den[k] += d
        rest = self.targets[V.REST].as_tuple()
        vals = [num[k] / den[k] if den[k] > 1e-9 else rest[k]
                for k in range(N_PARAMS)]
        return MouthParams(*vals)

    def _blend_linear(self, t_ms: float) -> MouthParams:
        """Baseline: triangular-window weighted average (the naive mixer
        the dominance model is gated against)."""
        wsum = 0.0
        acc = [0.0] * N_PARAMS
        for seg in self._near(t_ms, reach_ms=120.0):
            half = max(20.0, (seg.end_ms - seg.start_ms) / 2)
            d = abs(t_ms - seg.center_ms)
            w = max(0.0, 1.0 - d / (half + 60.0))
            if w <= 0:
                continue
            for k, v in enumerate(self.targets[seg.viseme].as_tuple()):
                acc[k] += w * v
            wsum += w
        if wsum <= 1e-9:
            return self.targets[V.REST]
        return MouthParams(*(a / wsum for a in acc))

    # ─── helpers ──────────────────────────────────────────

    def _near(self, t_ms: float, reach_ms: float) -> List[Segment]:
        return [s for s in self.segments
                if s.start_ms - reach_ms <= t_ms <= s.end_ms + reach_ms]

    def _slew_clamp(self, t_ms: float, p: MouthParams) -> MouthParams:
        """No physically impossible jumps: each parameter's per-ms delta
        is clamped to MAX_SLEW_PER_MS (≈ full excursion in 80 ms).
        Stateful; callers must evaluate in forward time order (the
        renderer does; QC re-evaluates statelessly)."""
        if self._last is None:
            self._last = (t_ms, p)
            return p
        t_prev, prev = self._last
        dt = t_ms - t_prev
        if dt <= 0:          # out-of-order query (QC probes): stateless
            return p
        max_d = MAX_SLEW_PER_MS * dt
        vals = []
        for a, b in zip(prev.as_tuple(), p.as_tuple(), strict=True):
            d = b - a
            vals.append(a + max(-max_d, min(max_d, d)))
        out = MouthParams(*vals)
        self._last = (t_ms, out)
        return out

    def reset_state(self) -> None:
        self._last = None


# ═══════════════════════════════════════════
# Parameter → contour (C¹ map used by the rasterizer and by QC's
# discriminability metric)
# ═══════════════════════════════════════════

def lip_contour(p: MouthParams, n: int = 48
                ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Normalized outer/inner lip contours for a parameter vector.

    Face-normalized space: mouth box is [-1, 1] × [-1, 1]. The renderer
    maps this through the rig's registered transform; QC uses it raw for
    class-separation measurements. C¹ in p by construction (all terms
    polynomial/trig in p's components).
    """
    half_w = 0.35 + 0.55 * p.width - 0.28 * p.round
    aperture = 0.05 + 0.85 * p.jaw * (1.0 - 0.65 * p.press)
    protrude = 0.25 * p.round
    corner_y = -0.30 * p.pull

    outer: List[Tuple[float, float]] = []
    inner: List[Tuple[float, float]] = []
    for i in range(n):
        a = 2 * math.pi * i / n
        c = math.cos(a)
        s = math.sin(a)
        # superellipse-ish blend: rounding pushes toward a circle
        shape = 1.0 - 0.35 * p.round * (1 - abs(c))
        x = half_w * c * shape
        thick = 0.16 * (1.0 + 0.8 * p.press) * (1.0 - 0.3 * p.round)
        y_out = (aperture / 2 + thick) * s + corner_y * c * c
        y_in = (aperture / 2) * s * (1.0 - 0.55 * p.press) + corner_y * c * c
        outer.append((x, y_out + protrude * 0.1 * s))
        inner.append((x * (1.0 - 0.12), y_in))
    return outer, inner


def contour_distance(a: MouthParams, b: MouthParams, n: int = 48) -> float:
    """Mean point-to-point distance between two parameter vectors'
    contours — the metric behind the viseme-discriminability gate
    (Part VIII): every class pair must exceed a separation threshold
    at phone scale."""
    ao, ai = lip_contour(a, n)
    bo, bi = lip_contour(b, n)
    d = 0.0
    # Both contours are generated at the same n; strict=True guards the
    # metric against ever comparing partial point sets.
    for (ax, ay), (bx, by) in zip(ao + ai, bo + bi, strict=True):
        d += math.hypot(ax - bx, ay - by)
    return d / (2 * n)


# ═══════════════════════════════════════════
# Parameter LRU (speech revisits similar shapes; hit rate stays high)
# ═══════════════════════════════════════════

class ParamLRU:
    """Rasterization cache keyed by quantized 5-D vectors."""

    def __init__(self, cap: int = 256):
        self._d: OrderedDict = OrderedDict()
        self.cap = cap
        self.hits = 0
        self.misses = 0

    def get_or(self, p: MouthParams, factory):
        key = p.quantized_key()
        if key in self._d:
            self._d.move_to_end(key)
            self.hits += 1
            return self._d[key]
        self.misses += 1
        val = factory()
        self._d[key] = val
        if len(self._d) > self.cap:
            self._d.popitem(last=False)
        return val

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0
