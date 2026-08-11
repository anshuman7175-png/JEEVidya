"""
JEEVidya — The Motion Grammar (Singularity Plan, Part XVI)
══════════════════════════════════════════════════════════
The twelve principles, encoded as a deterministic post-processor over
EVERY animation channel (head, brows, gaze, gestures, body) before
rendering. Human animators do this by instinct; here it is mechanical:

  • SecondOrderFilter  — follow-through & settle: every keyframe target
    is reached via a slightly-underdamped 2nd-order system (ζ ≈ 0.85):
    motions *arrive* with 1–2% overshoot and settle. The difference
    between tweened and alive.
  • AnticipationInjector — any motion above a velocity threshold gets an
    automatic small counter-motion first (head dips before it rises),
    amplitude ∝ main motion, capped, C¹-blended.
  • VerletChain        — spring-damper secondary motion for hair/costume:
    chains of 3–5 verlet points driven by head acceleration; follow
    through, overlap, settle with critically-damped wobble.
  • OUProcess          — Ornstein–Uhlenbeck idle noise replacing sines:
    mean-reverting, never visibly loops. Sines read as robotic in 10 s.
  • arc_lerp           — gaze/head targets travel along slight arcs,
    never straight lines.
  • smear decision     — directional blur taps when head angular velocity
    exceeds threshold, so fast turns stop strobing at 60 fps.
  • jerk/settle QC     — the gates of §XVI, exported for tools/face_qc.

Determinism (Law 4): every stochastic element is seeded; the physics
runs at a FIXED timestep derived from settings.FPS. Same inputs →
bit-identical channel outputs, covered by the frame-hash ledger.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import settings

# ═══════════════════════════════════════════
# Follow-through & settle (2nd-order filter)
# ═══════════════════════════════════════════


class SecondOrderFilter:
    """y follows x through a critically-tunable 2nd-order system.

    Parameterized the animator-friendly way:
        f     natural frequency (Hz) — how fast it responds
        zeta  damping ratio — < 1 overshoots (0.85 default: 1–2% overshoot)
        r     initial response — 0 smooth start, > 0 anticipates

    Semi-implicit Euler at the fixed frame timestep; unconditionally
    stable for the k-values used here and fully deterministic.
    """

    def __init__(self, f: float = 2.2, zeta: float = 0.85, r: float = 0.0,
                 x0: float = 0.0):
        w = 2.0 * math.pi * f
        self.k1 = zeta / (math.pi * f)
        self.k2 = 1.0 / (w * w)
        self.k3 = r * zeta / w
        self.y = x0
        self.yd = 0.0
        self._xp = x0

    def step(self, x: float, dt: Optional[float] = None) -> float:
        dt = dt if dt is not None else 1.0 / settings.FPS
        xd = (x - self._xp) / dt if dt > 0 else 0.0
        self._xp = x
        # clamp k2 for stability at large dt (draft 30 fps renders)
        k2 = max(self.k2, dt * dt / 2.0 + dt * self.k1 / 2.0, dt * self.k1)
        self.y = self.y + dt * self.yd
        self.yd = self.yd + dt * (x + self.k3 * xd - self.y
                                  - self.k1 * self.yd) / k2
        return self.y

    def reset(self, x0: float = 0.0) -> None:
        self.y = x0
        self.yd = 0.0
        self._xp = x0


# ═══════════════════════════════════════════
# Anticipation injector
# ═══════════════════════════════════════════


def inject_anticipation(track: np.ndarray, fps: Optional[int] = None,
                        velocity_threshold_ps: float = 1.5,
                        anticipation_frac: float = 0.12,
                        max_amp_frac: float = 0.25) -> np.ndarray:
    """Prepend a counter-motion before every large move in a sampled
    channel track (one value per frame).

    A 'move' is a contiguous run where |velocity| exceeds the per-second
    threshold. For each move we blend a raised-cosine dip of opposite
    sign into the 2–4 frames before its onset. Amplitude is
    `anticipation_frac` of the move's total delta, capped at
    `max_amp_frac` of the channel's dynamic range. C¹ by construction
    (raised cosine starts and ends with zero slope).
    """
    fps = fps or settings.FPS
    n = len(track)
    if n < 6:
        return track.copy()
    out = track.astype(np.float64).copy()
    v = np.gradient(out) * fps                     # per-second velocity
    rng_amp = float(np.ptp(out)) or 1.0
    fast = np.abs(v) > velocity_threshold_ps

    lead = max(2, min(4, settings.frames(3.0 / 60.0, fps)))  # 2–4 frames
    i = 1
    while i < n:
        if fast[i] and not fast[i - 1]:            # onset of a move
            j = i
            while j < n and fast[j]:
                j += 1
            delta = out[min(j, n - 1)] - out[i]
            amp = -math.copysign(
                min(abs(delta) * anticipation_frac, rng_amp * max_amp_frac),
                delta)
            s = max(0, i - lead)
            for k in range(s, i):
                t = (k - s + 1) / (i - s)          # 0→1 across the lead-in
                w = 0.5 - 0.5 * math.cos(2.0 * math.pi * t)  # up & back down
                out[k] += amp * w
            i = j
        else:
            i += 1
    return out


# ═══════════════════════════════════════════
# Verlet chains — hair / costume secondary motion
# ═══════════════════════════════════════════


@dataclass
class VerletChain:
    """3–5 point chain hanging from an anchor that follows the head.
    Driven purely by anchor acceleration + gravity; segment lengths are
    hard constraints (2 relaxation passes). Deterministic fixed timestep.
    """
    n_points: int = 4
    segment_len: float = 12.0            # px in head space
    damping: float = 0.88                # velocity retained per step
    gravity: float = 340.0               # px/s² in head space
    stiffness_passes: int = 2
    sway_gain: float = 1.0               # how strongly anchor accel couples

    def __post_init__(self):
        ys = np.arange(self.n_points, dtype=np.float64) * self.segment_len
        self.pos = np.stack([np.zeros(self.n_points), ys], axis=1)
        self.prev = self.pos.copy()
        self._anchor_prev = np.zeros(2)
        self._anchor_vel_prev = np.zeros(2)

    def step(self, anchor_xy: Tuple[float, float],
             dt: Optional[float] = None) -> np.ndarray:
        """Advance one frame; returns (n_points, 2) offsets RELATIVE to
        the anchor — composite these under the hair layer."""
        dt = dt if dt is not None else 1.0 / settings.FPS
        anchor = np.asarray(anchor_xy, dtype=np.float64)
        vel = (anchor - self._anchor_prev) / dt
        accel = (vel - self._anchor_vel_prev) / dt
        self._anchor_prev = anchor
        self._anchor_vel_prev = vel

        # Verlet integrate free points (index 0 is pinned to the anchor)
        force = np.array([-accel[0] * self.sway_gain, self.gravity])
        cur = self.pos.copy()
        self.pos[1:] = (cur[1:] + (cur[1:] - self.prev[1:]) * self.damping
                        + force * dt * dt)
        self.prev = cur
        self.pos[0] = anchor

        # Distance constraints
        for _ in range(self.stiffness_passes):
            for i in range(1, self.n_points):
                d = self.pos[i] - self.pos[i - 1]
                dist = float(np.linalg.norm(d)) or 1e-9
                corr = d * (1.0 - self.segment_len / dist)
                if i == 1:
                    self.pos[i] -= corr
                else:
                    self.pos[i] -= corr * 0.5
                    self.pos[i - 1] += corr * 0.5
        return self.pos - anchor

    def settled(self, tol: float = 0.05) -> bool:
        return bool(np.max(np.abs(self.pos - self.prev)) < tol)


# ═══════════════════════════════════════════
# OU idle noise — sines are robotic, OU never repeats
# ═══════════════════════════════════════════


class OUProcess:
    """Seeded Ornstein–Uhlenbeck process: dx = θ(μ−x)dt + σ dW.
    Mean-reverting noise for breathe wobble / sway / gaze drift that
    never visibly loops. Exact discretization → step-size independent
    statistics, deterministic per seed (Law 4)."""

    def __init__(self, theta: float = 1.4, sigma: float = 1.0,
                 mu: float = 0.0, seed: int = 0, x0: Optional[float] = None):
        self.theta, self.sigma, self.mu = theta, sigma, mu
        self.rng = np.random.default_rng(seed)
        self.x = mu if x0 is None else x0

    def step(self, dt: Optional[float] = None) -> float:
        dt = dt if dt is not None else 1.0 / settings.FPS
        e = math.exp(-self.theta * dt)
        var = (self.sigma ** 2) * (1.0 - e * e) / (2.0 * self.theta)
        self.x = self.mu + (self.x - self.mu) * e \
            + math.sqrt(max(var, 0.0)) * self.rng.standard_normal()
        return self.x

    def track(self, n_frames: int, dt: Optional[float] = None) -> np.ndarray:
        return np.array([self.step(dt) for _ in range(n_frames)])


# ═══════════════════════════════════════════
# Arc enforcement
# ═══════════════════════════════════════════


def arc_lerp(p0: Tuple[float, float], p1: Tuple[float, float], t: float,
             bow: float = 0.12) -> Tuple[float, float]:
    """Interpolate between two 2-D targets along a slight quadratic arc
    (bow ∝ distance, perpendicular to travel). Straight-line
    interpolation is the single biggest 'computer did this' tell."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return p1
    # perpendicular unit, biased upward (screen-space y-down)
    px, py = -dy / dist, dx / dist
    if py > 0:
        px, py = -px, -py
    h = bow * dist * 4.0 * t * (1.0 - t)          # quadratic bump, 0 at ends
    return (x0 + dx * t + px * h, y0 + dy * t + py * h)


# ═══════════════════════════════════════════
# Smear discipline
# ═══════════════════════════════════════════


def smear_taps(angular_velocity_dps: float,
               threshold_dps: float = 240.0,
               max_taps: int = 3) -> int:
    """How many directional blur taps the head layer should get this
    frame. 0 during held poses — smear on a hold is a QC violation."""
    if abs(angular_velocity_dps) <= threshold_dps:
        return 0
    over = min(abs(angular_velocity_dps) / threshold_dps - 1.0, 2.0)
    return min(max_taps, 1 + int(over * 1.5))


def apply_directional_smear(img, dx: float, dy: float, taps: int):
    """2–3 tap directional box blur on a PIL RGBA layer, motion-aligned.
    Cheap: sums shifted copies. Only ever called on the head layer."""
    if taps <= 0:
        return img
    from PIL import Image
    base = np.asarray(img, dtype=np.float32)
    acc = base.copy()
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    step = max(1.0, norm / (taps + 1))
    for i in range(1, taps + 1):
        sx, sy = int(round(-ux * step * i)), int(round(-uy * step * i))
        shifted = np.roll(np.roll(base, sy, axis=0), sx, axis=1)
        acc += shifted
    acc /= (taps + 1)
    return Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8), "RGBA")


# ═══════════════════════════════════════════
# The grammar — one object per character, all channels
# ═══════════════════════════════════════════

# Per-channel tuning: (frequency Hz, zeta). Head is weightier than brows.
CHANNEL_TUNING: Dict[str, Tuple[float, float]] = {
    "head_tilt": (2.0, 0.85), "head_nod": (2.4, 0.85),
    "head_yaw": (1.8, 0.88), "sway": (1.2, 0.90), "bounce": (2.2, 0.85),
    "brow": (3.5, 0.80), "eye_dx": (6.0, 0.95), "eye_dy": (6.0, 0.95),
    "gesture": (2.6, 0.82), "breathe": (0.8, 0.95),
}


class MotionGrammar:
    """Stateful per-character post-processor. Call `filter(channel, x)`
    once per frame per channel, in a fixed channel order (dict iteration
    over CHANNEL_TUNING is insertion-ordered → deterministic)."""

    def __init__(self, seed: int = 0):
        self.filters: Dict[str, SecondOrderFilter] = {
            name: SecondOrderFilter(f=f, zeta=z)
            for name, (f, z) in CHANNEL_TUNING.items()}
        self.idle = {
            "breathe": OUProcess(theta=0.9, sigma=0.35, seed=seed * 31 + 1),
            "sway": OUProcess(theta=0.7, sigma=0.5, seed=seed * 31 + 2),
            "gaze_drift": OUProcess(theta=2.2, sigma=0.15, seed=seed * 31 + 3),
        }
        self.hair = VerletChain()
        self._history: Dict[str, List[float]] = {}

    def filter(self, channel: str, target: float,
               dt: Optional[float] = None) -> float:
        f = self.filters.get(channel)
        if f is None:
            f = self.filters[channel] = SecondOrderFilter()
        y = f.step(target, dt)
        self._history.setdefault(channel, []).append(y)
        if len(self._history[channel]) > 240:
            self._history[channel] = self._history[channel][-240:]
        return y

    def idle_offset(self, kind: str, scale: float = 1.0,
                    dt: Optional[float] = None) -> float:
        return self.idle[kind].step(dt) * scale

    def reset(self) -> None:
        for f in self.filters.values():
            f.reset()
        self._history.clear()


# ═══════════════════════════════════════════
# QC gates (§XVI) — consumed by tools/face_qc.py + tests
# ═══════════════════════════════════════════


def jerk_budget(track: Sequence[float], fps: Optional[int] = None) -> float:
    """Max |third derivative| of a sampled channel, units/s³. The gate
    bounds this per channel — jerk spikes are what eyes read as 'pop'."""
    x = np.asarray(track, dtype=np.float64)
    if len(x) < 4:
        return 0.0
    f = float(fps or settings.FPS)
    return float(np.max(np.abs(np.diff(x, n=3))) * f ** 3)


def settle_cycles(track: Sequence[float], tail_frames: int = 60) -> int:
    """Zero-crossing count of the detrended tail — a channel that
    oscillates more than 3 visible cycles after arrival fails the settle
    gate (underdamped ζ drifted too low)."""
    x = np.asarray(track, dtype=np.float64)[-tail_frames:]
    if len(x) < 8:
        return 0
    d = x - x.mean()
    amp = np.max(np.abs(d))
    if amp < 1e-6:
        return 0
    sig = d / amp
    crossings = int(np.sum(np.abs(np.diff(np.signbit(
        np.where(np.abs(sig) < 0.02, 0.0, sig)))) > 0))
    return crossings // 2


def verify_motion(tracks: Dict[str, Sequence[float]],
                  fps: Optional[int] = None,
                  jerk_limits: Optional[Dict[str, float]] = None) -> List[str]:
    """Run the §XVI gates over per-channel frame tracks. Returns a list
    of violations (empty = green). Limits are per-second-cubed and scale
    with FPS automatically because jerk_budget converts units."""
    default_limit = 6.0e4
    limits = jerk_limits or {}
    violations: List[str] = []
    for name, track in tracks.items():
        j = jerk_budget(track, fps)
        lim = limits.get(name, default_limit)
        if j > lim:
            violations.append(
                f"jerk budget: channel '{name}' jerk {j:.3g} > {lim:.3g}")
        c = settle_cycles(track)
        if c > 3:
            violations.append(
                f"settle: channel '{name}' oscillates {c} cycles (> 3)")
    return violations


__all__ = ["SecondOrderFilter", "inject_anticipation", "VerletChain",
           "OUProcess", "arc_lerp", "smear_taps",
           "apply_directional_smear", "MotionGrammar", "CHANNEL_TUNING",
           "jerk_budget", "settle_cycles", "verify_motion"]
