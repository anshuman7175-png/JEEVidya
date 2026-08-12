"""
JEEVidya V5 — Cinematic Camera Dynamics
═══════════════════════════════════════
The single biggest tell of amateur motion graphics is LINEAR MOVEMENT.
This module makes the camera a physical object:

  • Critically-damped springs (second-order dynamics) smooth every
    parameter — position, scale, opacity — with natural overshoot on
    cuts and zero overshoot on settles. The same math Apple/AE use.
  • The camera is NEVER still: layered incommensurate sines produce
    organic hand-held micro-drift (a poor man's Perlin, deterministic).
  • Every shot gets a slow breathing push-in (documentary zoom): a
    static frame that is still 1%/second alive.
  • Cuts carry energy: punch-in impulses, directional whip-blur frames,
    and decaying shake impulses on emphasis beats.

Everything is a pure function of (frame, seed) — Tier 0 cache-safe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image


# ═══════════════════════════════════════════
# SECOND-ORDER SPRING (the professional easing)
# ═══════════════════════════════════════════

class Spring:
    """Critically-damped-ish spring. zeta<1 → cinematic overshoot."""

    def __init__(self, value: float, omega: float = 10.0, zeta: float = 0.85,
                 fps: int = 30):
        self.x = float(value)
        self.v = 0.0
        self.omega = omega          # responsiveness (rad/s)
        self.zeta = zeta            # damping: 0.7-0.9 = tasteful overshoot
        self.dt = 1.0 / fps

    def snap(self, value: float) -> None:
        self.x, self.v = float(value), 0.0

    def kick(self, impulse: float) -> None:
        self.v += impulse

    def step(self, target: float) -> float:
        # Semi-implicit Euler on x'' = ω²(t−x) − 2ζωx'
        a = self.omega * self.omega * (target - self.x) \
            - 2.0 * self.zeta * self.omega * self.v
        self.v += a * self.dt
        self.x += self.v * self.dt
        return self.x


class SpringVec:
    """A dict of named springs stepped toward a dict of targets."""

    def __init__(self, values: Dict[str, float], omega: float = 10.0,
                 zeta: float = 0.85, fps: int = 30):
        self.springs = {k: Spring(v, omega, zeta, fps)
                        for k, v in values.items()}

    def snap(self, values: Dict[str, float]) -> None:
        for k, v in values.items():
            if k in self.springs:
                self.springs[k].snap(v)

    def step(self, targets: Dict[str, float]) -> Dict[str, float]:
        return {k: s.step(targets.get(k, s.x))
                for k, s in self.springs.items()}


# ═══════════════════════════════════════════
# ORGANIC MICRO-DRIFT (never a dead frame)
# ═══════════════════════════════════════════

def micro_drift(frame: int, seed: int, amp: float = 3.0
                ) -> Tuple[float, float, float]:
    """(dx, dy, d_zoom): layered incommensurate sines ≈ hand-held life.
    Deterministic in (frame, seed)."""
    p = (seed % 977) * 0.618
    t = frame
    dx = (math.sin(t * 0.023 + p) * 0.6 + math.sin(t * 0.071 + p * 2) * 0.3
          + math.sin(t * 0.013 + p * 3) * 0.5) * amp
    dy = (math.sin(t * 0.019 + p * 4) * 0.5 + math.sin(t * 0.083 + p * 5) * 0.25
          + math.sin(t * 0.011 + p * 6) * 0.55) * amp * 0.8
    dz = (math.sin(t * 0.009 + p * 7) * 0.5 + 0.5) * 0.006   # 0..0.6% zoom sway
    return dx, dy, dz


# ═══════════════════════════════════════════
# THE CAMERA
# ═══════════════════════════════════════════

@dataclass
class CutState:
    frames_since_cut: int = 9999
    direction: float = 0.0          # −1 left, +1 right (whip direction)
    punch: float = 0.0              # punch-in strength at the cut


class CameraDynamics:
    """Spring-smoothed shot parameters + push-in + drift + impacts.

    Feed it the PRESET (target) params per character each frame; it
    returns physically-smoothed params plus a global frame transform.
    """

    # Documentary push-in: asymptotic, NOT linear. A linear 1%/s zoom is
    # unbounded — on a 60 s span it reached ×1.6, scaling anchor points
    # away from center until heads (and whole characters) left the frame.
    # The exponential approach keeps the frame alive with the same
    # initial feel but can never exceed PUSH_IN_MAX.
    PUSH_IN_MAX = 0.045             # ≤ 4.5% total zoom, ever
    PUSH_IN_TAU_S = 9.0             # time constant: ~63% there at 9 s

    def __init__(self, width: int, height: int, seed: int = 0,
                 fps: int = 30, energy: float = 0.6):
        self.width, self.height = width, height
        self.fps = fps
        self.seed = seed
        self.energy = energy
        self._chars: Dict[str, SpringVec] = {}
        self.cut = CutState()
        self._shake = Spring(0.0, omega=22.0, zeta=0.25, fps=fps)
        self._frame = 0

    # ─── Lifecycle per segment ─────────────────────────────

    def begin_segment(self, prev_params: Dict[str, Dict[str, float]],
                      seed: int) -> None:
        """Seed springs AT the previous shot's params so the cut ANIMATES
        into the new shot (springs overshoot tastefully)."""
        self.seed = seed
        self._frame = 0
        for key, params in prev_params.items():
            if key not in self._chars:
                self._chars[key] = SpringVec(params, omega=11.0, zeta=0.8,
                                             fps=self.fps)
            else:
                self._chars[key].snap(params)
        self._shake.snap(0.0)

    def on_cut(self, from_x: float, to_x: float, hard: bool = True) -> None:
        self.cut = CutState(
            frames_since_cut=0,
            direction=math.copysign(1.0, to_x - from_x) if to_x != from_x else 0.0,
            punch=0.05 + 0.05 * self.energy if hard else 0.0)

    def impulse(self, strength: float = 1.0) -> None:
        """Emphasis beat: decaying shake kick (numbers, reveals)."""
        self._shake.kick(strength * 55.0)

    # ─── Per-frame ─────────────────────────────────────────

    def smooth_char(self, key: str, target: Dict[str, float]
                    ) -> Dict[str, float]:
        sv = self._chars.get(key)
        if sv is None:
            sv = self._chars[key] = SpringVec(target, omega=11.0, zeta=0.8,
                                              fps=self.fps)
        return sv.step(target)

    def frame_transform(self) -> Dict[str, float]:
        """Global (dx, dy, zoom, blur) for this frame; call once/frame."""
        f = self._frame
        self._frame += 1
        self.cut.frames_since_cut += 1

        dx, dy, dz = micro_drift(f, self.seed, amp=2.0 + 2.5 * self.energy)
        push = 1.0 + self.PUSH_IN_MAX * (
            1.0 - math.exp(-(f / self.fps) / self.PUSH_IN_TAU_S))

        # Punch-in decays over ~8 frames after a cut
        k = self.cut.frames_since_cut
        punch = self.cut.punch * math.exp(-k / 4.0) if k < 20 else 0.0

        shake = self._shake.step(0.0)
        dx += shake * 0.10
        dy += shake * 0.06

        # Whip blur amount for the first 3 frames of a directional cut
        blur = 0.0
        if k < 3 and abs(self.cut.direction) > 0:
            blur = (3 - k) / 3.0

        return {"dx": dx, "dy": dy, "zoom": push + punch + dz,
                "whip_blur": blur, "whip_dir": self.cut.direction}


# ═══════════════════════════════════════════
# WHIP-PAN DIRECTIONAL BLUR (numpy, 3 rolls)
# ═══════════════════════════════════════════

def whip_blur(frame: Image.Image, amount: float,
              direction: float = 1.0) -> Image.Image:
    """Horizontal motion-blur streak for cut frames. amount 0..1."""
    if amount <= 0.01:
        return frame
    arr = np.asarray(frame.convert("RGB"), dtype=np.float32)
    shift = max(2, int(28 * amount))
    s = int(math.copysign(shift, direction))
    acc = arr.copy()
    acc += np.roll(arr, s // 2, axis=1)
    acc += np.roll(arr, s, axis=1)
    acc += np.roll(arr, -s // 3, axis=1)
    return Image.fromarray((acc / 4.0).astype(np.uint8))


def apply_frame_transform(pos_x: float, pos_y: float, scale: float,
                          xform: Dict[str, float],
                          cx: float, cy: float) -> Tuple[float, float, float]:
    """Apply the global camera transform to a world-space anchor point."""
    zoom = xform["zoom"]
    x = cx + (pos_x - cx) * zoom + xform["dx"]
    y = cy + (pos_y - cy) * zoom + xform["dy"]
    return x, y, scale * zoom
