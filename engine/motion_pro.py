"""
JEEVidya V5 — Character Motion (the Disney layer)
═════════════════════════════════════════════════
Amplitude-flapping cutouts read as AI. Characters with WEIGHT read as
animated. This module gives every character second-order body dynamics
and the classical animation principles, derived from physics rather
than keyframes:

  squash & stretch   vertical velocity compresses/extends the sprite
                     (volume-conserving: sx·sy ≈ 1)
  lean               horizontal acceleration tilts the body into the
                     move (anticipation) and past it (follow-through)
  asymmetric breath  inhale 40% / exhale 60% of the cycle — metronomic
                     sine breathing is the #2 amateur tell
  speech energy      voice loudness feeds a spring, so the body surges
                     INTO stressed syllables and settles after — with
                     lag (follow-through), never instantaneous
  weight shifts      slow seeded sways so idle characters never freeze
  landing recoil     impulses (gestures, emphasis) ring through the
                     spring and die out naturally

Everything is seeded and stepped at fixed dt → bit-identical re-renders.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from engine.cinematics import Spring


@dataclass
class BodyState:
    """Everything the renderer needs to place one character this frame."""
    dx: float = 0.0        # position offset (px, world space at scale=1)
    dy: float = 0.0
    lean: float = 0.0      # degrees; + = leaning screen-right
    sx: float = 1.0        # squash/stretch scales
    sy: float = 1.0
    energy: float = 0.0    # 0..1 smoothed speech energy (drives fx)


class CharacterMotion:
    """Per-character body dynamics. Step once per frame."""

    def __init__(self, seed: int, fps: int = 30, side: float = 1.0):
        self.fps = fps
        self.side = side                       # −1 screen-left, +1 right
        self.rng = random.Random(seed)
        self.phase = self.rng.uniform(0, math.tau)

        # Springs: position responds slower than the voice (weight),
        # velocity of the springs drives lean/squash (physics, not hacks)
        self._y = Spring(0.0, omega=9.0, zeta=0.55, fps=fps)
        self._x = Spring(0.0, omega=6.0, zeta=0.6, fps=fps)
        self._energy = Spring(0.0, omega=7.0, zeta=1.0, fps=fps)  # no ring
        self._frame = 0

    # ─── Impulses ──────────────────────────────────────────

    def hit(self, strength: float = 1.0) -> None:
        """Emphasis / gesture landing: kick the vertical spring."""
        self._y.kick(-140.0 * strength)

    # ─── Per-frame step ────────────────────────────────────

    def step(self, is_speaking: bool, loudness: float,
             emotion: str = "neutral") -> BodyState:
        """
        loudness: 0..1 normalized voice level this frame.
        Returns the body state for THIS frame.
        """
        f = self._frame
        self._frame += 1
        t = f / self.fps

        # Smoothed speech energy (attack fast via spring target jumps)
        energy = self._energy.step(loudness if is_speaking else 0.0)
        energy = max(0.0, min(1.0, energy))

        # ── Breathing: asymmetric cycle (inhale 40%, exhale 60%) ──
        rate = 0.22 + 0.10 * energy            # excited = faster breath
        cyc = (t * rate + self.phase) % 1.0
        breath = (math.sin(cyc / 0.4 * math.pi) if cyc < 0.4
                  else math.sin(math.pi + (cyc - 0.4) / 0.6 * math.pi))
        breath_dy = breath * 2.6

        # ── Speech surge: body pushes INTO loud syllables, with lag ──
        target_dy = -energy * 14.0             # rise when projecting
        target_dx = self.side * energy * 4.0 \
            + math.sin(t * 0.31 + self.phase) * 2.2      # idle weight shift
        if emotion in ("amazed", "dramatic"):
            target_dy -= 8.0
        dy = self._y.step(target_dy)
        dx = self._x.step(target_dx)

        # ── Physics-derived principles ──
        vy = self._y.v / self.fps              # px per frame
        vx = self._x.v / self.fps

        # Squash & stretch from vertical velocity (volume conserving)
        stretch = max(-0.06, min(0.06, -vy * 0.010))
        sy = 1.0 + stretch
        sx = 1.0 / sy

        # Lean INTO horizontal movement (anticipation/follow-through)
        lean = max(-6.0, min(6.0, vx * 2.2 + self.side * energy * 1.2))

        return BodyState(dx=dx, dy=dy + breath_dy, lean=lean,
                         sx=sx, sy=sy, energy=energy)


# ═══════════════════════════════════════════
# BLINK SCHEDULER (Poisson gaps + double blinks)
# ═══════════════════════════════════════════

class BlinkScheduler:
    """Human blink statistics: mean gap ~3.2s, 15% double-blinks,
    6-frame close-open curve. Seeded → deterministic. Returns eyelid
    closure 0(open)..1(closed) per frame; consumers that have lid art
    (Tier 1 rigs) apply it, others ignore it."""

    CLOSE_FRAMES = (0.25, 0.75, 1.0, 0.6, 0.25, 0.0)

    def __init__(self, seed: int, fps: int = 30):
        self.rng = random.Random(seed)
        self.fps = fps
        self._next = self._gap()
        self._blink_frame = -1

    def _gap(self) -> int:
        return int(self.rng.expovariate(1 / 3.2) * self.fps) + self.fps

    def closure(self, frame: int) -> float:
        if self._blink_frame < 0 and frame >= self._next:
            self._blink_frame = 0
        if self._blink_frame >= 0:
            i = self._blink_frame
            self._blink_frame += 1
            if i < len(self.CLOSE_FRAMES):
                return self.CLOSE_FRAMES[i]
            self._blink_frame = -1
            # occasional double blink
            self._next = frame + (int(0.25 * self.fps)
                                  if self.rng.random() < 0.15
                                  else self._gap())
        return 0.0


# ═══════════════════════════════════════════
# SPRITE TRANSFORM (squash + lean, cached-size aware)
# ═══════════════════════════════════════════

def transform_sprite(img, sx: float, sy: float, lean_deg: float):
    """Apply squash/stretch + lean to a character sprite.
    Rotation uses expand=False around the bottom-center pivot region
    (cheap approximation that keeps feet planted)."""
    from PIL import Image
    w, h = img.size
    nw, nh = max(1, int(w * sx)), max(1, int(h * sy))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
    if abs(lean_deg) > 0.3:
        img = img.rotate(lean_deg, resample=Image.Resampling.BILINEAR,
                         center=(nw // 2, int(nh * 0.9)), expand=False)
    return img
