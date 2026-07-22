"""Easing curves, smoothers and springs shared by every animation system."""
from __future__ import annotations

import math


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def clamp01(v: float) -> float:
    return clamp(v, 0.0, 1.0)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_sine(t: float) -> float:
    return 0.5 - 0.5 * math.cos(clamp01(t) * math.pi)


def ease_in_out_quad(t: float) -> float:
    t = clamp01(t)
    return 2.0 * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0


def raised_cosine(t: float) -> float:
    """Smooth S-curve used for viseme crossfades and pose blends."""
    return 0.5 - 0.5 * math.cos(clamp01(t) * math.pi)


def bell(t: float, sharpness: float = 1.5) -> float:
    """0->1->0 bump for gesture impulses."""
    return math.sin(clamp01(t) * math.pi) ** sharpness


def gaussian_bump(t_ms: float, center_ms: float, width_ms: float) -> float:
    d = (t_ms - center_ms) / max(1.0, width_ms)
    return math.exp(-d * d)


class Chase:
    """Per-frame exponential smoother. rate is fraction of gap closed per frame.

    Different channels use different rates so the head traces arcs instead of
    every axis snapping in lockstep (animation principle: arcs).
    """

    __slots__ = ("rate", "value")

    def __init__(self, rate: float, value: float = 0.0):
        self.rate = rate
        self.value = value

    def step(self, target: float) -> float:
        self.value += (target - self.value) * self.rate
        return self.value


class DampedSpring:
    """Second-order spring, zeta~0.85 gives a professional hint of overshoot."""

    __slots__ = ("freq", "zeta", "value", "vel")

    def __init__(self, freq: float = 1.4, zeta: float = 0.85, value: float = 0.0):
        self.freq = freq
        self.zeta = zeta
        self.value = value
        self.vel = 0.0

    def snap(self, value: float) -> None:
        self.value = value
        self.vel = 0.0

    def step(self, target: float, dt: float) -> float:
        w = 2.0 * math.pi * self.freq
        accel = w * w * (target - self.value) - 2.0 * self.zeta * w * self.vel
        self.vel += accel * dt
        self.value += self.vel * dt
        return self.value


class DecayImpulse:
    """Accumulates impulses that exponentially decay -- used for emphasis tilts."""

    __slots__ = ("value", "half_life_frames")

    def __init__(self, half_life_frames: float = 9.0):
        self.value = 0.0
        self.half_life_frames = half_life_frames

    def kick(self, amount: float) -> None:
        self.value += amount

    def step(self) -> float:
        self.value *= 0.5 ** (1.0 / self.half_life_frames)
        return self.value
