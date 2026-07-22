"""
JEEVidya — Easing Functions
Smooth interpolation functions for professional-grade animations.
Each function takes t in [0, 1] and returns a value in [0, 1].
"""
import math
from typing import Callable

EasingFunc = Callable[[float], float]


def ease_linear(t: float) -> float:
    """Linear interpolation — constant speed."""
    return t


def ease_in_quad(t: float) -> float:
    """Accelerating from zero velocity."""
    return t * t


def ease_out_quad(t: float) -> float:
    """Decelerating to zero velocity."""
    return t * (2 - t)


def ease_in_out_quad(t: float) -> float:
    """Acceleration until halfway, then deceleration."""
    if t < 0.5:
        return 2 * t * t
    return -1 + (4 - 2 * t) * t


def ease_in_cubic(t: float) -> float:
    """Cubic acceleration."""
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """Cubic deceleration — the most natural feel."""
    t -= 1
    return t * t * t + 1


def ease_in_out_cubic(t: float) -> float:
    """Smooth cubic acceleration/deceleration."""
    if t < 0.5:
        return 4 * t * t * t
    t -= 1
    return 1 + 4 * t * t * t


def ease_out_back(t: float) -> float:
    """Overshoot then settle — satisfying 'pop' effect."""
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)


def ease_out_elastic(t: float) -> float:
    """Elastic spring effect — use sparingly for emphasis."""
    if t == 0 or t == 1:
        return t
    c4 = (2 * math.pi) / 3
    return pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1


def ease_out_bounce(t: float) -> float:
    """Bouncing effect."""
    n1 = 7.5625
    d1 = 2.75
    if t < 1 / d1:
        return n1 * t * t
    elif t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375


def ease_in_out_sine(t: float) -> float:
    """Sinusoidal ease — very smooth, natural feeling."""
    return -(math.cos(math.pi * t) - 1) / 2


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def interpolate(start: float, end: float, t: float, easing: EasingFunc = ease_out_cubic) -> float:
    """Interpolate between start and end using the given easing function."""
    t = clamp(t)
    return start + (end - start) * easing(t)


def interpolate_color(
    color_start: tuple, color_end: tuple, t: float, easing: EasingFunc = ease_out_cubic
) -> tuple:
    """Interpolate between two RGB(A) color tuples."""
    t = clamp(t)
    eased_t = easing(t)
    return tuple(
        int(s + (e - s) * eased_t)
        for s, e in zip(color_start, color_end)
    )
