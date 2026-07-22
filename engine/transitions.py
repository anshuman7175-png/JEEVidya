"""
Gudiya & Chintu — Transition Library
Frame-interpolated cinematic transitions with easing.
Each function returns intermediate values for smooth animation.
"""
import math
from typing import Tuple

from engine.easing import ease_out_cubic, ease_out_back, ease_in_out_cubic


def whoosh_slide(frame: int, total_frames: int, start_x: int, end_x: int,
                 y: int) -> Tuple[int, int, float]:
    """
    Character slides in from off-screen.
    Returns (x, y, scale) for this frame.
    Uses ease_out_cubic for a decelerating slide.
    """
    t = min(1.0, frame / max(1, total_frames))
    eased = ease_out_cubic(t)
    x = int(start_x + (end_x - start_x) * eased)
    # Slight scale overshoot: 1.05 → 1.0
    scale = 1.0 + (1.0 - eased) * 0.05
    return x, y, scale


def scale_pop(frame: int, total_frames: int) -> float:
    """
    Element pops in with overshoot bounce.
    Returns scale factor: 0.0 → 1.15 → 1.0
    Uses ease_out_back for the overshoot.
    """
    t = min(1.0, frame / max(1, total_frames))
    return ease_out_back(t)


def fade_cut(frame: int, total_frames: int) -> Tuple[float, float]:
    """
    Cross-dissolve between two layers.
    Returns (old_opacity, new_opacity) for this frame.
    """
    t = min(1.0, frame / max(1, total_frames))
    eased = ease_in_out_cubic(t)
    return 1.0 - eased, eased


def punch_zoom(frame: int, total_frames: int, base_scale: float = 1.0,
               target_scale: float = 1.3) -> float:
    """
    Quick zoom-in emphasis, then return to normal.
    First half zooms in, second half zooms back.
    """
    t = min(1.0, frame / max(1, total_frames))
    if t < 0.5:
        # Zoom in
        phase_t = t * 2
        eased = ease_out_cubic(phase_t)
        return base_scale + (target_scale - base_scale) * eased
    else:
        # Zoom back
        phase_t = (t - 0.5) * 2
        eased = ease_out_cubic(phase_t)
        return target_scale - (target_scale - base_scale) * eased


def camera_transition(frame: int, total_frames: int,
                      from_preset: dict, to_preset: dict) -> dict:
    """
    Smoothly interpolate between two camera shot presets.
    Each preset is {"x": int, "y": int, "scale": float, "opacity": float}.
    Returns interpolated preset.
    """
    t = min(1.0, frame / max(1, total_frames))
    eased = ease_out_cubic(t)

    return {
        "x": from_preset["x"] + (to_preset["x"] - from_preset["x"]) * eased,
        "y": from_preset["y"] + (to_preset["y"] - from_preset["y"]) * eased,
        "scale": from_preset["scale"] + (to_preset["scale"] - from_preset["scale"]) * eased,
        "opacity": from_preset["opacity"] + (to_preset["opacity"] - from_preset["opacity"]) * eased,
    }
