"""
JEEVidya V5 — Pose Library (Tier 1)
═══════════════════════════════════
Auto-discovers pose images from assets/characters/<name>/poses/ and
provides cross-faded body-image switching keyed to gesture triggers.

Each pose is a full-body PNG; the library crops the TORSO region using
the same rig.json joints so it drops straight into the BoneEngine's
torso slot. The HEAD is shared across all poses (only arms/torso differ).

Degrades gracefully: if no poses/ directory exists, returns the default
torso and the puppet renders exactly as before (single-pose mode).
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config import settings

# Timing is derived from the ACTUAL frame rate, read lazily because
# settings.FPS is reassigned after module import (see settings.py).
# Hardcoded frame counts calibrated for 30 fps halve every duration at
# FPS=60 — sub-200ms full-body cross-fades read as strobing image
# swaps; real weight shifts take 350–500ms.

def _fps() -> float:
    """Current frame rate, read lazily from settings (never cached)."""
    try:
        fps = float(getattr(settings, "FPS", 30.0))
    except (TypeError, ValueError):
        fps = 30.0
    return fps if fps > 0 else 30.0


def blend_frames() -> int:
    """Cross-fade duration in frames: 400 ms at the current FPS."""
    return max(2, round(_fps() * 0.40))


def min_hold_frames() -> int:
    """Minimum frames a pose must be HELD (fully committed) before
    another transition may begin: ~3.5 s at the current FPS. This is
    the guard against rapid pose thrash while allowing natural speech
    gesticulation without hands flashing or blinking."""
    return max(10, round(_fps() * 3.5))

# Default pose when nothing is triggered
DEFAULT_POSE = "neutral"


class PoseLibrary:
    """Loads and manages pose variants for one character."""

    def __init__(self, character: str, rig_scale: float = 1.0):
        self.character = character
        self.scale = rig_scale
        self._poses: Dict[str, Image.Image] = {}   # pose_name → full body RGBA
        self._torsos: Dict[str, Image.Image] = {}   # pose_name → cropped torso
        self._available: List[str] = []

        poses_dir = os.path.join(settings.CHARACTERS_DIR, character, "poses")
        if not os.path.isdir(poses_dir):
            return

        for fname in sorted(os.listdir(poses_dir)):
            if not fname.lower().endswith(".png"):
                continue
            name = os.path.splitext(fname)[0]
            path = os.path.join(poses_dir, fname)
            try:
                img = Image.open(path).convert("RGBA")
                self._poses[name] = img
                self._available.append(name)
            except Exception as e:
                print(f"  [PoseLib] Failed to load {path}: {e}")

        if self._available:
            print(f"  [PoseLib] {character}: {len(self._available)} poses loaded "
                  f"({', '.join(self._available)})")

    @property
    def has_poses(self) -> bool:
        return len(self._available) > 1

    @property
    def pose_names(self) -> List[str]:
        return list(self._available)

    def get_body(self, name: str) -> Optional[Image.Image]:
        """Get the full body image for a pose (for V2 pro path)."""
        return self._poses.get(name) or self._poses.get(DEFAULT_POSE)

    def get_torso(self, name: str, neck_y: float,
                  torso_offset: Tuple[float, float],
                  original_torso: Image.Image) -> Image.Image:
        """Get the cropped torso layer for a pose (for BoneEngine path).
        Crops full width (0 to body.width) so extended arms and hands are
        never sliced off at the narrow default torso bounding box."""
        if name not in self._poses:
            return original_torso

        if name in self._torsos:
            return self._torsos[name]

        body = self._poses[name]
        ox, oy = torso_offset
        tw, th = original_torso.size

        # Crop from x=0 to full body width to keep hands intact
        s = self.scale
        if s != 0 and s != 1.0:
            crop_y = int(oy / s)
            crop_h = int(th / s)
        else:
            crop_y = int(oy)
            crop_h = th

        crop_y = max(0, min(crop_y, body.height - 1))
        crop_b = min(body.height, crop_y + crop_h)

        if crop_b <= crop_y:
            return original_torso

        # Full horizontal span (x=0 to body.width)
        cropped = body.crop((0, crop_y, body.width, crop_b))

        # Scale height to match original torso height while preserving full width
        target_w = int(body.width * s) if (s != 0 and s != 1.0) else body.width
        target_h = th
        if cropped.size != (target_w, target_h):
            cropped = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)

        self._torsos[name] = cropped
        return cropped

    def blended_body(self, from_pose: str, to_pose: str,
                     blend_t: float) -> Optional[Image.Image]:
        """Select body for pose transition. Snaps cleanly at midpoint t >= 0.5
        to guarantee 100% solid opacity with zero ghosting or blur/shadow artifacts."""
        img_from = self._poses.get(from_pose)
        img_to = self._poses.get(to_pose)
        if img_to is None:
            return img_from
        if img_from is None or blend_t >= 0.5:
            return img_to
        return img_from


class PoseState:
    """Tracks the cross-fade state for one character's body pose.

    Animation principles applied:
      • Slow-in / slow-out: raised-cosine S-curve on blend_t so transitions
        accelerate into and decelerate out of the midpoint — never a linear ramp.
      • Displacement-adaptive speed: big silhouette changes (arms up→down)
        get 3-frame fast fades to hide ghost overlap; subtle shifts get 7-frame
        graceful eases.
      • ±1 frame jitter: breaks metronomic regularity, the #1 AI tell.
      • Anti-ping-pong: tracks recent poses to prevent A→B→A cycling.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self.current: str = DEFAULT_POSE
        self.target: str = DEFAULT_POSE
        self._blend_frame: int = 0
        self._blend_total: int = blend_frames()
        self._rng = rng or random.Random(42)
        self._recent: list = []  # last 3 poses — anti-ping-pong
        self._hold_frames: int = min_hold_frames()  # frames since last commit

    @property
    def blend_t(self) -> float:
        """Eased blend: Ken Perlin's smootherstep S-curve (zero 1st and 2nd derivatives).
        Provides a perfectly smooth, continuous transition without abrupt acceleration."""
        if self.current == self.target:
            return 1.0
        x = min(1.0, max(0.0, self._blend_frame / max(1, self._blend_total)))
        # smootherstep: 6x^5 - 15x^4 + 10x^3
        return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

    @property
    def is_blending(self) -> bool:
        return self.current != self.target and self._blend_frame < self._blend_total

    def set_target(self, pose: str, displacement: float = 0.5) -> None:
        """Request transition. displacement 0..1 controls speed:
        high (big pose change) → fast fade; low (subtle) → slow graceful."""
        if pose == self.target:
            return
        if pose == self.current:
            self.target = pose
            self._blend_frame = self._blend_total
            return

        # Minimum-hold gate: refuse mid-blend interruptions AND rapid
        # re-targeting.
        if self.is_blending or self._hold_frames < min_hold_frames():
            return

        self.target = pose
        self._blend_frame = 0
        self._hold_frames = 0

        # Smooth keyframe transitions (0.24s - 0.32s, ~14-20 frames @ 60fps).
        # Eliminates the flashing/blinking hand artifact by providing a
        # continuous Perlin smootherstep dissolve between silhouettes.
        if displacement > 0.7:
            base_s = 0.24
        elif displacement > 0.3:
            base_s = 0.28
        else:
            base_s = 0.32
        base = max(6, round(_fps() * base_s))
        self._blend_total = max(6, base + self._rng.choice([-1, 0, 1]))

        # Track for anti-ping-pong
        self._recent.append(pose)
        if len(self._recent) > 3:
            self._recent.pop(0)

    def would_pingpong(self, pose: str) -> bool:
        """Check if selecting this pose would create A→B→A cycling."""
        if len(self._recent) >= 2 and pose == self._recent[-2]:
            return True
        return False

    def step(self) -> Tuple[str, str, float]:
        """Advance one frame. Returns (from_pose, to_pose, eased_blend_t)."""
        self._hold_frames += 1
        if self.current != self.target:
            self._blend_frame += 1
            if self._blend_frame >= self._blend_total:
                self.current = self.target
        return self.current, self.target, self.blend_t

