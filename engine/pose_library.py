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

from PIL import Image

from config import settings

# Cross-fade duration in frames (at 30fps → ~12 frames ≈ 400ms).
# Sub-200ms full-body cross-fades read as strobing image swaps; real
# weight shifts take 350–500ms.
BLEND_FRAMES = 12

# Minimum frames a pose must be HELD (fully committed) before another
# transition may begin (at 30fps → ~1.3s). This is the single strongest
# guard against rapid pose thrash: no matter how many gestures fire,
# the body settles into each stance long enough to be READ.
MIN_HOLD_FRAMES = 40

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
        """Cross-fade between two full body images. blend_t: 0=from, 1=to."""
        img_from = self._poses.get(from_pose)
        img_to = self._poses.get(to_pose)
        if img_to is None:
            return img_from
        if img_from is None or blend_t >= 0.99:
            return img_to
        if blend_t <= 0.01:
            return img_from

        # Alpha-based cross-fade
        result = img_from.copy()
        # Modulate the target's alpha by blend_t
        to_copy = img_to.copy()
        r, g, b, a = to_copy.split()
        a = a.point(lambda p: int(p * blend_t))
        to_copy = Image.merge("RGBA", (r, g, b, a))
        result.alpha_composite(to_copy)
        return result


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
        self._blend_total: int = BLEND_FRAMES
        self._rng = rng or random.Random(42)
        self._recent: list = []  # last 3 poses — anti-ping-pong
        self._hold_frames: int = MIN_HOLD_FRAMES  # frames since last commit

    @property
    def blend_t(self) -> float:
        """Eased blend: raised-cosine S-curve (slow-in, slow-out).
        This is the single biggest upgrade — linear ramps look like
        'switching a picture'; eased ramps look like weight shifting."""
        if self.current == self.target:
            return 1.0
        raw = min(1.0, self._blend_frame / max(1, self._blend_total))
        # Raised cosine: 0→1 with zero derivative at endpoints
        return 0.5 - 0.5 * math.cos(raw * math.pi)

    @property
    def is_blending(self) -> bool:
        return self.current != self.target and self._blend_frame < self._blend_total

    def set_target(self, pose: str, displacement: float = 0.5) -> None:
        """Request transition. displacement 0..1 controls speed:
        high (big pose change) → fast fade; low (subtle) → slow graceful.

        Requests arriving before MIN_HOLD_FRAMES have elapsed since the
        last transition are DROPPED — the pose must land, be held, and be
        read before the body is allowed to move again. Gesture triggers
        re-fire every frame while active, so a dropped request that still
        matters simply succeeds once the hold expires."""
        if pose == self.target:
            return
        if pose == self.current:
            self.target = pose
            self._blend_frame = self._blend_total
            return

        # Minimum-hold gate: refuse mid-blend interruptions AND rapid
        # re-targeting. Without this, keyword + beat + rotation triggers
        # stack into a pose swap every few hundred milliseconds.
        if self.is_blending or self._hold_frames < MIN_HOLD_FRAMES:
            return

        self.target = pose
        self._blend_frame = 0
        self._hold_frames = 0

        # Displacement-adaptive frame count + jitter (all slowed 2.4×:
        # sub-200ms full-body fades read as a strobing slideshow)
        if displacement > 0.7:
            base = 8     # big change: quicker to hide ghost overlap
        elif displacement > 0.3:
            base = 12    # medium: smooth default
        else:
            base = 16    # subtle: graceful ease
        # ±2 frame jitter (breaks regularity)
        self._blend_total = max(6, base + self._rng.choice([-2, -1, 0, 1, 2]))

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

