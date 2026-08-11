"""
JEEVidya — Unified Head Assembly Transform (Terminal Plan, Part VI)
═══════════════════════════════════════════════════════════════════
THE fix for D4 ("render() ignores head_yaw/tilt/nod; _compose_head /
_staged_head are dead code") and for cumulative resample softness.

Doctrine (Law 1): exactly ONE head-compose path. This module computes
ONE composed affine per frame; the head plate is resampled EXACTLY
ONCE (BICUBIC). No chained rotate→resize→transform. The composed
matrix is also exposed analytically so QC (Part VIII) can PREDICT
where the mouth centroid and iris centers must land on rendered
pixels — registration is checked against math, not vibes.

Pipeline per frame (Part VI):

  M = T(sway, bounce)                              # screen-space drift
    ∘ M_pose(t)          slerp(θ), lerp(s, tx, ty) # D1: head rides the
                                                   #     pose cross-fade
    ∘ R(head_tilt + physics_overshoot)             # roll
    ∘ Nod(head_nod)      vertical squash + shift   # pitch, 2.5D
    ∘ Yaw(head_yaw)      cos-squash + parallax     # yaw, 2.5D
    ∘ Shear_hair         inside head space

Yaw parallax: features live on a sphere, not a plane. Before the
affine, the face-feature layer is shifted by face_dx = yaw · half_w ·
PARALLAX_GAIN in head space (replacing the dead `face_dx = 0.0`), and
the whole head gets a horizontal cosine squash — the two cheapest
signals the brain uses to read "the head TURNED" from flat art.

Caching: two-level LRU. Level 1 (face-channel key → composed plate)
lives in the head assembler; level 2 here: (plate id, quantized
affine) → transformed RGBA.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from engine.registration import SimilarityTransform

# 2.5D gains — conservative; QC registration tolerances gate any change.
YAW_PARALLAX_GAIN = 0.32     # feature shift as fraction of half-width per unit yaw
YAW_SQUASH_GAIN = 0.10       # horizontal cos-squash at |yaw| = 1
NOD_SQUASH_GAIN = 0.06       # vertical squash at |nod| = 1
NOD_SHIFT_GAIN = 0.18        # vertical feature shift per unit nod (× half-height)

# Affine quantization for the transform cache: 1/64 px translation,
# ~0.0005 rad rotation. Below any perceptible threshold at 1080p.
_Q_T = 1 / 64
_Q_R = 5e-4
_Q_S = 1 / 512


@dataclass(frozen=True)
class HeadPose:
    """The per-frame animation channels that move the head as a unit.

    yaw / nod / tilt ∈ [-1, 1] (normalized art-space turns, not degrees)
    sway / bounce in screen px; overshoot in radians (spring physics).
    """
    yaw: float = 0.0
    nod: float = 0.0
    tilt: float = 0.0          # roll, radians
    sway: float = 0.0
    bounce: float = 0.0
    overshoot: float = 0.0     # physics roll overshoot, radians
    hair_shear: float = 0.0


@dataclass(frozen=True)
class ComposedAffine:
    """A 2×3 affine (head space → body space) + the feature-layer
    pre-shift that must be applied INSIDE head space before the affine."""
    m: Tuple[float, float, float, float, float, float]  # a b c / d e f
    face_dx: float     # feature-layer x pre-shift (head space px)
    face_dy: float     # feature-layer y pre-shift (head space px)

    @property
    def matrix(self) -> np.ndarray:
        a, b, c, d, e, f = self.m
        return np.array([[a, b, c], [d, e, f]], dtype=np.float64)

    def apply_point(self, x: float, y: float) -> Tuple[float, float]:
        a, b, c, d, e, f = self.m
        return (a * x + b * y + c, d * x + e * y + f)

    def apply_feature_point(self, x: float, y: float) -> Tuple[float, float]:
        """Predict where a FEATURE (mouth centroid, iris center) lands:
        pre-shift in head space, then the composed affine. This is the
        analytic prediction QC checks rendered pixels against."""
        return self.apply_point(x + self.face_dx, y + self.face_dy)

    def quantized_key(self) -> Tuple:
        a, b, c, d, e, f = self.m
        return (round(a / _Q_S), round(b / _Q_S), round(c / _Q_T),
                round(d / _Q_S), round(e / _Q_S), round(f / _Q_T),
                round(self.face_dx / _Q_T), round(self.face_dy / _Q_T))


def _mat(a, b, c, d, e, f) -> np.ndarray:
    return np.array([[a, b, c], [d, e, f], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_about(theta: float, cx: float, cy: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return (_mat(1, 0, cx, 0, 1, cy)
            @ _mat(c, -s, 0, s, c, 0)
            @ _mat(1, 0, -cx, 0, 1, -cy))


def _scale_about(sx: float, sy: float, cx: float, cy: float) -> np.ndarray:
    return (_mat(1, 0, cx, 0, 1, cy)
            @ _mat(sx, 0, 0, 0, sy, 0)
            @ _mat(1, 0, -cx, 0, 1, -cy))


def _shear_x_about(k: float, cy: float) -> np.ndarray:
    """Horizontal shear proportional to height above cy (hair sway)."""
    return _mat(1, -k, k * cy, 0, 1, 0)


def compose(pose_xform: SimilarityTransform,
            head: HeadPose,
            head_size: Tuple[float, float],
            pivot: Tuple[float, float]) -> ComposedAffine:
    """Build THE single affine for this frame.

    pose_xform : interpolated canonical→pose similarity (already lerped
                 with the SAME eased blend_t as the body cross-fade —
                 the head travels in lockstep through every transition)
    head_size  : (w, h) of the head plate in head space
    pivot      : rotation pivot in head space (neck joint, usually
                 bottom-center of the plate)
    """
    w, h = head_size
    px, py = pivot

    # 2.5D channels — squashes about the pivot, in head space
    yaw_sx = 1.0 - YAW_SQUASH_GAIN * abs(head.yaw)
    nod_sy = 1.0 - NOD_SQUASH_GAIN * abs(head.nod)
    M_25d = _scale_about(yaw_sx, nod_sy, px, py)

    # Roll: art roll (from the registered pose) is inside pose_xform's θ;
    # the ANIMATION roll (tilt + physics overshoot) composes on top.
    M_roll = _rot_about(head.tilt + head.overshoot, px, py)

    # Hair shear inside head space, before everything else
    M_hair = _shear_x_about(head.hair_shear, py) if abs(head.hair_shear) > 1e-6 \
        else np.eye(3)

    # Pose similarity: head space → body space (D1 fix rides here)
    P = np.vstack([pose_xform.matrix, [0.0, 0.0, 1.0]])

    # Screen-space drift last
    M_drift = _mat(1, 0, head.sway, 0, 1, head.bounce)

    M = M_drift @ P @ M_roll @ M_25d @ M_hair
    a, b, c = M[0]
    d, e, f = M[1]

    # Feature-layer parallax (pre-shift in head space, replaces face_dx=0.0)
    face_dx = head.yaw * (w / 2.0) * YAW_PARALLAX_GAIN
    face_dy = head.nod * (h / 2.0) * NOD_SHIFT_GAIN
    return ComposedAffine(m=(float(a), float(b), float(c),
                             float(d), float(e), float(f)),
                          face_dx=float(face_dx), face_dy=float(face_dy))


# ═══════════════════════════════════════════
# Level-2 transform cache: (plate id, quantized affine) → transformed head
# ═══════════════════════════════════════════

class TransformCache:
    def __init__(self, max_items: int = 256):
        self._d: "OrderedDict[Tuple, Tuple[Image.Image, Tuple[int, int]]]" \
            = OrderedDict()
        self._max = max_items
        self.hits = 0
        self.misses = 0

    def transform(self, plate: Image.Image, plate_key: Tuple,
                  aff: ComposedAffine,
                  out_size: Tuple[int, int]) -> Image.Image:
        """Apply the composed affine to the plate with EXACTLY ONE
        BICUBIC resample, returning a canvas of out_size.

        PIL's Image.transform maps OUTPUT→INPUT, so we hand it the
        inverse of the head→body matrix.
        """
        key = (plate_key, aff.quantized_key(), out_size)
        hit = self._d.get(key)
        if hit is not None:
            self._d.move_to_end(key)
            self.hits += 1
            return hit[0]
        self.misses += 1

        M = np.vstack([aff.matrix, [0.0, 0.0, 1.0]])
        Minv = np.linalg.inv(M)
        coeffs = (Minv[0, 0], Minv[0, 1], Minv[0, 2],
                  Minv[1, 0], Minv[1, 1], Minv[1, 2])
        out = plate.transform(out_size, Image.AFFINE, coeffs,
                              resample=Image.BICUBIC)
        self._d[key] = (out, out_size)
        if len(self._d) > self._max:
            self._d.popitem(last=False)
        return out


def shift_features_subpixel(layer: Image.Image, dx: float, dy: float
                            ) -> Image.Image:
    """Sub-pixel shift of the feature layer INSIDE head space (parallax).
    One AFFINE pass, BICUBIC — never an int()-rounded paste."""
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return layer
    return layer.transform(layer.size, Image.AFFINE,
                           (1, 0, -dx, 0, 1, -dy), resample=Image.BICUBIC)


__all__ = ["HeadPose", "ComposedAffine", "compose", "TransformCache",
           "shift_features_subpixel",
           "YAW_PARALLAX_GAIN", "YAW_SQUASH_GAIN",
           "NOD_SQUASH_GAIN", "NOD_SHIFT_GAIN"]
