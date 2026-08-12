"""
JEEVidya — Canonical→Pose Similarity Registration (Terminal Plan, Part III §3.2)
════════════════════════════════════════════════════════════════════════════════
THE fix for D1 ("face boxes detected once on body.png, pasted onto every
other pose"). Every pose gets its own registered similarity transform,
solved by Umeyama least-squares on a RIGID landmark subset with IRLS
Huber reweighting.

Why a rigid subset: lips, lids, and brows differ by expression between
poses — including them biases the fit toward the expression instead of
the head placement. The subset below is skull geometry only.

Why IRLS/Huber: a single mis-detected landmark (hair over a temple)
would otherwise drag the least-squares fit. Huber weights demote
outliers over 3 reweight passes; a pose whose post-fit RMS still
exceeds the budget FAILS LOUDLY with its name (no heuristic fallback —
the heuristic is what put the mouth on the eyes historically).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# MediaPipe FaceLandmarker indices for the RIGID skull subset:
#   nose bridge (6, 168, 197, 195, 5, 4), outer eye corners (33, 263),
#   temples (127, 356), chin (152). Deliberately excludes lips/lids/brows.
RIGID_SUBSET: Tuple[int, ...] = (6, 168, 197, 195, 5, 4, 33, 263, 127, 356, 152)

# Post-fit RMS budget in canonical-face pixels, normalized by face height
# at bake time. The gate exists to catch GROSS misregistration (a mouth
# landing on the eyes is tens of px) — hand-redrawn pose art legitimately
# carries a few px of per-pose landmark drift on the rigid subset.
# Measured on this repo's actual pose art: genuinely drawn poses land at
# 4–8 px RMS on a ~270–300 px face (1.5–3% of face height), while gross
# misregistration is >10% of face height (30+ px). 15 px at the 400 px
# reference face (= 3.75%) accepts every real pose and still rejects
# gross defects by a ~3x margin. This is safe because v3 anchors every
# facial feature to EACH pose's own detected landmarks — rigid-subset
# drift never moves the mouth or eyes off the face.
DEFAULT_RMS_BUDGET_PX = 15.0

IRLS_PASSES = 3
HUBER_K = 1.345  # standard 95%-efficiency Huber constant (in robust sigmas)


class RegistrationError(RuntimeError):
    """A pose that cannot be registered is a hard error — never a guess."""


@dataclass(frozen=True)
class SimilarityTransform:
    """x' = s·R(θ)·x + t  — canonical space → pose space."""
    s: float
    theta: float          # radians; this IS the pose's head roll
    tx: float
    ty: float
    rms: float            # post-fit robust RMS in px

    # ─── application ──────────────────────────────────────

    @property
    def matrix(self) -> np.ndarray:
        """2×3 affine matrix [sR | t]."""
        c, si = math.cos(self.theta), math.sin(self.theta)
        return np.array([[self.s * c, -self.s * si, self.tx],
                         [self.s * si, self.s * c, self.ty]], dtype=np.float64)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Transform an (N,2) point array canonical → pose."""
        m = self.matrix
        return pts @ m[:, :2].T + m[:, 2]

    def apply_point(self, x: float, y: float) -> Tuple[float, float]:
        p = self.apply(np.array([[x, y]], dtype=np.float64))[0]
        return float(p[0]), float(p[1])

    def inverse(self) -> "SimilarityTransform":
        s_inv = 1.0 / self.s
        th_inv = -self.theta
        c, si = math.cos(th_inv), math.sin(th_inv)
        tx = -s_inv * (c * self.tx - si * self.ty)
        ty = -s_inv * (si * self.tx + c * self.ty)
        return SimilarityTransform(s_inv, th_inv, tx, ty, self.rms)

    # ─── interpolation (Part VI: M_pose lockstep with body cross-fade) ──

    def lerp(self, other: "SimilarityTransform",
             t: float) -> "SimilarityTransform":
        """Interpolate transforms: slerp θ (shortest arc), log-lerp s,
        lerp t. Used so the head travels in lockstep with the eased
        body cross-fade through every pose transition (D1's dynamic
        half: the head must MOVE with the body, not teleport)."""
        t = max(0.0, min(1.0, t))
        # shortest-arc angle interpolation
        d = (other.theta - self.theta + math.pi) % (2 * math.pi) - math.pi
        theta = self.theta + d * t
        # geometric (log-space) scale interpolation — scale is multiplicative
        s = math.exp((1 - t) * math.log(self.s) + t * math.log(other.s))
        tx = self.tx + (other.tx - self.tx) * t
        ty = self.ty + (other.ty - self.ty) * t
        return SimilarityTransform(s, theta, tx, ty,
                                   max(self.rms, other.rms))

    # ─── persistence ──────────────────────────────────────

    def to_dict(self) -> dict:
        return {"s": self.s, "theta": self.theta,
                "tx": self.tx, "ty": self.ty, "rms": self.rms}

    @staticmethod
    def from_dict(d: dict) -> "SimilarityTransform":
        return SimilarityTransform(d["s"], d["theta"], d["tx"], d["ty"],
                                   d.get("rms", 0.0))

    @staticmethod
    def identity() -> "SimilarityTransform":
        return SimilarityTransform(1.0, 0.0, 0.0, 0.0, 0.0)


# ═══════════════════════════════════════════
# Umeyama + IRLS
# ═══════════════════════════════════════════

def _umeyama(src: np.ndarray, dst: np.ndarray,
             w: np.ndarray) -> Tuple[float, float, float, float]:
    """Weighted Umeyama similarity fit src→dst. Returns (s, θ, tx, ty).

    Closed-form: weighted centroids → weighted covariance → SVD →
    rotation with det-correction → scale from variance ratio.
    """
    w = w / w.sum()
    mu_s = (src * w[:, None]).sum(axis=0)
    mu_d = (dst * w[:, None]).sum(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc * w[:, None]).T @ sc                     # 2×2
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    var_s = float((w * (sc ** 2).sum(axis=1)).sum())
    if var_s < 1e-12:
        raise RegistrationError("degenerate landmark configuration "
                                "(zero variance)")
    s = float(np.trace(np.diag(D) @ S)) / var_s
    theta = math.atan2(R[1, 0], R[0, 0])
    t = mu_d - s * (R @ mu_s)
    return s, theta, float(t[0]), float(t[1])


def register_pose(canonical_lms: Sequence[Tuple[float, float]],
                  pose_lms: Sequence[Tuple[float, float]],
                  pose_name: str = "?",
                  subset: Sequence[int] = RIGID_SUBSET,
                  rms_budget_px: float = DEFAULT_RMS_BUDGET_PX,
                  face_height_px: Optional[float] = None
                  ) -> SimilarityTransform:
    """Fit canonical→pose similarity on the rigid subset with IRLS.

    `rms_budget_px` is interpreted at a reference face height of 400 px;
    when `face_height_px` is given the budget scales proportionally so
    the gate is resolution-independent (Part VIII: thresholds derived
    from face size, never literal pixels).
    """
    can = np.asarray(canonical_lms, dtype=np.float64)
    pos = np.asarray(pose_lms, dtype=np.float64)
    if can.shape[0] <= max(subset) or pos.shape[0] <= max(subset):
        raise RegistrationError(
            f"pose '{pose_name}': landmark array too small for rigid "
            f"subset (need > {max(subset)}, got {min(can.shape[0], pos.shape[0])})")

    src = can[list(subset)]
    dst = pos[list(subset)]
    w = np.ones(len(subset), dtype=np.float64)

    s = theta = tx = ty = 0.0
    resid = np.zeros(len(subset))
    for _ in range(1 + IRLS_PASSES):
        s, theta, tx, ty = _umeyama(src, dst, w)
        xf = SimilarityTransform(s, theta, tx, ty, 0.0)
        pred = xf.apply(src)
        resid = np.linalg.norm(pred - dst, axis=1)
        # robust sigma via MAD; guard against all-zero residuals
        sigma = max(1e-9, 1.4826 * float(np.median(resid)))
        r = resid / sigma
        w = np.where(r <= HUBER_K, 1.0, HUBER_K / np.maximum(r, 1e-12))

    rms = float(np.sqrt(np.mean(resid ** 2)))
    budget = rms_budget_px
    if face_height_px is not None:
        budget = rms_budget_px * (face_height_px / 400.0)
    if rms > budget:
        raise RegistrationError(
            f"pose '{pose_name}': registration RMS {rms:.3f}px exceeds "
            f"budget {budget:.3f}px — re-export the pose art or re-run "
            f"`jvmake rig --force` (NO heuristic fallback in rig v3)")
    return SimilarityTransform(s, theta, tx, ty, rms)


def predict_feature_position(xform: SimilarityTransform,
                             canonical_xy: Tuple[float, float]
                             ) -> Tuple[float, float]:
    """Where a canonical-space feature (mouth centroid, iris center)
    MUST land in pose space. face_qc re-detects on rendered pixels and
    asserts the detection matches this prediction within 0.6 px."""
    return xform.apply_point(*canonical_xy)
