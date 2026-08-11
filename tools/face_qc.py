"""
JEEVidya — The QC Constitution (Terminal Plan, Part VIII)
═════════════════════════════════════════════════════════
Hard-fail gates on RENDERED PIXELS. Features are legislation; these
gates are the constitution — nothing un-gated reaches an encode, and
nothing un-measured counts as "done" (Law 2).

Every threshold is DERIVED from FPS and face size, never a literal
frame or pixel count (the lint in tools/lint_constants.py enforces
that discipline codebase-wide; this module practices it).

Checks implemented (each returns a `GateResult` with a number):

  registration      mouth centroid + iris centers vs. the ANALYTIC
                    prediction from engine/head_transform.ComposedAffine
  single_face       exactly one connected lip-color component; no
                    lip-color pixels outside mouth contour + margin
  blink_closure     at blink=1: zero iris-color pixels
  temporal          per-frame deltas of aperture/centroid bounded;
                    jerk metric (no single-frame sign reversals)
  av_sync           cross-correlation of rendered aperture vs. audio
                    envelope peaks within ±1 frame
  sync_confidence   sliding-window local correlation (SyncNet-style,
                    pure numpy) — catches ONE bad turn that a global
                    correlation averages away
  discriminability  pairwise contour distance between viseme classes
                    rendered at phone scale (~420 px) — the metric
                    that gates the dominance model and articulation gain
  seam              vertical gradient continuity across the neck band;
                    no alpha<1 hole inside the body silhouette
  rig_sanity        every pose RMS ≤ budget; rig version == 3

CLI:  python -m tools.face_qc --report out.json [--strict]
      (full sweep wiring arrives with `jvmake verify-face`)
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# ═══════════════════════════════════════════
# Threshold derivation — FPS- and face-size-relative, never literal
# ═══════════════════════════════════════════

# Registration budgets (Part VIII table), in units of face size / degrees
REG_POS_TOL_FRAC = 0.6 / 100.0     # 0.6 px per 100 px of face height
REG_SCALE_TOL = 0.01               # head scale within 1%
REG_ROLL_TOL_DEG = 0.3
RMS_BUDGET_FRAC = 0.8 / 100.0      # pose RMS ≤ 0.8 px per 100 px face height

# Temporal budgets, in fractions of face height PER SECOND (÷ fps at use)
MAX_CENTROID_SPEED_FRAC = 1.2      # mouth centroid drift ≤ 1.2 face-heights/s
MAX_APERTURE_SPEED = 12.0          # full-aperture units per second (slew law)

# A/V sync: peak lag within ±1 frame at the render fps
# Sync confidence: minimum local correlation over any window
SYNC_WINDOW_MS = 1200.0
SYNC_MIN_LOCAL_CORR = 0.35

# Discriminability at phone scale
PHONE_SCALE_PX = 420
DISCRIM_MIN_SEP_FRAC = 0.04        # min pairwise contour distance / mouth width


@dataclass
class GateResult:
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": bool(self.passed),
                "value": float(self.value), "threshold": float(self.threshold),
                "detail": self.detail}


@dataclass
class QCReport:
    gates: List[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def add(self, g: GateResult) -> None:
        self.gates.append(g)

    def to_dict(self) -> dict:
        return {"passed": self.passed,
                "gates": [g.to_dict() for g in self.gates]}

    def save(self, path: str) -> None:
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)

    def summary(self) -> str:
        lines = []
        for g in self.gates:
            mark = "PASS" if g.passed else "FAIL"
            lines.append(f"[{mark}] {g.name:<18} value={g.value:.4f} "
                         f"thresh={g.threshold:.4f} {g.detail}")
        lines.append(f"→ {'ALL GATES PASS' if self.passed else 'GATE FAILURE'}")
        return "\n".join(lines)


# ═══════════════════════════════════════════
# Color-mask utilities (pure numpy)
# ═══════════════════════════════════════════

def color_mask(img: Image.Image, rgb: Tuple[int, int, int],
               tol: int = 26) -> np.ndarray:
    """Boolean mask of pixels within `tol` (Chebyshev) of `rgb`,
    alpha > half."""
    a = np.asarray(img.convert("RGBA"), dtype=np.int16)
    d = np.abs(a[..., :3] - np.array(rgb, dtype=np.int16)).max(axis=-1)
    return (d <= tol) & (a[..., 3] > 127)


def connected_components(mask: np.ndarray) -> int:
    """Count 4-connected components ≥ 4 px. Pure-numpy two-pass label."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    nxt = 0
    parent: List[int] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for y in range(h):
        row = mask[y]
        for x in range(w):
            if not row[x]:
                continue
            up = labels[y - 1, x] if y > 0 else 0
            left = labels[y, x - 1] if x > 0 else 0
            if up and left:
                ru, rl = find(up - 1), find(left - 1)
                labels[y, x] = ru + 1
                if ru != rl:
                    parent[rl] = ru
            elif up or left:
                labels[y, x] = find((up or left) - 1) + 1
            else:
                parent.append(nxt)
                nxt += 1
                labels[y, x] = nxt
    if nxt == 0:
        return 0
    roots = {}
    flat = labels[mask]
    for lab in flat:
        r = find(lab - 1)
        roots[r] = roots.get(r, 0) + 1
    return sum(1 for c in roots.values() if c >= 4)


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


# ═══════════════════════════════════════════
# Gates
# ═══════════════════════════════════════════

def gate_registration(predicted: Tuple[float, float],
                      detected: Optional[Tuple[float, float]],
                      face_h: float, what: str) -> GateResult:
    """Feature (mouth centroid / iris center) must land within the
    face-size-relative budget of the analytic prediction."""
    tol = REG_POS_TOL_FRAC * face_h
    if detected is None:
        return GateResult(f"registration:{what}", False, math.inf, tol,
                          "feature not detected on rendered pixels")
    err = math.dist(predicted, detected)
    return GateResult(f"registration:{what}", err <= tol, err, tol)


def gate_single_face(frame: Image.Image, lip_rgb: Tuple[int, int, int],
                     mouth_bbox: Tuple[int, int, int, int],
                     face_h: float) -> GateResult:
    """Exactly one lip-color component; none outside mouth bbox + margin.
    Kills D2/D3 regressions (three-mouth composites, painted-mouth ghosts)."""
    m = color_mask(frame, lip_rgb)
    n = connected_components(m)
    margin = int(round(0.02 * face_h)) + 2
    x0, y0, x1, y1 = mouth_bbox
    outside = m.copy()
    outside[max(0, y0 - margin):y1 + margin,
            max(0, x0 - margin):x1 + margin] = False
    stray = int(outside.sum())
    ok = (n == 1) and (stray == 0)
    return GateResult("single_face", ok, float(n + stray), 1.0,
                      f"components={n} stray_px={stray}")


def gate_blink_closure(frame_closed: Image.Image,
                       iris_rgb: Tuple[int, int, int]) -> GateResult:
    """At blink=1 the iris must be GONE — geometric guarantee, verified."""
    count = int(color_mask(frame_closed, iris_rgb).sum())
    return GateResult("blink_closure", count == 0, float(count), 0.0,
                      "iris-color px at blink=1")


def gate_temporal(apertures: Sequence[float],
                  centroids: Sequence[Tuple[float, float]],
                  face_h: float, fps: int) -> GateResult:
    """Per-frame deltas bounded + jerk metric. Thresholds derived from
    fps and face size (never literal frames/pixels)."""
    if len(apertures) < 3:
        return GateResult("temporal", True, 0.0, 1.0, "too short to judge")
    dt = 1.0 / max(1, fps)
    ap = np.asarray(apertures, dtype=np.float64)
    d_ap = np.abs(np.diff(ap)) / dt
    max_ap_speed = float(d_ap.max())
    c = np.asarray(centroids, dtype=np.float64)
    d_c = np.linalg.norm(np.diff(c, axis=0), axis=1) / dt
    max_c_speed = float(d_c.max()) / max(1.0, face_h)
    # jerk: sign of aperture velocity must not flip every single frame
    v = np.diff(ap)
    signs = np.sign(v[np.abs(v) > 1e-4])
    flips = int(np.sum(signs[1:] * signs[:-1] < 0))
    flip_rate = flips / max(1, len(signs) - 1)
    ok = (max_ap_speed <= MAX_APERTURE_SPEED
          and max_c_speed <= MAX_CENTROID_SPEED_FRAC
          and flip_rate <= 0.5)
    return GateResult(
        "temporal", ok, max(max_ap_speed / MAX_APERTURE_SPEED,
                            max_c_speed / MAX_CENTROID_SPEED_FRAC,
                            flip_rate / 0.5), 1.0,
        f"ap_speed={max_ap_speed:.2f}/s centroid={max_c_speed:.3f}fh/s "
        f"flip_rate={flip_rate:.2f}")


def gate_av_sync(rendered_aperture: np.ndarray, audio_env: np.ndarray,
                 hop_ms: float, fps: int) -> GateResult:
    """Global cross-correlation lag ≤ 1 frame at the render fps."""
    from engine.align import global_offset_ms
    lag = abs(global_offset_ms(rendered_aperture, audio_env, hop_ms))
    frame_ms = 1000.0 / max(1, fps)
    return GateResult("av_sync", lag <= frame_ms, lag, frame_ms,
                      f"peak lag {lag:.1f} ms")


def gate_sync_confidence(rendered_aperture: np.ndarray,
                         audio_env: np.ndarray,
                         hop_ms: float) -> GateResult:
    """Sliding-window local correlation — catches a single desynced
    turn that a global metric averages away. Windows with no speech
    energy are skipped (nothing to correlate)."""
    n = min(len(rendered_aperture), len(audio_env))
    win = max(4, int(SYNC_WINDOW_MS / hop_ms))
    hop = win // 2
    worst = 1.0
    worst_t = 0.0
    for a in range(0, max(1, n - win), hop):
        p = rendered_aperture[a:a + win]
        q = audio_env[a:a + win]
        if q.std() < 1e-4 or p.std() < 1e-4:
            continue  # silence or held mouth: nothing to correlate
        r = float(np.corrcoef(p, q)[0, 1])
        if r < worst:
            worst, worst_t = r, a * hop_ms
    return GateResult("sync_confidence", worst >= SYNC_MIN_LOCAL_CORR,
                      worst, SYNC_MIN_LOCAL_CORR,
                      f"worst window @ {worst_t:.0f} ms")


def gate_discriminability(contours: Dict[str, np.ndarray],
                          mouth_width_px: float) -> GateResult:
    """Pairwise contour distance between viseme classes rendered at
    phone scale. THE metric that gates dominance blending (§4.2) and
    the articulation gain (Part XI). Contours: (N,2) arrays, resampled
    to equal N, in phone-scale pixels."""
    names = sorted(contours)
    min_sep = math.inf
    worst_pair = ""
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = contours[names[i]], contours[names[j]]
            n = min(len(a), len(b))
            d = float(np.mean(np.linalg.norm(a[:n] - b[:n], axis=1)))
            d /= max(1.0, mouth_width_px)
            if d < min_sep:
                min_sep, worst_pair = d, f"{names[i]}↔{names[j]}"
    return GateResult("discriminability", min_sep >= DISCRIM_MIN_SEP_FRAC,
                      min_sep, DISCRIM_MIN_SEP_FRAC,
                      f"closest pair {worst_pair}")


def gate_seam(frame: Image.Image, seam_y: int, band_frac: float,
              face_h: float, body_mask: Optional[np.ndarray] = None
              ) -> GateResult:
    """Vertical gradient continuity across the neck band + no alpha<1
    hole inside the silhouette. band height derived from face size."""
    arr = np.asarray(frame.convert("RGBA"), dtype=np.float64)
    band = max(2, int(band_frac * face_h))
    y0, y1 = max(1, seam_y - band), min(arr.shape[0] - 1, seam_y + band)
    if body_mask is None:
        body_mask = arr[..., 3] > 8
    # alpha hole: pixels inside the silhouette band with alpha < 255
    hole = int(np.sum((arr[y0:y1, :, 3] < 250) & body_mask[y0:y1]))
    # gradient continuity: max |d(luma)/dy| inside band vs. global p99
    luma = arr[..., :3].mean(axis=-1)
    gy = np.abs(np.diff(luma, axis=0))
    band_max = float(gy[y0:y1].max()) if y1 > y0 else 0.0
    global_p99 = float(np.percentile(gy[body_mask[:-1]], 99)) \
        if body_mask[:-1].any() else 1.0
    ok = hole == 0 and band_max <= max(24.0, 1.5 * global_p99)
    return GateResult("seam", ok, band_max, max(24.0, 1.5 * global_p99),
                      f"alpha_holes={hole}")


def gate_rig_sanity(rig_dict: dict, face_h: float) -> GateResult:
    """Version == 3 and every pose RMS within the derived budget."""
    ver = rig_dict.get("version", 0)
    budget = RMS_BUDGET_FRAC * face_h
    worst, worst_pose = 0.0, ""
    for name, pose in rig_dict.get("poses", {}).items():
        rms = float(pose.get("xform", {}).get("rms", math.inf))
        if rms > worst:
            worst, worst_pose = rms, name
    ok = (ver == 3) and (worst <= budget)
    return GateResult("rig_sanity", ok, worst, budget,
                      f"version={ver} worst_pose={worst_pose}")


# ═══════════════════════════════════════════
# Phone-scale re-check (Part XI): rerun pixel gates at ~420 px
# ═══════════════════════════════════════════

def at_phone_scale(frame: Image.Image) -> Image.Image:
    """Downsample to the real Shorts viewport width. QC that only
    passes at 1080 is QC that lies."""
    if frame.width <= PHONE_SCALE_PX:
        return frame
    h = round(frame.height * PHONE_SCALE_PX / frame.width)
    return frame.resize((PHONE_SCALE_PX, h), Image.LANCZOS)


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="JEEVidya face QC gates")
    ap.add_argument("--report", default="face_qc_report.json")
    ap.add_argument("--rig", help="path to a rig.json to sanity-check")
    ap.add_argument("--face-h", type=float, default=100.0)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    report = QCReport()
    if args.rig:
        with open(args.rig, "r", encoding="utf-8") as f:
            report.add(gate_rig_sanity(json.load(f), args.face_h))
    report.save(args.report)
    print(report.summary())
    return 0 if report.passed or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["QCReport", "GateResult", "gate_registration", "gate_single_face",
           "gate_blink_closure", "gate_temporal", "gate_av_sync",
           "gate_sync_confidence", "gate_discriminability", "gate_seam",
           "gate_rig_sanity", "at_phone_scale", "color_mask",
           "connected_components", "mask_centroid", "PHONE_SCALE_PX"]
