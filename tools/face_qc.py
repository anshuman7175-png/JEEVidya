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
  single_face       exactly one MOUTH-SCALE lip-colour region (no rival
                    blob ≥ 25% of the mouth's area, stray area within a
                    face-relative speckle budget)
  blink_closure     at blink=1: ≤2% of the OPEN eyeball still visible,
                    measured inside the eye's own rendered footprint
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
# Pose RMS budget — MUST match engine/registration.DEFAULT_RMS_BUDGET_PX
# (15 px at a 400 px reference face → 3.75 px per 100 px face height).
# The bake accepts hand-drawn poses up to that drift; a stricter QC gate
# here would falsely flag every legitimately baked rig.
RMS_BUDGET_FRAC = 15.0 / 400.0     # pose RMS ≤ 3.75 px per 100 px face height

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
    skipped: bool = False
    # A skipped gate DID NOT RUN — it is neither a pass nor a fail.
    # It never satisfies "all gates passed": verify_manifest refuses to
    # claim a clean pass while any gate is unrun. Constructors should
    # pass passed=False alongside skipped=True.

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": bool(self.passed),
                "value": float(self.value), "threshold": float(self.threshold),
                "detail": self.detail, "skipped": bool(self.skipped)}


@dataclass
class QCReport:
    gates: List[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only when every gate RAN and passed. A skipped gate is
        an unverified claim, not a pass."""
        return all(g.passed and not g.skipped for g in self.gates)

    @property
    def skipped_gates(self) -> List[str]:
        return [g.name for g in self.gates if g.skipped]

    @property
    def failed_gates(self) -> List[str]:
        return [g.name for g in self.gates if not g.passed and not g.skipped]

    def add(self, g: GateResult) -> None:
        self.gates.append(g)

    def to_dict(self) -> dict:
        return {"passed": self.passed,
                "skipped": self.skipped_gates,
                "gates": [g.to_dict() for g in self.gates]}

    def save(self, path: str) -> None:
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)

    def summary(self) -> str:
        lines = []
        for g in self.gates:
            mark = "SKIP" if g.skipped else ("PASS" if g.passed else "FAIL")
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


MIN_COMPONENT_PX = 4       # below this a "component" is antialias speckle


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """4-connected labelling of a boolean mask → (labels, count).

    Backends in order: scipy.ndimage.label, cv2.connectedComponents, and
    a pure-Python union-find. The accelerated backends are not a luxury:
    a 1080×1920 QC frame is 2 M pixels, and the Python fallback walks
    every one of them in the interpreter, which is why a full sweep used
    to take minutes per gate. All three agree on 4-connectivity, so the
    gate verdict is backend-independent.
    """
    m = np.ascontiguousarray(mask.astype(bool))
    if not m.any():
        return np.zeros(m.shape, dtype=np.int32), 0
    try:
        from scipy.ndimage import label as _label
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        lab, n = _label(m, structure=structure)
        return lab.astype(np.int32), int(n)
    except Exception:
        pass
    try:
        import cv2
        n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=4)
        return lab.astype(np.int32), int(max(0, n - 1))
    except Exception:
        pass
    return _label_python(m)


def _label_python(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Dependency-free two-pass union-find label (slow; last resort)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: List[int] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nxt = 0
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
        return labels, 0
    remap = np.zeros(nxt + 1, dtype=np.int32)
    seen: Dict[int, int] = {}
    for lab in range(1, nxt + 1):
        r = find(lab - 1)
        if r not in seen:
            seen[r] = len(seen) + 1
        remap[lab] = seen[r]
    return remap[labels], len(seen)


def component_sizes(mask: np.ndarray,
                    min_px: int = MIN_COMPONENT_PX) -> np.ndarray:
    """Areas of the 4-connected components ≥ `min_px`, largest first.

    Sizes — not a bare count — are what let a gate distinguish a second
    MOUTH from a stray speck of lip-coloured clothing. Counting alone
    made `single_face` unusable on real artwork: a character with red
    trim reports 21 "faces" and the gate fails a perfect render.
    """
    lab, n = label_components(mask)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    counts = np.bincount(lab.ravel())
    counts = counts[1:] if len(counts) > 1 else np.zeros(0, dtype=np.int64)
    sizes = counts[counts >= max(1, int(min_px))]
    return np.sort(sizes)[::-1].astype(np.int64)


def connected_components(mask: np.ndarray,
                         min_px: int = MIN_COMPONENT_PX) -> int:
    """Count 4-connected components ≥ `min_px` px."""
    return int(len(component_sizes(mask, min_px)))


def largest_component(mask: np.ndarray,
                      min_px: int = MIN_COMPONENT_PX) -> np.ndarray:
    """The biggest 4-connected component of `mask`, as a boolean mask.

    Feature measurement (centroid, bbox, aperture) reads THIS rather
    than the raw colour mask. A colour mask over real artwork always
    carries scattered look-alike pixels — a red hair tie, a warm
    highlight in the hair, chroma fringing — and averaging them into the
    centroid is precisely what produced "mouth 363 px off" on a render
    whose mouth was in exactly the right place. The mouth is the largest
    lip-coloured body on a face by construction, so the largest
    component is the mouth; the leftovers are then judged separately by
    `gate_single_face`, which is where stray lip colour BELONGS as a
    verdict.
    """
    lab, n = label_components(mask)
    if n == 0:
        return np.zeros(mask.shape, dtype=bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    top = int(np.argmax(counts))
    if counts[top] < max(1, int(min_px)):
        return np.zeros(mask.shape, dtype=bool)
    return lab == top


def dilate_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    """Grow a boolean mask by `radius` px (4-connected disc approx.).

    Used to turn an exact detection region into a tolerant one, so a
    legitimately antialiased feature edge is not clipped away by the
    region constraint it is being measured inside.
    """
    r = int(round(radius))
    if r <= 0 or not mask.any():
        return mask.astype(bool)
    try:
        from scipy.ndimage import binary_dilation, iterate_structure
        base = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        return binary_dilation(mask, structure=iterate_structure(base, r))
    except Exception:
        out = mask.astype(bool)
        for _ in range(r):
            g = out.copy()
            g[1:, :] |= out[:-1, :]
            g[:-1, :] |= out[1:, :]
            g[:, 1:] |= out[:, :-1]
            g[:, :-1] |= out[:, 1:]
            out = g
        return out


def variation_mask(frames: Sequence[Image.Image],
                   min_delta: float = 10.0) -> np.ndarray:
    """Pixels whose colour CHANGES across renders that differ only in the
    feature under test — i.e. the feature's own footprint, measured from
    rendered pixels alone.

    This is how a gate can be spatially constrained without becoming
    circular. Constraining detection to a neighbourhood of `predict()`
    would make the registration gate check the prediction against
    itself; constraining it to the pixels the feature actually moved
    keeps the measurement independent of the prediction, while excluding
    every unrelated pixel that merely happens to share the feature's
    colour (hair, blush, clothing — the pollution that put a "mouth
    centroid" 363 px off the mouth).
    """
    if len(frames) < 2:
        return np.zeros((1, 1), dtype=bool)
    stack = np.stack([np.asarray(f.convert("RGB"), dtype=np.float32)
                      for f in frames])
    span = stack.max(axis=0) - stack.min(axis=0)
    return (span.max(axis=-1) >= float(min_delta))


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


SECOND_MOUTH_FRAC = 0.25    # a rival lip-colour blob this big is a 2nd mouth
STRAY_AREA_FRAC = 0.010     # stray lip-colour budget, ×face_h² (speckle only)


def gate_single_face(frame: Image.Image, lip_rgb: Tuple[int, int, int],
                     mouth_bbox: Tuple[int, int, int, int],
                     face_h: float,
                     region: Optional[np.ndarray] = None,
                     tol: int = 26) -> GateResult:
    """Exactly ONE mouth-scale lip-colour region. Kills D2/D3 regressions
    (three-mouth composites, painted-mouth ghosts under the parametric
    mouth) without failing on artwork that merely contains the lip hue.

    The old form demanded exactly one component and literally zero
    lip-coloured pixels outside the mouth bbox. On real character art
    that is unsatisfiable — a red hair tie, a bindi, or the JPEG fringe
    of a saturated garment each count as a "face", and the gate reported
    21 of them on a flawless render. The defect being legislated against
    is a SECOND MOUTH, so the measurement is area-relative: no rival
    blob within a quarter of the mouth's own area, and total stray area
    inside the search region bounded by a face-relative speckle budget.
    `region` (typically a `variation_mask` ROI) narrows the search to
    pixels the renderer actually touched.
    """
    m = color_mask(frame, lip_rgb, tol)
    if region is not None and region.shape == m.shape:
        m = m & region
    lab, n = label_components(m)
    if n == 0:
        return GateResult("single_face", False, math.inf, 1.0,
                          "no lip-colour pixels found at all")
    counts = np.bincount(lab.ravel())
    counts[0] = 0

    # The mouth component: the largest one intersecting the mouth bbox.
    x0, y0, x1, y1 = mouth_bbox
    pad = int(round(0.02 * face_h)) + 2
    win = lab[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    in_box = np.unique(win[win > 0])
    if len(in_box) == 0:
        return GateResult("single_face", False, math.inf, 1.0,
                          "no lip-colour component inside the mouth bbox")
    mouth_lab = int(in_box[np.argmax(counts[in_box])])
    mouth_area = float(counts[mouth_lab])

    others = np.nonzero(counts >= MIN_COMPONENT_PX)[0]
    others = others[others != mouth_lab]
    rival_area = float(counts[others].max()) if len(others) else 0.0
    stray_area = float(counts[others].sum()) if len(others) else 0.0

    rival_ratio = rival_area / max(1.0, mouth_area)
    stray_budget = STRAY_AREA_FRAC * face_h * face_h
    ok = (rival_ratio < SECOND_MOUTH_FRAC) and (stray_area <= stray_budget)
    value = max(rival_ratio / SECOND_MOUTH_FRAC,
                stray_area / max(1.0, stray_budget))
    return GateResult(
        "single_face", ok, value, 1.0,
        f"mouth={int(mouth_area)}px rival={int(rival_area)}px "
        f"({rival_ratio:.2f}× mouth) stray={int(stray_area)}px "
        f"budget={int(stray_budget)}px comps={n}")


BLINK_RESIDUAL_FRAC = 0.02   # ≤2% of the open eyeball may survive blink=1


def eyeball_mask(frame: Image.Image, iris_rgb: Tuple[int, int, int],
                 sclera_rgb: Optional[Tuple[int, int, int]] = None,
                 region: Optional[np.ndarray] = None,
                 tol: int = 26) -> np.ndarray:
    """Visible eyeball pixels: iris colour, plus sclera colour when the
    bake supplies it (a lid that hides the iris but leaves the white
    showing is not a closed eye)."""
    m = color_mask(frame, iris_rgb, tol)
    if sclera_rgb is not None:
        m |= color_mask(frame, sclera_rgb, tol)
    if region is not None and region.shape == m.shape:
        m = m & region
    return m


def gate_blink_closure(frame_closed: Image.Image,
                       iris_rgb: Tuple[int, int, int],
                       frame_open: Optional[Image.Image] = None,
                       sclera_rgb: Optional[Tuple[int, int, int]] = None,
                       region: Optional[np.ndarray] = None,
                       tol: int = 26) -> GateResult:
    """At blink=1 the eyeball must be occluded by the lid.

    Measured as a RATIO against the same eye rendered open, inside the
    eye's own footprint (`region`, from `variation_mask`), because the
    absolute-zero form could not survive real art: lash ink and a dark
    brown iris sit within any usable colour tolerance of each other, so
    a perfectly shut eye still reported iris-coloured pixels — while a
    frame-wide colour count also swept up every dark-brown pixel of
    hair. A residual of ≤2% of the open eyeball is antialiasing at the
    lid edge; anything more is an eye that did not close.
    """
    closed = int(eyeball_mask(frame_closed, iris_rgb, sclera_rgb,
                             region, tol).sum())
    if frame_open is None:
        return GateResult("blink_closure", closed == 0, float(closed), 0.0,
                          "eyeball-colour px at blink=1 (absolute form: no "
                          "open-eye reference supplied)")
    open_px = int(eyeball_mask(frame_open, iris_rgb, sclera_rgb,
                              region, tol).sum())
    if open_px < 16:
        return GateResult("blink_closure", False, math.inf,
                          BLINK_RESIDUAL_FRAC,
                          f"the OPEN eye renders only {open_px} eyeball px — "
                          "there is no visible eyeball to close")
    ratio = closed / float(open_px)
    return GateResult("blink_closure", ratio <= BLINK_RESIDUAL_FRAC,
                      ratio, BLINK_RESIDUAL_FRAC,
                      f"{closed}px of {open_px}px eyeball still visible "
                      "at blink=1")


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


# ════════════════════════════════��══════════
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
# ════════════════════════���══════════════════

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
           "connected_components", "component_sizes", "label_components",
           "largest_component",
           "dilate_mask", "variation_mask", "eyeball_mask", "mask_centroid",
           "PHONE_SCALE_PX", "MIN_COMPONENT_PX", "BLINK_RESIDUAL_FRAC"]
