"""
JEEVidya — Face Verification Sweep (`jvmake verify-face`)
═════════════════════════════════════════════════════════
The enforcement layer the QC constitution (tools/face_qc.py) promises:
a deterministic sweep rendered on the REAL rig, with every core element
re-detected on rendered pixels and compared against the renderer's own
math. Any gate fails → non-zero exit → encode refused.

The sweep, per character:

  1. every baked viseme class HELD one frame
       → registration (mouth + both irises vs BoneEngine.predict),
         single_face, phone-scale discriminability of the rendered
         mouth contours
  2. a full blink ramp
       → blink_closure at blink=1 (zero iris pixels)
  3. one pose transition (canonical → the other registered pose, or a
     self-transition when the rig has a single pose)
       → per-frame MOUTH LOCK: rendered centroid must track
         predict() of the interpolated transform, every frame
       → temporal gate across the transition
  4. a synthetic speech line (default 6 s)
       → temporal, av_sync (peak lag ≤ 1 frame),
         sync_confidence (worst 1.2 s window)
  5. rig_sanity on the rig dict itself

Prediction and rendering share ONE channel mapping
(engine/bone_engine.BoneEngine._channels) and ONE affine
(engine/head_assembly.HeadAssembly.affine) — the gate cannot pass a lie
and the mouth mathematically cannot "fly".

Artifacts:  <out>/face_qc_report.json
            <out>/failures/<gate>_<frame>.png   (labelled failure strip)

Stdlib + numpy + PIL only. No ffmpeg, no mediapipe.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from config import settings
from engine.bone_engine import BoneEngine, PuppetPose
from engine.rig import Rig
from tools.face_qc import (GateResult, QCReport, at_phone_scale, color_mask,
                           gate_av_sync, gate_blink_closure,
                           gate_discriminability, gate_registration,
                           gate_rig_sanity, gate_single_face,
                           gate_sync_confidence, gate_temporal,
                           mask_centroid, PHONE_SCALE_PX)

# Sweep design constants — durations in SECONDS (frame counts are always
# derived from the live fps; never a literal frame count).
BLINK_S = 0.5              # half-second full blink ramp (down + up)
TRANSITION_S = 0.8         # pose cross-fade length in the sweep
SPEECH_S = 6.0             # synthetic speech line
CONTOUR_POINTS = 64        # resampled outer-contour samples per viseme
# Lip-color detection tolerance: wider than the QC default because baked
# art plates blend art pixels over the procedural lip fill.
LIP_TOL = 48


# ═══════════════════════════════════════════
# Detection on rendered pixels
# ═══════════════════════════════════════════

def _lip_mask(frame: Image.Image, lip_rgb: Tuple[int, int, int],
              shadow_rgb: Optional[Tuple[int, int, int]] = None
              ) -> np.ndarray:
    m = color_mask(frame, lip_rgb, tol=LIP_TOL)
    if shadow_rgb is not None:
        m |= color_mask(frame, shadow_rgb, tol=LIP_TOL)
    return m


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _mouth_contour(mask: np.ndarray, n: int = CONTOUR_POINTS
                   ) -> Optional[np.ndarray]:
    """Outer contour of the mouth mask as (n, 2) points, sampled at
    uniform angles around the centroid — rotation-consistent, so two
    visemes rendered at the same spot compare shape against shape."""
    c = mask_centroid(mask)
    if c is None:
        return None
    cx, cy = c
    ys, xs = np.nonzero(mask)
    ang = np.arctan2(ys - cy, xs - cx)
    rad = np.hypot(xs - cx, ys - cy)
    out = np.zeros((n, 2), dtype=np.float64)
    edges = np.linspace(-math.pi, math.pi, n + 1)
    idx = np.digitize(ang, edges) - 1
    idx = np.clip(idx, 0, n - 1)
    for k in range(n):
        sel = rad[idx == k]
        r = float(sel.max()) if len(sel) else 0.0
        theta = (edges[k] + edges[k + 1]) / 2.0
        out[k] = (cx + r * math.cos(theta), cy + r * math.sin(theta))
    # Express relative to centroid so contours compare SHAPE, not
    # placement (placement is the registration gate's job).
    out[:, 0] -= cx
    out[:, 1] -= cy
    return out


def _aperture(mask: np.ndarray, face_h: float) -> float:
    """Scale-free mouth openness proxy: vertical extent of the lip
    mask over face height."""
    bb = _mask_bbox(mask)
    if bb is None:
        return 0.0
    return (bb[3] - bb[1]) / max(1.0, face_h)


# ═══════════════════════════════════════════
# The sweep
# ═══════════════════════════════════════════

class SweepArtifacts:
    """Collects failure frames so a red build shows its defects."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.saved: List[str] = []

    def save(self, frame: Image.Image, tag: str) -> None:
        d = os.path.join(self.out_dir, "failures")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{tag}.png")
        frame.save(path)
        self.saved.append(path)


def _baked_viseme_classes(rig: Rig) -> List[str]:
    """Viseme classes with an art-fitted mouth target (the bake's
    output). Falls back to the full plate list when targets are empty."""
    names = [n for n in (rig.mouth_targets or {})]
    if not names:
        names = [n for n in (rig.visemes or {}) if not n.startswith("LID")]
    # Only classes the V enum knows can be driven through PuppetPose.
    from engine.visemes import V
    out = []
    for n in names:
        try:
            V(n)
            out.append(n)
        except ValueError:
            pass
    return sorted(set(out))


def run_sweep(character: str, out_dir: str,
              fps: Optional[int] = None,
              speech_s: float = SPEECH_S) -> QCReport:
    """Render the deterministic verification sweep for one character and
    run the full face-QC gate suite on the rendered frames."""
    fps = int(fps or settings.FPS)
    os.makedirs(out_dir, exist_ok=True)
    art = SweepArtifacts(out_dir)
    report = QCReport()

    rig = Rig.load(character)
    if not rig.is_v3():
        report.add(GateResult("rig_sanity", False, float(rig.version), 3.0,
                              f"{character}: rig version {rig.version} — "
                              "run `python3 jvmake.py rig --force`"))
        return report

    engine = BoneEngine(rig)
    if engine.assembly is None:
        report.add(GateResult("rig_sanity", False, 0.0, 1.0,
                              f"{character}: v3 rig built no HeadAssembly"))
        return report

    face_h = max(1.0, rig.head.face_height * engine.scale)
    palette = rig.head.palette or {}
    lip_rgb = tuple(palette.get("lip", (170, 80, 80)))
    shadow_rgb = tuple(palette["lip_shadow"]) if "lip_shadow" in palette \
        else None
    iris_rgb = tuple(palette.get("iris", (92, 58, 38)))

    # ─── 5 · rig sanity (cheap; run first so a broken rig
    #         fails before any rendering) ─────────────────
    report.add(gate_rig_sanity(rig.to_dict(), rig.head.face_height))

    # ─── 1 · every baked viseme class, held ───────────────
    contours: Dict[str, np.ndarray] = {}
    mouth_w_phone = 1.0
    for vname in _baked_viseme_classes(rig):
        pose = PuppetPose(viseme=vname, viseme_to=vname, mouth_open=1.0)
        frame = engine.render(pose)
        pred = engine.predict(pose)

        mask = _lip_mask(frame, lip_rgb, shadow_rgb)
        det = mask_centroid(mask)
        g = gate_registration(pred["mouth"], det, face_h,
                              f"mouth[{vname}]")
        report.add(g)
        if not g.passed:
            art.save(frame, f"registration_mouth_{vname}")

        bb = _mask_bbox(mask)
        if bb is not None:
            g = gate_single_face(frame, lip_rgb, bb, face_h)
            g.name = f"single_face[{vname}]"
            report.add(g)
            if not g.passed:
                art.save(frame, f"single_face_{vname}")
        else:
            report.add(GateResult(f"single_face[{vname}]", False,
                                  0.0, 1.0, "no lip pixels detected"))
            art.save(frame, f"single_face_{vname}")

        # phone-scale contour for the discriminability gate
        pframe = at_phone_scale(frame)
        pmask = _lip_mask(pframe, lip_rgb, shadow_rgb)
        pc = _mouth_contour(pmask)
        pbb = _mask_bbox(pmask)
        if pc is not None and pbb is not None:
            contours[vname] = pc
            mouth_w_phone = max(mouth_w_phone, float(pbb[2] - pbb[0]))

    if len(contours) >= 2:
        report.add(gate_discriminability(contours, mouth_w_phone))
    else:
        report.add(GateResult("discriminability", False, float(len(contours)),
                              2.0, "fewer than 2 viseme classes rendered "
                              "detectable mouth contours", skipped=False))

    # iris registration on the neutral held frame
    neutral = PuppetPose()
    frame = engine.render(neutral)
    pred = engine.predict(neutral)
    for eye in ("iris_l", "iris_r"):
        m = color_mask(frame, iris_rgb, tol=LIP_TOL)
        # split the iris mask by canvas midline of the two predictions
        mid_x = (pred["iris_l"][0] + pred["iris_r"][0]) / 2.0
        xs = np.zeros_like(m)
        if eye == "iris_l":
            xs[:, :int(mid_x)] = m[:, :int(mid_x)]
        else:
            xs[:, int(mid_x):] = m[:, int(mid_x):]
        g = gate_registration(pred[eye], mask_centroid(xs), face_h, eye)
        report.add(g)
        if not g.passed:
            art.save(frame, f"registration_{eye}")

    # ─── 2 · full blink ───────────────────────────────────
    n_blink = max(4, round(fps * BLINK_S))
    closed_frame = None
    for i in range(n_blink):
        t = i / max(1, n_blink - 1)
        b = 1.0 - abs(2.0 * t - 1.0)          # 0 → 1 → 0 triangle
        f = engine.render(PuppetPose(blink=b))
        if b >= 0.999:
            closed_frame = f
    if closed_frame is None:
        closed_frame = engine.render(PuppetPose(blink=1.0))
    g = gate_blink_closure(closed_frame, iris_rgb)
    report.add(g)
    if not g.passed:
        art.save(closed_frame, "blink_closure")

    # ─── 3 · pose transition + per-frame mouth lock ───────
    pose_names = sorted(rig.poses or {"neutral": None})
    from_pose = rig.canonical_pose if rig.canonical_pose in pose_names \
        else pose_names[0]
    to_pose = next((p for p in pose_names if p != from_pose), from_pose)
    n_tr = max(4, round(fps * TRANSITION_S))
    tr_apertures: List[float] = []
    tr_centroids: List[Tuple[float, float]] = []
    lock_worst = 0.0
    lock_tol = 0.6 / 100.0 * face_h * 2.0   # 2× reg budget across a blend
    lock_fail_frame = None
    for i in range(n_tr):
        t = i / max(1, n_tr - 1)
        pose = PuppetPose(viseme="MID_E", viseme_to="MID_E", mouth_open=0.6,
                          body_pose=from_pose, body_pose_to=to_pose,
                          body_pose_blend=t)
        f = engine.render(pose)
        pred = engine.predict(pose)
        mask = _lip_mask(f, lip_rgb, shadow_rgb)
        det = mask_centroid(mask)
        if det is None:
            lock_worst = math.inf
            lock_fail_frame = (f, i)
            continue
        err = math.dist(pred["mouth"], det)
        if err > lock_worst:
            lock_worst = err
            if err > lock_tol:
                lock_fail_frame = (f, i)
        tr_apertures.append(_aperture(mask, face_h))
        tr_centroids.append(det)
    g = GateResult("pose_mouth_lock", lock_worst <= lock_tol,
                   lock_worst, lock_tol,
                   f"{from_pose}→{to_pose}: worst rendered-vs-predicted "
                   "mouth centroid error across the blend")
    report.add(g)
    if not g.passed and lock_fail_frame is not None:
        art.save(lock_fail_frame[0], f"pose_mouth_lock_f{lock_fail_frame[1]}")
    if len(tr_apertures) >= 3:
        g = gate_temporal(tr_apertures, tr_centroids, face_h, fps)
        g.name = "temporal[transition]"
        report.add(g)

    # ─── 4 · synthetic speech line ────────────────────────
    n_sp = max(8, round(fps * speech_s))
    classes = _baked_viseme_classes(rig) or ["OPEN_A", "MID_E", "BILABIAL"]
    hop_ms = 1000.0 / fps
    env = np.zeros(n_sp, dtype=np.float64)
    apertures: List[float] = []
    centroids: List[Tuple[float, float]] = []
    # Deterministic syllabic envelope ~4 Hz with silences at both ends.
    for i in range(n_sp):
        t = i / fps
        e = max(0.0, math.sin(2 * math.pi * 4.0 * t)) \
            * (0.55 + 0.45 * math.sin(2 * math.pi * 0.35 * t + 1.0))
        if t < 0.4 or t > speech_s - 0.4:
            e = 0.0
        env[i] = e
    for i in range(n_sp):
        t = i / fps
        k = int(t * 3.0) % len(classes)           # new class ~every 333 ms
        k2 = (k + 1) % len(classes)
        blend = (t * 3.0) % 1.0
        pose = PuppetPose(viseme=classes[k], viseme_to=classes[k2],
                          viseme_blend=blend * 0.4, mouth_open=float(env[i]))
        f = engine.render(pose)
        mask = _lip_mask(f, lip_rgb, shadow_rgb)
        apertures.append(_aperture(mask, face_h))
        c = mask_centroid(mask)
        centroids.append(c if c is not None
                         else (centroids[-1] if centroids else (0.0, 0.0)))
    ap = np.asarray(apertures)
    report.add(gate_temporal(apertures, centroids, face_h, fps))
    report.add(gate_av_sync(ap, env, hop_ms, fps))
    report.add(gate_sync_confidence(ap, env, hop_ms))

    return report


def run_all(characters: Sequence[str], out_dir: str,
            fps: Optional[int] = None,
            speech_s: float = SPEECH_S) -> Tuple[bool, Dict[str, QCReport]]:
    """Sweep every character; write one combined report artifact."""
    os.makedirs(out_dir, exist_ok=True)
    reports: Dict[str, QCReport] = {}
    for name in characters:
        char_out = os.path.join(out_dir, name)
        reports[name] = run_sweep(name, char_out, fps=fps, speech_s=speech_s)
    combined = {"passed": all(r.passed for r in reports.values()),
                "characters": {n: r.to_dict() for n, r in reports.items()}}
    import json
    path = os.path.join(out_dir, "face_qc_report.json")
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    os.replace(tmp, path)
    return combined["passed"], reports


__all__ = ["run_sweep", "run_all", "SweepArtifacts",
           "BLINK_S", "TRANSITION_S", "SPEECH_S"]
