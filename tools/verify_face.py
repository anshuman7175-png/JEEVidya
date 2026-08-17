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
from engine.rig import Rig, rig_dir
from tools.face_qc import (GateResult, QCReport, at_phone_scale, color_mask,
                           dilate_mask, fit_fixed_axes_ellipse, gate_av_sync,
                           gate_blink_closure, gate_discriminability,
                           gate_registration, gate_rig_sanity,
                           gate_single_face, gate_sync_confidence,
                           gate_temporal, largest_component, mask_centroid,
                           variation_mask, PHONE_SCALE_PX)

# Sweep design constants — durations in SECONDS (frame counts are always
# derived from the live fps; never a literal frame count).
BLINK_S = 0.5              # half-second full blink ramp (down + up)
TRANSITION_S = 0.8         # pose cross-fade length in the sweep
SPEECH_S = 6.0             # synthetic speech line
CONTOUR_POINTS = 64        # resampled outer-contour samples per viseme
# Lip-color detection tolerance: wider than the QC default because baked
# art plates blend art pixels over the procedural lip fill.
LIP_TOL = 48
IRIS_TOL = 30              # irises are small; a loose tolerance swamps them
ROI_DELTA = 10.0           # RGB span that counts as "this pixel moved"
ROI_GROW_FRAC = 0.05       # ×face_h, ROI dilation so soft edges survive


def _lip_mask(frame: Image.Image, lip_rgb: Tuple[int, int, int],
              shadow_rgb: Optional[Tuple[int, int, int]] = None,
              region: Optional[np.ndarray] = None) -> np.ndarray:
    m = color_mask(frame, lip_rgb, tol=LIP_TOL)
    if shadow_rgb is not None:
        m |= color_mask(frame, shadow_rgb, tol=LIP_TOL)
    if region is not None and region.shape == m.shape:
        m = m & region
    return m


def _mouth_blob(frame: Image.Image, lip_rgb: Tuple[int, int, int],
                shadow_rgb: Optional[Tuple[int, int, int]] = None,
                region: Optional[np.ndarray] = None) -> np.ndarray:
    """The mouth as a single connected body of lip colour.

    Every measurement below (centroid, bbox, aperture, contour) reads
    this instead of the raw colour mask, so scattered look-alike pixels
    elsewhere in the artwork cannot drag the numbers. `single_face`
    still judges the leftovers, so nothing is swept under the rug.
    """
    return largest_component(_lip_mask(frame, lip_rgb, shadow_rgb, region))


# ── locating a rendered iris ────────────────────────────────
# Cap on the detector datum below, as a fraction of the eyeball's
# semi-minor axis. The datum corrects a soft-edge disagreement of a
# pixel or two; anything larger is a real defect and must fail rather
# than be calibrated away.
IRIS_DATUM_FRAC = 0.20


def _iris_center(mask: np.ndarray,
                 axes: Optional[Tuple[float, float, float]]
                 ) -> Optional[Tuple[float, float]]:
    """The centre of the eyeball whose visible pixels are in `mask`.

    NOT the centroid of `mask`. On this art the eyeball is a brown ring
    around a near-black pupil with a white highlight in the upper outer
    quadrant, so the iris-coloured pixels form a BOTTOM CRESCENT whose
    centroid sits ~14 px below the true centre on all four eyes, against
    a ~1.8 px budget (measured, renderer uninvolved: tools/
    dev_iris_control). The crescent's outer edge is however a true arc of
    the eyeball ellipse, and the bake measured that ellipse's axes from
    the same artwork, so the centre is recovered by fitting the known
    axes to the arc.
    """
    if axes is None:
        return None
    blob = largest_component(mask)
    if not blob.any():
        return None
    return fit_fixed_axes_ellipse(blob, (axes[0], axes[1]), axes[2])


def _iris_datum(plate: Image.Image, art: Dict[str, object],
                tol: int = IRIS_TOL) -> Optional[Tuple[float, float]]:
    """How far `_iris_center` reads from the baked eyeball centre on the
    UNTOUCHED head plate, in PLATE pixels.

    Takes the raw `rig.head.art_eye_*` dict, not the assembly's
    EyeGeometry: `HeadAssembly._scaled_eye` pre-multiplies the assembly's
    copy into CANVAS space, so using it here would measure a plate with
    canvas-space coordinates. That mistake is invisible on a character
    whose scale is exactly 1.0 (gudiya) and shifts every reading by ~17 px
    on one whose scale is not (chintu, 0.957) — so the two spaces are kept
    strictly apart: this function is plate-space only, and its caller
    scales the result once.

    Two estimators of one hand-painted edge do not agree to zero. The
    bake fits the eyeball's full ellipse; the gate sees only the pixels
    that survive a colour tolerance, and antialiasing against the sclera
    and lash insets that crescent inward by a pixel or two — which, fitted
    with fixed axes, biases the centre consistently upward (measured
    dy = -1.7 … -3.3 px on the four eyes, dx within ±2).
    
    Both sides of the gate must therefore measure the SAME way. This is
    the detector's own reading at rest, with the renderer not involved, so
    subtracting it cancels the estimator disagreement while leaving the
    1.76 px budget untouched — a real misplacement still scores in full,
    because this constant is fixed by the source art and does not move
    when the renderer puts an iris in the wrong place.
    
    Deliberately NOT solved by masking iris ∪ pupil to get a fuller disc:
    the pupil (1,0,0) and the lash (36,9,3) are both near-black and
    inseparable at any usable tolerance, so the union swallows the lash
    arcing over the eye — measured at 242% of the iris area with 19.8 px
    of centroid error on chintu's right eye. Rejected on measurement.
    """
    try:
        ax, ay = (float(v) for v in (art.get("iris_axes") or ()))
        cx, cy = (float(v) for v in (art.get("iris_c") or ()))
    except (TypeError, ValueError):
        return None
    if ax <= 0.0 or ay <= 0.0:
        return None
    rgb = (art.get("colors") or {}).get("iris")
    if rgb is None:
        return None
    angle = float(art.get("iris_angle") or 0.0)
    # A generous box around the known iris, so that "largest component"
    # cannot be won by some other feature of the face.
    pad = int(max(ax, ay) * 2.2)
    x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
    crop = plate.crop((x0, y0, int(cx + pad), int(cy + pad)))
    fit = _iris_center(color_mask(crop, tuple(rgb), tol=tol),
                       (ax, ay, angle))
    if fit is None:
        return None
    return (fit[0] + x0 - float(cx), fit[1] + y0 - float(cy))


def _roi_from_variation(frames: List[Image.Image], face_h: float
                        ) -> Optional[np.ndarray]:
    """A detection region derived from the RENDER itself: the bounding
    box of every pixel that changed across `frames`, grown by a
    face-relative margin.

    The frames handed in differ in exactly ONE driven channel (viseme,
    or blink), so the pixels that moved are that feature's own
    footprint. Using the filled bounding box rather than the raw
    difference keeps a feature's INVARIANT core (the lip centre that
    every viseme shares) inside the region, while still excluding the
    rest of the canvas. Nothing here consults `predict()`, so the
    registration gate stays an independent check rather than a tautology.
    """
    var = variation_mask(frames, ROI_DELTA)
    if var.shape == (1, 1) or not var.any():
        return None
    var = dilate_mask(var, ROI_GROW_FRAC * face_h)
    ys, xs = np.nonzero(var)
    region = np.zeros(var.shape, dtype=bool)
    region[int(ys.min()):int(ys.max()) + 1,
           int(xs.min()):int(xs.max()) + 1] = True
    return region


def _painted_region(frame: Image.Image, reference: Image.Image,
                    face_h: float) -> Optional[np.ndarray]:
    """Pixels where `frame` differs from a reference render — i.e. every
    pixel the face renderer actually PAINTED, anywhere on the canvas.

    This is the honest search region for the "is there a second mouth?"
    question. The gate used to scan the whole frame for lip COLOUR, on
    the theory that a ghost mouth could appear anywhere and a region
    would hide it. True, but unusable on this artwork: flat cel shading
    draws lips, skin, ears and clothing from one warm ramp, so lip
    colour matches 17.7% of chintu's body (measured) and the gate
    reported ~1000 components and a "rival blob 3× the mouth" on a
    render whose mouth was provably correct.

    Differencing against a reference keeps the whole canvas in scope —
    a ghost mouth on the shirt still shows up, because painting one
    CHANGES those pixels — while a cheek that merely shares the lip hue
    does not, because nothing painted it. So the gate keeps its full
    reach and loses only its false positives.

    The reference is a render of the same pose with the mouth closed and
    eyes open; anything the mouth/eye compositors added relative to it is
    by definition renderer output, not artwork.
    """
    a = np.asarray(frame.convert("RGB"), dtype=np.int16)
    b = np.asarray(reference.convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return None
    diff = np.abs(a - b).max(axis=-1) >= ROI_DELTA
    if not diff.any():
        return None
    # Grow so the antialiased rim of a painted feature counts as painted.
    return dilate_mask(diff, ROI_GROW_FRAC * face_h)


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
    sclera_rgb = tuple(palette["sclera"]) if "sclera" in palette else None

    # ─── 5 · rig sanity (cheap; run first so a broken rig
    #         fails before any rendering) ─────────────────
    report.add(gate_rig_sanity(rig.to_dict(), rig.head.face_height))

    # ─── 1 · every baked viseme class, held ───────────────
    # Two passes on purpose: render every viseme first so the mouth's own
    # footprint can be measured from how the render CHANGES between them
    # (`_roi_from_variation`), then gate inside that region. A held
    # closed mouth is included so the region covers the resting lips too.
    viseme_names = _baked_viseme_classes(rig)
    held: Dict[str, Tuple[Image.Image, Dict[str, Tuple[float, float]]]] = {}
    for vname in viseme_names:
        pose = PuppetPose(viseme=vname, viseme_to=vname, mouth_open=1.0)
        held[vname] = (engine.render(pose), engine.predict(pose))
    rest_pose = PuppetPose(mouth_open=0.0)
    rest_frame = engine.render(rest_pose)
    mouth_roi = _roi_from_variation(
        [f for f, _ in held.values()] + [rest_frame], face_h)
    if mouth_roi is None:
        report.add(GateResult(
            "mouth_motion", False, 0.0, ROI_DELTA,
            "no pixel changed across the baked viseme set — the mouth "
            "never opens (check the art-fitted mouth targets in "
            "rig.mouth_targets)"))
    else:
        report.add(GateResult("mouth_motion", True, float(mouth_roi.sum()),
                              1.0, "mouth footprint measured from the render"))

    contours: Dict[str, np.ndarray] = {}
    mouth_w_phone = 1.0
    for vname in viseme_names:
        frame, pred = held[vname]

        mask = _mouth_blob(frame, lip_rgb, shadow_rgb, mouth_roi)
        det = mask_centroid(mask)
        g = gate_registration(pred["mouth"], det, face_h,
                              f"mouth[{vname}]")
        report.add(g)
        if not g.passed:
            art.save(frame, f"registration_mouth_{vname}")

        bb = _mask_bbox(mask)
        if bb is not None:
            # Judged over the WHOLE canvas, but only on pixels the
            # renderer painted (vs. the mouth-closed reference). A ghost
            # mouth anywhere is still caught — painting one changes those
            # pixels — while a lip-hued cheek or shirt is not, because
            # nothing painted it. Searching raw lip colour instead made
            # this gate unsatisfiable on cel-shaded art.
            # Union with the mouth's own footprint: a REST frame barely
            # differs from the rest reference, so on its own `painted`
            # could exclude the very mouth the gate must find.
            painted = _painted_region(frame, rest_frame, face_h)
            if painted is not None and mouth_roi is not None:
                painted = painted | mouth_roi
            elif painted is None:
                painted = mouth_roi
            g = gate_single_face(frame, lip_rgb, bb, face_h,
                                 region=painted, tol=LIP_TOL)
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
        pmask = _mouth_blob(pframe, lip_rgb, shadow_rgb)
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

    # ─── 2 · full blink, then iris registration ───────────
    # The blink supplies the eyes' own footprint: the only pixels that
    # differ between eyes-open and eyes-shut ARE the eyes. Irises are
    # small — far smaller than a patch of same-toned hair — so unlike the
    # mouth they cannot be found by "largest component" alone, and this
    # region is what makes their registration measurable at all.
    neutral = PuppetPose()
    open_frame = engine.render(neutral)
    pred = engine.predict(neutral)

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

    eye_roi = _roi_from_variation([open_frame, closed_frame], face_h)
    if eye_roi is None:
        report.add(GateResult(
            "eye_motion", False, 0.0, ROI_DELTA,
            "no pixel changed between blink=0 and blink=1 — the eyes "
            "never move"))
    else:
        report.add(GateResult("eye_motion", True, float(eye_roi.sum()), 1.0,
                              "eye footprint measured from the render"))

    # The eyeball geometry the RENDERER uses. `HeadAssembly._scaled_eye`
    # has already mapped these into CANVAS pixels, so they are used
    # verbatim — multiplying by `engine.scale` again double-scales them.
    # The ellipse's angle needs no mapping here because every iris gate
    # below renders the neutral pose, where the head carries no rotation.
    S = float(engine.scale)
    geos = {"iris_l": engine.assembly.eyes.left.geo,
            "iris_r": engine.assembly.eyes.right.geo}
    # …while the datum is measured on the head PLATE, which is canonical
    # space. Hence the raw `art_eye_*` dicts for that side of the
    # measurement, scaled into canvas space exactly once, below.
    arts = {"iris_l": rig.head.art_eye_l, "iris_r": rig.head.art_eye_r}
    plate_path = os.path.join(rig_dir(character), "head_canonical.png")
    plate = Image.open(plate_path).convert("RGBA") \
        if os.path.exists(plate_path) else None

    iris_axes: Dict[str, Tuple[float, float, float]] = {}
    iris_datum: Dict[str, Tuple[float, float]] = {}
    for eye, geo in geos.items():
        ax, ay = geo.iris_axes
        if ax > 0.0 and ay > 0.0:
            iris_axes[eye] = (float(ax), float(ay), float(geo.iris_angle))
        art = arts.get(eye) or {}
        d = _iris_datum(plate, art) if plate is not None else None
        if d is None:
            # Without a datum the two sides of the gate measure
            # differently, and the gate would be scoring an estimator
            # disagreement as a renderer defect. Refuse to guess.
            report.add(GateResult(
                f"iris_datum[{eye}]", False, 0.0, 1.0,
                "could not locate the eyeball on the untouched head "
                "plate — the iris gate has no like-for-like reference "
                "(check art_eye_*.iris_axes / colors.iris in rig.json)"))
            continue
        # Cap in the same (plate) space the datum was measured in.
        cap = IRIS_DATUM_FRAC * min(
            float(v) for v in (art.get("iris_axes") or (1.0, 1.0)))
        mag = math.hypot(d[0], d[1])
        report.add(GateResult(
            f"iris_datum[{eye}]", mag <= cap, mag, cap,
            f"detector reads {mag:.2f}px from the baked centre on the "
            f"untouched plate (dx={d[0]:+.2f}, dy={d[1]:+.2f}) — a "
            f"soft-edge estimator offset, cancelled on both sides"))
        iris_datum[eye] = (d[0] * S, d[1] * S)

    mid_x = int((pred["iris_l"][0] + pred["iris_r"][0]) / 2.0)
    m_iris = color_mask(open_frame, iris_rgb, tol=IRIS_TOL)
    if eye_roi is not None:
        m_iris = m_iris & eye_roi
    for eye in ("iris_l", "iris_r"):
        # Split by the canvas midline between the two eyes, then take that
        # side's largest body of iris colour as this iris.
        #
        # Which side of the midline an eye falls on is read from ITS OWN
        # PREDICTION, never inferred from the name. `iris_l` is the
        # CHARACTER's left eye, so it renders on the viewer's RIGHT —
        # hardcoding `iris_l → left half` scored each eye against the
        # other eye's prediction and reported ~150 px of error on a render
        # whose irises were within a pixel or two of correct. A mirrored
        # character, or one drawn in three-quarter view, would flip the
        # sides again; the prediction is the only honest source.
        m = np.zeros_like(m_iris)
        if pred[eye][0] < mid_x:
            m[:, :mid_x] = m_iris[:, :mid_x]
        else:
            m[:, mid_x:] = m_iris[:, mid_x:]
        det = _iris_center(m, iris_axes.get(eye))
        # Compare like with like: the prediction is where the eyeball's
        # CENTRE must land, and `det` is where this detector reads that
        # centre, so the prediction carries the detector's own measured
        # offset. Without it the gate charges a 2 px soft-edge
        # disagreement to the renderer.
        dat = iris_datum.get(eye, (0.0, 0.0))
        expect = (pred[eye][0] + dat[0], pred[eye][1] + dat[1])
        g = gate_registration(expect, det, face_h, eye)
        report.add(g)
        if not g.passed:
            art.save(open_frame, f"registration_{eye}")

    g = gate_blink_closure(closed_frame, iris_rgb, frame_open=open_frame,
                           sclera_rgb=sclera_rgb, region=eye_roi,
                           tol=IRIS_TOL)
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
        # No ROI across a pose blend: the head travels, so the mouth's
        # footprint moves with it. `_mouth_blob` stays honest here — it
        # constrains by connectivity, not by position.
        mask = _mouth_blob(f, lip_rgb, shadow_rgb)
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
        mask = _mouth_blob(f, lip_rgb, shadow_rgb, mouth_roi)
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
