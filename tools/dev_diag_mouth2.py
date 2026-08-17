"""Diagnostic: is the mouth registration error a WRONG PLACE, or two
different definitions of "centroid"?

`MouthRaster.predicted_centroid` averages the outer contour's VERTICES.
The detector measures the AREA centroid of the painted lip body, which is
the outer polygon MINUS the punched-out aperture — a ring. Those are not
the same point, and the difference grows with the aperture.

This prints, per viseme, the rendered centroid against BOTH definitions:

  vtx   — mean of outer vertices          (what predict() returns today)
  ring  — area centroid of outer − inner  (what the pixels actually are)

Usage:  python3 -m tools.dev_diag_mouth2 [chintu gudiya]
"""
from __future__ import annotations

import math
import sys
from typing import List, Sequence, Tuple

from engine.bone_engine import BoneEngine, PuppetPose
from engine.mouth_model import lip_contour
from engine.rig import Rig
from tools.face_qc import mask_centroid
from tools.verify_face import (_baked_viseme_classes, _mouth_blob,
                               _roi_from_variation)


def poly_area_centroid(pts: Sequence[Tuple[float, float]]
                       ) -> Tuple[float, float, float]:
    """(cx, cy, signed_area) of a simple polygon."""
    a = 0.0
    cx = 0.0
    cy = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cr = x0 * y1 - x1 * y0
        a += cr
        cx += (x0 + x1) * cr
        cy += (y0 + y1) * cr
    a *= 0.5
    if abs(a) < 1e-9:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (sum(xs) / n, sum(ys) / n, 0.0)
    return (cx / (6.0 * a), cy / (6.0 * a), a)


def ring_centroid(outer: Sequence[Tuple[float, float]],
                  inner: Sequence[Tuple[float, float]]
                  ) -> Tuple[float, float]:
    ox, oy, oa = poly_area_centroid(outer)
    ix, iy, ia = poly_area_centroid(inner)
    oa, ia = abs(oa), abs(ia)
    if ia >= oa or oa - ia < 1e-6:
        return (ox, oy)
    w = oa - ia
    return ((ox * oa - ix * ia) / w, (oy * oa - iy * ia) / w)


def diag(character: str) -> None:
    rig = Rig.load(character)
    eng = BoneEngine(rig)
    asm = eng.assembly
    mr = asm.mouth
    face_h = max(1.0, rig.head.face_height * eng.scale)
    pal = rig.head.palette or {}
    lip = tuple(pal.get("lip", (170, 80, 80)))
    shadow = tuple(pal["lip_shadow"]) if "lip_shadow" in pal else None
    budget = 0.6 / 100.0 * face_h
    print(f"\n═══ {character}   budget={budget:.2f}px  "
          f"mouth.center={mr.center} mouth.scale={mr.scale}")

    names = _baked_viseme_classes(rig)
    frames = {}
    for v in names:
        frames[v] = eng.render(PuppetPose(viseme=v, viseme_to=v,
                                         mouth_open=1.0))
    rest = eng.render(PuppetPose(mouth_open=0.0))
    roi = _roi_from_variation(list(frames.values()) + [rest], face_h)

    print(f"    {'viseme':15s} {'err_vtx':>8s} {'err_ring':>9s} "
          f"{'d_vtx':>16s} {'d_ring':>16s}")
    for v in names:
        pose = PuppetPose(viseme=v, viseme_to=v, mouth_open=1.0)
        ch, head, fp, tp, t = eng._channels(pose.clamped(), None)
        aff = asm.affine(fp, tp, t, head)
        p = ch.mouth
        outer_n, inner_n = lip_contour(p)
        cx, cy = mr.center
        outer = [(cx + x * mr.scale, cy + y * mr.scale) for x, y in outer_n]
        inner = [(cx + x * mr.scale, cy + y * mr.scale) for x, y in inner_n]
        vtx = mr.predicted_centroid(p)
        rng = ring_centroid(outer, inner)
        pv = aff.apply_feature_point(*vtx)
        pr = aff.apply_feature_point(*rng)
        det = mask_centroid(_mouth_blob(frames[v], lip, shadow, roi))
        if det is None:
            print(f"    {v:15s} no mask")
            continue
        ev = math.dist(pv, det)
        er = math.dist(pr, det)
        print(f"    {v:15s} {ev:8.2f} {er:9.2f} "
              f"({pv[0]:6.1f},{pv[1]:6.1f}) ({pr[0]:6.1f},{pr[1]:6.1f})"
              f"   det=({det[0]:6.1f},{det[1]:6.1f})")


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["chintu", "gudiya"]):
        diag(c)
