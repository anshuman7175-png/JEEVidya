"""Diagnostic: what the mouth gates actually see.

Prints, per character:
  A. per viseme — predict()['mouth'] vs the rendered lip blob centroid, at
     FULL canvas scale (with the mouth ROI, as `registration:mouth` does)
     and the blob's area/bbox
  B. the SAME at phone scale WITHOUT a region, which is what feeds the
     discriminability gate — so a blob that is really the whole cheek
     shows up as an absurd area
  C. the pose blend used by `pose_mouth_lock`, frame by frame

Read-only: renders, measures, prints. No files written.

Usage:  python3 -m tools.dev_diag_mouth [chintu gudiya]
"""
from __future__ import annotations

import math
import sys
from typing import Dict

import numpy as np

from engine.bone_engine import BoneEngine, PuppetPose
from engine.rig import Rig
from tools.face_qc import at_phone_scale, mask_centroid
from tools.verify_face import (_baked_viseme_classes, _mask_bbox, _mouth_blob,
                               _roi_from_variation, _lip_mask)


def diag(character: str) -> None:
    rig = Rig.load(character)
    eng = BoneEngine(rig)
    face_h = max(1.0, rig.head.face_height * eng.scale)
    pal = rig.head.palette or {}
    lip = tuple(pal.get("lip", (170, 80, 80)))
    shadow = tuple(pal["lip_shadow"]) if "lip_shadow" in pal else None
    print(f"\n═══ {character}  face_h={face_h:.1f} lip={lip} "
          f"shadow={shadow} scale={eng.scale:.4f}")

    names = _baked_viseme_classes(rig)
    held: Dict[str, tuple] = {}
    for v in names:
        p = PuppetPose(viseme=v, viseme_to=v, mouth_open=1.0)
        held[v] = (eng.render(p), eng.predict(p))
    rest = eng.render(PuppetPose(mouth_open=0.0))
    roi = _roi_from_variation([f for f, _ in held.values()] + [rest], face_h)
    print(f"  mouth ROI px={0 if roi is None else int(roi.sum())} "
          f"bbox={None if roi is None else _mask_bbox(roi)}")

    print("\n  A. full scale, inside ROI")
    print(f"    {'viseme':16s} {'pred':>18s} {'det':>18s} {'err':>7s} "
          f"{'area':>7s}  bbox")
    for v in names:
        frame, pred = held[v]
        m = _mouth_blob(frame, lip, shadow, roi)
        det = mask_centroid(m)
        pm = pred["mouth"]
        err = math.dist(pm, det) if det else float("inf")
        d = f"({det[0]:.1f},{det[1]:.1f})" if det else "None"
        print(f"    {v:16s} ({pm[0]:7.1f},{pm[1]:7.1f}) {d:>18s} "
              f"{err:7.2f} {int(m.sum()):7d}  {_mask_bbox(m)}")

    print("\n  B. phone scale, NO region (feeds discriminability)")
    ph = at_phone_scale(rest)
    print(f"    phone canvas={ph.size}")
    for v in names:
        pframe = at_phone_scale(held[v][0])
        raw = _lip_mask(pframe, lip, shadow)
        m = _mouth_blob(pframe, lip, shadow)
        print(f"    {v:16s} raw={int(raw.sum()):7d} blob={int(m.sum()):7d} "
              f"bbox={_mask_bbox(m)}")

    print("\n  C. pose blend (pose_mouth_lock)")
    poses = sorted(rig.poses or {"neutral": None})
    a = rig.canonical_pose if rig.canonical_pose in poses else poses[0]
    b = next((p for p in poses if p != a), a)
    print(f"    {a} → {b}   tol={0.6 / 100.0 * face_h * 2.0:.2f}")
    for i in range(6):
        t = i / 5.0
        pose = PuppetPose(viseme="MID_E", viseme_to="MID_E", mouth_open=0.6,
                          body_pose=a, body_pose_to=b, body_pose_blend=t)
        f = eng.render(pose)
        pred = eng.predict(pose)
        m = _mouth_blob(f, lip, shadow)
        det = mask_centroid(m)
        pm = pred["mouth"]
        err = math.dist(pm, det) if det else float("inf")
        d = f"({det[0]:.1f},{det[1]:.1f})" if det else "None"
        print(f"    t={t:.2f} pred=({pm[0]:7.1f},{pm[1]:7.1f}) det={d:>18s} "
              f"err={err:8.2f} area={int(m.sum()):7d} bbox={_mask_bbox(m)}")


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["chintu", "gudiya"]):
        diag(c)
