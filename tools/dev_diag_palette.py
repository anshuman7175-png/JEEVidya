"""Diagnostic: can the mouth be separated from skin by COLOUR at all?

Prints the baked palette, the distance from each lip colour to every other
entry, and — inside the mouth's own measured footprint — how many pixels a
plain tolerance mask keeps versus a NEAREST-PALETTE rule (a pixel is mouth
only when its closest palette entry is a lip entry).

Usage:  python3 -m tools.dev_diag_palette [chintu gudiya]
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from engine.bone_engine import BoneEngine, PuppetPose
from engine.rig import Rig
from tools.face_qc import color_mask, mask_centroid
from tools.verify_face import (_baked_viseme_classes, _lip_mask, _mask_bbox,
                               _mouth_blob, _roi_from_variation, LIP_TOL)


def nearest_palette_mask(frame: Image.Image, palette: dict,
                         lip_keys) -> np.ndarray:
    """True where the pixel's nearest palette colour is one of `lip_keys`."""
    arr = np.asarray(frame.convert("RGB"), dtype=np.float32)
    keys = [k for k, v in palette.items()
            if isinstance(v, (list, tuple)) and len(v) >= 3]
    cols = np.array([[float(c) for c in palette[k][:3]] for k in keys],
                    dtype=np.float32)
    d = np.linalg.norm(arr[:, :, None, :] - cols[None, None, :, :], axis=-1)
    win = np.argmin(d, axis=-1)
    lip_idx = [i for i, k in enumerate(keys) if k in lip_keys]
    m = np.zeros(arr.shape[:2], dtype=bool)
    for i in lip_idx:
        m |= (win == i)
    return m


def diag(character: str) -> None:
    rig = Rig.load(character)
    eng = BoneEngine(rig)
    face_h = max(1.0, rig.head.face_height * eng.scale)
    pal = dict(rig.head.palette or {})
    print(f"\n═══ {character}")
    for k, v in pal.items():
        print(f"    {k:14s} {v}")
    lip = tuple(pal.get("lip", (170, 80, 80)))
    shadow = tuple(pal["lip_shadow"]) if "lip_shadow" in pal else None
    print("\n  distance from lip to every other entry "
          f"(tolerance in use = {LIP_TOL}, per-channel):")
    for k, v in pal.items():
        if not isinstance(v, (list, tuple)) or len(v) < 3:
            continue
        d = np.linalg.norm(np.array(lip[:3], dtype=float)
                           - np.array(v[:3], dtype=float))
        cheb = max(abs(int(lip[i]) - int(v[i])) for i in range(3))
        print(f"    lip↔{k:14s} euclid={d:7.1f} chebyshev={cheb:4d}"
              f"{'   ← inside tol' if cheb <= LIP_TOL else ''}")

    names = _baked_viseme_classes(rig)
    held = {}
    for v in names:
        held[v] = eng.render(PuppetPose(viseme=v, viseme_to=v,
                                       mouth_open=1.0))
    rest = eng.render(PuppetPose(mouth_open=0.0))
    roi = _roi_from_variation(list(held.values()) + [rest], face_h)
    print(f"\n  mouth ROI px={int(roi.sum())} bbox={_mask_bbox(roi)}")

    lip_keys = {"lip", "lip_shadow", "mouth", "teeth", "tongue"}
    for v in ("REST", "OPEN_A"):
        if v not in held:
            continue
        f = held[v]
        tolm = _lip_mask(f, lip, shadow, roi)
        blob = _mouth_blob(f, lip, shadow, roi)
        npm = nearest_palette_mask(f, pal, lip_keys) & roi
        print(f"  {v:8s} tol_mask={int(tolm.sum()):6d} blob={int(blob.sum()):6d}"
              f" nearest={int(npm.sum()):6d} "
              f"nearest_bbox={_mask_bbox(npm)} "
              f"nearest_centroid={mask_centroid(npm)}")
        # what the nearest rule says over the WHOLE canvas
        npm_all = nearest_palette_mask(f, pal, lip_keys)
        print(f"           nearest over whole canvas={int(npm_all.sum()):7d}"
              f" bbox={_mask_bbox(npm_all)}")


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["chintu", "gudiya"]):
        diag(c)
