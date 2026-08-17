"""Diagnostic: WHAT is inside the measured mouth blob?

For each character and a couple of visemes, take the blob that
`tools.verify_face` actually measures and classify every pixel in it by
its NEAREST baked palette entry. If the blob is honestly the lip body,
almost every pixel classifies as `lip`/`lip_shadow`. If skin is leaking
in through the Chebyshev tolerance, the histogram says so outright.

Also prints, for reference, how far each palette entry is from the lip
colours, and the analytic lip-polygon bbox for the same viseme, so an
over-large blob is visible as a number rather than an impression.

Usage:  python3 -m tools.dev_diag_mouth5 [chintu gudiya]
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from engine.bone_engine import BoneEngine, PuppetPose
from engine.rig import Rig
from tools.verify_face import (LIP_TOL, _lip_mask, _mask_bbox, _mouth_blob,
                               _roi_from_variation)


def _palette_keys(pal: dict):
    keys = [k for k, v in pal.items()
            if isinstance(v, (list, tuple)) and len(v) >= 3]
    cols = np.array([[float(c) for c in pal[k][:3]] for k in keys],
                    dtype=np.float32)
    return keys, cols


def _nearest_hist(frame: Image.Image, mask: np.ndarray, pal: dict):
    keys, cols = _palette_keys(pal)
    arr = np.asarray(frame.convert("RGB"), dtype=np.float32)
    px = arr[mask]                       # (N,3)
    if px.size == 0:
        return {}
    d = np.linalg.norm(px[:, None, :] - cols[None, :, :], axis=-1)
    win = np.argmin(d, axis=1)
    out = {}
    for i, k in enumerate(keys):
        n = int((win == i).sum())
        if n:
            out[k] = n
    return out


def diag(character: str) -> None:
    rig = Rig.load(character)
    eng = BoneEngine(rig)
    pal = dict(rig.head.palette or {})
    lip = tuple(int(c) for c in pal.get("lip", (170, 80, 80))[:3])
    shadow = (tuple(int(c) for c in pal["lip_shadow"][:3])
              if "lip_shadow" in pal else None)
    print(f"\n═══ {character}   lip={lip} shadow={shadow} LIP_TOL={LIP_TOL}")

    for k, v in pal.items():
        if not isinstance(v, (list, tuple)) or len(v) < 3:
            continue
        cl = max(abs(int(lip[i]) - int(v[i])) for i in range(3))
        cs = (max(abs(int(shadow[i]) - int(v[i])) for i in range(3))
              if shadow else 999)
        flag = " ← WITHIN TOLERANCE" if min(cl, cs) <= LIP_TOL else ""
        print(f"    {k:12s} cheb(lip)={cl:4d} cheb(shadow)={cs:4d}{flag}")

    face_h = max(1.0, rig.head.face_height * eng.scale)
    vis_names = ("REST", "OPEN_A", "WIDE_E", "ROUND_O")
    frames = {}
    for vis in vis_names:
        try:
            frames[vis] = eng.render(PuppetPose(viseme=vis))
        except Exception as exc:                       # pragma: no cover
            print(f"    {vis:8s} render failed: {exc}")
    roi = _roi_from_variation(list(frames.values()), face_h)
    for vis, frame in frames.items():
        raw = _lip_mask(frame, lip, shadow, roi)
        blob = _mouth_blob(frame, lip, shadow, roi)
        hist = _nearest_hist(frame, blob, pal)
        tot = max(1, int(blob.sum()))
        lipish = hist.get("lip", 0) + hist.get("lip_shadow", 0)
        print(f"    {vis:8s} raw={int(raw.sum()):6d} blob={tot:6d} "
              f"bbox={_mask_bbox(blob)}  lip-ish={100.0 * lipish / tot:5.1f}%")
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:6]
        print("             nearest-palette: "
              + "  ".join(f"{k}={100.0 * n / tot:.1f}%" for k, n in top))


def main(argv):
    for c in (argv or ["chintu", "gudiya"]):
        diag(c)


if __name__ == "__main__":
    main(sys.argv[1:])
