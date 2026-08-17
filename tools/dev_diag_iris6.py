"""dev probe: why does chintu's iris_r still read 3.37px out?

Enumerates every iris-coloured component on the RENDERED neutral frame in
the right eye's half, and reports each one's rim-residual fit, so we can
see which component the selector actually picks and what the alternatives
would have given. Nothing here is a gate; it only prints.

    .venv/bin/python -m tools.dev_diag_iris6 chintu
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image

from engine.bone_engine import BoneEngine, PuppetPose
from engine.rig import Rig, rig_dir
from tools.face_qc import (color_mask, fit_fixed_axes_ellipse_ex,
                           label_components, dilate_mask, variation_mask)

IRIS_TOL = 30
ROI_DELTA = 10.0
ROI_GROW_FRAC = 0.05


def _roi(frames, face_h):
    var = variation_mask(frames, ROI_DELTA)
    if var.shape == (1, 1) or not var.any():
        return None
    var = dilate_mask(var, ROI_GROW_FRAC * face_h)
    ys, xs = np.nonzero(var)
    region = np.zeros(var.shape, dtype=bool)
    region[int(ys.min()):int(ys.max()) + 1,
           int(xs.min()):int(xs.max()) + 1] = True
    return region


def main(character: str) -> None:
    rig = Rig.load(character)
    engine = BoneEngine(rig)
    face_h = max(1.0, rig.head.face_height * engine.scale)
    palette = rig.head.palette or {}
    iris_rgb = tuple(palette.get("iris", (92, 58, 38)))

    neutral = PuppetPose()
    open_frame = engine.render(neutral)
    closed = engine.render(PuppetPose(blink=1.0))
    pred = engine.predict(neutral)
    eye_roi = _roi([open_frame, closed], face_h)

    geos = {"iris_l": engine.assembly.eyes.left.geo,
            "iris_r": engine.assembly.eyes.right.geo}

    m_iris = color_mask(open_frame, iris_rgb, tol=IRIS_TOL)
    if eye_roi is not None:
        m_iris = m_iris & eye_roi
    mid_x = int((pred["iris_l"][0] + pred["iris_r"][0]) / 2.0)

    print(f"=== {character} : iris component census ===")
    print(f"iris_rgb={iris_rgb} face_h={face_h:.1f} mid_x={mid_x} "
          f"scale={engine.scale:.4f}")

    for eye in ("iris_l", "iris_r"):
        geo = geos[eye]
        ax, ay = geo.iris_axes
        angle = float(geo.iris_angle)
        m = np.zeros_like(m_iris)
        if pred[eye][0] < mid_x:
            m[:, :mid_x] = m_iris[:, :mid_x]
        else:
            m[:, mid_x:] = m_iris[:, mid_x:]

        print(f"\n--- {eye}  axes=({ax:.2f},{ay:.2f}) angle={angle:.2f} "
              f"pred=({pred[eye][0]:.2f},{pred[eye][1]:.2f})")
        lab, n = label_components(m)
        rows = []
        for k in range(1, n + 1):
            comp = (lab == k)
            area = int(comp.sum())
            if area < 4:
                continue
            ys, xs = np.nonzero(comp)
            w = int(xs.max() - xs.min() + 1)
            h = int(ys.max() - ys.min() + 1)
            got = fit_fixed_axes_ellipse_ex(comp, (ax, ay), angle)
            if got is None:
                rows.append((area, w, h, None, None, None, None))
                continue
            (cx, cy), err, nrim = got
            d = math.dist((cx, cy), pred[eye])
            rows.append((area, w, h, cx, cy, err, nrim, d))
        rows.sort(key=lambda r: -r[0])
        for r in rows:
            if r[3] is None:
                print(f"  area={r[0]:6d} {r[1]:3d}x{r[2]:3d}  fit=NONE")
            else:
                area, w, h, cx, cy, err, nrim, d = r
                print(f"  area={area:6d} {w:3d}x{h:3d}  "
                      f"c=({cx:7.2f},{cy:7.2f}) rim_err={err:.4f} "
                      f"n_rim={nrim:4d}  dist_to_pred={d:7.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "chintu")
