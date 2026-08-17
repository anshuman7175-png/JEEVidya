"""Compare candidate iris detectors against the current one.

Throwaway diagnostic (dev_*), not part of the gate. Answers one
question with numbers: which search-region policy locates the rendered
iris on all four eyes, without consulting predict() for the region.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.bone_engine import BoneEngine, PuppetPose            # noqa: E402
from engine.rig import Rig, rig_dir                              # noqa: E402
from tools.face_qc import (                                      # noqa: E402
    color_mask, dilate_mask, largest_component, variation_mask,
)
from tools.verify_face import (                                  # noqa: E402
    IRIS_TOL, ROI_DELTA, ROI_GROW_FRAC, _iris_center, _iris_datum,
    _mask_bbox, _roi_from_variation,
)


def box_mask(shape, cx, cy, pad):
    m = np.zeros(shape, dtype=bool)
    y0, y1 = max(0, int(cy - pad)), min(shape[0], int(cy + pad) + 1)
    x0, x1 = max(0, int(cx - pad)), min(shape[1], int(cx + pad) + 1)
    m[y0:y1, x0:x1] = True
    return m


def run(character: str) -> None:
    rig = Rig.load(character)
    eng = BoneEngine(rig)
    S = float(eng.scale)
    face_h = max(1.0, rig.head.face_height * S)
    tol_px = 0.6 / 100.0 * face_h

    openf = eng.render(PuppetPose())
    closed = eng.render(PuppetPose(blink=1.0))
    pred = eng.predict(PuppetPose())

    pal = rig.head.palette or {}
    pal_iris = tuple(pal.get("iris", (92, 58, 38)))
    arts = {"iris_l": rig.head.art_eye_l, "iris_r": rig.head.art_eye_r}
    geos = {"iris_l": eng.assembly.eyes.left.geo,
            "iris_r": eng.assembly.eyes.right.geo}
    plate = Image.open(os.path.join(rig_dir(character),
                                    "head_canonical.png")).convert("RGBA")

    shared_roi = _roi_from_variation([openf, closed], face_h)
    var = variation_mask([openf, closed], ROI_DELTA)
    var = dilate_mask(var, ROI_GROW_FRAC * face_h)
    mid_x = int((pred["iris_l"][0] + pred["iris_r"][0]) / 2.0)

    print(f"\n===== {character}  face_h={face_h:.1f} tol={tol_px:.2f}px "
          f"scale={S:.3f} =====")

    for eye in ("iris_l", "iris_r"):
        art = arts[eye] or {}
        geo = geos[eye]
        ax, ay = geo.iris_axes
        axes = (float(ax), float(ay), float(geo.iris_angle))
        eye_iris = tuple((art.get("colors") or {}).get("iris") or pal_iris)
        eye_scl = (art.get("colors") or {}).get("sclera")
        d = _iris_datum(plate, art) or (0.0, 0.0)
        expect = (pred[eye][0] + d[0] * S, pred[eye][1] + d[1] * S)
        left_side = pred[eye][0] < mid_x

        def half(m):
            out = np.zeros_like(m)
            if left_side:
                out[:, :mid_x] = m[:, :mid_x]
            else:
                out[:, mid_x:] = m[:, mid_x:]
            return out

        def score(label, mask):
            det = _iris_center(mask, axes)
            n = int(largest_component(mask).sum())
            if det is None:
                print(f"  {label:<34} NO DETECTION")
                return
            err = math.dist(expect, det)
            flag = "PASS" if err <= tol_px else "FAIL"
            bb = _mask_bbox(largest_component(mask))
            wh = f"{bb[2]-bb[0]}x{bb[3]-bb[1]}" if bb else "-"
            print(f"  {label:<34} err={err:7.2f} {flag}  blob={n:5d} {wh}")

        print(f" -- {eye}  palette_iris={pal_iris} eye_iris={eye_iris} "
              f"axes=({ax:.1f},{ay:.1f})")

        # A: today's detector
        mA = half(color_mask(openf, pal_iris, tol=IRIS_TOL) & shared_roi)
        score("A current (palette+sharedROI)", mA)

        # B: per-eye colour only
        mB = half(color_mask(openf, eye_iris, tol=IRIS_TOL) & shared_roi)
        score("B per-eye colour", mB)

        # C: per-eye colour + per-eye blink-variation bbox
        vh = half(var)
        bb = _mask_bbox(vh)
        if bb:
            reg = np.zeros_like(vh)
            reg[bb[1]:bb[3], bb[0]:bb[2]] = True
            mC = color_mask(openf, eye_iris, tol=IRIS_TOL) & reg
            score("C +per-eye blink bbox", mC)

        # D: per-eye colour + box around that eye's sclera blob
        if eye_scl is not None:
            sm = largest_component(half(
                color_mask(openf, tuple(eye_scl), tol=IRIS_TOL)))
            if sm.any():
                ys, xs = np.nonzero(sm)
                cx, cy = xs.mean(), ys.mean()
                pad = max(ax, ay) * 2.2
                mD = color_mask(openf, eye_iris, tol=IRIS_TOL) & \
                    box_mask(sm.shape, cx, cy, pad)
                score("D +sclera-anchored box", mD)
            else:
                print("  D +sclera-anchored box            NO SCLERA BLOB")


if __name__ == "__main__":
    for c in ("chintu", "gudiya"):
        run(c)
