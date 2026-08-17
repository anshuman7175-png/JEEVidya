"""Break chintu's right-eye iris mask into components and show them."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.bone_engine import BoneEngine, PuppetPose            # noqa: E402
from engine.rig import Rig, rig_dir                              # noqa: E402
from tools.face_qc import (                                      # noqa: E402
    color_mask, dilate_mask, variation_mask,
)
from tools.verify_face import (                                  # noqa: E402
    IRIS_TOL, ROI_DELTA, ROI_GROW_FRAC, _iris_center, _iris_datum,
    _mask_bbox,
)


def components(mask: np.ndarray):
    """All 4-connected components, largest first, via scipy-free BFS."""
    lab = np.zeros(mask.shape, dtype=np.int32)
    out = []
    cur = 0
    idx = np.argwhere(mask)
    for sy, sx in idx:
        if lab[sy, sx]:
            continue
        cur += 1
        stack = [(int(sy), int(sx))]
        lab[sy, sx] = cur
        pix = []
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] \
                        and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = cur
                    stack.append((ny, nx))
        m = np.zeros(mask.shape, dtype=bool)
        for y, x in pix:
            m[y, x] = True
        out.append(m)
    out.sort(key=lambda m: -int(m.sum()))
    return out


def run(character: str, eye: str) -> None:
    rig = Rig.load(character)
    eng = BoneEngine(rig)
    S = float(eng.scale)
    face_h = max(1.0, rig.head.face_height * S)
    tol_px = 0.6 / 100.0 * face_h
    openf = eng.render(PuppetPose())
    closed = eng.render(PuppetPose(blink=1.0))
    pred = eng.predict(PuppetPose())

    art = (rig.head.art_eye_l if eye == "iris_l" else rig.head.art_eye_r) or {}
    geo = (eng.assembly.eyes.left.geo if eye == "iris_l"
           else eng.assembly.eyes.right.geo)
    ax, ay = geo.iris_axes
    axes = (float(ax), float(ay), float(geo.iris_angle))
    eye_iris = tuple((art.get("colors") or {}).get("iris"))
    plate = Image.open(os.path.join(rig_dir(character),
                                    "head_canonical.png")).convert("RGBA")
    d = _iris_datum(plate, art) or (0.0, 0.0)
    expect = (pred[eye][0] + d[0] * S, pred[eye][1] + d[1] * S)

    var = dilate_mask(variation_mask([openf, closed], ROI_DELTA),
                      ROI_GROW_FRAC * face_h)
    mid_x = int((pred["iris_l"][0] + pred["iris_r"][0]) / 2.0)
    vh = np.zeros_like(var)
    if pred[eye][0] < mid_x:
        vh[:, :mid_x] = var[:, :mid_x]
    else:
        vh[:, mid_x:] = var[:, mid_x:]
    bb = _mask_bbox(vh)
    reg = np.zeros_like(vh)
    reg[bb[1]:bb[3], bb[0]:bb[2]] = True
    print(f"{character} {eye}: blink bbox {bb} "
          f"({bb[2]-bb[0]}x{bb[3]-bb[1]}), axes=({ax:.1f},{ay:.1f}), "
          f"expect=({expect[0]:.1f},{expect[1]:.1f}), tol={tol_px:.2f}")

    m = color_mask(openf, eye_iris, tol=IRIS_TOL) & reg
    comps = components(m)
    vis = openf.convert("RGB").copy()
    dr = ImageDraw.Draw(vis)
    palette = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0)]
    arr = np.asarray(vis).copy()
    for i, c in enumerate(comps[:6]):
        cb = _mask_bbox(c)
        fit = _iris_center(c, axes)
        err = math.dist(expect, fit) if fit else float("nan")
        print(f"   comp{i}: n={int(c.sum()):5d} bbox={cb} "
              f"({cb[2]-cb[0]}x{cb[3]-cb[1]}) fit_err={err:.2f}")
        arr[c] = palette[i % len(palette)]
    vis = Image.fromarray(arr)
    dr = ImageDraw.Draw(vis)
    dr.line([expect[0] - 7, expect[1], expect[0] + 7, expect[1]],
            fill=(255, 255, 255), width=2)
    dr.line([expect[0], expect[1] - 7, expect[0], expect[1] + 7],
            fill=(255, 255, 255), width=2)
    dr.rectangle([bb[0], bb[1], bb[2], bb[3]], outline=(0, 128, 255), width=2)
    cx, cy = expect
    crop = vis.crop((int(cx - 110), int(cy - 110), int(cx + 110),
                     int(cy + 110))).resize((660, 660), Image.NEAREST)
    os.makedirs("output/diag_iris", exist_ok=True)
    p = f"output/diag_iris/comps_{character}_{eye}.png"
    crop.save(p)
    print(f"   -> {p}")


if __name__ == "__main__":
    run("chintu", "iris_r")
    run("chintu", "iris_l")
