"""Per-component iris evidence for all four eyes.

Answers, with numbers rather than assertion, the question the iris
registration gate turns on: of the several blobs of iris-coloured paint
inside one eye's footprint, WHICH is the eyeball, and does a
prediction-independent criterion pick it?

For every component it prints
    area, bbox, rim_err (mean |r−1| of the retained rim points),
    the fitted centre, and that centre's distance from the prediction,
and marks which component `largest_component` (the old rule) and
`fit_ellipse_best_component` (the rim-residual rule) each select.

The reg-err column is printed for information only — selection never
reads it, or the gate would be checking `predict()` against itself.

    .venv/bin/python -m tools.dev_diag_iris5
"""
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
    CAND_AREA_HI_FRAC, CAND_AREA_LO_FRAC, MIN_COMPONENT_PX, RIM_ERR_MAX,
    color_mask, dilate_mask, fit_ellipse_best_component,
    fit_fixed_axes_ellipse_ex, label_components, largest_component,
    variation_mask,
)
from tools.verify_face import (                                  # noqa: E402
    IRIS_TOL, ROI_DELTA, ROI_GROW_FRAC, _iris_datum, _mask_bbox,
)

PALETTE = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0),
           (255, 128, 0), (128, 128, 255)]


def probe(character: str, eye: str) -> None:
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
    ax, ay = (float(v) for v in geo.iris_axes)
    angle = float(geo.iris_angle)
    iris_rgb = tuple((art.get("colors") or {}).get("iris"))
    plate = Image.open(os.path.join(rig_dir(character),
                                    "head_canonical.png")).convert("RGBA")
    d = _iris_datum(plate, art) or (0.0, 0.0)
    expect = (pred[eye][0] + d[0] * S, pred[eye][1] + d[1] * S)

    var = dilate_mask(variation_mask([openf, closed], ROI_DELTA),
                      ROI_GROW_FRAC * face_h)
    mid_x = int((pred["iris_l"][0] + pred["iris_r"][0]) / 2.0)
    half = np.zeros_like(var)
    if pred[eye][0] < mid_x:
        half[:, :mid_x] = var[:, :mid_x]
    else:
        half[:, mid_x:] = var[:, mid_x:]
    bb = _mask_bbox(half)
    region = np.zeros_like(half)
    region[bb[1]:bb[3], bb[0]:bb[2]] = True

    mask = color_mask(openf, iris_rgb, tol=IRIS_TOL) & region
    ell_area = math.pi * ax * ay
    lo = max(float(MIN_COMPONENT_PX), CAND_AREA_LO_FRAC * ell_area)
    hi = CAND_AREA_HI_FRAC * ell_area
    print(f"\n{character} {eye}: axes=({ax:.1f},{ay:.1f}) angle={angle:.1f} "
          f"ellipse_area={ell_area:.0f} cand_window=[{lo:.0f},{hi:.0f}] "
          f"expect=({expect[0]:.1f},{expect[1]:.1f}) tol={tol_px:.2f}px")

    lab, n = label_components(mask)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    order = [int(k) for k in np.argsort(counts)[::-1] if counts[k] > 0][:6]
    big = largest_component(mask)
    big_id = int(lab[big][0]) if big.any() else -1
    best = fit_ellipse_best_component(mask, (ax, ay), angle)

    arr = np.asarray(openf.convert("RGB")).copy()
    for i, lb in enumerate(order):
        comp = lab == lb
        area = int(counts[lb])
        cb = _mask_bbox(comp)
        got = fit_fixed_axes_ellipse_ex(comp, (ax, ay), angle)
        in_win = lo <= area <= hi
        if got is None:
            print(f"   comp{i} id={lb}: n={area:5d} bbox={cb} "
                  f"({cb[2]-cb[0]}x{cb[3]-cb[1]}) rim=unfittable "
                  f"{'(in window)' if in_win else '(out of window)'}")
        else:
            (cx, cy), rim, nrim = got
            err = math.dist(expect, (cx, cy))
            tags = []
            if lb == big_id:
                tags.append("LARGEST")
            if best is not None and abs(best[1] - rim) < 1e-12 \
                    and abs(best[0][0] - cx) < 1e-9:
                tags.append("PICKED")
            if not in_win:
                tags.append("out-of-window")
            if rim > RIM_ERR_MAX:
                tags.append("rim>max")
            print(f"   comp{i} id={lb}: n={area:5d} bbox={cb} "
                  f"({cb[2]-cb[0]}x{cb[3]-cb[1]}) rim={rim:.4f} "
                  f"nrim={nrim:3d} fit=({cx:.1f},{cy:.1f}) "
                  f"reg_err={err:6.2f}px  {' '.join(tags)}")
        arr[comp] = PALETTE[i % len(PALETTE)]

    if best is None:
        print("   PICKED: none — gate reports not-measurable")
    else:
        (cx, cy), rim, area = best
        print(f"   PICKED: n={area} rim={rim:.4f} fit=({cx:.1f},{cy:.1f}) "
              f"reg_err={math.dist(expect, (cx, cy)):.2f}px "
              f"tol={tol_px:.2f}px "
              f"{'PASS' if math.dist(expect, (cx, cy)) <= tol_px else 'FAIL'}")

    vis = Image.fromarray(arr)
    dr = ImageDraw.Draw(vis)
    dr.line([expect[0] - 8, expect[1], expect[0] + 8, expect[1]],
            fill=(255, 255, 255), width=2)
    dr.line([expect[0], expect[1] - 8, expect[0], expect[1] + 8],
            fill=(255, 255, 255), width=2)
    if best is not None:
        cx, cy = best[0]
        dr.ellipse([cx - ax, cy - ay, cx + ax, cy + ay],
                   outline=(0, 0, 255), width=2)
    cx, cy = expect
    crop = vis.crop((int(cx - 110), int(cy - 110), int(cx + 110),
                     int(cy + 110))).resize((660, 660), Image.NEAREST)
    os.makedirs("output/diag_iris", exist_ok=True)
    p = f"output/diag_iris/rim_{character}_{eye}.png"
    crop.save(p)
    print(f"   -> {p}")


if __name__ == "__main__":
    for ch in ("chintu", "gudiya"):
        for e in ("iris_l", "iris_r"):
            probe(ch, e)
