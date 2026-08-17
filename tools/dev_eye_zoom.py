"""Dump a magnified crop around each measured eye (developer tool).

Writes <out>/<character>_<side>.png at 6× nearest-neighbour so the lash,
the lens, the frame and the eyelid rows can be told apart by eye — the walk
diagnostics in tools/dev_lid_walk.py say WHAT the tests read, this says what
the artist actually drew there.

    python -m tools.dev_eye_zoom [--out DIR] [character ...]
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image

from engine.rig import Rig
from tools import art_eyes as A
from tools import rig_v3 as R
from tools.rig_builder import _detect_landmarks

ZOOM = 6


def dump(name: str, out: str) -> None:
    body = Image.open(f"assets/characters/{name}/body.png").convert("RGBA")
    lms = np.asarray(_detect_landmarks(body), dtype=np.float64)
    fh = R.face_height(lms)
    arr = np.asarray(body)
    hmask = R.head_mask(lms, arr[..., 3])
    ys, xs = np.nonzero(hmask > 0.01)
    m = int(math.ceil(R.HEAD_PLATE_MARGIN * fh))
    x0, x1 = int(xs.min()) - m, int(xs.max()) + 1 + m
    y0, y1 = int(ys.min()) - m, int(ys.max()) + 1 + m
    try:
        seam_y = float(Rig.load(name).joints.get("neck", (0.0, y1))[1])
    except Exception:
        seam_y = float(y1)
    hr, _ = R.complementary_ramps(hmask, seam_y, R.SEAM_BAND * fh)
    hf = arr.copy().astype(np.float32)
    hf[..., 3] = hf[..., 3] * hr
    crop = R.crop_padded(np.clip(hf, 0, 255).astype(np.uint8), x0, y0, x1, y1)

    plate_lms = lms - np.array([x0, y0], dtype=np.float64)
    seeds = ((R._pick(plate_lms, R.LID_UPPER_L + R.LID_LOWER_L).mean(axis=0),
              "l"),
             (R._pick(plate_lms, R.LID_UPPER_R + R.LID_LOWER_R).mean(axis=0),
              "r"))
    os.makedirs(out, exist_ok=True)
    for seed, side in seeds:
        eye = A.measure_eye(crop, (seed[0], seed[1]), fh, f"eye_{side}")
        ap = np.asarray(eye.aperture, dtype=np.float64)
        pad = int(0.09 * fh)
        cx0 = max(0, int(ap[:, 0].min()) - pad)
        cx1 = min(crop.shape[1], int(ap[:, 0].max()) + pad)
        cy0 = max(0, int(ap[:, 1].min()) - pad)
        cy1 = min(crop.shape[0], int(ap[:, 1].max()) + pad)
        sub = Image.fromarray(crop[cy0:cy1, cx0:cx1], "RGBA")
        big = sub.resize((sub.width * ZOOM, sub.height * ZOOM), Image.NEAREST)
        # overlay: aperture rim (green), iris circle (cyan)
        from PIL import ImageDraw
        dr = ImageDraw.Draw(big)
        dr.polygon([((p[0] - cx0) * ZOOM, (p[1] - cy0) * ZOOM) for p in ap],
                   outline=(0, 255, 0, 255))
        ix, iy, ir = eye.iris_c[0], eye.iris_c[1], eye.iris_r
        dr.ellipse([((ix - ir) - cx0) * ZOOM, ((iy - ir) - cy0) * ZOOM,
                    ((ix + ir) - cx0) * ZOOM, ((iy + ir) - cy0) * ZOOM],
                   outline=(0, 220, 255, 255))
        path = os.path.join(out, f"{name}_{side}.png")
        big.save(path)
        # and a clean copy with no overlay
        sub.resize((sub.width * ZOOM, sub.height * ZOOM), Image.NEAREST) \
            .save(os.path.join(out, f"{name}_{side}_clean.png"))
        print(f"  {path}  crop=({cx0},{cy0})-({cx1},{cy1}) "
              f"ap y={ap[:, 1].min():.0f}..{ap[:, 1].max():.0f}")


if __name__ == "__main__":
    args = list(sys.argv[1:])
    out = "output/eye_zoom"
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        del args[i:i + 2]
    for c in (args or ("chintu", "gudiya")):
        dump(c, out)
