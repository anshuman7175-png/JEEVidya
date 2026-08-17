"""Bake-faithful dump of the eyelid walk (developer tool, renders nothing).

Unlike tools/dev_lid_probe.py this does not need a baked rig: it repeats the
bake's own head crop and eye measurement from body.png, then prints every row
the lid walk sees with the tests that accept or reject it.

    python -m tools.dev_lid_walk [character ...]
"""
from __future__ import annotations

import math
import sys

import numpy as np
from PIL import Image

from engine.rig import Rig
from tools import art_eyes as A
from tools import rig_v3 as R
from tools.rig_builder import _detect_landmarks


def probe(name: str) -> None:
    body = Image.open(
        f"assets/characters/{name}/body.png").convert("RGBA")
    lms = np.asarray(_detect_landmarks(body), dtype=np.float64)
    fh = R.face_height(lms)
    arr = np.asarray(body)
    hmask = R.head_mask(lms, arr[..., 3])
    ys, xs = np.nonzero(hmask > 0.01)
    margin = int(math.ceil(R.HEAD_PLATE_MARGIN * fh))
    x0, x1 = int(xs.min()) - margin, int(xs.max()) + 1 + margin
    y0, y1 = int(ys.min()) - margin, int(ys.max()) + 1 + margin
    try:
        seam_y = float(Rig.load(name).joints.get("neck", (0.0, y1))[1])
    except Exception:
        seam_y = float(y1)
    hr, _ = R.complementary_ramps(hmask, seam_y, R.SEAM_BAND * fh)
    hf = arr.copy().astype(np.float32)
    hf[..., 3] = hf[..., 3] * hr
    crop = R.crop_padded(np.clip(hf, 0, 255).astype(np.uint8), x0, y0, x1, y1)
    print(f"\n=== {name}: fh={fh:.0f} margin={margin} "
          f"plate={crop.shape[1]}x{crop.shape[0]} "
          f"border={R.border_opaque_counts(crop)}")

    plate_lms = lms - np.array([x0, y0], dtype=np.float64)
    sl = R._pick(plate_lms, R.LID_UPPER_L + R.LID_LOWER_L).mean(axis=0)
    sr = R._pick(plate_lms, R.LID_UPPER_R + R.LID_LOWER_R).mean(axis=0)
    for seed, label in ((sl, "eye_l"), (sr, "eye_r")):
        eye = A.measure_eye(crop, (seed[0], seed[1]), fh, label)
        ap = np.asarray(eye.aperture, dtype=np.float64)
        gain = float(eye.tone_gain)
        rgb = crop[..., :3].astype(np.float32)
        ex0 = max(0, int(math.floor(ap[:, 0].min())) - 3)
        ex1 = min(crop.shape[1], int(math.ceil(ap[:, 0].max())) + 3)
        top = int(math.floor(ap[:, 1].min()))
        bot = int(math.ceil(ap[:, 1].max()))
        ap_h = max(3, bot - top)
        print(f"\n {label}: gain={gain:.3f} x={ex0}..{ex1} y={top}..{bot} "
              f"h={ap_h} w={ap[:, 0].max() - ap[:, 0].min():.0f} "
              f"iris_c={np.round(eye.iris_c, 1)} iris_r={eye.iris_r:.1f} "
              f"gaze={[round(v, 1) for v in eye.gaze_box]}")
        rows, ref = A._lid_skin_below(rgb, ex0, ex1, bot, ap_h, gain, label)
        print(f"   below_ref={ref.round(0)} n={rows.shape[0]} "
              f"near={rows[0].round(0)} far={rows[-1].round(0)} "
              f"skin_like={[int(A._lid_skin_like(r, ref)) for r in rows]}")
        print(f"   {'y':>5} {'dy':>3} {'ink%':>6} {'tone':<18} skin_like")
        for dy in range(0, min(top, int(ap_h * 1.2)) + 1):
            yy = top - dy
            if yy < 0:
                break
            row = rgb[yy, ex0:ex1]
            ink = float(np.mean(A._ink(row, gain)))
            t = A._row_tone(row, gain)
            sk = "-" if t is None else int(A._lid_skin_like(t, ref))
            print(f"   {yy:>5} {dy:>3} {ink * 100:6.1f} "
                  f"{'—' if t is None else str(t.round(0)):<18} {sk}")


if __name__ == "__main__":
    for c in (sys.argv[1:] or ("chintu", "gudiya")):
        probe(c)
