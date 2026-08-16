"""Numeric probe: how much of the drawn eye aperture does the closing lid
actually cover, per closure step, and is the lid margin well-formed?

Answers three questions with numbers rather than eyeballing a strip:

  1. Is the lid margin x-monotonic?  The polar (radius+angle) interpolation
     in _lid_path rotates points about the eyeball centre, which can reorder
     them in x. A non-monotonic margin makes the lid POLYGON self-intersect,
     and PIL fills a bowtie by the even-odd rule -- so the crossed lobe is
     left EMPTY. That is what a ragged crescent looks like.
  2. What fraction of the aperture is still uncovered at each closure, and
     where is it?  At closure 1 this must be 0.
  3. Do any iris pixels survive in the RENDERED patch at closure 1?
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import eye_model as em          # noqa: E402
from engine.head_assembly import HeadAssembly  # noqa: E402
from engine.rig import Rig                  # noqa: E402

S = 4


def poly_mask(size, pts) -> np.ndarray:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon([tuple(map(float, p)) for p in pts], fill=255)
    return np.asarray(m) > 127


def probe(char: str) -> None:
    rig = Rig.load(char)
    asm = HeadAssembly(rig, scale=1.0, fps=60, seed=char)
    print(f"\n{'=' * 66}\n{char}\n{'=' * 66}")

    for label, rast in (("L", asm.eyes.left), ("R", asm.eyes.right)):
        geo = rast.geo
        print(f"\n-- eye {label}: measured={geo.measured} "
              f"iris_c={tuple(round(v, 1) for v in geo.iris_c)} "
              f"axes={tuple(round(v, 1) for v in geo.axes)} "
              f"angle={geo.iris_angle:.1f} "
              f"lid_img={'yes' if rast.lid is not None else 'NO'}")

        # Patch grid, exactly as EyeRasterizer builds it.
        x0, y0 = rast._x0, rast._y0
        w = int(np.ceil((rast._x1 - x0) * S))
        h = int(np.ceil((rast._y1 - y0) * S))
        size = (max(w, 2), max(h, 2))

        def T(pts):
            return [((px - x0) * S, (py - y0) * S) for px, py in pts]

        ap_mask = rast._ap
        ap_area = int(ap_mask.sum())
        print(f"   aperture area {ap_area / S / S:.0f}px^2   "
              f"columns {int(rast._col_any.sum())}")

        for c in (0.0, 0.25, 0.5, 0.7, 0.85, 1.0):
            rows = rast._margin_rows(c)
            edge = np.asarray(rast._edge_points(rows), dtype=np.float64)
            back = int((np.diff(edge[:, 0]) < -1e-9).sum()) if len(edge) else 0

            cover = rast._cover(rows)
            open_px = ap_mask & ~cover
            frac = open_px.sum() / max(ap_area, 1)

            where = ""
            if open_px.any():
                ys, xs = np.nonzero(open_px)
                where = (f" bbox x{xs.min() / S + x0:.0f}..{xs.max() / S + x0:.0f}"
                         f" y{ys.min() / S + y0:.0f}..{ys.max() / S + y0:.0f}")
            flag = "  <== BOWTIE" if back else ""
            print(f"   closure {c:.2f}: x-backsteps {back:2d}  "
                  f"aperture uncovered {frac * 100:6.2f}%{where}{flag}")

        # Rendered pixels at full closure: does the artwork's iris survive?
        st = em.EyeState(blink_l=1.0, blink_r=1.0)
        patch, _ = rast.render(st, left=(label == "L"))
        a = np.asarray(patch.convert("RGBA"))
        alpha = a[..., 3]
        small_ap = np.asarray(Image.fromarray(
            (ap_mask * 255).astype(np.uint8)).resize(patch.size,
                                                     Image.NEAREST)) > 127
        holes = int((small_ap & (alpha < 32)).sum())
        print(f"   RENDER closure 1: transparent aperture pixels {holes} "
              f"of {int(small_ap.sum())}"
              f"{'   <== iris shows through' if holes else '   ok'}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["chintu", "gudiya"]
    for n in names:
        probe(n)
