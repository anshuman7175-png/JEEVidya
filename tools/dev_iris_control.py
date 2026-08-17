"""Dev control: is the iris-registration error the RENDERER's or the
DETECTOR's?

Applies the QC gate's own detector (largest component of an
iris-coloured mask) to the UNTOUCHED head plate, in plate space, and
compares it with `iris_c` — the value the bake measured from that very
artwork and that the predictor propagates.

The renderer is not involved, so any error printed here is the
detector's own bias and cannot be fixed by changing the renderer.

    python -m tools.dev_iris_control
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from tools.face_qc import color_mask, label_components
from tools.verify_face import IRIS_TOL


def control(character: str) -> None:
    d = f"assets/characters/{character}/rig"
    rig = json.load(open(os.path.join(d, "rig.json")))
    head = rig["head"]
    palette = head.get("palette") or {}
    plate = Image.open(os.path.join(d, "head_canonical.png")).convert("RGBA")
    print(f"═══ {character}  plate={plate.size}  "
          f"palette iris={tuple(palette.get('iris', ()))}")

    for side in ("l", "r"):
        eye = head.get(f"art_eye_{side}")
        if not eye:
            continue
        cx, cy = eye["iris_c"]
        ax, ay = eye["iris_axes"]
        # Search a generous box around the known iris so no other feature
        # can win the "largest component" contest — this isolates the
        # estimator's bias from the gate's separate side-assignment bug.
        pad = int(max(ax, ay) * 2.2)
        x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        x1, y1 = int(cx + pad), int(cy + pad)
        crop = plate.crop((x0, y0, x1, y1))

        for name, rgb in (("palette", tuple(palette.get("iris", (92, 58, 38)))),
                          ("per-eye", tuple(eye["colors"]["iris"]))):
            m = color_mask(crop, rgb, tol=IRIS_TOL)
            lab, n = label_components(m)
            if n == 0:
                print(f"  eye_{side} [{name:7s}] rgb={rgb}: NO iris-coloured "
                      f"pixels at all")
                continue
            counts = np.bincount(lab.ravel())
            counts[0] = 0
            top = int(np.argmax(counts))
            ys, xs = np.nonzero(lab == top)
            dcx, dcy = float(xs.mean()) + x0, float(ys.mean()) + y0
            bx0, bx1 = int(xs.min()) + x0, int(xs.max()) + x0
            by0, by1 = int(ys.min()) + y0, int(ys.max()) + y0
            err = ((dcx - cx) ** 2 + (dcy - cy) ** 2) ** 0.5
            full_w, full_h = 2.0 * ax, 2.0 * ay
            print(f"  eye_{side} [{name:7s}] rgb={rgb}")
            print(f"      bake iris_c=({cx:7.1f},{cy:7.1f})  "
                  f"axes={ax:.1f}×{ay:.1f} → full {full_w:.0f}×{full_h:.0f}px "
                  f"(area {np.pi*ax*ay:.0f}px)")
            print(f"      detected  =({dcx:7.1f},{dcy:7.1f})  "
                  f"visible area={counts[top]}px "
                  f"({counts[top]/(np.pi*ax*ay)*100:.0f}% of the iris)")
            print(f"      visible bbox={bx1-bx0+1}×{by1-by0+1}px "
                  f"rows {by0}..{by1} (iris spans "
                  f"{cy-ay:.0f}..{cy+ay:.0f})")
            print(f"      ERROR {err:.1f}px  (dx={dcx-cx:+.1f}, "
                  f"dy={dcy-cy:+.1f})")


def main() -> None:
    for c in ("chintu", "gudiya"):
        control(c)


if __name__ == "__main__":
    main()
