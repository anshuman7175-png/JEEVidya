"""Dev probe: what actually occupies the upper two-thirds of the iris?

`dev_iris_control` proved the iris-colour detector finds only a BOTTOM
CRESCENT (17–24% of the ellipse area, full width, one third of the
height) and therefore carries a systematic ~+14 px downward centroid
bias on untouched artwork — a detector defect, not a renderer one.

This answers the next question: WHY. It walks the iris ellipse row by
row in plate space and reports, per row, how the pixels distribute over
the palette's named colours plus a catch-all, so the occluding feature
is identified by measurement rather than assumed. It also dumps a
magnified crop with the bake's ellipse and centre drawn, so the shape
can be seen.

    python -m tools.dev_iris_anatomy [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw

ZOOM = 6


def _nearest(px: np.ndarray, palette: Dict[str, Tuple[int, int, int]]
             ) -> Tuple[str, float]:
    """Name the palette entry closest to `px`, with its RGB distance."""
    best, bd = "?", 1e9
    for name, rgb in palette.items():
        d = float(np.linalg.norm(px.astype(float) - np.array(rgb, float)))
        if d < bd:
            best, bd = name, d
    return best, bd


def anatomy(character: str, out_dir: str) -> None:
    d = f"assets/characters/{character}/rig"
    rig = json.load(open(os.path.join(d, "rig.json")))
    head = rig["head"]
    palette = {k: tuple(v) for k, v in (head.get("palette") or {}).items()}
    plate = Image.open(os.path.join(d, "head_canonical.png")).convert("RGB")
    arr = np.asarray(plate)
    print(f"═══ {character}  plate={plate.size}")
    print(f"    palette: " + ", ".join(f"{k}={v}" for k, v in palette.items()))

    for side in ("l", "r"):
        eye = head.get(f"art_eye_{side}")
        if not eye:
            continue
        cx, cy = eye["iris_c"]
        ax, ay = eye["iris_axes"]
        per_eye = {k: tuple(v) for k, v in (eye.get("colors") or {}).items()}
        print(f"\n  ── eye_{side}: iris_c=({cx:.1f},{cy:.1f}) "
              f"axes={ax:.1f}x{ay:.1f}  per-eye colors={per_eye}")

        # Row-by-row composition strictly INSIDE the fitted ellipse.
        y0, y1 = int(round(cy - ay)), int(round(cy + ay))
        print(f"     row   inside  " + "  ".join(f"{k:>8s}"
                                                 for k in per_eye) + "     other")
        for y in range(y0, y1 + 1):
            if not (0 <= y < arr.shape[0]):
                continue
            dy = (y - cy) / ay
            if abs(dy) > 1.0:
                continue
            half = ax * float(np.sqrt(max(0.0, 1.0 - dy * dy)))
            xa, xb = int(round(cx - half)), int(round(cx + half))
            xa, xb = max(0, xa), min(arr.shape[1] - 1, xb)
            if xb < xa:
                continue
            row = arr[y, xa:xb + 1]
            counts = {k: 0 for k in per_eye}
            other = 0
            for px in row:
                name, dist = _nearest(px, per_eye)
                if dist <= 40.0:
                    counts[name] += 1
                else:
                    other += 1
            n = len(row)
            cells = "  ".join(f"{counts[k] / n * 100:7.0f}%" for k in per_eye)
            print(f"    {y:5d}  {n:5d}   {cells}   {other / n * 100:6.0f}%")

        # Magnified crop with the fitted ellipse drawn.
        pad = int(max(ax, ay) * 1.9)
        bx0, by0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        bx1, by1 = int(cx + pad), int(cy + pad)
        crop = plate.crop((bx0, by0, bx1, by1)).resize(
            ((bx1 - bx0) * ZOOM, (by1 - by0) * ZOOM), Image.NEAREST)
        dr = ImageDraw.Draw(crop)
        ex0 = (cx - ax - bx0) * ZOOM
        ey0 = (cy - ay - by0) * ZOOM
        ex1 = (cx + ax - bx0) * ZOOM
        ey1 = (cy + ay - by0) * ZOOM
        dr.ellipse([ex0, ey0, ex1, ey1], outline=(0, 255, 0), width=2)
        mx, my = (cx - bx0) * ZOOM, (cy - by0) * ZOOM
        dr.line([mx - 14, my, mx + 14, my], fill=(255, 0, 255), width=2)
        dr.line([mx, my - 14, mx, my + 14], fill=(255, 0, 255), width=2)
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f"{character}_eye_{side}_anatomy.png")
        crop.save(p)
        print(f"     → {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("characters", nargs="*", default=["chintu", "gudiya"])
    ap.add_argument("--out", default="/tmp/irisanat")
    a = ap.parse_args()
    for c in a.characters:
        anatomy(c, a.out)


if __name__ == "__main__":
    main()
