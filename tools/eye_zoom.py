"""Zoom hard on one eye across states, beside the untouched artwork.

The face strip shows that the eye no longer breaks the head, but at
head scale it cannot show whether the eye the RENDERER paints matches
the eye the ARTIST drew. This puts the plate crop (ground truth) next
to the rendered states at high magnification, which is the only way to
judge "the sclera is too wide" or "the iris sits off-centre".

    python tools/eye_zoom.py gudiya /tmp/agent-browser/gudiya_eye.png
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.bone_engine import BoneEngine, PuppetPose  # noqa: E402
from engine.rig import Rig  # noqa: E402

CELL = 300
PAD = 6
LABEL_H = 20


def _eye_box(rig, pad_ratio: float = 1.9) -> Tuple[int, int, int, int]:
    """Square box in BODY pixels around both eyes."""
    h = rig.head
    ox, oy = h.offset
    xs, ys = [], []
    for side in ("art_eye_l", "art_eye_r"):
        d = getattr(h, side, None)
        ap = (d or {}).get("aperture") if isinstance(d, dict) else None
        if ap:
            xs += [p[0] + ox for p in ap]
            ys += [p[1] + oy for p in ap]
    if not xs:
        for e in (h.iris_l, h.iris_r):
            xs += [e[0] + ox - e[2], e[0] + ox + e[2]]
            ys += [e[1] + oy - e[2], e[1] + oy + e[2]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) * pad_ratio / 2
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def _cell(img: Image.Image, box, label: str) -> Image.Image:
    crop = img.crop(box).convert("RGB").resize((CELL, CELL), Image.NEAREST)
    out = Image.new("RGB", (CELL, CELL + LABEL_H), (24, 24, 28))
    out.paste(crop, (0, 0))
    ImageDraw.Draw(out).text((6, CELL + 4), label, fill=(235, 235, 240))
    return out


def zoom(character: str, out_path: str) -> str:
    rig = Rig.load(character)
    eng = BoneEngine(rig)
    box = _eye_box(rig)

    cells: List[Image.Image] = []

    # Ground truth: the staged body art, never touched by the eye renderer.
    body_path = os.path.join(os.path.dirname(
        os.path.dirname(rig.head.plate if os.path.isabs(rig.head.plate)
                        else os.path.abspath(rig.head.plate))), "body.png")
    for cand in (body_path,
                 f"assets/characters/{character}/body.png",
                 f"assets/characters/{character}/poses/neutral.png"):
        if os.path.exists(cand):
            cells.append(_cell(Image.open(cand).convert("RGBA"), box,
                               "ART (ground truth)"))
            break

    d = rig.head.face_height * 0.06
    for label, kw in (("rest", {}), ("blink 0.35", {"blink": 0.35}),
                      ("blink 0.7", {"blink": 0.7}),
                      ("blink 1.0", {"blink": 1.0}),
                      ("look L", {"eye_dx": -d}), ("look R", {"eye_dx": d}),
                      ("look up", {"eye_dy": -d}),
                      ("look down", {"eye_dy": d}),
                      ("lid 0.4", {"lid": 0.4}), ("brow", {"brow": 0.9})):
        cells.append(_cell(eng.render(PuppetPose(**kw)), box, label))

    cols = min(6, len(cells))
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * CELL + (cols + 1) * PAD,
                             rows * (CELL + LABEL_H) + (rows + 1) * PAD),
                      (16, 16, 20))
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        sheet.paste(c, (PAD + col * (CELL + PAD),
                        PAD + r * (CELL + LABEL_H + PAD)))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path)
    return out_path


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "gudiya"
    dest = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/agent-browser/{name}_eye.png"
    print(zoom(name, dest))
