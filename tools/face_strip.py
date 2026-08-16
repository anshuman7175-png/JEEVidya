"""Render a labelled contact strip of one character's face states.

A debugging aid, not part of the pipeline: it renders the states a human
needs to SEE to judge the face (rest, blink sweep, every viseme, gaze)
and crops each to the head at a readable scale. Reviewing rendered
pixels is the only check that catches "the eye looks like a rectangle" —
no numeric gate expresses that.

    python tools/face_strip.py chintu /tmp/agent-browser/chintu.png
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.bone_engine import BoneEngine, PuppetPose  # noqa: E402
from engine.rig import Rig  # noqa: E402

CELL = 320          # px per cell in the output strip
PAD = 8
LABEL_H = 22


def _head_box(rig, frame_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """A generous square box around the head in body pixels."""
    h = rig.head
    ox, oy = h.offset
    fh = h.face_height
    cx = ox + sum(p[0] for p in h.lip_outer) / len(h.lip_outer)
    cy = oy + sum(p[1] for p in h.lip_outer) / len(h.lip_outer)
    half = fh * 0.95
    cy -= fh * 0.28          # centre on the eyes, not the mouth
    x0 = max(0, int(cx - half))
    y0 = max(0, int(cy - half))
    x1 = min(frame_size[0], int(cx + half))
    y1 = min(frame_size[1], int(cy + half))
    return x0, y0, x1, y1


def _cell(img: Image.Image, box, label: str) -> Image.Image:
    crop = img.crop(box).convert("RGB")
    crop = crop.resize((CELL, CELL), Image.LANCZOS)
    out = Image.new("RGB", (CELL, CELL + LABEL_H), (24, 24, 28))
    out.paste(crop, (0, 0))
    ImageDraw.Draw(out).text((6, CELL + 5), label, fill=(235, 235, 240))
    return out


def strip(character: str, out_path: str,
          states: Optional[List[Tuple[str, dict]]] = None) -> str:
    rig = Rig.load(character)
    eng = BoneEngine(rig)

    if states is None:
        d = rig.head.face_height * 0.06 if rig.head else 8.0
        states = [("rest", {}),
                  ("blink 0.5", {"blink": 0.5}),
                  ("blink 1.0", {"blink": 1.0}),
                  ("look L", {"eye_dx": -d}),
                  ("look R", {"eye_dx": d}),
                  ("look up", {"eye_dy": -d}),
                  ("lid 0.4", {"lid": 0.4}),
                  ("brow up", {"brow": 0.9})]
        # LID* entries are lid sprites, not mouth shapes.
        for v in sorted(n for n in (rig.visemes or {})
                        if not n.startswith("LID")):
            states.append((v, {"viseme": v, "mouth_open": 1.0}))

    cells: List[Image.Image] = []
    box = None
    for label, kw in states:
        pose = PuppetPose(**kw) if kw else PuppetPose()
        frame = eng.render(pose)
        if box is None:
            box = _head_box(rig, frame.size)
        cells.append(_cell(frame, box, label))

    cols = min(6, len(cells))
    rows = (len(cells) + cols - 1) // cols
    w = cols * CELL + (cols + 1) * PAD
    h = rows * (CELL + LABEL_H) + (rows + 1) * PAD
    sheet = Image.new("RGB", (w, h), (16, 16, 20))
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        sheet.paste(c, (PAD + col * (CELL + PAD),
                        PAD + r * (CELL + LABEL_H + PAD)))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path)
    return out_path


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "chintu"
    dest = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/agent-browser/{name}.png"
    print(strip(name, dest))
