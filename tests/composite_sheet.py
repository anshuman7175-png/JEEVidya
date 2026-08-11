"""
Visual QA: composite every baked viseme sprite onto the neutral body
exactly per the BoneEngine contract (centered-x on mouth center, sprite
top at mouth_center_y - 0.42*sprite_h), then crop the face region and
tile the results into one labeled contact sheet per character.

Run:  .venv/bin/python tests/composite_sheet.py
Out:  /tmp/agent-browser/composite_<name>.png
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

from engine.rig import Rig, rig_dir

OUT_DIR = "/tmp/agent-browser"
CELL = 300  # face-crop cell size on the sheet


def sheet(character: str) -> str:
    rig = Rig.load(character)
    d = rig_dir(character)
    body = Image.open(os.path.join(
        os.path.dirname(d), "body.png")).convert("RGBA")

    x0, y0, x1, y1 = rig.box("mouth")
    mcx, mcy = (x0 + x1) / 2, (y0 + y1) / 2

    # Face crop window around the mouth (wide enough to show cheeks,
    # chin, and anywhere an occluder could flash)
    fw = int((x1 - x0) * 4.5)
    fx0, fy0 = int(mcx - fw / 2), int(mcy - fw * 0.75)
    fx1, fy1 = int(mcx + fw / 2), int(mcy + fw * 0.55)

    names = [k for k in rig.visemes if not k.startswith("LID_")]
    names.sort()
    cols = 4
    rows = (len(names) + cols - 1) // cols
    grid = Image.new("RGBA", (cols * CELL, rows * (CELL + 22)),
                     (34, 34, 40, 255))
    dr = ImageDraw.Draw(grid)

    for i, name in enumerate(names):
        canvas = body.copy()
        sp = Image.open(os.path.join(d, rig.visemes[name])).convert("RGBA")
        mw, mh = sp.size
        canvas.alpha_composite(
            sp, dest=(int(mcx - mw / 2), int(mcy - mh * 0.42)))
        face = canvas.crop((fx0, fy0, fx1, fy1)).resize(
            (CELL, CELL), Image.Resampling.LANCZOS)
        cx = (i % cols) * CELL
        cy = (i // cols) * (CELL + 22)
        grid.alpha_composite(face, dest=(cx, cy))
        dr.text((cx + 6, cy + CELL + 4), name, fill=(255, 255, 255, 255))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"composite_{character}.png")
    grid.convert("RGB").save(out)
    print(f"  [QA] {character}: {out} ({len(names)} visemes)")
    return out


if __name__ == "__main__":
    for name in ("chintu", "gudiya"):
        sheet(name)
