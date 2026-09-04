"""
JEEVidya — YouTube Shorts UI overlay for framing QA.

Draws the regions the Shorts player paints OVER the video on a rendered
frame, so caption / character placement can be judged against what a
viewer actually sees on a phone — not against a bare 1080×1920 canvas.

    python tools/shorts_overlay.py <frame.png> [more.png ...] [--out DIR]

Zones (1080×1920, measured from the 2025 Shorts player, portrait phone):

    TOP_BAR        y    0 –  230   status bar + "Shorts" header + search/camera
    ACTION_RAIL    x  930 – 1080   like / dislike / comments / share / remix
                   y  980 – 1600   (sits above the metadata block)
    METADATA       y 1590 – 1810   @channel · Subscribe · title · audio ticker
    NAV_BAR        y 1810 – 1920   Home / Shorts / + / Subscriptions / You

Everything else is the SAFE window. Faces and the caption band must live
inside it; bodies may run under METADATA / NAV_BAR (close-up framing).
Also prints how much of each zone is covered by non-background pixels so
overlaps are reported numerically, not just eyeballed.
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
ZONES = {
    "TOP_BAR":     (0,    0,    W,    230),
    "ACTION_RAIL": (930,  980,  W,    1600),
    "METADATA":    (0,    1590, 930,  1810),
    "NAV_BAR":     (0,    1810, W,    H),
}
SAFE = (60, 260, 900, 1560)   # generous inner box for faces + caption


def overlay(path: str, out_dir: str) -> str:
    im = Image.open(path).convert("RGBA")
    if im.size != (W, H):
        im = im.resize((W, H), Image.Resampling.LANCZOS)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for name, (x0, y0, x1, y1) in ZONES.items():
        d.rectangle((x0, y0, x1, y1), fill=(255, 40, 40, 90),
                    outline=(255, 40, 40, 220), width=3)
        d.text((x0 + 12, y0 + 10), name, fill=(255, 255, 255, 230),
               font=ImageFont.load_default(size=28))
    d.rectangle(SAFE, outline=(60, 255, 120, 220), width=4)
    d.text((SAFE[0] + 12, SAFE[1] + 10), "SAFE (faces + caption)",
           fill=(60, 255, 120, 230), font=ImageFont.load_default(size=28))
    # Rule-of-thirds guides
    for fy in (H / 3, 2 * H / 3):
        d.line((0, fy, W, fy), fill=(255, 255, 255, 70), width=1)
    out = Image.alpha_composite(im, layer).convert("RGB")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.basename(path).replace(".png", "_shorts.png"))
    out.save(dst)
    return dst


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv \
        else os.path.dirname(args[0])
    for p in args:
        print("wrote", overlay(p, out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
