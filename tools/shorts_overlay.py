"""
JEEVidya — YouTube Shorts UI overlay for framing QA.

Draws the regions the Shorts player paints OVER the video on a rendered
frame, so caption / character placement can be judged against what a
viewer actually sees on a phone — not against a bare 1080×1920 canvas.

    python tools/shorts_overlay.py <frame.png> [more.png ...] [--out DIR]

Zones (1080×1920). Published 2025–26 Shorts safe-zone guides disagree by
device and app version (top 150–250, rail 120–200 px wide, bottom 350–420),
so these take the CONSERVATIVE end of each range — the same numbers used by
config/settings.py, config/brand.py and tests/test_naturalism.py:

    TOP_BAR        y    0 –  250   status bar + "Shorts" header + search/camera
    ACTION_RAIL    x  900 – 1080   like / dislike / comments / share / remix
                   y  900 – 1560   (stacked just above the metadata block)
    METADATA       y 1500 – 1810   @channel · Subscribe · title · hashtags · audio
    NAV_BAR        y 1810 – 1920   Home / Shorts / + / Subscriptions / You

Everything else is the SAFE window. Faces and the caption band must live
inside it; bodies may run under METADATA / NAV_BAR (close-up framing).
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
ZONES = {
    "TOP_BAR":     (0,    0,    W,    250),
    "ACTION_RAIL": (900,  900,  W,    1560),
    "METADATA":    (0,    1500, 900,  1810),
    "NAV_BAR":     (0,    1810, W,    H),
}
SAFE = (60, 280, 880, 1480)   # inner box for faces + caption


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
    argv = sys.argv[1:]
    out_dir = None
    if "--out" in argv:
        i = argv.index("--out")
        out_dir = argv[i + 1]
        del argv[i:i + 2]          # don't treat the directory as a frame
    args = [a for a in argv if not a.startswith("--")]
    if out_dir is None:
        out_dir = os.path.dirname(args[0])
    os.makedirs(out_dir, exist_ok=True)
    for p in args:
        print("wrote", overlay(p, out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
