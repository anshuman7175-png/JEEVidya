"""
JEEVidya V5 — Motif Forge (Tier 2)
══════════════════════════════════
25 procedural vector motifs (atoms, planets, integrals, beakers, gears…)
drawn entirely with PIL primitives in DNA colors. Zero downloaded assets,
infinitely recolorable, cached per (name, size, color).

Every motif draws inside a circle of radius r centered at (c, c) on a
transparent canvas — a uniform contract so the physics world can fling
any of them around interchangeably.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFilter

Color = Tuple[int, int, int]

# ─── Drawing helpers ────────────────────────────────────────

def _poly_points(cx, cy, r, n, rot=0.0):
    return [(cx + r * math.cos(2 * math.pi * i / n + rot),
             cy + r * math.sin(2 * math.pi * i / n + rot)) for i in range(n)]


def _wave(d, cx, cy, r, color, w, freq=2.0, amp=0.35):
    pts = [(cx - r + 2 * r * i / 40,
            cy + math.sin(i / 40 * math.pi * 2 * freq) * r * amp)
           for i in range(41)]
    d.line(pts, fill=color, width=w)


# ─── The 25 motifs ──────────────────────────────────────────
# Each: fn(draw, c, r, color, w) drawing on an RGBA canvas.

def _atom(d, c, r, col, w):
    d.ellipse((c - r * .12, c - r * .12, c + r * .12, c + r * .12), fill=col)
    for rot in (0, math.pi / 3, -math.pi / 3):
        bbox = (c - r, c - r * .38, c + r, c + r * .38)
        e = Image.new("RGBA", (int(2 * r + 4),) * 2, (0, 0, 0, 0))
        ed = ImageDraw.Draw(e)
        ed.ellipse((2, r * .62, 2 * r + 2, r * 1.38), outline=col, width=w)
        e = e.rotate(math.degrees(rot), resample=Image.Resampling.BILINEAR)
        d._image.alpha_composite(e, (int(c - r - 2), int(c - r - 2)))
    _ = bbox


def _planet(d, c, r, col, w):
    d.ellipse((c - r * .55, c - r * .55, c + r * .55, c + r * .55),
              outline=col, width=w)
    d.arc((c - r, c - r * .35, c + r, c + r * .35), 200, 340, fill=col, width=w)
    d.arc((c - r, c - r * .35, c + r, c + r * .35), 20, 160, fill=col, width=w)


def _helix(d, c, r, col, w):
    for phase in (0.0, math.pi):
        pts = [(c + math.sin(t / 14 * math.pi * 2 + phase) * r * .5,
                c - r + 2 * r * t / 28) for t in range(29)]
        d.line(pts, fill=col, width=w)
    for t in range(2, 28, 5):
        y = c - r + 2 * r * t / 28
        x1 = c + math.sin(t / 14 * math.pi * 2) * r * .5
        x2 = c + math.sin(t / 14 * math.pi * 2 + math.pi) * r * .5
        d.line((x1, y, x2, y), fill=col, width=max(1, w - 1))


def _beaker(d, c, r, col, w):
    d.line((c - r * .35, c - r, c - r * .35, c - r * .2), fill=col, width=w)
    d.line((c + r * .35, c - r, c + r * .35, c - r * .2), fill=col, width=w)
    d.polygon([(c - r * .35, c - r * .2), (c - r * .8, c + r),
               (c + r * .8, c + r), (c + r * .35, c - r * .2)],
              outline=col, width=w)
    d.line((c - r * .55, c + r * .45, c + r * .55, c + r * .45),
           fill=col, width=w)


def _gear(d, c, r, col, w):
    n = 8
    for i in range(n):
        a = 2 * math.pi * i / n
        d.line((c + math.cos(a) * r * .6, c + math.sin(a) * r * .6,
                c + math.cos(a) * r, c + math.sin(a) * r), fill=col, width=w + 2)
    d.ellipse((c - r * .6, c - r * .6, c + r * .6, c + r * .6),
              outline=col, width=w)
    d.ellipse((c - r * .22, c - r * .22, c + r * .22, c + r * .22),
              outline=col, width=w)


def _waveform(d, c, r, col, w):
    _wave(d, c, c, r, col, w, freq=2.5, amp=0.45)


def _vector(d, c, r, col, w):
    d.line((c - r * .8, c + r * .8, c + r * .7, c - r * .7), fill=col, width=w)
    d.polygon([(c + r * .8, c - r * .8), (c + r * .35, c - r * .65),
               (c + r * .65, c - r * .35)], fill=col)


def _integral(d, c, r, col, w):
    d.arc((c - r * .5, c - r, c + r * .3, c - r * .3), 180, 330, fill=col, width=w)
    d.line((c, c - r * .62, c, c + r * .62), fill=col, width=w)
    d.arc((c - r * .3, c + r * .3, c + r * .5, c + r), 0, 150, fill=col, width=w)


def _sigma(d, c, r, col, w):
    d.line((c + r * .6, c - r * .8, c - r * .6, c - r * .8), fill=col, width=w)
    d.line((c - r * .6, c - r * .8, c + r * .1, c), fill=col, width=w)
    d.line((c + r * .1, c, c - r * .6, c + r * .8), fill=col, width=w)
    d.line((c - r * .6, c + r * .8, c + r * .6, c + r * .8), fill=col, width=w)


def _lens(d, c, r, col, w):
    d.arc((c - r, c - r, c + r * .4, c + r), 290, 70, fill=col, width=w)
    d.arc((c - r * .4, c - r, c + r, c + r), 110, 250, fill=col, width=w)
    d.line((c - r, c, c - r * .5, c), fill=col, width=max(1, w - 1))
    d.line((c + r * .5, c, c + r, c), fill=col, width=max(1, w - 1))


def _magnet(d, c, r, col, w):
    d.arc((c - r * .7, c - r, c + r * .7, c + r * .4), 180, 360, fill=col, width=w + 2)
    d.line((c - r * .7, c - r * .3, c - r * .7, c + r * .3), fill=col, width=w + 2)
    d.line((c + r * .7, c - r * .3, c + r * .7, c + r * .3), fill=col, width=w + 2)
    for k in (0.55, 0.8):
        d.arc((c - r * k, c + r * .1, c + r * k, c + r), 20, 160,
              fill=col, width=max(1, w - 1))


def _pendulum(d, c, r, col, w):
    d.line((c - r * .6, c - r, c + r * .6, c - r), fill=col, width=w)
    d.line((c, c - r, c + r * .35, c + r * .5), fill=col, width=max(1, w - 1))
    d.ellipse((c + r * .2, c + r * .35, c + r * .5, c + r * .65), fill=col)
    d.arc((c - r * .5, c - r * .1, c + r * .5, c + r * .9), 60, 120,
          fill=col, width=1)


def _spring(d, c, r, col, w):
    pts = [(c - r + 2 * r * t / 30,
            c + math.sin(t * math.pi / 3) * r * .3) for t in range(31)]
    d.line(pts, fill=col, width=w)


def _orbit(d, c, r, col, w):
    d.ellipse((c - r, c - r * .45, c + r, c + r * .45), outline=col, width=w)
    d.ellipse((c - r * .18, c - r * .18, c + r * .18, c + r * .18), fill=col)
    d.ellipse((c + r * .68, c - r * .32, c + r * .92, c - r * .08), fill=col)


def _prism(d, c, r, col, w):
    d.polygon(_poly_points(c, c, r * .8, 3, -math.pi / 2), outline=col, width=w)
    d.line((c - r, c + r * .1, c - r * .28, c + r * .1), fill=col, width=max(1, w - 1))
    for dy in (-0.12, 0.02, 0.16):
        d.line((c + r * .3, c + r * .05, c + r, c + r * (dy + .2)),
               fill=col, width=max(1, w - 1))


def _graph(d, c, r, col, w):
    d.line((c - r, c + r, c - r, c - r), fill=col, width=w)
    d.line((c - r, c + r, c + r, c + r), fill=col, width=w)
    pts = [(c - r + 2 * r * t / 20,
            c + r - (2 * r * (t / 20) ** 2)) for t in range(21)]
    d.line(pts, fill=col, width=max(1, w - 1))


def _molecule(d, c, r, col, w):
    nodes = [(c, c - r * .7), (c - r * .8, c + r * .4), (c + r * .8, c + r * .4)]
    for i in range(3):
        for j in range(i + 1, 3):
            d.line((*nodes[i], *nodes[j]), fill=col, width=max(1, w - 1))
    for x, y in nodes:
        d.ellipse((x - r * .18, y - r * .18, x + r * .18, y + r * .18), fill=col)


def _rocket(d, c, r, col, w):
    d.polygon([(c, c - r), (c - r * .3, c - r * .2), (c - r * .3, c + r * .5),
               (c + r * .3, c + r * .5), (c + r * .3, c - r * .2)],
              outline=col, width=w)
    d.polygon([(c - r * .3, c + r * .2), (c - r * .6, c + r * .7),
               (c - r * .3, c + r * .5)], fill=col)
    d.polygon([(c + r * .3, c + r * .2), (c + r * .6, c + r * .7),
               (c + r * .3, c + r * .5)], fill=col)
    d.line((c, c + r * .55, c, c + r * .95), fill=col, width=w)


def _bulb(d, c, r, col, w):
    d.ellipse((c - r * .5, c - r * .9, c + r * .5, c + r * .1),
              outline=col, width=w)
    d.line((c - r * .25, c + r * .25, c + r * .25, c + r * .25), fill=col, width=w)
    d.line((c - r * .2, c + r * .45, c + r * .2, c + r * .45), fill=col, width=w)
    for a in range(0, 360, 60):
        ar = math.radians(a)
        d.line((c + math.cos(ar) * r * .65, c - r * .4 + math.sin(ar) * r * .65,
                c + math.cos(ar) * r * .85, c - r * .4 + math.sin(ar) * r * .85),
               fill=col, width=max(1, w - 1))


def _flask(d, c, r, col, w):
    d.line((c - r * .18, c - r, c - r * .18, c - r * .2), fill=col, width=w)
    d.line((c + r * .18, c - r, c + r * .18, c - r * .2), fill=col, width=w)
    d.ellipse((c - r * .75, c - r * .35, c + r * .75, c + r),
              outline=col, width=w)
    d.line((c - r * .55, c + r * .35, c + r * .55, c + r * .35), fill=col, width=w)


def _wave_packet(d, c, r, col, w):
    pts = []
    for t in range(41):
        x = -1 + 2 * t / 40
        env = math.exp(-4 * x * x)
        pts.append((c + x * r, c + math.sin(x * 12) * env * r * .6))
    d.line(pts, fill=col, width=w)


def _axis3d(d, c, r, col, w):
    for a in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6):
        d.line((c, c, c + math.cos(a) * r, c + math.sin(a) * r), fill=col, width=w)
        d.ellipse((c + math.cos(a) * r - 3, c + math.sin(a) * r - 3,
                   c + math.cos(a) * r + 3, c + math.sin(a) * r + 3), fill=col)


def _pi(d, c, r, col, w):
    d.arc((c - r * .9, c - r * .75, c + r * .9, c + r * 1.4), 200, 320,
          fill=col, width=w)
    d.line((c - r * .45, c - r * .5, c - r * .45, c + r * .7), fill=col, width=w)
    d.line((c + r * .45, c - r * .5, c + r * .38, c + r * .7), fill=col, width=w)


def _infinity(d, c, r, col, w):
    d.ellipse((c - r, c - r * .4, c, c + r * .4), outline=col, width=w)
    d.ellipse((c, c - r * .4, c + r, c + r * .4), outline=col, width=w)


def _triangle_rule(d, c, r, col, w):
    d.polygon([(c - r * .8, c + r * .6), (c + r * .8, c + r * .6),
               (c + r * .8, c - r * .7)], outline=col, width=w)
    d.arc((c - r * .95, c + r * .3, c - r * .35, c + r * .9), 300, 360,
          fill=col, width=max(1, w - 1))


def _hourglass(d, c, r, col, w):
    d.polygon([(c - r * .6, c - r), (c + r * .6, c - r), (c, c)],
              outline=col, width=w)
    d.polygon([(c - r * .6, c + r), (c + r * .6, c + r), (c, c)],
              outline=col, width=w)
    d.polygon([(c - r * .2, c + r * .95), (c + r * .2, c + r * .95),
               (c, c + r * .45)], fill=col)


def _satellite(d, c, r, col, w):
    d.rectangle((c - r * .25, c - r * .25, c + r * .25, c + r * .25),
                outline=col, width=w)
    for s in (-1, 1):
        x1, x2 = c + s * r * .35, c + s * r * .95
        d.rectangle((min(x1, x2), c - r * .5, max(x1, x2), c + r * .5),
                    outline=col, width=max(1, w - 1))
        d.line((c + s * r * .25, c, c + s * r * .35, c), fill=col, width=w)


def _cell(d, c, r, col, w):
    d.ellipse((c - r, c - r * .8, c + r, c + r * .8), outline=col, width=w)
    d.ellipse((c - r * .35, c - r * .3, c + r * .15, c + r * .2),
              outline=col, width=w)
    d.ellipse((c + r * .35, c + r * .25, c + r * .6, c + r * .5), fill=col)
    d.ellipse((c - r * .6, c + r * .3, c - r * .4, c + r * .5), fill=col)


MOTIFS: Dict[str, Callable] = {
    "atom": _atom, "planet": _planet, "helix": _helix, "beaker": _beaker,
    "gear": _gear, "waveform": _waveform, "vector": _vector,
    "integral": _integral, "sigma": _sigma, "lens": _lens, "magnet": _magnet,
    "pendulum": _pendulum, "spring": _spring, "orbit": _orbit,
    "prism": _prism, "graph": _graph, "molecule": _molecule,
    "rocket": _rocket, "bulb": _bulb, "flask": _flask,
    "wave_packet": _wave_packet, "axis3d": _axis3d, "pi": _pi,
    "infinity": _infinity, "triangle": _triangle_rule,
    "hourglass": _hourglass, "satellite": _satellite, "cell": _cell,
}
MOTIF_NAMES: List[str] = sorted(MOTIFS.keys())

_cache: Dict[Tuple[str, int, Color, int], Image.Image] = {}


def render_motif(name: str, size: int, color: Color,
                 alpha: int = 255, glow: bool = False) -> Image.Image:
    """Render a motif as an RGBA sprite of `size`×`size`. Cached."""
    key = (name, size, color, alpha)
    img = _cache.get(key)
    if img is None:
        fn = MOTIFS.get(name, _atom)
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d._image = img          # some motifs need alpha_composite access
        c, r = size / 2, size * 0.42
        w = max(2, size // 28)
        fn(d, c, r, color + (alpha,), w)
        if glow:
            halo = img.filter(ImageFilter.GaussianBlur(size // 16))
            img = Image.alpha_composite(halo, img)
        _cache[key] = img
    return img


def forge_sheet(size: int = 128, color: Color = (0, 212, 255)) -> Image.Image:
    """Debug contact sheet of every motif (jvmake forge --motifs)."""
    cols = 7
    rows = (len(MOTIF_NAMES) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * size, rows * size), (12, 12, 40, 255))
    for i, name in enumerate(MOTIF_NAMES):
        sheet.alpha_composite(render_motif(name, size, color),
                              ((i % cols) * size, (i // cols) * size))
    return sheet
