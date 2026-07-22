"""
JEEVidya V5 — Thumbnail Forge (Tier 4)
══════════════════════════════════════
Best-frame detection + title card. Scoring per candidate frame:

    sharpness (Laplacian variance) × colorfulness × brightness-sanity
    × position prior (20-70% of runtime — past the hook, before the CTA)

The winner gets the DNA-colored title card: gradient energy bar, huge
stroked headline, channel chip. Pure PIL + numpy.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine.render_fast import resolve_devanagari_font

if TYPE_CHECKING:
    from engine.visual_dna import VisualDNA


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _extract_candidates(video_path: str, every_s: float = 0.75,
                        max_frames: int = 60) -> List[Tuple[float, Image.Image]]:
    """(position 0..1, frame) pairs sampled across the video."""
    with tempfile.TemporaryDirectory() as tmp:
        pattern = os.path.join(tmp, "c_%03d.png")
        subprocess.run(
            [_ffmpeg_exe(), "-y", "-i", video_path,
             "-vf", f"fps=1/{every_s}", "-frames:v", str(max_frames), pattern],
            capture_output=True, check=True)
        names = sorted(os.listdir(tmp))
        out = []
        for i, name in enumerate(names):
            img = Image.open(os.path.join(tmp, name)).convert("RGB")
            img.load()
            out.append((i / max(1, len(names) - 1), img))
        return out


def _score(pos: float, img: Image.Image) -> float:
    small = img.resize((180, 320))
    arr = np.asarray(small, dtype=np.float32)
    gray = arr.mean(axis=2)

    # Sharpness: variance of a 4-neighbour Laplacian
    lap = (4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
           - gray[1:-1, :-2] - gray[1:-1, 2:])
    sharp = float(lap.var()) / 1000.0

    # Colorfulness (Hasler–Süsstrunk, simplified)
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colorful = float(np.hypot(rg.std(), yb.std())) / 60.0

    # Brightness sanity: punish near-black / blown frames
    mean = gray.mean() / 255.0
    bright = 1.0 - abs(mean - 0.42) * 2.2

    # Position prior: peak interest zone after the hook, before the CTA
    prior = 1.0 - abs(pos - 0.45) * 1.2

    return max(0.0, sharp) * max(0.1, colorful) * max(0.1, bright) \
        * max(0.2, prior)


def best_frame(video_path: str) -> Image.Image:
    candidates = _extract_candidates(video_path)
    if not candidates:
        raise RuntimeError(f"No frames extractable from {video_path}")
    return max(candidates, key=lambda c: _score(*c))[1]


def make_thumbnail(video_path: str, title: str, out_path: str,
                   dna: Optional["VisualDNA"] = None) -> str:
    """Best frame + DNA title card → 1080x1920 JPEG."""
    frame = best_frame(video_path).resize((1080, 1920),
                                          Image.Resampling.LANCZOS)
    frame = frame.convert("RGBA")

    primary = dna.palette["primary"] if dna else (0, 212, 255)
    accent = dna.palette["accent"] if dna else (255, 51, 102)

    # Darken the title zone for contrast
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle((0, 0, 1080, 560), fill=(0, 0, 12, 150))
    overlay = overlay.filter(ImageFilter.GaussianBlur(2))
    frame = Image.alpha_composite(frame, overlay)
    d = ImageDraw.Draw(frame)

    # Energy bar: primary→accent gradient strip
    for x in range(1080):
        t = x / 1080
        col = tuple(int(primary[i] * (1 - t) + accent[i] * t) for i in range(3))
        d.line((x, 560, x, 588), fill=col + (255,))

    # Headline: wrap to ≤3 lines, huge, stroked
    font = resolve_devanagari_font(96)
    words, lines, cur = title.split(), [], ""
    for w in words:
        test = f"{cur} {w}".strip()
        if d.textlength(test, font=font) > 980 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)
    y = 90
    for line in lines[:3]:
        x = (1080 - d.textlength(line, font=font)) / 2
        d.text((x, y), line, font=font, fill=(255, 255, 255),
               stroke_width=6, stroke_fill=(0, 0, 0))
        y += 128

    # Channel chip
    chip_font = resolve_devanagari_font(40)
    d.rounded_rectangle((30, 1800, 420, 1880), radius=40,
                        fill=(0, 0, 0, 190), outline=primary + (255,), width=3)
    d.text((60, 1818), "Gudiya & Chintu", font=chip_font,
           fill=primary + (255,))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frame.convert("RGB").save(out_path, "JPEG", quality=92)
    return out_path
