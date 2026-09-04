"""
JEEVidya — Visual verification harness (no audio, no TTS, no network).

Renders a handful of FULL-RESOLUTION frames straight through the V5
compositor with synthetic word timings, so the caption band, both
characters, the pose cut and the lens stack can be inspected pixel by
pixel before a real render is committed.

    python tools/verify_frames.py [out_dir] [--res 1.0]

Writes:
    <out>/girl_t<ms>.png    girl speaking, several timestamps
    <out>/boy_t<ms>.png     boy speaking
    <out>/contact.png       2x3 contact sheet at 25 %
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from config import settings  # noqa: E402


def _vtt(words, path):
    """Write an edge-tts-style word VTT (one cue per word)."""
    def ts(ms):
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for text, a, b in words:
            f.write(f"{ts(a)} --> {ts(b)}\n{text}\n\n")


class _Sink:
    def __init__(self):
        self.frames = []

    def write_frame(self, img):
        self.frames.append(img.copy())


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else "/tmp/agent-browser/verify"
    res = 1.0
    if "--res" in sys.argv:
        res = float(sys.argv[sys.argv.index("--res") + 1])
    os.makedirs(out, exist_ok=True)

    from engine.visual_dna import VisualDNA
    from pipeline.compositor_v5 import StreamingCompositor
    from pipeline.timeline import Timeline

    tmp = tempfile.mkdtemp(prefix="jv_verify_")
    girl_words = [("क्या", 0, 300), ("आपको", 300, 650), ("पता", 650, 950),
                  ("है", 950, 1150), ("220", 1150, 1600), ("किलोमीटर", 1600, 2200),
                  ("प्रति", 2200, 2500), ("घंटा?", 2500, 2900)]
    boy_words = [("Arre", 0, 350), ("wow!", 350, 800), ("Formula", 800, 1300),
                 ("kya", 1300, 1550), ("hai", 1550, 1800), ("iska?", 1800, 2300)]
    g_vtt, b_vtt = os.path.join(tmp, "g.vtt"), os.path.join(tmp, "b.vtt")
    _vtt(girl_words, g_vtt)
    _vtt(boy_words, b_vtt)

    turns = [
        {"turn_id": 1, "speaker": "girl", "duration_ms": 3400,
         "emotion": "curious", "shot_type": "two_shot", "vtt": g_vtt,
         "text": "क्या आपको पता है 220 किलोमीटर प्रति घंटा?"},
        {"turn_id": 2, "speaker": "boy", "duration_ms": 2800,
         "emotion": "amazed", "shot_type": "two_shot", "vtt": b_vtt,
         "text": "Arre wow! Formula kya hai iska?"},
    ]
    timeline = Timeline(turns, fps=settings.FPS)
    dna = VisualDNA.from_title("VERIFY FRAMES")
    comp = StreamingCompositor(res_scale=res, dna=dna)
    print(f"  caption font={comp.caption_style.font_px}px stroke="
          f"{comp.caption_style.stroke_px}px backend="
          f"{comp.caption_renderer.backend} clear_y={comp.caption_clear_y}")

    seed = "a" * 64
    sink = _Sink()
    n0 = comp.render_segment(timeline, 0, sink, seed, max_frames=None)
    sink2 = _Sink()
    comp.render_segment(timeline, 1, sink2, seed[::-1], prev_shot="two_shot",
                        max_frames=None)

    fps = settings.FPS
    picks = {}
    for ms in (120, 700, 1230, 2000, 2650):
        f = min(len(sink.frames) - 1, round(ms * fps / 1000))
        picks[f"girl_t{ms:04d}"] = sink.frames[f]
    for ms in (400, 1400):
        f = min(len(sink2.frames) - 1, round(ms * fps / 1000))
        picks[f"boy_t{ms:04d}"] = sink2.frames[f]

    for name, img in picks.items():
        img.convert("RGB").save(os.path.join(out, f"{name}.png"))

    # Contact sheet
    thumbs = [im.convert("RGB").resize((im.width // 4, im.height // 4),
                                       Image.Resampling.LANCZOS)
              for im in picks.values()]
    tw, th = thumbs[0].size
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, th * rows), (0, 0, 0))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.save(os.path.join(out, "contact.png"))
    print(f"  wrote {len(picks)} frames + contact.png → {out} "
          f"({n0} frames rendered for turn 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
