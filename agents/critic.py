"""
JEEVidya V5 — Vision Critic Agent (Tier 3)
══════════════════════════════════════════
The system reviews its own videos before you do:

  1. Sample 12 frames from a (preview) render via ffmpeg
  2. Send them to Gemini vision with a hard visual-QC rubric
  3. Get back a STRUCTURED defect list
  4. Map each defect to a concrete parameter fix (move caption Y,
     shrink hologram, raise contrast…) the pipeline can apply
  5. Save a per-video critic report next to the output

Humans stay in the loop for FACTS (per the QC philosophy); the Critic
owns legibility, overlap, contrast, dead frames, and mangled Devanagari.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.llm import LLM

N_SAMPLE_FRAMES = 12

VISION_RUBRIC = """You are a merciless visual QC reviewer for a Hinglish
educational YouTube Shorts channel (1080x1920 vertical). You will receive
sampled frames from one video, in chronological order.

Check EVERY frame for these defect types:
- caption_illegible: caption text too small / low contrast / clipped
- caption_overlap: captions overlapping a character's face or a formula
- devanagari_mangled: Hindi text showing boxes, torn glyphs, or gibberish
- low_contrast: subject barely separated from background
- dead_frame: nothing meaningful on screen (empty/near-black/stuck)
- element_clipped: formula, character, or diagram cut off at an edge
- hologram_oversized: formula panel dominating > 60% of frame height

Return JSON only:
{"defects": [{"frame_index": n, "type": "<one of the types>",
  "severity": "low"|"medium"|"high", "note": "one specific sentence"}],
 "overall": "pass"|"fix_needed", "summary": "one paragraph"}
If everything is clean return {"defects": [], "overall": "pass", ...}."""

# Defect type → concrete pipeline parameter fix (what auto-repair applies)
FIX_MAP: Dict[str, Dict[str, Any]] = {
    "caption_illegible": {"param": "caption_font_scale", "delta": +0.15,
                          "hint": "increase CAPTION_FONT_SIZE ~15%"},
    "caption_overlap": {"param": "caption_y_position", "delta": +0.05,
                        "hint": "push CAPTION_Y_POSITION lower"},
    "devanagari_mangled": {"param": "font_fallback", "delta": None,
                           "hint": "run jvmake doctor — Devanagari font "
                                   "missing; captions falling back"},
    "low_contrast": {"param": "grade_contrast", "delta": +0.08,
                     "hint": "raise DNA grade curve strength"},
    "dead_frame": {"param": "motif_density", "delta": +0.2,
                   "hint": "raise DNA motif_density / check explanation "
                           "visual_elements"},
    "element_clipped": {"param": "explain_margin", "delta": +0.04,
                        "hint": "shrink explanation content area"},
    "hologram_oversized": {"param": "hologram_scale", "delta": -0.15,
                           "hint": "reduce formula panel max width"},
}


@dataclass
class CriticReport:
    video: str
    overall: str = "pass"
    summary: str = ""
    defects: List[Dict[str, Any]] = field(default_factory=list)
    fixes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"video": self.video, "overall": self.overall,
                "summary": self.summary, "defects": self.defects,
                "fixes": self.fixes}

    @property
    def needs_fix(self) -> bool:
        return self.overall == "fix_needed" and bool(self.fixes)


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def sample_frames(video_path: str, n: int = N_SAMPLE_FRAMES,
                  max_side: int = 640) -> List[bytes]:
    """Evenly sample n JPEG frames from a video (downscaled for vision)."""
    with tempfile.TemporaryDirectory() as tmp:
        pattern = os.path.join(tmp, "f_%03d.jpg")
        # thumbnail-quality decimation: pick n frames spread over duration
        cmd = [_ffmpeg_exe(), "-y", "-i", video_path,
               "-vf", f"select='not(mod(n\\,{max(1, _frame_stride(video_path, n))}))',"
                      f"scale=-2:{max_side}",
               "-vsync", "vfr", "-frames:v", str(n), "-q:v", "5", pattern]
        subprocess.run(cmd, capture_output=True, check=True)
        frames = []
        for name in sorted(os.listdir(tmp)):
            with open(os.path.join(tmp, name), "rb") as f:
                frames.append(f.read())
        return frames


def _frame_stride(video_path: str, n: int) -> int:
    """Total frames / n, probed cheaply from ffmpeg's own metadata."""
    out = subprocess.run([_ffmpeg_exe(), "-i", video_path],
                         capture_output=True, text=True).stderr
    import re
    dur = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    fps = re.search(r"(\d+(?:\.\d+)?) fps", out)
    if not dur:
        return 30
    h, m, s = float(dur.group(1)), float(dur.group(2)), float(dur.group(3))
    total = (h * 3600 + m * 60 + s) * (float(fps.group(1)) if fps else 30)
    return max(1, int(total / max(1, n)))


class Critic:
    """The vision QC loop."""

    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or LLM()

    def review(self, video_path: str,
               save_report: bool = True) -> CriticReport:
        """Sample → judge → map fixes → persist report."""
        report = CriticReport(video=video_path)

        print(f"  [Critic] Sampling {N_SAMPLE_FRAMES} frames: "
              f"{os.path.basename(video_path)}")
        frames = sample_frames(video_path)
        if not frames:
            report.overall = "fix_needed"
            report.summary = "No frames could be sampled — encode problem."
            return report

        print(f"  [Critic] Reviewing {len(frames)} frames with vision model")
        result = self.llm.generate_json(
            f"These are {len(frames)} chronological frames from one video. "
            "Apply the rubric.", system=VISION_RUBRIC,
            images_jpeg=frames, temperature=0.1)

        report.overall = result.get("overall", "pass")
        report.summary = result.get("summary", "")
        report.defects = [d for d in result.get("defects", [])
                          if isinstance(d, dict) and d.get("type") in FIX_MAP]

        # Defects → parameter fixes (dedup by param, worst severity wins)
        by_param: Dict[str, Dict[str, Any]] = {}
        sev_rank = {"low": 0, "medium": 1, "high": 2}
        for d in report.defects:
            fix = dict(FIX_MAP[d["type"]])
            fix.update({"defect": d["type"],
                        "severity": d.get("severity", "medium"),
                        "frames": [d.get("frame_index")]})
            prev = by_param.get(fix["param"])
            if prev is None:
                by_param[fix["param"]] = fix
            else:
                prev["frames"].append(d.get("frame_index"))
                if sev_rank.get(fix["severity"], 1) > sev_rank.get(
                        prev["severity"], 1):
                    prev["severity"] = fix["severity"]
        report.fixes = list(by_param.values())

        if save_report:
            path = os.path.splitext(video_path)[0] + ".critic.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"  [Critic] Report: {path} — {report.overall} "
                  f"({len(report.defects)} defects, {len(report.fixes)} fixes)")
        return report

    def apply_fixes(self, report: CriticReport,
                    dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """Bake auto-fixable deltas into the dialogue's render overrides.
        Returns the modified dialogue (re-render picks them up; Tier 0
        keys change → only affected artifacts rebuild)."""
        overrides = dialogue.setdefault("render_overrides", {})
        for fix in report.fixes:
            if fix["delta"] is None:
                continue                        # human-action fixes
            key = fix["param"]
            overrides[key] = round(overrides.get(key, 0.0) + fix["delta"], 4)
        return dialogue
