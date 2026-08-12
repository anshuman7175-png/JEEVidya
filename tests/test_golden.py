"""Tier 5 — Golden-frame regression: SSIM ≥ 0.98 against reference frames.

First run (or `JV_UPDATE_GOLDEN=1`) writes the references; every later
run asserts the renderer still produces the same pictures. This is what
makes an 8,000-line renderer safe to evolve.

Skips gracefully when character assets aren't present (CI-safe).
"""
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
SSIM_THRESHOLD = 0.98


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM (luma), sufficient for whole-frame regression."""
    a = a.astype(np.float64).mean(axis=2)
    b = b.astype(np.float64).mean(axis=2)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / \
           ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))


def _have_assets() -> bool:
    d = os.path.join(settings.CHARACTERS_DIR, "gudiya")
    return os.path.isdir(d) and any(
        f.startswith(("body", "original")) for f in os.listdir(d))


def _render_reference_frames():
    """Render 3 deterministic frames straight through the compositor."""
    from engine.visual_dna import VisualDNA
    from pipeline.compositor_v5 import StreamingCompositor
    from pipeline.timeline import Timeline

    dna = VisualDNA.from_title("GOLDEN REFERENCE")
    comp = StreamingCompositor(res_scale=0.25, dna=dna)
    turns = [
        {"turn_id": 1, "speaker": "boy", "duration_ms": 2000,
         "emotion": "explaining", "shot_type": "two_shot", "vtt": None},
        {"turn_id": 2, "speaker": "explanation", "duration_ms": 2000,
         "emotion": "neutral", "shot_type": "fullscreen_explain", "vtt": None,
         "visual_elements": [
             {"action": "draw_circle",
              "params": {"radius": 150, "color": "primary"}},
             {"action": "show_formula",
              "params": {"latex": "$E = mc^2$", "y": 300}}]},
    ]
    timeline = Timeline(turns, fps=settings.FPS)

    class Collect:
        frames = []

        def write_frame(self, img):
            self.frames.append(img.copy() if hasattr(img, "copy") else img)

    sink = Collect()
    seed = "f" * 64
    comp.render_segment(timeline, 0, sink, seed, max_frames=2)
    comp.render_segment(timeline, 1, sink, seed, max_frames=1)
    return [f if isinstance(f, Image.Image) else Image.fromarray(f)
            for f in sink.frames]


@pytest.mark.skipif(not _have_assets(),
                    reason="character assets not present")
def test_golden_frames():
    frames = _render_reference_frames()
    assert len(frames) == 3
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    update = os.environ.get("JV_UPDATE_GOLDEN") == "1"
    for i, frame in enumerate(frames):
        ref_path = os.path.join(GOLDEN_DIR, f"golden_{i:02d}.png")
        if update or not os.path.exists(ref_path):
            frame.convert("RGB").save(ref_path)
            continue
        ref = np.asarray(Image.open(ref_path).convert("RGB"))
        cur = np.asarray(frame.convert("RGB"))
        assert ref.shape == cur.shape, f"frame {i} changed size"
        score = ssim(ref, cur)
        assert score >= SSIM_THRESHOLD, \
            (f"frame {i} drifted: SSIM={score:.4f} < {SSIM_THRESHOLD}. "
             "If intentional, re-baseline with JV_UPDATE_GOLDEN=1")


@pytest.mark.skipif(not _have_assets(),
                    reason="character assets not present")
def test_render_is_deterministic():
    """Same inputs → byte-identical pixels (the Tier 0 contract)."""
    a = _render_reference_frames()
    b = _render_reference_frames()
    for i, (fa, fb) in enumerate(zip(a, b, strict=True)):
        assert np.array_equal(np.asarray(fa.convert("RGB")),
                              np.asarray(fb.convert("RGB"))), \
            f"frame {i} is non-deterministic"
