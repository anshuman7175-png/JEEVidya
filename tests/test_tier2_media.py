"""Tier 2 — Motifs, grade, audio forge: fast sanity + determinism."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.post_production import ColorGrade
from engine.visual_dna import VisualDNA
from tools.motif_forge import MOTIF_NAMES, render_motif


def test_all_25_plus_motifs_render():
    assert len(MOTIF_NAMES) >= 25
    for name in MOTIF_NAMES:
        img = render_motif(name, 96, (0, 212, 255))
        assert img.size == (96, 96) and img.mode == "RGBA"
        # something visible was drawn
        assert np.asarray(img)[..., 3].sum() > 0, f"motif '{name}' is empty"


def test_motif_cache_returns_same_object():
    a = render_motif("atom", 64, (255, 0, 0))
    b = render_motif("atom", 64, (255, 0, 0))
    assert a is b


def test_grade_is_deterministic_and_shaped():
    dna = VisualDNA.from_title("Thermodynamics")
    g1 = ColorGrade(dna, 120, 200)
    g2 = ColorGrade(dna, 120, 200)
    frame = Image.new("RGB", (120, 200), (100, 110, 140))
    out1 = np.asarray(g1.apply(frame, 3))
    out2 = np.asarray(g2.apply(frame, 3))
    assert np.array_equal(out1, out2)            # cache-safe
    # vignette: corners darker than center
    assert out1[0, 0].mean() < out1[100, 60].mean()


def test_audio_forge_sfx_shapes():
    from tools.audio_forge import (SR, loudness_lufs, master, sfx_bass_drop,
                                   sfx_pop, sfx_riser, sfx_whoosh)
    for fn in (sfx_whoosh, sfx_pop, sfx_riser, sfx_bass_drop):
        x = fn()
        assert x.dtype == np.float32 and len(x) > SR // 20
        y = master(x)
        assert float(np.abs(y).max()) <= 1.0     # never clips
        assert -18.0 < loudness_lufs(y) < -8.0   # near the -14 target


def test_bgm_deterministic_per_dna():
    from tools.audio_forge import forge_bgm
    dna = VisualDNA.from_title("Waves")
    a = forge_bgm(dna, seconds=2.0)
    b = forge_bgm(dna, seconds=2.0)
    assert np.array_equal(a, b)
    c = forge_bgm(VisualDNA.from_title("Heat"), seconds=2.0)
    assert not np.array_equal(a, c)
