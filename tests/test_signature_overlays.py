"""
Tests for Visual Bible Signature Brand Overlays (§5).
"""
import os
import shutil
import tempfile
import pytest
from PIL import Image

from engine.signature_overlays import (
    ConceptCard, VoronoiShatter, LiveCounter, GudiyasNotebook, TeaserCard
)


def test_concept_card_belief_and_fix():
    b_card = ConceptCard(text="GRAVITY KHATAM HO GAYI?", kind="belief")
    img_b = b_card.render_full_frame(t_seconds=0.5, duration_seconds=3.0)
    assert isinstance(img_b, Image.Image)
    assert img_b.size == (1080, 1920)
    assert img_b.mode == "RGBA"

    f_card = ConceptCard(text="FREE FALL MEIN WEIGHTLESSNESS HOTA HAI", kind="fix")
    img_f = f_card.render_full_frame(t_seconds=1.0, duration_seconds=3.0)
    assert isinstance(img_f, Image.Image)
    assert img_f.size == (1080, 1920)


def test_voronoi_shatter_animation():
    shatter = VoronoiShatter(text="WRONG!")
    # Beginning of crack lines
    frame_0 = shatter.render_full_frame(t_seconds=0.1, duration_seconds=1.8)
    assert isinstance(frame_0, Image.Image)
    assert frame_0.size == (1080, 1920)

    # Mid shatter tumble
    frame_mid = shatter.render_full_frame(t_seconds=0.8, duration_seconds=1.8)
    assert isinstance(frame_mid, Image.Image)
    assert frame_mid.size == (1080, 1920)


def test_live_counter_numeric_growth():
    counter = LiveCounter(target_value="8.7", unit="m/s²", source="ISS Orbit (400 km)", from_val=9.8)
    # Start: ~9.8
    img_start = counter.render(progress=0.0)
    assert isinstance(img_start, Image.Image)

    # End: 8.7
    img_end = counter.render(progress=1.0)
    assert isinstance(img_end, Image.Image)

    # Full-frame wrapper
    full = counter.render_full_frame(t_seconds=1.2, duration_seconds=2.8)
    assert full.size == (1080, 1920)


def test_gudiyas_notebook_and_png_export():
    tmp_dir = tempfile.mkdtemp()
    try:
        nb = GudiyasNotebook(
            title="Gravitational Force Formula",
            formula_latex=r"g = \frac{GM}{(R+h)^2}",
            chapter="Gravitation & Motion",
            tags="JEE · NEET"
        )
        full = nb.render_full_frame(t_seconds=1.0, duration_seconds=3.5, auto_export_dir=tmp_dir)
        assert isinstance(full, Image.Image)
        assert full.size == (1080, 1920)

        # Check that high-res social card was exported
        exported_files = [f for f in os.listdir(tmp_dir) if f.endswith(".png")]
        assert len(exported_files) >= 1
        export_path = os.path.join(tmp_dir, exported_files[0])
        with Image.open(export_path) as ex_img:
            assert ex_img.size == (1080, 1350)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_teaser_card():
    teaser = TeaserCard(
        next_episode_num=2,
        next_question="Lift ki rassi tooti toh kya hoga?",
        pinned_comment="Apna answer comment karo!"
    )
    full = teaser.render_full_frame(t_seconds=0.5, duration_seconds=3.0)
    assert isinstance(full, Image.Image)
    assert full.size == (1080, 1920)


def test_half_resolution_previews():
    b_card = ConceptCard(text="GRAVITY TEST", kind="belief")
    img_half = b_card.render_full_frame(t_seconds=0.5, canvas_w=540, canvas_h=960, res_scale=0.5)
    assert img_half.size == (270, 480) or img_half.size == (540, 960)


def test_card_emoji_stripping_and_safe_zone_clearance():
    """Verify that cards strip emojis and stay strictly within the upper Idea Zone (y <= 450)."""
    # Text with heavy emojis that would previously cause tofu box
    card = ConceptCard(text="🚀 ❓ KYA GRAVITY KHATAM HO GAYI? ⚡", kind="belief")
    assert "🚀" not in card.text
    assert "⚡" not in card.text
    assert "GRAVITY" in card.text

    full = card.render_full_frame(t_seconds=0.5, duration_seconds=3.0, canvas_w=1080, canvas_h=1920)
    # Check bounding box of non-zero pixels
    bbox = full.getbbox()
    assert bbox is not None
    # Bottom of card must be <= 460 to give >270px head clearance
    assert bbox[3] <= 460, f"Card bottom at {bbox[3]} is too low!"

    # Test teaser card
    teaser = TeaserCard(next_episode_num=2, next_question="❓ Lift gir rahi hai! 🚀", pinned_comment="Comment karo! 💡")
    assert "❓" not in teaser.next_question
    t_full = teaser.render_full_frame(t_seconds=0.5, duration_seconds=3.0, canvas_w=1080, canvas_h=1920)
    t_bbox = t_full.getbbox()
    assert t_bbox is not None
    assert t_bbox[3] <= 460, f"Teaser bottom at {t_bbox[3]} is too low!"

