"""
Tests for StageGeometry and Visual Bible 3-Stage Layout System (§3).
"""
import pytest
from engine.stages import StageGeometry, StageLayout


def test_stage_geometry_initialization():
    sg = StageGeometry(width=1080, height=1920, res_scale=1.0)
    assert sg.width == 1080
    assert sg.height == 1920
    assert sg.res_scale == 1.0


def test_stage_duo_layout():
    sg = StageGeometry()
    layout = sg.get_layout("DUO", t_seconds=0.0)
    assert isinstance(layout, StageLayout)
    assert layout.stage_name == "DUO"
    assert layout.alpha_lab == 0.0
    # Both characters in dialogue framing
    assert layout.gudiya_x < layout.chintu_x
    assert 1200 <= layout.gudiya_y <= 1700
    assert 1200 <= layout.chintu_y <= 1700


def test_stage_lab_layout_safe_zones():
    sg = StageGeometry()
    layout = sg.get_layout("LAB", t_seconds=1.0)
    assert layout.stage_name == "LAB"
    assert layout.alpha_lab == 1.0

    # Top 50% reserved for simulation canvas
    vx1, vy1, vx2, vy2 = layout.viewport_rect
    assert vx1 == 0
    assert vy1 == 0
    assert vx2 == 1080
    assert vy2 == 820

    # Characters must sit below simulation canvas (y >= 900)
    assert layout.gudiya_y >= 960
    assert layout.chintu_y >= 960
    assert layout.gudiya_scale <= 0.80
    assert layout.chintu_scale <= 0.80


def test_stage_proof_layout():
    sg = StageGeometry()
    layout = sg.get_layout("PROOF", t_seconds=0.5)
    assert layout.stage_name == "PROOF"
    # In hero proof stage, characters step aside/shrink
    assert layout.gudiya_scale <= 0.50
    assert layout.chintu_scale <= 0.50


def test_stage_transition_cubic_easing():
    sg = StageGeometry()
    # At t=0, starting from DUO transition to LAB
    l_start = sg.transition_layout("DUO", "LAB", progress=0.0)
    assert l_start.alpha_lab == 0.0

    l_mid = sg.transition_layout("DUO", "LAB", progress=0.5)
    assert 0.0 < l_mid.alpha_lab < 1.0

    l_end = sg.transition_layout("DUO", "LAB", progress=1.0)
    assert l_end.alpha_lab == 1.0


def test_res_scale_scaling():
    sg_half = StageGeometry(width=540, height=960, res_scale=0.5)
    layout = sg_half.get_layout("LAB", t_seconds=1.0)
    vx1, vy1, vx2, vy2 = layout.viewport_rect
    assert vx2 == 540
    assert vy2 == 410
