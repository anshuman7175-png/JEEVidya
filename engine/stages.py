"""
JEEVidya — Stage & Geometry Coordinator (Visual Bible v1 §2)
═════════════════════════════════════════════════════════════
Defines the three visual stages (DUO, LAB, PROOF) for 1080x1920 vertical canvas:
  • DUO: Full-body / three-quarter characters, idea zone for cards, captions.
  • LAB: Split-stage: top 50% procedural physics canvas, bottom waist-up busts.
  • PROOF: Full-screen hero plate / simulation with Live Counter and speaker chip.

Enforces YouTube Shorts UI safe-zones:
  • Right button rail: x >= 930 for y in [1000, 1600]
  • Bottom metadata: y >= 1540
  • Top header: y <= 150
  • Safe text box: x in [70, 1010], y in [200, 1500]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class StageType(str, Enum):
    DUO = "duo"
    LAB = "lab"
    PROOF = "proof"


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out for smooth cinematic stage transitions."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - math.pow(-2.0 * t + 2.0, 3) / 2.0


def ease_out_back(t: float, overshoot: float = 1.2) -> float:
    """Overshoot ease for bouncy card and character pop-ins."""
    t = max(0.0, min(1.0, t))
    c1 = overshoot
    c3 = c1 + 1.0
    return 1.0 + c3 * math.pow(t - 1.0, 3) + c1 * math.pow(t - 1.0, 2)


@dataclass
class CharacterStagePlacement:
    """Placement, scale, and visibility of a character on the canvas."""
    x: float
    y: float
    scale: float
    opacity: float = 1.0
    brightness: float = 1.0
    is_bust: bool = False   # True in LAB (waist-up framing)


@dataclass
class StageLayout:
    """Active frame metrics for rendering characters and simulation viewports."""
    stage_name: str
    viewport_rect: Tuple[int, int, int, int]
    gudiya_x: float
    gudiya_y: float
    gudiya_scale: float
    chintu_x: float
    chintu_y: float
    chintu_scale: float
    alpha_lab: float = 0.0


@dataclass
class StageGeometry:
    """Canvas layout metrics for a given stage."""
    stage_type: StageType = StageType.DUO
    width: int = 1080
    height: int = 1920
    res_scale: float = 1.0

    # Zone Y-boundaries (unscaled 1080x1920 coordinates)
    idea_zone_top: float = 240.0
    idea_zone_bottom: float = 640.0
    canvas_zone_top: float = 180.0
    canvas_zone_bottom: float = 1000.0
    caption_y_top: float = 990.0
    caption_y_bottom: float = 1160.0

    # Character placements (unscaled coordinates)
    gudiya: CharacterStagePlacement = field(default_factory=lambda: CharacterStagePlacement(x=300.0, y=1480.0, scale=0.98))
    chintu: CharacterStagePlacement = field(default_factory=lambda: CharacterStagePlacement(x=780.0, y=1480.0, scale=0.88))

    def get_layout(self, stage_name: str, t_seconds: float = 0.0, camera_submode: str = "wide") -> StageLayout:
        """Compute StageLayout for DUO, LAB, or PROOF with current res_scale."""
        norm = stage_name.strip().lower()
        st = StageType.LAB if norm == "lab" else (StageType.PROOF if norm == "proof" else StageType.DUO)
        geom = get_stage_definition(st, camera_submode=camera_submode).scale_to(self.res_scale)
        rs = self.res_scale
        if st == StageType.LAB:
            return StageLayout(
                stage_name="LAB",
                viewport_rect=(0, 0, int(1080 * rs), int(820 * rs)),
                gudiya_x=geom.gudiya.x,
                gudiya_y=geom.gudiya.y,
                gudiya_scale=geom.gudiya.scale / rs if rs else geom.gudiya.scale,
                chintu_x=geom.chintu.x,
                chintu_y=geom.chintu.y,
                chintu_scale=geom.chintu.scale / rs if rs else geom.chintu.scale,
                alpha_lab=1.0
            )
        elif st == StageType.PROOF:
            return StageLayout(
                stage_name="PROOF",
                viewport_rect=(0, 0, int(1080 * rs), int(1920 * rs)),
                gudiya_x=geom.gudiya.x,
                gudiya_y=geom.gudiya.y,
                gudiya_scale=geom.gudiya.scale,
                chintu_x=geom.chintu.x,
                chintu_y=geom.chintu.y,
                chintu_scale=geom.chintu.scale,
                alpha_lab=0.0
            )
        else:
            return StageLayout(
                stage_name="DUO",
                viewport_rect=(0, 0, 0, 0),
                gudiya_x=geom.gudiya.x,
                gudiya_y=geom.gudiya.y,
                gudiya_scale=geom.gudiya.scale,
                chintu_x=geom.chintu.x,
                chintu_y=geom.chintu.y,
                chintu_scale=geom.chintu.scale,
                alpha_lab=0.0
            )

    def transition_layout(self, from_stage: str, to_stage: str, progress: float) -> StageLayout:
        """Cubic ease-in-out transition between two stage layouts."""
        p = ease_in_out_cubic(progress)
        l_from = self.get_layout(from_stage)
        l_to = self.get_layout(to_stage)
        return StageLayout(
            stage_name=to_stage if p >= 0.5 else from_stage,
            viewport_rect=l_to.viewport_rect if p >= 0.5 else l_from.viewport_rect,
            gudiya_x=l_from.gudiya_x + (l_to.gudiya_x - l_from.gudiya_x) * p,
            gudiya_y=l_from.gudiya_y + (l_to.gudiya_y - l_from.gudiya_y) * p,
            gudiya_scale=l_from.gudiya_scale + (l_to.gudiya_scale - l_from.gudiya_scale) * p,
            chintu_x=l_from.chintu_x + (l_to.chintu_x - l_from.chintu_x) * p,
            chintu_y=l_from.chintu_y + (l_to.chintu_y - l_from.chintu_y) * p,
            chintu_scale=l_from.chintu_scale + (l_to.chintu_scale - l_from.chintu_scale) * p,
            alpha_lab=l_from.alpha_lab + (l_to.alpha_lab - l_from.alpha_lab) * p
        )

    def scale_to(self, res_scale: float) -> "StageGeometry":
        """Return a scaled copy for half-resolution draft preview."""
        rs = res_scale
        return StageGeometry(
            stage_type=self.stage_type,
            width=int(self.width * rs) // 2 * 2,
            height=int(self.height * rs) // 2 * 2,
            res_scale=rs,
            idea_zone_top=self.idea_zone_top * rs,
            idea_zone_bottom=self.idea_zone_bottom * rs,
            canvas_zone_top=self.canvas_zone_top * rs,
            canvas_zone_bottom=self.canvas_zone_bottom * rs,
            caption_y_top=self.caption_y_top * rs,
            caption_y_bottom=self.caption_y_bottom * rs,
            gudiya=CharacterStagePlacement(
                x=self.gudiya.x * rs,
                y=self.gudiya.y * rs,
                scale=self.gudiya.scale * rs,
                opacity=self.gudiya.opacity,
                brightness=self.gudiya.brightness,
                is_bust=self.gudiya.is_bust
            ),
            chintu=CharacterStagePlacement(
                x=self.chintu.x * rs,
                y=self.chintu.y * rs,
                scale=self.chintu.scale * rs,
                opacity=self.chintu.opacity,
                brightness=self.chintu.brightness,
                is_bust=self.chintu.is_bust
            )
        )


def get_stage_definition(stage: StageType, camera_submode: str = "wide") -> StageGeometry:
    """
    Get the canonical geometry for a stage and camera sub-mode.
    Sub-modes:
      • 'wide': standard balanced two-shot
      • 'close_gudiya': Gudiya 1.35x, Chintu pushed to right edge
      • 'close_chintu': Chintu 1.35x, Gudiya pushed to left edge
      • 'pan_gudiya': 40px camera pan toward Gudiya
      • 'pan_chintu': 40px camera pan toward Chintu
    """
    if stage == StageType.DUO:
        if camera_submode == "close_gudiya":
            return StageGeometry(
                stage_type=StageType.DUO,
                idea_zone_top=240.0,
                idea_zone_bottom=640.0,
                caption_y_top=990.0,
                caption_y_bottom=1160.0,
                gudiya=CharacterStagePlacement(x=380.0, y=1520.0, scale=1.18, opacity=1.0, brightness=1.0),
                chintu=CharacterStagePlacement(x=960.0, y=1520.0, scale=0.72, opacity=0.65, brightness=0.75)
            )
        elif camera_submode == "close_chintu":
            return StageGeometry(
                stage_type=StageType.DUO,
                idea_zone_top=240.0,
                idea_zone_bottom=640.0,
                caption_y_top=990.0,
                caption_y_bottom=1160.0,
                gudiya=CharacterStagePlacement(x=120.0, y=1520.0, scale=0.72, opacity=0.65, brightness=0.75),
                chintu=CharacterStagePlacement(x=700.0, y=1520.0, scale=1.18, opacity=1.0, brightness=1.0)
            )
        elif camera_submode == "pan_gudiya":
            return StageGeometry(
                stage_type=StageType.DUO,
                idea_zone_top=240.0,
                idea_zone_bottom=640.0,
                caption_y_top=990.0,
                caption_y_bottom=1160.0,
                gudiya=CharacterStagePlacement(x=330.0, y=1480.0, scale=0.92, opacity=1.0, brightness=1.0),
                chintu=CharacterStagePlacement(x=830.0, y=1480.0, scale=0.84, opacity=0.85, brightness=0.82)
            )
        elif camera_submode == "pan_chintu":
            return StageGeometry(
                stage_type=StageType.DUO,
                idea_zone_top=240.0,
                idea_zone_bottom=640.0,
                caption_y_top=990.0,
                caption_y_bottom=1160.0,
                gudiya=CharacterStagePlacement(x=250.0, y=1480.0, scale=0.84, opacity=0.85, brightness=0.82),
                chintu=CharacterStagePlacement(x=750.0, y=1480.0, scale=0.92, opacity=1.0, brightness=1.0)
            )
        else:  # 'wide'
            return StageGeometry(
                stage_type=StageType.DUO,
                idea_zone_top=240.0,
                idea_zone_bottom=640.0,
                caption_y_top=990.0,
                caption_y_bottom=1160.0,
                gudiya=CharacterStagePlacement(x=300.0, y=1480.0, scale=0.98, opacity=1.0, brightness=1.0),
                chintu=CharacterStagePlacement(x=780.0, y=1480.0, scale=0.88, opacity=1.0, brightness=1.0)
            )

    elif stage == StageType.LAB:
        # Split screen: top is interactive canvas (y 0 to 820), bottom corners are character presenter busts (y 1140 to 1860)
        # Anchoring at y=1860 with scale ~0.66 leaves clean band below captions (y ~ 1018-1060).
        g_scale = 0.68
        c_scale = 0.62
        g_bright = 1.0
        c_bright = 1.0

        if camera_submode == "close_gudiya":
            g_scale = 0.74
            c_scale = 0.56
            c_bright = 0.75
        elif camera_submode == "close_chintu":
            c_scale = 0.68
            g_scale = 0.60
            g_bright = 0.75

        return StageGeometry(
            stage_type=StageType.LAB,
            canvas_zone_top=180.0,
            canvas_zone_bottom=840.0,
            caption_y_top=960.0,
            caption_y_bottom=1120.0,
            gudiya=CharacterStagePlacement(x=190.0, y=1720.0, scale=g_scale, opacity=1.0, brightness=g_bright, is_bust=True),
            chintu=CharacterStagePlacement(x=890.0, y=1720.0, scale=c_scale, opacity=1.0, brightness=c_bright, is_bust=True)
        )

    elif stage == StageType.PROOF:
        # Fullscreen hero plate: characters present as avatar chips next to caption, bodies invisible
        return StageGeometry(
            stage_type=StageType.PROOF,
            canvas_zone_top=180.0,
            canvas_zone_bottom=1920.0,
            caption_y_top=1160.0,
            caption_y_bottom=1330.0,
            gudiya=CharacterStagePlacement(x=290.0, y=2100.0, scale=0.4, opacity=0.0, brightness=0.0),
            chintu=CharacterStagePlacement(x=790.0, y=2100.0, scale=0.4, opacity=0.0, brightness=0.0)
        )

    raise ValueError(f"Unknown StageType: {stage}")


def interpolate_stage_geometry(geom_from: StageGeometry, geom_to: StageGeometry, progress: float) -> StageGeometry:
    """
    Interpolate smoothly between two stage geometries using cubic easing.
    Handles transitions between DUO, LAB, and PROOF.
    """
    p = ease_in_out_cubic(progress)

    def _interp_char(c1: CharacterStagePlacement, c2: CharacterStagePlacement) -> CharacterStagePlacement:
        return CharacterStagePlacement(
            x=c1.x + (c2.x - c1.x) * p,
            y=c1.y + (c2.y - c1.y) * p,
            scale=c1.scale + (c2.scale - c1.scale) * p,
            opacity=c1.opacity + (c2.opacity - c1.opacity) * p,
            brightness=c1.brightness + (c2.brightness - c1.brightness) * p,
            is_bust=c2.is_bust if p >= 0.5 else c1.is_bust
        )

    return StageGeometry(
        stage_type=geom_to.stage_type if p >= 0.5 else geom_from.stage_type,
        width=geom_to.width,
        height=geom_to.height,
        res_scale=geom_to.res_scale,
        idea_zone_top=geom_from.idea_zone_top + (geom_to.idea_zone_top - geom_from.idea_zone_top) * p,
        idea_zone_bottom=geom_from.idea_zone_bottom + (geom_to.idea_zone_bottom - geom_from.idea_zone_bottom) * p,
        canvas_zone_top=geom_from.canvas_zone_top + (geom_to.canvas_zone_top - geom_from.canvas_zone_top) * p,
        canvas_zone_bottom=geom_from.canvas_zone_bottom + (geom_to.canvas_zone_bottom - geom_from.canvas_zone_bottom) * p,
        caption_y_top=geom_from.caption_y_top + (geom_to.caption_y_top - geom_from.caption_y_top) * p,
        caption_y_bottom=geom_from.caption_y_bottom + (geom_to.caption_y_bottom - geom_from.caption_y_bottom) * p,
        gudiya=_interp_char(geom_from.gudiya, geom_to.gudiya),
        chintu=_interp_char(geom_from.chintu, geom_to.chintu)
    )
