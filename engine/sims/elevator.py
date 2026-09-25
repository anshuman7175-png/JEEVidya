"""
JEEVidya — Elevator & Free-Fall Simulation Engine (Visual Bible v1 §6.2)
═════════════════════════════════════════════════════════════════════════
Renders broadcast-grade elevator mechanics for Visual Bible Lab Canvas:
  • Shaft: Glowing neon guide rails and deep-space dark frosted shaft.
  • Cabin: Modern chamfered glass/steel elevator cabin with suspension cables.
  • Floor: Platform with glowing digital LED weighing scale readout.
  • Occupant: Stylized vector figure standing or floating in zero-g.
  • Vectors:
      - Normal force N (Cyan #00E5FF) pushing upward from scale.
      - Gravity force mg (Coral #FF5252) pulling downward from centre of mass.
  • Digital Scale Readout:
      - 'static': 60.0 kg (N = mg)
      - 'cable_snap': Snapped cable with sparks, dropping to 0.00 kg (N = 0)
      - 'free_fall': 0.00 kg (Occupant floating, N = 0, Weightlessness)
      - 'accelerating_up': 75.0 kg (N > mg, Apparent Weight increases)
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from config import settings


def _load_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    """Load bundled Google Fonts with graceful OS system font fallbacks."""
    paths = [
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", font_name),
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", "Mukta-ExtraBold.ttf"),
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", "Baloo2-Variable.ttf"),
        f"/System/Library/Fonts/Supplemental/{font_name}",
        f"/Library/Fonts/{font_name}",
        f"/System/Library/Fonts/{font_name}",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_shadow_text(draw: ImageDraw.Draw, x: float, y: float, text: str,
                      font: ImageFont.FreeTypeFont, fill: Tuple[int, int, int],
                      anchor: str = "mm", rs: float = 1.0) -> None:
    """Draw text with a clean single-offset drop shadow (not 8-directional stamp)."""
    off = max(1, int(2 * rs))
    draw.text((x + off, y + off), text, font=font, fill=(0, 0, 0, 100), anchor=anchor)
    draw.text((x, y), text, font=font, fill=fill + (255,), anchor=anchor)


def _draw_hud_vector(draw: ImageDraw.Draw, x1: float, y1: float, x2: float, y2: float,
                     color: Tuple[int, int, int], label: str = "",
                     width: int = 4, rs: float = 1.0,
                     label_pos: Optional[Tuple[float, float]] = None) -> None:
    """Draw sleek vector arrow with tapered stroke, glowing arrowhead, and clean drop-shadowed label."""
    # Outer glow
    draw.line([(x1, y1), (x2, y2)], fill=color + (90,), width=width + int(4 * rs))
    # Core arrow
    draw.line([(x1, y1), (x2, y2)], fill=color + (255,), width=width)

    # Arrowhead
    angle = math.atan2(y2 - y1, x2 - x1)
    head_len = 15 * rs
    p1 = (x2 - head_len * math.cos(angle - 0.40), y2 - head_len * math.sin(angle - 0.40))
    p2 = (x2 - head_len * math.cos(angle + 0.40), y2 - head_len * math.sin(angle + 0.40))
    draw.polygon([(x2, y2), p1, p2], fill=color + (255,))

    if label:
        font_main = _load_font("Mukta-ExtraBold.ttf", max(11, int(15 * rs)))
        if label_pos is not None:
            lx, ly = label_pos
        else:
            pill_dist = 22 * rs
            lx = x2 + pill_dist * math.cos(angle)
            ly = y2 + pill_dist * math.sin(angle)
        _draw_shadow_text(draw, lx, ly, label, font_main, fill=color, anchor="mm", rs=rs)


class ElevatorSimulation:
    """Broadcast-grade 60fps elevator & weighing scale physics simulation."""

    def __init__(self, mode: str = "static", **kwargs: Any):
        self.mode = mode
        self.kwargs = kwargs

    def render(self, t_seconds: float, params: Dict[str, Any],
               width: int = 1080, height: int = 820, res_scale: float = 1.0) -> Image.Image:
        """
        Render a single simulation frame for the Lab canvas.
        Dimensions: width x height (already scaled to current rendering resolution).
        """
        rs = res_scale
        w_px = int(width)
        h_px = int(height)
        canvas = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)

        mode = params.get("mode", self.mode)
        snap_time = float(params.get("snap_time", 1.5))
        stage_start_sec = float(params.get("stage_start_ms", 0.0)) / 1000.0
        t_rel = max(0.0, t_seconds - stage_start_sec) if t_seconds >= stage_start_sec else t_seconds

        # Determine state based on mode and time
        is_free_fall = (mode == "cable_snap" and t_rel >= snap_time) or mode == "free_fall"
        is_accel_up = (mode == "accelerating_up")

        center_x = w_px / 2
        center_y = (h_px / 2) + 20 * rs

        # ── 1. Top Chapter Title ──
        font_tag = _load_font("Inter-Black.ttf", max(16, int(20 * rs)))
        d.text((center_x, 32 * rs), "Apparent Weight & Free Fall",
               font=font_tag, fill=(140, 215, 255, 255), anchor="mm",
               stroke_width=max(1, int(2 * rs)), stroke_fill=(0, 0, 0, 120))

        # ── 2. Elevator Shaft Architecture ──
        shaft_w = 490.0 * rs
        rail_left = center_x - shaft_w / 2
        rail_right = center_x + shaft_w / 2

        # Vertical guide columns with metallic rim
        d.line([(rail_left, 50 * rs), (rail_left, h_px)], fill=(40, 70, 110, 140), width=int(6 * rs))
        d.line([(rail_left + 4 * rs, 50 * rs), (rail_left + 4 * rs, h_px)], fill=(62, 224, 255, 80), width=int(2 * rs))

        d.line([(rail_right, 50 * rs), (rail_right, h_px)], fill=(40, 70, 110, 140), width=int(6 * rs))
        d.line([(rail_right - 4 * rs, 50 * rs), (rail_right - 4 * rs, h_px)], fill=(62, 224, 255, 80), width=int(2 * rs))

        # ── 3. Cabin Dynamics ──
        cabin_w = 420.0 * rs
        cabin_h = 510.0 * rs

        cabin_y = center_y
        if is_free_fall:
            dt = t_rel - snap_time if mode == "cable_snap" else t_rel
            cabin_y += (80.0 * (dt ** 1.55)) * rs
            cabin_y = min(cabin_y, h_px - cabin_h / 2 + 50 * rs)
        elif is_accel_up:
            cabin_y -= (22.0 * math.sin(t_rel * 2.5)) * rs

        cabin_top = cabin_y - cabin_h / 2
        cabin_bottom = cabin_y + cabin_h / 2
        cabin_left = center_x - cabin_w / 2
        cabin_right = center_x + cabin_w / 2

        # ── 4. Suspension Cable Rig ──
        cable_col = (190, 220, 255, 220)
        cable_top_y = 52 * rs

        # Pre-snap tension window: cables vibrate in the 0.5s before snap_time
        tension_window = 0.5  # seconds of pre-snap shake
        is_tensing = (mode == "cable_snap" and not is_free_fall and
                      t_rel >= (snap_time - tension_window) and t_rel < snap_time)

        if is_free_fall and mode == "cable_snap":
            # Snapped cable ends with dynamic sparks
            snap_y = cabin_top - 45 * rs
            d.line([(center_x - 14 * rs, cable_top_y), (center_x - 14 * rs, snap_y - 18 * rs)], fill=cable_col, width=int(4 * rs))
            d.line([(center_x + 14 * rs, cable_top_y), (center_x + 14 * rs, snap_y - 12 * rs)], fill=cable_col, width=int(4 * rs))

            d.line([(center_x - 9 * rs, snap_y + 12 * rs), (center_x - 9 * rs, cabin_top)], fill=cable_col, width=int(4 * rs))
            d.line([(center_x + 9 * rs, snap_y + 20 * rs), (center_x + 9 * rs, cabin_top)], fill=cable_col, width=int(4 * rs))

            # Energetic spark burst
            d.ellipse((center_x - 20 * rs, snap_y - 20 * rs, center_x + 20 * rs, snap_y + 20 * rs),
                      fill=(255, 220, 50, 220))
            d.ellipse((center_x - 9 * rs, snap_y - 9 * rs, center_x + 9 * rs, snap_y + 9 * rs),
                      fill=(255, 255, 255, 255))
            for sp_dx, sp_dy in [(-24, -12), (26, -15), (-20, 18), (22, 20), (-28, 5), (30, -7)]:
                d.line([(center_x, snap_y), (center_x + sp_dx * rs, snap_y + sp_dy * rs)],
                       fill=(255, 235, 100, 255), width=max(1, int(2 * rs)))
        elif not is_free_fall:
            if is_tensing:
                # Tension shake: cables vibrate laterally with increasing intensity
                progress = (t_rel - (snap_time - tension_window)) / tension_window  # 0→1
                amplitude = progress * 8.0 * rs  # max 8px lateral shake at snap moment
                freq = 18.0 + progress * 30.0    # frequency ramps from 18Hz to 48Hz
                shake_x = amplitude * math.sin(t_rel * freq * 2 * math.pi)

                # Stress tint: cables shift toward red as snap approaches
                stress_r = int(190 + progress * 65)   # 190→255
                stress_g = int(220 - progress * 120)   # 220→100
                stress_b = int(255 - progress * 155)   # 255→100
                stress_col = (min(255, stress_r), max(0, stress_g), max(0, stress_b), 240)

                d.line([(center_x - 14 * rs + shake_x, cable_top_y),
                        (center_x - 14 * rs + shake_x * 0.3, cabin_top)],
                       fill=stress_col, width=int(4 * rs))
                d.line([(center_x + 14 * rs - shake_x, cable_top_y),
                        (center_x + 14 * rs - shake_x * 0.3, cabin_top)],
                       fill=stress_col, width=int(4 * rs))
            else:
                # Normal dual heavy-duty suspension cables
                d.line([(center_x - 14 * rs, cable_top_y), (center_x - 14 * rs, cabin_top)], fill=cable_col, width=int(4 * rs))
                d.line([(center_x + 14 * rs, cable_top_y), (center_x + 14 * rs, cabin_top)], fill=cable_col, width=int(4 * rs))

        # ── 5. Panoramic Glass Elevator Cabin ──
        # Translucent dark interior
        d.rounded_rectangle(
            (cabin_left, cabin_top, cabin_right, cabin_bottom),
            radius=int(18 * rs),
            fill=(10, 20, 38, 225),
            outline=(50, 130, 210, 220),
            width=int(3 * rs)
        )

        # Specular glass reflection streak (diagonal sheen across glass)
        d.polygon([
            (cabin_left + 30 * rs, cabin_top),
            (cabin_left + 110 * rs, cabin_top),
            (cabin_left + 55 * rs, cabin_bottom),
            (cabin_left, cabin_bottom - 70 * rs)
        ], fill=(255, 255, 255, 15))

        # Soft ceiling downlight
        d.polygon([
            (center_x - 60 * rs, cabin_top + 16 * rs),
            (center_x + 60 * rs, cabin_top + 16 * rs),
            (center_x + 130 * rs, cabin_bottom - 20 * rs),
            (center_x - 130 * rs, cabin_bottom - 20 * rs)
        ], fill=(210, 240, 255, 18))

        # Ceiling light fixture
        d.rounded_rectangle(
            (center_x - 65 * rs, cabin_top + 4 * rs, center_x + 65 * rs, cabin_top + 16 * rs),
            radius=int(4 * rs),
            fill=(255, 255, 255, 240)
        )

        # ── 6. Digital Scale Platform on Cabin Floor ──
        scale_w = 230.0 * rs
        scale_h = 36.0 * rs
        scale_x1 = center_x - scale_w / 2
        scale_x2 = center_x + scale_w / 2
        scale_y1 = cabin_bottom - scale_h - 16 * rs
        scale_y2 = cabin_bottom - 16 * rs

        # Scale platform — simple dark plate, no neon outline
        d.rounded_rectangle((scale_x1, scale_y1, scale_x2, scale_y2), radius=int(6 * rs),
                            fill=(20, 30, 50, 255))

        # Digital scale readout
        if is_free_fall:
            scale_val_str = "0.00 kg"
            scale_col = (255, 69, 58)
        elif is_accel_up:
            scale_val_str = "75.0 kg"
            scale_col = (52, 199, 89)
        else:
            scale_val_str = "60.0 kg"
            scale_col = (0, 229, 255)

        # Scale readout — just the number, no inner glass box
        font_digits = _load_font("Inter-Black.ttf", max(12, int(19 * rs)))
        d.text((center_x, (scale_y1 + scale_y2) / 2),
               scale_val_str, font=font_digits, fill=scale_col + (255,), anchor="mm",
               stroke_width=1, stroke_fill=(0, 0, 0, 120))

        # Scale label — simple, no dot-separator
        font_scale_lbl = _load_font("Inter-Bold.ttf", max(10, int(13 * rs)))
        d.text((center_x, scale_y2 + 9 * rs), "Apparent Weight",
               font=font_scale_lbl, fill=(120, 180, 235, 180), anchor="mm",
               stroke_width=1, stroke_fill=(0, 0, 0, 80))

        # ── 7. Occupant: Actual Chintu character PNG (not PIL stick figure) ──
        char_path = os.path.join(settings.PROJECT_ROOT, "assets", "characters", "chintu", "body.png")
        if not hasattr(self, "_char_img"):
            try:
                self._char_img = Image.open(char_path).convert("RGBA")
            except Exception:
                self._char_img = None

        if self._char_img is not None:
            # Scale character to fit inside the cabin (~55% of cabin height)
            char_target_h = int(cabin_h * 0.55)
            char_aspect = self._char_img.width / self._char_img.height
            char_target_w = int(char_target_h * char_aspect)
            char_scaled = self._char_img.resize(
                (max(2, char_target_w), max(2, char_target_h)),
                Image.Resampling.LANCZOS)

            # Position: standing on scale, floating up in free fall
            occupant_float = 0.0
            if is_free_fall:
                dt = t_rel - snap_time if mode == "cable_snap" else t_rel
                occupant_float = min(50.0 * rs, dt * 40.0 * rs)

            char_x = int(center_x - char_target_w / 2)
            char_feet_y = int(scale_y1 - occupant_float)
            char_y = int(char_feet_y - char_target_h)

            if canvas.mode != "RGBA":
                canvas = canvas.convert("RGBA")
            canvas.alpha_composite(char_scaled, (char_x, char_y))
            d = ImageDraw.Draw(canvas)

            # Reference points for force vectors
            torso_y = char_y + int(char_target_h * 0.45)
            foot_y = char_feet_y
        else:
            # Fallback: simple dark silhouette if PNG missing
            torso_y = scale_y1 - int(100 * rs)
            foot_y = scale_y1

        # ── 8. Force Vectors (simplified: 2 arrows, 2 labels each) ──
        fbd_left_x = center_x - 155 * rs
        fbd_right_x = center_x + 155 * rs

        font_fbd_title = _load_font("Inter-Black.ttf", max(14, int(16 * rs)))
        font_fbd_val = _load_font("Inter-Black.ttf", max(16, int(22 * rs)))

        # LEFT: Normal Force
        if not is_free_fall:
            n_len = 94.0 * rs if is_accel_up else 74.0 * rs
            _draw_hud_vector(d, fbd_left_x, foot_y, fbd_left_x, foot_y - n_len,
                             color=(0, 229, 255), label="", width=int(3 * rs), rs=rs)
            _draw_shadow_text(d, fbd_left_x, foot_y - n_len - 20 * rs, "N = 588 N",
                              font_fbd_val, fill=(0, 229, 255), anchor="mm", rs=rs)
            _draw_shadow_text(d, fbd_left_x, foot_y + 18 * rs, "NORMAL FORCE",
                              font_fbd_title, fill=(140, 215, 255), anchor="mm", rs=rs)
        else:
            _draw_shadow_text(d, fbd_left_x, torso_y - 10 * rs, "N = 0",
                              font_fbd_val, fill=(255, 69, 58), anchor="mm", rs=rs)
            _draw_shadow_text(d, fbd_left_x, torso_y + 16 * rs, "WEIGHTLESS",
                              font_fbd_title, fill=(255, 216, 77), anchor="mm", rs=rs)

        # RIGHT: Gravity
        mg_len = 74.0 * rs
        _draw_hud_vector(d, fbd_right_x, torso_y, fbd_right_x, torso_y + mg_len,
                         color=(255, 82, 82), label="", width=int(3 * rs), rs=rs)
        _draw_shadow_text(d, fbd_right_x, torso_y + mg_len + 20 * rs, "mg = 588 N",
                          font_fbd_val, fill=(255, 82, 82), anchor="mm", rs=rs)
        _draw_shadow_text(d, fbd_right_x, torso_y - 20 * rs, "GRAVITY",
                          font_fbd_title, fill=(255, 140, 140), anchor="mm", rs=rs)

        # ── 9. Cabin Status — text only, no box ──
        if is_free_fall:
            status_txt = "Free Fall"
            status_col = (255, 90, 90)
        elif is_accel_up:
            status_txt = "Accelerating Up"
            status_col = (92, 240, 138)
        else:
            status_txt = "At Rest"
            status_col = (140, 215, 255)

        lintel_y = cabin_top + 26 * rs
        font_status = _load_font("Inter-Black.ttf", max(14, int(18 * rs)))
        d.text((center_x, lintel_y), status_txt, font=font_status,
               fill=status_col + (255,), anchor="mm",
               stroke_width=max(1, int(2 * rs)), stroke_fill=(0, 0, 0, 160))

        return canvas
