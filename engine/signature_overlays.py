"""
JEEVidya — Signature Brand Overlays (Visual Bible v1 §5)
════════════════════════════════════════════════════════
Implements the signature visual inventions for Gudiya & Chintu:
  1. Belief Card: Frosted glass panel with coral-pink chamfer.
  2. Voronoi Glass Shatter: Crack lines -> 16 gravity-tumbling shards -> Red X stamp.
  3. Fix Card: Corrected aha claim with emerald-green chamfer.
  4. Live Digital Counter: 160px tabular numerals with ease-out-expo.
  5. Gudiya's Notebook: Lined-paper card with LaTeX formula and auto PNG export.
  6. Teaser Card: Next episode silhouette & pinned comment challenge.
"""
from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config import brand, settings
from engine.math_renderer import MathRenderer


def _get_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Load font with graceful fallback."""
    font_paths = [
        os.path.join(settings.FONTS_DIR, name),
        os.path.join(settings.FONTS_DIR, "Mukta-ExtraBold.ttf"),
        os.path.join(settings.FONTS_DIR, "Baloo2-Variable.ttf"),
        os.path.join(settings.FONTS_DIR, "Montserrat-Bold.ttf"),
        os.path.join(settings.FONTS_DIR, "Inter-Bold.ttf"),
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ══════════════════════════════════════════════════════════════
# 1. BELIEF CARD & FIX CARD (BROADCAST GLASSMORPHISM)
# ══════════════════════════════════════════════════════════════

def strip_emojis(text: str) -> str:
    """Strip Unicode emojis and special pictographs that render as missing glyph boxes in PIL."""
    if not text:
        return ""
    # Strip high-surrogate emojis
    s = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # Strip common symbol pictographs
    s = re.sub(r'[\u2600-\u27bf\u2300-\u23ff\u2b50\u2b55]', '', s)
    return s.strip()


class ConceptCard:
    """Clean floating typography for misconceptions (pink) or corrected facts (green) — zero boxes."""

    def __init__(self, text: str, card_type: str = "belief", kind: Optional[str] = None, width: int = 960, height: int = 190):
        self.raw_text = text or ""
        self.text = strip_emojis(self.raw_text)
        self.card_type = (kind or card_type or "belief").lower()
        self.width = width
        self.height = height

        if "fix" in self.card_type or "rule" in self.card_type or "truth" in self.card_type:
            self.glow_color = (0, 235, 125)          # Neon Emerald #00EB7D
            self.badge_title = "THE REAL TRUTH"
        elif "question" in self.card_type:
            self.glow_color = (255, 215, 0)          # Neon Amber #FFD700
            self.badge_title = "BIG QUESTION"
        else:  # "belief" / misconception
            self.glow_color = (255, 90, 120)         # Neon Coral-Pink #FF5A78
            self.badge_title = "COMMON MYTH"

    def render(self, progress: float = 1.0, show_stamp: bool = True) -> Image.Image:
        """Render text overlay directly — NO card box, NO container rectangle.
        Clean floating typography with dark stroke for contrast against any background.
        """
        margin = 20
        w, h = self.width, self.height
        canvas = Image.new("RGBA", (w + margin * 2, h + margin * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)

        h_cx = margin + w // 2

        # 1. Floating Category Label
        cat_font = _get_font("Inter-Black.ttf", 18)
        d.text((h_cx, margin + 26), f"●  {self.badge_title}",
               font=cat_font, fill=self.glow_color + (int(255 * progress),), anchor="mm",
               stroke_width=2, stroke_fill=(0, 0, 0, int(180 * progress)))

        # 2. Main Claim Text — auto-wrapped to 2 balanced lines so ZERO letters cut off
        display_text = self.text.strip()
        words = display_text.split()
        max_text_w = w - 60
        lines: List[str] = []
        chosen_font = None

        font_44 = _get_font("Inter-Black.ttf", 44)
        single_bbox = d.textbbox((0, 0), display_text, font=font_44)
        single_w = single_bbox[2] - single_bbox[0]

        if single_w <= 700 or len(words) <= 3:
            lines = [display_text]
            chosen_font = font_44
        else:
            mid = (len(words) + 1) // 2
            line1 = " ".join(words[:mid])
            line2 = " ".join(words[mid:])
            f_size = 42
            f = _get_font("Inter-Black.ttf", f_size)
            while f_size > 26:
                bb1 = d.textbbox((0, 0), line1, font=f)
                bb2 = d.textbbox((0, 0), line2, font=f)
                if max(bb1[2] - bb1[0], bb2[2] - bb2[0]) <= max_text_w:
                    break
                f_size -= 2
                f = _get_font("Inter-Black.ttf", f_size)
            lines = [line1, line2]
            chosen_font = f

        text_alpha = int(255 * progress)
        stroke_alpha = int(230 * progress)

        if len(lines) == 1:
            d.text((h_cx, margin + 95), lines[0], font=chosen_font,
                   fill=(255, 255, 255, text_alpha), anchor="mm",
                   stroke_width=3, stroke_fill=(0, 0, 0, stroke_alpha))
        else:
            start_y = margin + 80
            spacing = 50
            for i, line in enumerate(lines[:2]):
                d.text((h_cx, start_y + i * spacing), line, font=chosen_font,
                       fill=(255, 255, 255, text_alpha), anchor="mm",
                       stroke_width=3, stroke_fill=(0, 0, 0, stroke_alpha))

        return canvas

    def render_full_frame(self, t_seconds: float, duration_seconds: float = 3.0,
                          canvas_w: int = 1080, canvas_h: int = 1920,
                          res_scale: float = 1.0) -> Image.Image:
        """Render card composited onto full-screen canvas with guaranteed head clearance (bottom <= 460)."""
        rs = res_scale
        full = Image.new("RGBA", (int(canvas_w * rs), int(canvas_h * rs)), (0, 0, 0, 0))
        if t_seconds < 0 or t_seconds > duration_seconds:
            return full

        # Pop-in spring progression (0.0 to 0.35s)
        in_dur = 0.35
        out_dur = 0.30
        if t_seconds < in_dur:
            p = t_seconds / in_dur
            progress = 1.0 - math.pow(1.0 - p, 3)
        elif t_seconds > duration_seconds - out_dur:
            p = (duration_seconds - t_seconds) / out_dur
            progress = max(0.0, p)
        else:
            progress = 1.0

        card_img = self.render(progress=progress, show_stamp=True)
        if rs != 1.0:
            cw = int(card_img.width * rs)
            ch = int(card_img.height * rs)
            card_img = card_img.resize((cw, ch), Image.Resampling.BILINEAR)

        # Center horizontally, place top at y = 175 -> bottom reaches <= 410 (assert bbox[3] <= 460 passes easily!)
        cx = int((canvas_w * rs - card_img.width) / 2)
        cy = int(175 * rs)
        full.paste(card_img, (cx, cy), card_img)
        return full


# ══════════════════════════════════════════════════════════════
# 2. VORONOI GLASS SHATTER ENGINE
# ══════════════════════════════════════════════════════════════

@dataclass
class GlassShard:
    polygon: List[Tuple[float, float]]
    centroid: Tuple[float, float]
    vx: float
    vy: float
    rot_speed: float
    color: Tuple[int, int, int]


class VoronoiShatter:
    """Precomputed deterministic glass shatter animation for 'Wrong!' stings."""

    def __init__(self, text: str = "WRONG!", width: int = 880, height: int = 200, shard_count: int = 18, seed: int = 42):
        self.text = text
        self.width = width
        self.height = height
        self.shard_count = shard_count
        self.shards: List[GlassShard] = []
        self._init_shards(seed)

    def _init_shards(self, seed: int) -> None:
        rng = random.Random(seed)
        cols = 5
        rows = 3
        cell_w = self.width / cols
        cell_h = self.height / rows

        grid_points = []
        for r in range(rows):
            for c in range(cols):
                px = (c + rng.uniform(0.15, 0.85)) * cell_w
                py = (r + rng.uniform(0.15, 0.85)) * cell_h
                grid_points.append((px, py))

        # Build triangular / polygonal shards around grid cells
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                cx, cy = grid_points[idx]
                half_w = cell_w * rng.uniform(0.45, 0.70)
                half_h = cell_h * rng.uniform(0.45, 0.70)

                poly = [
                    (cx - half_w, cy - half_h * rng.uniform(0.7, 1.2)),
                    (cx + half_w * rng.uniform(0.6, 1.3), cy - half_h),
                    (cx + half_w, cy + half_h * rng.uniform(0.8, 1.2)),
                    (cx - half_w * rng.uniform(0.5, 1.1), cy + half_h)
                ]

                # Physics velocities: explode outward radially from center
                dx = cx - (self.width / 2)
                dy = cy - (self.height / 2)
                dist = max(1.0, math.sqrt(dx * dx + dy * dy))

                speed = rng.uniform(350.0, 950.0)
                self.shards.append(GlassShard(
                    polygon=poly,
                    centroid=(cx, cy),
                    vx=(dx / dist) * speed + rng.uniform(-100, 100),
                    vy=(dy / dist) * speed - rng.uniform(150, 450),  # upward pop before falling
                    rot_speed=rng.uniform(-380.0, 380.0),
                    color=(
                        rng.randint(220, 255),
                        rng.randint(235, 255),
                        rng.randint(245, 255)
                    )
                ))

    def render(self, elapsed_s: float, card_image: Optional[Image.Image] = None) -> Tuple[Image.Image, bool]:
        """
        Render shatter frame at elapsed seconds since 'Wrong!'.
        Returns (RGBA overlay image, is_finished).
        """
        duration = 0.60  # 600ms total
        if elapsed_s >= duration:
            return Image.new("RGBA", (self.width + 100, self.height + 400), (0, 0, 0, 0)), True

        canvas = Image.new("RGBA", (self.width + 100, self.height + 400), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)

        gravity = 2400.0  # px/s^2
        alpha_progress = max(0.0, 1.0 - (elapsed_s / duration))

        # Phase 1: First 3 frames (0.05s) show glowing spiderweb crack lines
        if elapsed_s < 0.06:
            for s in self.shards:
                poly = [(p[0] + 50, p[1] + 50) for p in s.polygon]
                d.polygon(poly, outline=(255, 255, 255, 240), width=2)
            return canvas, False

        # Phase 2: Shards fly outward & rotate
        t = elapsed_s - 0.05
        for s in self.shards:
            dx = s.vx * t
            dy = s.vy * t + 0.5 * gravity * t * t
            rot_deg = s.rot_speed * t
            rot_rad = math.radians(rot_deg)

            cx, cy = s.centroid
            poly_trans = []
            for px, py in s.polygon:
                # Rotate around centroid
                rel_x = px - cx
                rel_y = py - cy
                rx = rel_x * math.cos(rot_rad) - rel_y * math.sin(rot_rad)
                ry = rel_x * math.sin(rot_rad) + rel_y * math.cos(rot_rad)
                poly_trans.append((cx + rx + dx + 50, cy + ry + dy + 50))

            shard_alpha = int(220 * alpha_progress)
            d.polygon(poly_trans, fill=s.color + (int(shard_alpha * 0.45),), outline=(255, 255, 255, shard_alpha), width=2)

        # Phase 3: Big Red X Stamp (appears at 0.08s with ease-out-back bounce)
        if elapsed_s > 0.08:
            stamp_t = min(1.0, (elapsed_s - 0.08) / 0.20)
            scale = 1.4 - 0.4 * stamp_t
            red_alpha = int(255 * min(1.0, (elapsed_s - 0.08) / 0.10))

            x_font = _get_font("Montserrat-Bold.ttf", int(140 * scale))
            center_x = (self.width + 100) // 2
            center_y = (self.height + 100) // 2
            d.text((center_x, center_y), "✗", font=x_font, fill=(255, 59, 59, red_alpha), anchor="mm")

        return canvas, False

    def render_full_frame(self, t_seconds: float, duration_seconds: float = 1.8,
                          canvas_w: int = 1080, canvas_h: int = 1920,
                          res_scale: float = 1.0) -> Image.Image:
        """Render full-frame Voronoi shatter animation."""
        rs = res_scale
        full = Image.new("RGBA", (int(canvas_w * rs), int(canvas_h * rs)), (0, 0, 0, 0))
        if t_seconds < 0 or t_seconds > duration_seconds:
            return full

        overlay, is_finished = self.render(t_seconds)
        if is_finished:
            return full
        if rs != 1.0:
            ow, oh = int(overlay.width * rs), int(overlay.height * rs)
            overlay = overlay.resize((ow, oh), Image.Resampling.BILINEAR)

        cx = int((canvas_w * rs - overlay.width) / 2)
        cy = int(420 * rs - overlay.height / 2)
        full.paste(overlay, (cx, cy), overlay)
        return full


# ══════════════════════════════════════════════════════════════
# 3. LIVE DIGITAL COUNTER
# ══════════════════════════════════════════════════════════════

class LiveCounter:
    """Tabular numeral counter with ease-out-expo tick acceleration."""

    def __init__(self, target_value: str, unit: str = "", source: str = "", from_val: float = 0.0,
                 subtitle: Optional[str] = None):
        self.target_str = target_value.strip()
        self.unit = unit.strip()
        self.source = source.strip()
        self.from_val = from_val
        self.subtitle = subtitle.strip() if subtitle else None

        # Extract numeric component if possible
        clean_num = target_value.replace(",", "").strip()
        try:
            self.target_num = float(clean_num)
            self.is_numeric = True
        except ValueError:
            self.target_num = 0.0
            self.is_numeric = False

    def render(self, progress: float = 1.0) -> Image.Image:
        """Render the counter card with ease-out-expo number growth."""
        p = max(0.0, min(1.0, progress))
        # Ease out expo
        t = 1.0 if p == 1.0 else 1.0 - math.pow(2.0, -10.0 * p)

        if self.is_numeric:
            current_num = self.from_val + (self.target_num - self.from_val) * t
            if "." in self.target_str:
                decimals = len(self.target_str.split(".")[1])
                disp_str = f"{current_num:.{decimals}f}"
            else:
                # Format with Indian grouping if target had commas or is >= 100,000
                int_val = int(round(current_num))
                if "," in self.target_str or int_val >= 100000:
                    disp_str = f"{int_val:,}"
                else:
                    disp_str = str(int_val)
        else:
            disp_str = self.target_str

        # No box — text floats directly on the scene
        w, h = 680, 180
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)

        center_x = w // 2

        # Source header — small text with stroke for contrast, no box
        src_text = self.source if self.source else "Gravity at ISS altitude"
        src_font = _get_font("Inter-Black.ttf", 16)
        d.text((center_x, 28), src_text.upper(), font=src_font,
               fill=(140, 215, 255, 220), anchor="mm",
               stroke_width=2, stroke_fill=(0, 0, 0, 120))

        # Big number — the hero element
        num_font = _get_font("Inter-Black.ttf", 82)
        unit_font = _get_font("Inter-Bold.ttf", 36)

        num_bbox = num_font.getbbox(disp_str)
        num_w = num_bbox[2] - num_bbox[0]
        unit_w = 0
        if self.unit:
            u_bbox = unit_font.getbbox(self.unit)
            unit_w = u_bbox[2] - u_bbox[0] + 16

        total_lockup_w = num_w + unit_w
        start_x = center_x - total_lockup_w // 2

        num_x = start_x + num_w // 2
        d.text((num_x, 98), disp_str, font=num_font,
               fill=(255, 216, 77, 255), anchor="mm",
               stroke_width=3, stroke_fill=(0, 0, 0, 180))

        if self.unit:
            unit_x = start_x + num_w + 12 + (unit_w - 16) // 2
            d.text((unit_x, 102), self.unit, font=unit_font,
                   fill=(240, 248, 255, 255), anchor="mm",
                   stroke_width=2, stroke_fill=(0, 0, 0, 140))

        # Subtitle — optional context line
        if self.subtitle:
            sub_font = _get_font("Inter-Bold.ttf", 16)
            d.text((center_x, 150), self.subtitle,
                   font=sub_font, fill=(160, 230, 200, 200), anchor="mm",
                   stroke_width=2, stroke_fill=(0, 0, 0, 100))

        return canvas

    def render_full_frame(self, t_seconds: float, duration_seconds: float = 2.8,
                          canvas_w: int = 1080, canvas_h: int = 1920,
                          res_scale: float = 1.0) -> Image.Image:
        """Render LiveCounter composited onto full-screen frame in the clean zone below the simulation."""
        rs = res_scale
        full = Image.new("RGBA", (int(canvas_w * rs), int(canvas_h * rs)), (0, 0, 0, 0))
        if t_seconds < 0 or t_seconds > duration_seconds:
            return full

        # Smooth pop-in spring easing
        in_dur = 0.30
        out_dur = 0.25
        if t_seconds < in_dur:
            p = t_seconds / in_dur
            progress = 1.0 - math.pow(1.0 - p, 3)
        elif t_seconds > duration_seconds - out_dur:
            p = (duration_seconds - t_seconds) / out_dur
            progress = max(0.0, p)
        else:
            progress = 1.0

        card_img = self.render(progress=progress)
        if rs != 1.0:
            cw, ch = int(card_img.width * rs), int(card_img.height * rs)
            card_img = card_img.resize((cw, ch), Image.Resampling.BILINEAR)

        # Position at y = 960 (safe band between simulation at y <= 820 and character busts at y >= 1350)
        target_y = 960 if canvas_h >= 1600 else 500
        cx = int((canvas_w * rs - card_img.width) / 2)
        cy = int(target_y * rs - card_img.height / 2)
        full.paste(card_img, (cx, cy), card_img)
        return full


# ══════════════════════════════════════════════════════════════
# 4. GUDIYA'S NOTEBOOK (EXAM STAMP)
# ══════════════════════════════════════════════════════════════

class GudiyasNotebook:
    """Lined paper card with handwritten formula, NCERT chapter, and JEE/NEET badges."""

    def __init__(self, title: str, formula_latex: str, chapter: str, tags: str = "JEE · NEET"):
        self.title = strip_emojis(title or "")
        self.formula_latex = formula_latex or ""
        self.chapter = strip_emojis(chapter or "")
        self.tags = strip_emojis(tags or "JEE · NEET")

    def render(self, width: int = 920, height: int = 410) -> Image.Image:
        """Render the complete notebook card with zero text cut-off and clean headroom."""
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)

        # Lined-paper yellow/cream card with soft drop outline
        card_box = (15, 12, width - 15, height - 12)
        d.rounded_rectangle(card_box, radius=22, fill=(252, 248, 235, 252), outline=(210, 195, 160, 255), width=3)

        # Pink vertical margin line on left
        margin_x = 90
        d.line((margin_x, 20, margin_x, height - 20), fill=(245, 130, 150, 220), width=3)

        # Faint horizontal ruled lines
        for y in range(70, height - 30, 42):
            d.line((margin_x + 10, y, width - 35, y), fill=(190, 215, 240, 160), width=1)

        # Title: Gudiya's Handwritten Takeaway (dynamically wrapped so NO words get cut out)
        max_title_w = width - margin_x - 55
        raw_title = self.title.strip()
        full_title = raw_title if raw_title.upper().startswith("NOTE:") else f"NOTE: {raw_title}"

        # Word wrap to ensure full visibility without clipping
        title_font = _get_font("Montserrat-Bold.ttf", 30)
        words = full_title.split()
        lines: List[str] = []
        cur_line: List[str] = []
        for word in words:
            test_line = " ".join(cur_line + [word])
            bbox = d.textbbox((0, 0), test_line, font=title_font)
            if bbox[2] - bbox[0] > max_title_w and cur_line:
                lines.append(" ".join(cur_line))
                cur_line = [word]
            else:
                cur_line.append(word)
        if cur_line:
            lines.append(" ".join(cur_line))

        # Render 1 or 2 lines cleanly
        if len(lines) == 1:
            d.text((margin_x + 22, 54), lines[0], font=title_font, fill=(20, 25, 45, 255), anchor="lm")
        else:
            t_font_2 = _get_font("Montserrat-Bold.ttf", 25)
            d.text((margin_x + 22, 42), lines[0], font=t_font_2, fill=(20, 25, 45, 255), anchor="lm")
            d.text((margin_x + 22, 72), lines[1], font=t_font_2, fill=(20, 25, 45, 255), anchor="lm")

        # LaTeX Formula in center
        if self.formula_latex:
            formula_img = MathRenderer.render(self.formula_latex, color=(15, 25, 55), font_size=38)
            if formula_img:
                max_fw = width - margin_x - 65
                max_fh = height - 170
                scale = 1.0
                if formula_img.width > max_fw:
                    scale = min(scale, max_fw / formula_img.width)
                if formula_img.height > max_fh:
                    scale = min(scale, max_fh / formula_img.height)
                if scale < 1.0:
                    nw = max(1, int(formula_img.width * scale))
                    nh = max(1, int(formula_img.height * scale))
                    formula_img = formula_img.resize((nw, nh), Image.Resampling.LANCZOS)

                # Vertically center between title area and bottom chips
                content_top = 88 if len(lines) > 1 else 76
                content_bottom = height - 56
                fy = content_top + max(0, (content_bottom - content_top - formula_img.height) // 2)
                fx = margin_x + 28
                canvas.alpha_composite(formula_img, dest=(fx, fy))

        # Bottom Tags: Handwritten pen ink on notebook paper (NO digital UI pill boxes!)
        chip_y = height - 42
        chip_font = _get_font("Montserrat-Bold.ttf", 20)

        # 1. Exam badge (anchored right, handwritten in crimson ink)
        tag_text = f"[{self.tags or 'JEE · NEET'}]"
        d.text((width - 40, chip_y), tag_text, font=chip_font, fill=(200, 45, 45, 255), anchor="rm")

        # 2. Chapter label (anchored left, handwritten in deep navy ink, dynamically truncated if needed)
        chap_raw = self.chapter or "NCERT Class 11 Physics"
        chap_short = re.sub(r'\(.*?\)', '', chap_raw).strip()
        chap_label = chap_short if chap_short.upper().startswith("CH") else f"CH: {chap_short}"

        tag_bbox = d.textbbox((0, 0), tag_text, font=chip_font)
        max_chap_w = (width - 40 - (tag_bbox[2] - tag_bbox[0]) - 30) - (margin_x + 22)
        chap_bbox = d.textbbox((0, 0), f"• {chap_label}", font=chip_font)
        while (chap_bbox[2] - chap_bbox[0] > max_chap_w) and len(chap_label) > 8:
            chap_label = chap_label[:-4].strip() + "..."
            chap_bbox = d.textbbox((0, 0), f"• {chap_label}", font=chip_font)

        d.text((margin_x + 22, chip_y), f"• {chap_label}", font=chip_font, fill=(35, 65, 140, 255), anchor="lm")

        return canvas

    def export_to_image(self, output_path: str) -> str:
        """Export as high-resolution 1080x1350 image for YouTube Community tab."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img = self.render(width=1000, height=480)

        # Paste onto square/portrait social card
        card_full = Image.new("RGBA", (1080, 1350), (11, 16, 32, 255))
        d = ImageDraw.Draw(card_full)

        header_font = _get_font("Montserrat-Bold.ttf", 52)
        d.text((540, 180), "GUDIYA'S PHYSICS NOTEBOOK", font=header_font, fill=(255, 216, 77, 255), anchor="mm")

        sub_font = _get_font("Montserrat-Bold.ttf", 34)
        d.text((540, 245), "NCERT Class 11 Physics — Short Notes", font=sub_font, fill=(180, 205, 245, 220), anchor="mm")

        # Paste notebook in middle
        card_full.alpha_composite(img, dest=(40, 390))

        # Bottom prompt
        save_font = _get_font("Montserrat-Bold.ttf", 32)
        d.text((540, 1150), "Save this for your quick revision!", font=save_font, fill=(255, 255, 255, 240), anchor="mm")

        card_full.convert("RGB").save(output_path, quality=95)
        return output_path

    def render_full_frame(self, t_seconds: float, duration_seconds: float = 3.5,
                          canvas_w: int = 1080, canvas_h: int = 1920,
                          res_scale: float = 1.0, auto_export_dir: Optional[str] = None) -> Image.Image:
        """Render GudiyasNotebook onto full-screen canvas with guaranteed head clearance (bottom <= 600)."""
        rs = res_scale
        full = Image.new("RGBA", (int(canvas_w * rs), int(canvas_h * rs)), (0, 0, 0, 0))
        if t_seconds < 0 or t_seconds > duration_seconds:
            return full

        if auto_export_dir and not getattr(self, "_exported", False):
            safe_title = "".join(c for c in self.title if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
            out_file = os.path.join(auto_export_dir, f"notebook_{safe_title}.png")
            try:
                self.export_to_image(out_file)
                self._exported = True
            except Exception as e:
                print(f"  [Notebook] Community export warning: {e}")

        in_dur = 0.4
        progress = min(1.0, t_seconds / in_dur) if t_seconds < in_dur else 1.0
        if t_seconds > duration_seconds - 0.3:
            progress = max(0.0, (duration_seconds - t_seconds) / 0.3)

        card_img = self.render(width=920, height=410)
        if rs != 1.0:
            cw, ch = int(card_img.width * rs), int(card_img.height * rs)
            card_img = card_img.resize((cw, ch), Image.Resampling.BILINEAR)

        slide_offset = int((1.0 - progress) * 25 * rs)
        cx = int((canvas_w * rs - card_img.width) / 2)
        # Position card centered at y = 390 -> top = 390 - 205 = 185, bottom = 185 + 410 = 595.
        # Character heads start at y >= 740, providing 145px clean headroom clearance!
        cy = int(390 * rs - card_img.height / 2) + slide_offset
        full.paste(card_img, (cx, cy), card_img)
        return full


# ══════════════════════════════════════════════════════════════
# 5. TEASER CARD (SEAMLESS LOOP)
# ══════════════════════════════════════════════════════════════

class TeaserCard:
    """End-card previewing next episode with auto-wrapping, zero container boxes, and clean head clearance."""

    def __init__(self, next_episode_num: Optional[int] = None, next_question: str = "", pinned_comment: str = "", badge_label: Optional[str] = None):
        self.next_episode_num = next_episode_num
        self.next_question = strip_emojis(next_question or "")
        self.pinned_comment = strip_emojis(pinned_comment or "")
        self.badge_label = badge_label

    def render(self, width: int = 860, height: int = 220) -> Image.Image:
        """Render pure floating typography with dark stroke/shadow — NO container box, NO pill buttons."""
        margin = 15
        w, h = width, height
        canvas = Image.new("RGBA", (w + margin * 2, h + margin * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)

        h_cx = margin + w // 2

        # 1. Header: Floating neon cyan label (NO pill box)
        if self.badge_label:
            header_text = self.badge_label.replace("[»]", "").strip()
            if not header_text.startswith("»"):
                header_text = f"»  {header_text}"
        elif self.next_episode_num:
            header_text = f"»  UP NEXT · EPISODE #{self.next_episode_num}"
        else:
            header_text = "»  UP NEXT"

        header_font = _get_font("Inter-Black.ttf", 18)
        d.text((h_cx, margin + 20), header_text,
               font=header_font, fill=(0, 225, 255, 255), anchor="mm",
               stroke_width=2, stroke_fill=(0, 0, 0, 220))

        # 2. Main Question / Topic: Split on colon if present, or wrap cleanly
        q_text = self.next_question.strip()
        lines: List[Tuple[str, ImageFont.FreeTypeFont, Tuple[int, int, int]]] = []

        if ":" in q_text:
            parts = q_text.split(":", 1)
            topic_part = parts[0].strip()
            detail_part = parts[1].strip()
            f_top = _get_font("Inter-Black.ttf", 32)
            lines.append((topic_part, f_top, (255, 220, 60)))  # Gold topic title
            # Wrap detail part cleanly
            words = detail_part.split()
            dw_lines = []
            cur_w: List[str] = []
            for w_item in words:
                if len(" ".join(cur_w + [w_item])) > 28 and cur_w:
                    dw_lines.append(" ".join(cur_w))
                    cur_w = [w_item]
                else:
                    cur_w.append(w_item)
            if cur_w:
                dw_lines.append(" ".join(cur_w))
            f_det = _get_font("Inter-Black.ttf", 26)
            for dl in dw_lines[:2]:
                lines.append((dl, f_det, (255, 255, 255)))
        else:
            words = q_text.split()
            wrap_lines = []
            cur_w = []
            for w_item in words:
                if len(" ".join(cur_w + [w_item])) > 28 and cur_w:
                    wrap_lines.append(" ".join(cur_w))
                    cur_w = [w_item]
                else:
                    cur_w.append(w_item)
            if cur_w:
                wrap_lines.append(" ".join(cur_w))
            f_q = _get_font("Inter-Black.ttf", 28 if len(wrap_lines) <= 2 else 24)
            for wl in wrap_lines[:2]:
                lines.append((wl, f_q, (255, 255, 255)))

        # Render question lines centered vertically
        start_qy = margin + 64
        q_spacing = 38
        for idx, (ltxt, lfont, lcolor) in enumerate(lines):
            ly = start_qy + idx * q_spacing
            d.text((h_cx, ly), ltxt, font=lfont, fill=lcolor + (255,), anchor="mm",
                   stroke_width=3, stroke_fill=(0, 0, 0, 230))

        # 3. Bottom hint: Floating warm amber comment challenge (NO pill box)
        last_y = start_qy + max(0, len(lines) - 1) * q_spacing
        comment_text = self.pinned_comment or "Answer in the comments! 👇"
        c_font = _get_font("Inter-Bold.ttf", 16)
        c_y = last_y + 44
        d.text((h_cx, c_y), comment_text, font=c_font,
               fill=(255, 214, 80, 255), anchor="mm",
               stroke_width=2, stroke_fill=(0, 0, 0, 200))

        return canvas

    def render_full_frame(self, t_seconds: float, duration_seconds: float = 3.0,
                          canvas_w: int = 1080, canvas_h: int = 1920,
                          res_scale: float = 1.0) -> Image.Image:
        """Render TeaserCard onto full-screen canvas with zero boxes and clean safe zone clearance (bottom <= 460)."""
        rs = res_scale
        full = Image.new("RGBA", (int(canvas_w * rs), int(canvas_h * rs)), (0, 0, 0, 0))
        if t_seconds < 0 or t_seconds > duration_seconds:
            return full

        # Pop-in spring progression
        in_dur = 0.35
        out_dur = 0.30
        if t_seconds < in_dur:
            p = t_seconds / in_dur
            progress = 1.0 - math.pow(1.0 - p, 3)
        elif t_seconds > duration_seconds - out_dur:
            p = (duration_seconds - t_seconds) / out_dur
            progress = max(0.0, p)
        else:
            progress = 1.0

        card_img = self.render(width=860, height=220)
        if rs != 1.0:
            cw = int(card_img.width * rs)
            ch = int(card_img.height * rs)
            card_img = card_img.resize((cw, ch), Image.Resampling.BILINEAR)

        # Center horizontally, place top at y = 175 -> text content spans y ~ 195 to 370.
        # This guarantees bbox bottom <= 460 for tests, and gives >370px head clearance!
        cx = int((canvas_w * rs - card_img.width) / 2)
        cy = int(175 * rs)
        full.paste(card_img, (cx, cy), card_img)
        return full

