"""
JEEVidya — Canvas Renderer
Core frame renderer using Pillow. Handles drawing geometric primitives,
text, formulas, and compositing them onto a branded background.
Coordinate system: (0,0) at center, x: -540 to 540, y: -960 to 960
"""
import math
from typing import List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from config import brand, settings


class Canvas:
    """
    High-level drawing surface for a single video frame.
    All coordinates use a centered system where (0,0) is the middle of the frame.
    """

    def __init__(self, width: int = None, height: int = None):
        self.width = width or settings.WIDTH
        self.height = height or settings.HEIGHT
        self.cx = self.width // 2   # Center x in pixel coords
        self.cy = self.height // 2  # Center y in pixel coords

        # Main canvas (RGBA for transparency compositing)
        self.image = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

        # Font cache
        self._font_cache: dict = {}

    def to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert centered coords to pixel coords. Y is inverted (positive = up)."""
        px = int(self.cx + x)
        py = int(self.cy - y)  # Invert Y so positive goes up
        return (px, py)

    def from_pixel(self, px: int, py: int) -> Tuple[float, float]:
        """Convert pixel coords to centered coords."""
        x = px - self.cx
        y = self.cy - py
        return (x, y)

    # === Background ===

    def draw_gradient_background(self) -> None:
        """Draw the branded dark gradient background."""
        gradient_stops = brand.bg_gradient_colors()
        for y in range(self.height):
            t = y / self.height
            # Find the two stops we're between
            color = self._interpolate_gradient(t, gradient_stops)
            self.draw.line([(0, y), (self.width, y)], fill=color + (255,))

    def draw_grid(self, spacing: int = 80, opacity: float = None) -> None:
        """Draw a subtle background grid for structural reference."""
        if opacity is None:
            opacity = brand.OPACITY_STRUCTURE
        alpha = int(255 * opacity)
        color = brand.GRID_COLOR + (alpha,)

        # Vertical lines
        for x in range(0, self.width, spacing):
            self.draw.line([(x, 0), (x, self.height)], fill=color, width=1)
        # Horizontal lines
        for y in range(0, self.height, spacing):
            self.draw.line([(0, y), (self.width, y)], fill=color, width=1)

    # === Geometric Primitives ===

    def draw_circle(
        self, cx: float, cy: float, radius: float,
        color: Tuple[int, int, int] = None,
        opacity: float = 1.0,
        width: int = 3,
        fill_color: Optional[Tuple[int, int, int]] = None,
        fill_opacity: float = 0.15,
        progress: float = 1.0,
    ) -> None:
        """
        Draw a circle. If progress < 1.0, draws a partial circle (arc).
        
        Args:
            cx, cy: Center in centered coords
            radius: Radius in pixels
            color: Stroke color (default: PRIMARY)
            opacity: Stroke opacity
            width: Stroke width
            fill_color: Optional fill color
            fill_opacity: Fill opacity
            progress: 0.0 to 1.0, how much of the circle to draw
        """
        color = color or brand.PRIMARY
        alpha = int(255 * opacity)
        px, py = self.to_pixel(cx, cy)

        if progress >= 1.0:
            # Full circle
            bbox = [px - radius, py - radius, px + radius, py + radius]
            if fill_color:
                fill_alpha = int(255 * fill_opacity)
                self.draw.ellipse(bbox, fill=fill_color + (fill_alpha,),
                                  outline=color + (alpha,), width=width)
            else:
                self.draw.ellipse(bbox, outline=color + (alpha,), width=width)
        else:
            # Partial circle — draw as arc
            end_angle = progress * 360 - 90  # Start from top
            bbox = [px - radius, py - radius, px + radius, py + radius]
            self.draw.arc(bbox, start=-90, end=int(end_angle),
                          fill=color + (alpha,), width=width)

    def draw_line(
        self, x1: float, y1: float, x2: float, y2: float,
        color: Tuple[int, int, int] = None,
        opacity: float = 1.0,
        width: int = 3,
        progress: float = 1.0,
    ) -> None:
        """Draw a line segment with optional progressive drawing."""
        color = color or brand.PRIMARY
        alpha = int(255 * opacity)
        px1, py1 = self.to_pixel(x1, y1)
        px2, py2 = self.to_pixel(x2, y2)

        if progress < 1.0:
            # Partial line
            px2 = int(px1 + (px2 - px1) * progress)
            py2 = int(py1 + (py2 - py1) * progress)

        self.draw.line([(px1, py1), (px2, py2)],
                       fill=color + (alpha,), width=width)

    def draw_arrow(
        self, x1: float, y1: float, x2: float, y2: float,
        color: Tuple[int, int, int] = None,
        opacity: float = 1.0,
        width: int = 3,
        head_size: int = 15,
        progress: float = 1.0,
    ) -> None:
        """Draw an arrow from (x1,y1) to (x2,y2) with arrowhead."""
        color = color or brand.PRIMARY
        alpha = int(255 * opacity)

        # Draw the line part
        self.draw_line(x1, y1, x2, y2, color, opacity, width, progress)

        if progress >= 0.8:  # Show arrowhead near the end
            px2, py2 = self.to_pixel(x2, y2)
            px1, py1 = self.to_pixel(x1, y1)

            # Calculate arrowhead
            dx = px2 - px1
            dy = py2 - py1
            length = math.sqrt(dx * dx + dy * dy)
            if length == 0:
                return
            dx /= length
            dy /= length

            # Arrowhead points
            ax = px2 - head_size * dx + head_size * 0.5 * dy
            ay = py2 - head_size * dy - head_size * 0.5 * dx
            bx = px2 - head_size * dx - head_size * 0.5 * dy
            by = py2 - head_size * dy + head_size * 0.5 * dx

            self.draw.polygon(
                [(px2, py2), (int(ax), int(ay)), (int(bx), int(by))],
                fill=color + (alpha,)
            )

    def draw_arc(
        self, cx: float, cy: float, radius: float,
        start_angle: float, end_angle: float,
        color: Tuple[int, int, int] = None,
        opacity: float = 1.0,
        width: int = 3,
        progress: float = 1.0,
    ) -> None:
        """Draw an arc. Angles in degrees, 0=right, 90=up."""
        color = color or brand.SECONDARY
        alpha = int(255 * opacity)
        px, py = self.to_pixel(cx, cy)
        bbox = [px - radius, py - radius, px + radius, py + radius]

        # Convert to PIL angle system (0=right, positive=clockwise)
        pil_start = -end_angle  # Flip because PIL Y is inverted
        pil_end = -start_angle
        
        if progress < 1.0:
            pil_end = pil_start + (pil_end - pil_start) * progress

        self.draw.arc(bbox, start=int(pil_start), end=int(pil_end),
                      fill=color + (alpha,), width=width)

    def draw_dotted_line(
        self, x1: float, y1: float, x2: float, y2: float,
        color: Tuple[int, int, int] = None,
        opacity: float = 0.6,
        width: int = 2,
        dash_length: int = 10,
        gap_length: int = 8,
        progress: float = 1.0,
    ) -> None:
        """Draw a dotted/dashed line."""
        color = color or brand.TEXT_DIM
        alpha = int(255 * opacity)
        px1, py1 = self.to_pixel(x1, y1)
        px2, py2 = self.to_pixel(x2, y2)

        if progress < 1.0:
            px2 = int(px1 + (px2 - px1) * progress)
            py2 = int(py1 + (py2 - py1) * progress)

        dx = px2 - px1
        dy = py2 - py1
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            return

        nx, ny = dx / length, dy / length
        pos = 0
        drawing = True

        while pos < length:
            segment = dash_length if drawing else gap_length
            end_pos = min(pos + segment, length)

            if drawing:
                sx = int(px1 + nx * pos)
                sy = int(py1 + ny * pos)
                ex = int(px1 + nx * end_pos)
                ey = int(py1 + ny * end_pos)
                self.draw.line([(sx, sy), (ex, ey)],
                               fill=color + (alpha,), width=width)

            pos = end_pos
            drawing = not drawing

    def draw_angle_marker(
        self, cx: float, cy: float,
        angle1: float, angle2: float,
        radius: float = 30,
        color: Tuple[int, int, int] = None,
        opacity: float = 0.8,
        label: str = None,
    ) -> None:
        """Draw an angle arc marker between two angles, with optional label."""
        color = color or brand.SECONDARY
        self.draw_arc(cx, cy, radius, angle1, angle2, color, opacity, width=2)

        if label:
            # Place label at midpoint of arc
            mid_angle = math.radians((angle1 + angle2) / 2)
            lx = cx + (radius + 20) * math.cos(mid_angle)
            ly = cy + (radius + 20) * math.sin(mid_angle)
            self.draw_text(lx, ly, label, color=color, opacity=opacity,
                          font_size=brand.FONT_SIZE_LABEL, anchor='center')

    # === Text Rendering ===

    def _get_font(self, font_name: str = None, size: int = None) -> ImageFont.FreeTypeFont:
        """Get a cached font instance."""
        font_name = font_name or brand.FONT_MAIN
        size = size or brand.FONT_SIZE_BODY

        cache_key = (font_name, size)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        # Try to load the font
        font_paths = [
            font_name,
            f"/System/Library/Fonts/{font_name}.ttc",
            f"/System/Library/Fonts/{font_name}.ttf",
            f"/System/Library/Fonts/Supplemental/{font_name}.ttf",
            f"/System/Library/Fonts/Supplemental/{font_name}.ttc",
            f"/Library/Fonts/{font_name}.ttf",
        ]

        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, size)
                break
            except (OSError, IOError):
                continue

        if font is None:
            # Try common macOS font paths
            common_fonts = [
                "/System/Library/Fonts/Menlo.ttc",
                "/System/Library/Fonts/SFNSMono.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
            ]
            for path in common_fonts:
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except (OSError, IOError):
                    continue

        if font is None:
            font = ImageFont.load_default()

        self._font_cache[cache_key] = font
        return font

    def draw_text(
        self, x: float, y: float, text: str,
        color: Tuple[int, int, int] = None,
        opacity: float = 1.0,
        font_size: int = None,
        font_name: str = None,
        anchor: str = 'center',
        progress: float = 1.0,
        max_width: int = None,
    ) -> None:
        """
        Draw text at the given centered coordinates.
        
        Args:
            anchor: 'center', 'left', 'right', 'top_left', etc.
            progress: 0.0 to 1.0, for typewriter effect (shows partial text)
            max_width: Maximum width in pixels before wrapping
        """
        color = color or brand.TEXT_WHITE
        alpha = int(255 * opacity)
        font_size = font_size or brand.FONT_SIZE_BODY

        # Detect if text contains Devanagari
        has_devanagari = any('\u0900' <= c <= '\u097F' for c in text)
        if has_devanagari and font_name is None:
            font_name = brand.FONT_HINDI

        font = self._get_font(font_name, font_size)

        # Apply typewriter progress
        if progress < 1.0:
            visible_chars = max(1, int(len(text) * progress))
            text = text[:visible_chars]

        # Handle text wrapping
        if max_width:
            lines = self._wrap_text(text, font, max_width)
        else:
            lines = [text]

        # Calculate total text block size
        line_height = font_size + 8
        total_height = line_height * len(lines)

        px, py = self.to_pixel(x, y)

        for i, line in enumerate(lines):
            try:
                bbox = font.getbbox(line)
                tw = bbox[2] - bbox[0]
            except Exception:
                tw = len(line) * font_size // 2

            # Position based on anchor
            if anchor == 'center':
                tx = px - tw // 2
                ty = py - total_height // 2 + i * line_height
            elif anchor == 'left':
                tx = px
                ty = py - total_height // 2 + i * line_height
            elif anchor == 'right':
                tx = px - tw
                ty = py - total_height // 2 + i * line_height
            else:
                tx = px
                ty = py + i * line_height

            self.draw.text((tx, ty), line, fill=color + (alpha,), font=font)

    def draw_formula(
        self, x: float, y: float, latex: str,
        color: Tuple[int, int, int] = None,
        opacity: float = 1.0,
        scale: float = 1.0,
        progress: float = 1.0,
    ) -> None:
        """Draw a LaTeX formula at the given position."""
        from engine.math_renderer import MathRenderer

        color = color or brand.TEXT_WHITE
        formula_img = MathRenderer.render(latex, color=color, dpi=150)

        if scale != 1.0:
            new_size = (int(formula_img.width * scale), int(formula_img.height * scale))
            formula_img = formula_img.resize(new_size, Image.LANCZOS)

        # Apply opacity
        if opacity < 1.0:
            alpha_channel = formula_img.split()[3]
            alpha_channel = alpha_channel.point(lambda p: int(p * opacity))
            formula_img.putalpha(alpha_channel)

        # Apply progress (reveal from left)
        if progress < 1.0:
            visible_width = max(1, int(formula_img.width * progress))
            formula_img = formula_img.crop((0, 0, visible_width, formula_img.height))

        # Position
        px, py = self.to_pixel(x, y)
        paste_x = px - formula_img.width // 2
        paste_y = py - formula_img.height // 2

        self.image.paste(formula_img, (paste_x, paste_y), formula_img)
        # Refresh draw handle after paste
        self.draw = ImageDraw.Draw(self.image)

    # === Effects ===

    def draw_glow(
        self, x: float, y: float, radius: float,
        color: Tuple[int, int, int] = None,
        intensity: float = 0.3,
    ) -> None:
        """Draw a soft glow/bloom effect at the given position."""
        color = color or brand.GLOW_COLOR
        glow_layer = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)

        px, py = self.to_pixel(x, y)
        alpha = int(255 * intensity)

        # Draw concentric circles with decreasing opacity
        for r in range(int(radius), 0, -3):
            a = int(alpha * (r / radius) * 0.5)
            bbox = [px - r, py - r, px + r, py + r]
            glow_draw.ellipse(bbox, fill=color + (a,))

        # Blur the glow
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=radius // 3))

        # Composite
        self.image = Image.alpha_composite(self.image, glow_layer)
        self.draw = ImageDraw.Draw(self.image)

    def draw_sun_rays(
        self, x: float, y: float,
        direction_angle: float = -90,
        num_rays: int = 5,
        ray_length: float = 400,
        spacing: float = 60,
        color: Tuple[int, int, int] = None,
        opacity: float = 0.6,
        progress: float = 1.0,
    ) -> None:
        """Draw parallel sun rays (for physics diagrams)."""
        color = color or brand.SECONDARY
        angle_rad = math.radians(direction_angle)
        perp_rad = angle_rad + math.pi / 2

        for i in range(num_rays):
            offset = (i - num_rays // 2) * spacing
            start_x = x + offset * math.cos(perp_rad)
            start_y = y + offset * math.sin(perp_rad)
            end_x = start_x + ray_length * math.cos(angle_rad)
            end_y = start_y + ray_length * math.sin(angle_rad)

            self.draw_arrow(
                start_x, start_y, end_x, end_y,
                color=color, opacity=opacity, width=2,
                head_size=10, progress=progress
            )

    # === Compositing ===

    def composite_onto(self, background: Image.Image) -> Image.Image:
        """Composite this canvas onto a background image."""
        result = background.copy().convert('RGBA')
        result = Image.alpha_composite(result, self.image)
        return result

    def get_image(self) -> Image.Image:
        """Get the current canvas image."""
        return self.image.copy()

    def clear(self) -> None:
        """Clear the canvas."""
        self.image = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

    # === Helpers ===

    @staticmethod
    def _interpolate_gradient(t: float, stops: list) -> Tuple[int, int, int]:
        """Interpolate a color from gradient stops."""
        for i in range(len(stops) - 1):
            if stops[i][0] <= t <= stops[i + 1][0]:
                local_t = (t - stops[i][0]) / (stops[i + 1][0] - stops[i][0])
                c1 = stops[i][1]
                c2 = stops[i + 1][1]
                return tuple(int(c1[j] + (c2[j] - c1[j]) * local_t) for j in range(3))
        return stops[-1][1]

    @staticmethod
    def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """Wrap text to fit within max_width pixels."""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            try:
                bbox = font.getbbox(test_line)
                tw = bbox[2] - bbox[0]
            except Exception:
                tw = len(test_line) * 20

            if tw <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return lines if lines else [text]
