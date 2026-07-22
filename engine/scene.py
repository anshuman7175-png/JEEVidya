"""
JEEVidya — Scene Renderer
Renders animation timelines into sequences of PIL frames.
This is the bridge between the animation system and the video compositor.
"""
import math
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter
from config import brand, settings
from engine.canvas import Canvas
from engine.animator import AnimationTimeline, SceneElement


class SceneRenderer:
    """
    Renders a complete scene (AnimationTimeline) into a sequence of PIL frames.
    Each frame composites all visible elements at that point in time.
    """

    def __init__(self, timeline: AnimationTimeline):
        self.timeline = timeline
        self.width = settings.WIDTH
        self.height = settings.HEIGHT
        self.fps = settings.FPS

        # Pre-render the background once (it's static)
        self._background = self._create_background()

    def render_frame(self, time: float) -> Image.Image:
        """
        Render a single frame at the given time.
        Returns a complete RGBA image ready for video encoding.
        """
        # Start with background
        frame = self._background.copy()
        canvas = Canvas(self.width, self.height)

        # Get all visible elements at this time
        elements = self.timeline.get_elements_at(time)

        # Draw each element
        for elem in elements:
            self._draw_element(canvas, elem, time)

        # Composite canvas onto background
        frame = Image.alpha_composite(frame, canvas.get_image())

        return frame

    def render_all_frames(self) -> List[Image.Image]:
        """Render all frames for the entire scene duration."""
        total_frames = int(self.timeline.duration * self.fps)
        frames = []

        for i in range(total_frames):
            time = i / self.fps
            frame = self.render_frame(time)
            frames.append(frame.convert('RGB'))

            # Progress indication
            if i % (self.fps * 2) == 0:
                pct = (i / total_frames) * 100
                print(f"  Rendering: {pct:.0f}% ({i}/{total_frames} frames)")

        return frames

    def _draw_element(self, canvas: Canvas, elem: SceneElement, time: float) -> None:
        """Draw a single element on the canvas at the given time."""
        # Get common animated properties
        opacity = elem.get_property('opacity', time, 1.0)
        progress = elem.get_property('progress', time, 1.0)
        scale = elem.get_property('scale', time, 1.0)
        x = elem.get_property('x', time, elem.draw_params.get('x', 0))
        y = elem.get_property('y', time, elem.draw_params.get('y', 0))

        if opacity <= 0.01:
            return

        element_type = elem.element_type

        if element_type == 'circle':
            radius = elem.draw_params.get('radius', 100) * scale
            color = elem.draw_params.get('color', brand.PRIMARY)
            width = elem.draw_params.get('width', 3)
            fill_color = elem.draw_params.get('fill_color', None)
            fill_opacity = elem.draw_params.get('fill_opacity', 0.15)
            canvas.draw_circle(x, y, radius, color, opacity, width,
                             fill_color, fill_opacity, progress)

        elif element_type == 'line':
            x2 = elem.get_property('x2', time, elem.draw_params.get('x2', 0))
            y2 = elem.get_property('y2', time, elem.draw_params.get('y2', 0))
            color = elem.draw_params.get('color', brand.PRIMARY)
            width = elem.draw_params.get('width', 3)
            canvas.draw_line(x, y, x2, y2, color, opacity, width, progress)

        elif element_type == 'arrow':
            x2 = elem.get_property('x2', time, elem.draw_params.get('x2', 0))
            y2 = elem.get_property('y2', time, elem.draw_params.get('y2', 0))
            color = elem.draw_params.get('color', brand.PRIMARY)
            width = elem.draw_params.get('width', 3)
            canvas.draw_arrow(x, y, x2, y2, color, opacity, width, progress=progress)

        elif element_type == 'arc':
            radius = elem.draw_params.get('radius', 50) * scale
            start_angle = elem.draw_params.get('start_angle', 0)
            end_angle = elem.draw_params.get('end_angle', 90)
            color = elem.draw_params.get('color', brand.SECONDARY)
            width = elem.draw_params.get('width', 3)
            canvas.draw_arc(x, y, radius, start_angle, end_angle,
                          color, opacity, width, progress)

        elif element_type == 'dotted_line':
            x2 = elem.get_property('x2', time, elem.draw_params.get('x2', 0))
            y2 = elem.get_property('y2', time, elem.draw_params.get('y2', 0))
            color = elem.draw_params.get('color', brand.TEXT_DIM)
            canvas.draw_dotted_line(x, y, x2, y2, color, opacity, progress=progress)

        elif element_type == 'text':
            text = elem.draw_params.get('text', '')
            color = elem.draw_params.get('color', brand.TEXT_WHITE)
            font_size = int(elem.draw_params.get('font_size', brand.FONT_SIZE_BODY) * scale)
            font_name = elem.draw_params.get('font_name', None)
            anchor = elem.draw_params.get('anchor', 'center')
            max_width = elem.draw_params.get('max_width', None)
            canvas.draw_text(x, y, text, color, opacity, font_size,
                           font_name, anchor, progress, max_width)

        elif element_type == 'formula':
            latex = elem.draw_params.get('latex', '')
            color = elem.draw_params.get('color', brand.TEXT_WHITE)
            canvas.draw_formula(x, y, latex, color, opacity, scale, progress)

        elif element_type == 'glow':
            radius = elem.draw_params.get('radius', 80) * scale
            color = elem.draw_params.get('color', brand.GLOW_COLOR)
            intensity = opacity * elem.draw_params.get('intensity', 0.3)
            canvas.draw_glow(x, y, radius, color, intensity)

        elif element_type == 'sun_rays':
            direction = elem.draw_params.get('direction_angle', -90)
            num_rays = elem.draw_params.get('num_rays', 5)
            ray_length = elem.draw_params.get('ray_length', 400)
            spacing = elem.draw_params.get('spacing', 60)
            color = elem.draw_params.get('color', brand.SECONDARY)
            canvas.draw_sun_rays(x, y, direction, num_rays, ray_length,
                               spacing, color, opacity, progress)

        elif element_type == 'angle_marker':
            radius = elem.draw_params.get('radius', 30) * scale
            angle1 = elem.draw_params.get('angle1', 0)
            angle2 = elem.draw_params.get('angle2', 90)
            color = elem.draw_params.get('color', brand.SECONDARY)
            label = elem.draw_params.get('label', None)
            canvas.draw_angle_marker(x, y, angle1, angle2, radius,
                                    color, opacity, label)

        elif element_type == 'measurement':
            x2 = elem.draw_params.get('x2', 0)
            y2 = elem.draw_params.get('y2', 0)
            label = elem.draw_params.get('label', '')
            color = elem.draw_params.get('color', brand.SECONDARY)
            # Draw measurement line with endpoints
            if progress > 0.1:
                canvas.draw_line(x, y, x2, y2, color, opacity * 0.6, 2, progress)
                # Draw tick marks at endpoints
                if progress > 0.5:
                    canvas.draw_text(
                        (x + x2) / 2, (y + y2) / 2 + 30,
                        label, color, opacity, brand.FONT_SIZE_LABEL, anchor='center'
                    )

    def _create_background(self) -> Image.Image:
        """Create the static branded background image."""
        canvas = Canvas(self.width, self.height)
        canvas.draw_gradient_background()

        # Add subtle grid
        canvas.draw_grid(spacing=90, opacity=0.04)

        # Add subtle vignette effect
        vignette = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        vignette_draw = ImageDraw.Draw(vignette)

        # Radial gradient vignette
        cx, cy = self.width // 2, self.height // 2
        max_dist = math.sqrt(cx * cx + cy * cy)

        for radius in range(int(max_dist), 0, -5):
            t = radius / max_dist
            if t > 0.6:
                alpha = int(80 * ((t - 0.6) / 0.4) ** 2)
                vignette_draw.ellipse(
                    [cx - radius, cy - radius, cx + radius, cy + radius],
                    fill=(0, 0, 0, alpha)
                )

        bg = canvas.get_image()
        bg = Image.alpha_composite(bg, vignette)

        return bg
