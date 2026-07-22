"""
JEEVidya — Math Formula Renderer
Renders LaTeX math strings to transparent PIL Images using matplotlib's mathtext.
No LaTeX installation required — uses matplotlib's built-in parser.
"""
import io
from typing import Optional, Tuple

from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.mathtext import math_to_image
import matplotlib as mpl

# Configure mathtext for clean rendering
mpl.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern — classic LaTeX look
mpl.rcParams['mathtext.default'] = 'regular'


class MathRenderer:
    """Renders LaTeX formulas to PIL Images with transparency."""

    _cache: dict = {}

    @classmethod
    def render(
        cls,
        latex: str,
        color: Tuple[int, int, int] = (240, 240, 255),
        dpi: int = 200,
        font_size: int = 24,
    ) -> Image.Image:
        """
        Render a LaTeX string to a PIL Image with transparent background.

        Args:
            latex: LaTeX math string (e.g., r"$C = 2\\pi R$")
            color: RGB text color
            dpi: Resolution (higher = sharper but larger)
            font_size: Base font size

        Returns:
            PIL.Image.Image in RGBA mode with transparent background
        """
        cache_key = (latex, color, dpi, font_size)
        if cache_key in cls._cache:
            return cls._cache[cache_key].copy()

        # Ensure $ delimiters
        if not latex.startswith('$'):
            latex = f'${latex}$'

        # Set color for matplotlib
        r, g, b = [c / 255.0 for c in color]
        mpl.rcParams['text.color'] = (r, g, b)

        # Render to buffer
        buf = io.BytesIO()
        try:
            fig, ax = plt.subplots(figsize=(10, 2), dpi=dpi)
            fig.patch.set_alpha(0.0)
            ax.set_alpha(0.0)
            ax.axis('off')

            ax.text(
                0.5, 0.5, latex,
                transform=ax.transAxes,
                fontsize=font_size,
                color=(r, g, b),
                ha='center', va='center',
                math_fontfamily='cm'
            )

            fig.savefig(buf, format='png', transparent=True,
                        bbox_inches='tight', pad_inches=0.1, dpi=dpi)
            plt.close(fig)
        except Exception as e:
            plt.close('all')
            # Fallback: render as plain text image
            return cls._render_plain_text(latex.strip('$'), color, font_size)

        buf.seek(0)
        img = Image.open(buf).convert('RGBA')

        # Trim transparent borders
        img = cls._trim_transparent(img)

        # Cache the result
        cls._cache[cache_key] = img.copy()
        return img

    @staticmethod
    def _trim_transparent(img: Image.Image, padding: int = 10) -> Image.Image:
        """Remove transparent borders from an RGBA image."""
        if img.mode != 'RGBA':
            return img

        # Get the alpha channel
        alpha = img.split()[3]
        bbox = alpha.getbbox()
        if bbox is None:
            return img

        # Add padding
        x0 = max(0, bbox[0] - padding)
        y0 = max(0, bbox[1] - padding)
        x1 = min(img.width, bbox[2] + padding)
        y1 = min(img.height, bbox[3] + padding)

        return img.crop((x0, y0, x1, y1))

    @staticmethod
    def _render_plain_text(
        text: str,
        color: Tuple[int, int, int],
        font_size: int
    ) -> Image.Image:
        """Fallback: render plain text as an image using PIL."""
        from PIL import ImageDraw, ImageFont

        # Estimate size
        width = len(text) * font_size
        height = font_size * 2
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("Menlo", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        draw.text((10, height // 4), text, fill=color + (255,), font=font)
        return MathRenderer._trim_transparent(img)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the formula cache."""
        cls._cache.clear()
