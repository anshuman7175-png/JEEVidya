"""
Gudiya & Chintu — Floating Particle System
Renders tiny glowing dots that drift upward, creating a living background.
Colors cycle through cyan, gold, pink, green at varying opacity.
"""
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFilter

from config import settings, brand


@dataclass
class Particle:
    """A single floating particle."""
    x: float
    y: float
    speed: float           # Pixels per frame (upward)
    size: int
    color: Tuple[int, int, int, int]  # RGBA
    phase: float           # For oscillation offset
    lifetime: int = 0

    def update(self) -> None:
        """Move particle upward with gentle horizontal oscillation."""
        self.y -= self.speed
        self.x += math.sin(self.lifetime * 0.03 + self.phase) * 0.5
        self.lifetime += 1

    def is_dead(self) -> bool:
        """Particle has drifted off screen."""
        return self.y < -20


class ParticleSystem:
    """Manages a pool of floating glow particles."""

    def __init__(self, width: int = settings.WIDTH, height: int = settings.HEIGHT,
                 rng=None, palette=None):
        """
        rng: optional random.Random instance. jvmake passes a per-segment
        seeded RNG so a turn's particle field is a pure function of its
        content key — bit-identical re-renders, cacheable segments.
        palette: optional RGBA color list (Visual DNA particle colors).
        """
        self.width = width
        self.height = height
        self._rng = rng if rng is not None else random
        self._palette = palette or brand.PARTICLE_COLORS
        self.particles: List[Particle] = []
        self._seed_initial()

    def _seed_initial(self) -> None:
        """Seed particles across the screen for frame 0."""
        for _ in range(settings.PARTICLE_COUNT):
            self.particles.append(self._spawn(randomize_y=True))

    def _spawn(self, randomize_y: bool = False) -> Particle:
        """Create a new particle at a random position."""
        rng = self._rng
        return Particle(
            x=rng.randint(0, self.width),
            y=rng.randint(0, self.height) if randomize_y else self.height + 10,
            speed=rng.uniform(settings.PARTICLE_SPEED_MIN, settings.PARTICLE_SPEED_MAX),
            size=rng.randint(settings.PARTICLE_SIZE_MIN, settings.PARTICLE_SIZE_MAX),
            color=rng.choice(self._palette),
            phase=rng.uniform(0, math.pi * 2),
        )

    def update(self) -> List[Particle]:
        """Advance all particles by one frame, respawning dead ones."""
        alive = []
        for p in self.particles:
            p.update()
            if p.is_dead():
                alive.append(self._spawn())
            else:
                alive.append(p)
        self.particles = alive
        return self.particles

    def render(self, base: Image.Image) -> Image.Image:
        """
        Render all particles onto a base image.
        Uses a glow layer (blurred circles) composited at low opacity.
        """
        # Create a transparent overlay for particles
        overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for p in self.particles:
            x, y = int(p.x), int(p.y)
            s = p.size
            r, g, b, a = p.color

            # Draw core dot
            draw.ellipse(
                (x - s, y - s, x + s, y + s),
                fill=(r, g, b, a)
            )

        # Apply glow: blur the particle layer
        glow_layer = overlay.filter(ImageFilter.GaussianBlur(radius=settings.PARTICLE_GLOW_RADIUS))

        # Composite glow first (behind), then sharp particles on top
        result = base.copy()
        if result.mode != 'RGBA':
            result = result.convert('RGBA')
        result = Image.alpha_composite(result, glow_layer)
        result = Image.alpha_composite(result, overlay)

        return result
