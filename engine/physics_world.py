"""
JEEVidya V5 — Unified Physics World (Tier 2)
════════════════════════════════════════════
ONE verlet simulation drives everything that moves in the background:
floating motifs, emotion VFX bursts, knowledge-transfer arcs, and (via
the same integrator) the puppet spring chains.

Deterministic by construction: seeded RNG, fixed timestep — a segment's
physics is a pure function of (seed, frame_count, reaction schedule),
which is exactly what the Tier 0 cache requires.

Keyword reactions are REAL physics, not canned animations:
    "gravity"  → the world's gravity vector triples for a second
    "speed"    → global velocity multiplier surges
    "zero/space" → weightlessness
    "boom"     → radial impulse burst
fired at the VTT word timestamp by the compositor.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DT = 1.0            # fixed step = one frame (all constants tuned to it)
DAMPING = 0.985


@dataclass
class Body:
    """A verlet point mass with render metadata."""
    x: float
    y: float
    px: float          # previous position (velocity is implicit)
    py: float
    size: float = 4.0
    color: Tuple[int, int, int, int] = (255, 255, 255, 120)
    kind: str = "dot"          # "dot" | motif name | "spark"
    spin: float = 0.0          # radians/frame (motifs rotate)
    angle: float = 0.0
    life: int = -1             # frames to live; -1 = immortal (recycled)
    mass: float = 1.0
    depth: float = 1.0         # z: <1 near camera, >1 far (parallax)

    @property
    def vx(self) -> float:
        return self.x - self.px

    @property
    def vy(self) -> float:
        return self.y - self.py


@dataclass
class Spring:
    """Distance constraint between two bodies (or body↔anchor)."""
    a: Body
    b: Body
    rest: float
    stiffness: float = 0.5


# Keyword → world reaction. Values are (duration_frames, effect fn name).
REACTION_WORDS: Dict[str, str] = {
    "gravity": "heavy", "gravitational": "heavy", "gurutvakarshan": "heavy",
    "speed": "surge", "fast": "surge", "velocity": "surge", "tez": "surge",
    "zero": "weightless", "space": "weightless", "antariksh": "weightless",
    "boom": "burst", "blast": "burst", "explosion": "burst", "dhamaka": "burst",
    "energy": "excite", "power": "excite", "force": "excite",
}


class PhysicsWorld:
    """Seeded, fixed-step verlet world."""

    def __init__(self, width: int, height: int, seed: int = 0,
                 gravity: Tuple[float, float] = (0.0, 0.0)):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.base_gravity = gravity
        self.gravity = list(gravity)
        self.speed_mult = 1.0
        self.bodies: List[Body] = []
        self.springs: List[Spring] = []
        self.frame = 0
        # Active timed effects: name → frames remaining
        self._effects: Dict[str, int] = {}

    # ─── Population ────────────────────────────────────────

    def spawn(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0,
              **kw) -> Body:
        b = Body(x=x, y=y, px=x - vx, py=y - vy, **kw)
        self.bodies.append(b)
        return b

    def spawn_drifter(self, kind: str, size: float,
                      color: Tuple[int, int, int, int]) -> Body:
        """A slow ambient floater on a REAL depth plane: far motifs are
        smaller, fainter, slower — and parallax against camera drift."""
        r = self.rng
        depth = r.uniform(0.65, 1.9)
        rr, g, b, a = color
        return self.spawn(
            r.uniform(0, self.width), r.uniform(0, self.height),
            vx=r.uniform(-0.3, 0.3) / depth,
            vy=r.uniform(-0.65, -0.15) / depth,
            size=size / depth,
            color=(rr, g, b, max(12, int(a / depth))),
            kind=kind, depth=depth,
            spin=r.uniform(-0.01, 0.01) / depth,
            angle=r.uniform(0, math.tau))

    def add_chain(self, anchor: Body, n: int, link: float,
                  stiffness: float = 0.6, **kw) -> List[Body]:
        """Verlet spring chain hanging off an anchor (hair, dupatta)."""
        prev = anchor
        chain = []
        for i in range(n):
            b = self.spawn(anchor.x, anchor.y + link * (i + 1), **kw)
            self.springs.append(Spring(prev, b, rest=link,
                                       stiffness=stiffness))
            chain.append(b)
            prev = b
        return chain

    def burst(self, x: float, y: float, count: int = 24,
              speed: float = 9.0,
              colors: Optional[Sequence[Tuple[int, int, int, int]]] = None,
              life: int = 26) -> None:
        """Radial spark explosion (emotion VFX, reveals)."""
        colors = colors or [(255, 215, 0, 220), (255, 255, 255, 200)]
        for _ in range(count):
            a = self.rng.uniform(0, math.tau)
            v = self.rng.uniform(0.35, 1.0) * speed
            self.spawn(x, y, vx=math.cos(a) * v, vy=math.sin(a) * v,
                       size=self.rng.uniform(2, 5),
                       color=self.rng.choice(list(colors)),
                       kind="spark", life=life)

    def arc_stream(self, x1: float, y1: float, x2: float, y2: float,
                   count: int = 10,
                   color: Tuple[int, int, int, int] = (0, 212, 255, 200),
                   life: int = 30) -> None:
        """Knowledge-transfer arc: sparks lobbed from speaker to listener.
        Initial velocity solves the ballistic arc under current gravity."""
        g = max(0.05, self.gravity[1] + 0.25)
        for i in range(count):
            t = life * (0.7 + 0.3 * self.rng.random())
            vx = (x2 - x1) / t
            vy = (y2 - y1) / t - 0.5 * g * t
            jitter = self.rng.uniform(-0.5, 0.5)
            self.spawn(x1, y1, vx=vx + jitter * 0.3, vy=vy + jitter,
                       size=self.rng.uniform(2.5, 4.5), color=color,
                       kind="spark", life=int(t) + 4)

    # ─── Keyword reactions ─────────────────────────────────

    def react(self, word: str,
              at: Optional[Tuple[float, float]] = None) -> Optional[str]:
        """Fire the physical reaction a spoken word deserves (if any)."""
        w = "".join(ch for ch in word.lower() if ch.isalpha())
        effect = REACTION_WORDS.get(w)
        if effect is None:
            return None
        if effect == "burst":
            x, y = at or (self.width / 2, self.height * 0.4)
            self.burst(x, y)
        else:
            self._effects[effect] = {"heavy": 30, "surge": 24,
                                     "weightless": 45, "excite": 30}[effect]
        return effect

    def _apply_effects(self) -> None:
        gx, gy = self.base_gravity
        self.speed_mult = 1.0
        if self._effects.get("heavy", 0) > 0:
            gy = gy * 3.0 + 0.5
        if self._effects.get("weightless", 0) > 0:
            gx, gy = 0.0, -0.04
        if self._effects.get("surge", 0) > 0:
            self.speed_mult = 1.9
        if self._effects.get("excite", 0) > 0:
            self.speed_mult = max(self.speed_mult, 1.4)
        self.gravity = [gx, gy]
        for k in list(self._effects):
            self._effects[k] -= 1
            if self._effects[k] <= 0:
                del self._effects[k]

    # ─── Integration ───────────────────────────────────────

    def step(self) -> None:
        self.frame += 1
        self._apply_effects()
        gx, gy = self.gravity
        sm = self.speed_mult

        alive: List[Body] = []
        for b in self.bodies:
            if b.life > 0:
                b.life -= 1
                if b.life == 0:
                    continue
            # Verlet: x' = x + (x - px)*damping*speed + a*dt²
            nx = b.x + (b.x - b.px) * DAMPING * sm + gx * DT * DT
            ny = b.y + (b.y - b.py) * DAMPING * sm + gy * DT * DT
            b.px, b.py = b.x, b.y
            b.x, b.y = nx, ny
            b.angle += b.spin * sm

            # Immortal drifters wrap; mortals just fly off
            if b.life < 0:
                if b.y < -60:
                    b.y += self.height + 120
                    b.py = b.y - b.vy
                if b.x < -60:
                    b.x += self.width + 120
                    b.px = b.x - b.vx
                elif b.x > self.width + 60:
                    b.x -= self.width + 120
                    b.px = b.x - b.vx
            alive.append(b)
        self.bodies = alive

        # Satisfy spring constraints (2 relaxation passes)
        for _ in range(2):
            for s in self.springs:
                dx, dy = s.b.x - s.a.x, s.b.y - s.a.y
                dist = math.hypot(dx, dy) or 1e-6
                diff = (dist - s.rest) / dist * s.stiffness * 0.5
                s.a.x += dx * diff
                s.a.y += dy * diff
                s.b.x -= dx * diff
                s.b.y -= dy * diff

    def sparks(self) -> List[Body]:
        return [b for b in self.bodies if b.kind == "spark"]

    def drifters(self) -> List[Body]:
        return [b for b in self.bodies if b.kind not in ("spark",)]
