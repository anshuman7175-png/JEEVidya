"""
JEEVidya V5 — Visual DNA (Tier 2)
═════════════════════════════════
Every topic gets a unique, deterministic visual identity: a versioned
genome seeded by sha256(title). The same title always renders the same
video (Tier 0 cache keys embed the genes); a different title gets a
different palette, energy, cut-rate, motif set, grade, and BGM.

Genes are TUNABLE: the Tier 4 flywheel learns which gene values your
audience rewards and passes overrides into `from_title(tuning=...)`.
"""
from __future__ import annotations

import colorsys
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

DNA_VERSION = "dna-v2"

HOOK_STYLES = ("question", "shock_number", "challenge", "myth_bust")
BGM_MODES = ("major", "minor", "dorian", "lydian")

# Gene search space — the flywheel's bandit arms live over these bins.
GENE_SPACE: Dict[str, Any] = {
    "hue": (0.0, 1.0),            # base hue of the whole identity
    "energy": (0.35, 1.0),        # drives motion, cut-rate, BGM tempo
    "cut_rate": (0.6, 1.6),       # shot-change frequency multiplier
    "motif_density": (0.0, 1.0),  # background motif count
    "hook_style": HOOK_STYLES,
    "grade_warmth": (-0.5, 0.5),  # split-tone shadow↔highlight tilt
    "grain": (0.0, 0.35),
    "vignette": (0.15, 0.55),
    "bgm_mode": BGM_MODES,
    "bgm_root": (45, 57),         # MIDI A2..A3
}


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _hsv(h: float, s: float, v: float) -> Tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, _clip(s, 0, 1), _clip(v, 0, 1))
    return int(r * 255), int(g * 255), int(b * 255)


@dataclass
class VisualDNA:
    """A versioned, serializable visual genome for one video."""
    title: str
    seed: int
    genes: Dict[str, Any]
    version: str = DNA_VERSION

    # ─── Construction ──────────────────────────────────────

    @classmethod
    def from_title(cls, title: str,
                   tuning: Optional[Dict[str, Any]] = None) -> "VisualDNA":
        """Deterministic genome from a title, with optional flywheel
        overrides (tuning wins over the seeded roll, gene by gene)."""
        seed = int(hashlib.sha256(title.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        genes: Dict[str, Any] = {
            "hue": rng.random(),
            "energy": rng.uniform(*GENE_SPACE["energy"]),
            "cut_rate": rng.uniform(*GENE_SPACE["cut_rate"]),
            "motif_density": rng.uniform(*GENE_SPACE["motif_density"]),
            "hook_style": rng.choice(HOOK_STYLES),
            "grade_warmth": rng.uniform(*GENE_SPACE["grade_warmth"]),
            "grain": rng.uniform(*GENE_SPACE["grain"]),
            "vignette": rng.uniform(*GENE_SPACE["vignette"]),
            "bgm_mode": rng.choice(BGM_MODES),
            "bgm_root": rng.randint(*GENE_SPACE["bgm_root"]),
        }
        if tuning:
            for k, v in tuning.items():
                if k in genes and v is not None:
                    genes[k] = v
        return cls(title=title, seed=seed, genes=genes)

    @classmethod
    def from_dialogue(cls, dialogue: Dict[str, Any]) -> "VisualDNA":
        """DNA for a dialogue JSON. An embedded 'dna' block (saved by the
        factory / .jvproj bundles) wins over a fresh title roll — that is
        what makes re-renders bit-identical months later."""
        embedded = dialogue.get("dna")
        if isinstance(embedded, dict) and "genes" in embedded:
            return cls.from_dict(embedded)
        return cls.from_title(dialogue.get("title", "Untitled"),
                              tuning=dialogue.get("dna_tuning"))

    # ─── Derived phenotype (what the renderer consumes) ────

    @property
    def palette(self) -> Dict[str, Tuple[int, int, int]]:
        """Cinematic dark palette derived from the hue gene.
        Analogous body colors + a complementary accent — always legible
        on a dark background, never random mud."""
        h = self.genes["hue"]
        e = self.genes["energy"]
        return {
            "bg_top": _hsv(h, 0.55, 0.14),
            "bg_bottom": _hsv(h + 0.06, 0.50, 0.26),
            "primary": _hsv(h + 0.46, 0.85, 0.98),       # complementary pop
            "secondary": _hsv(h + 0.11, 0.75, 0.98),     # analogous warm
            "accent": _hsv(h + 0.54, 0.90, 0.98),
            "success": _hsv(h + 0.33, 0.80, 0.92),
            "glow": _hsv(h + 0.46, 0.60, 1.0),
            "chalk": _hsv(h + 0.02, 0.08, 0.88),
            "energy_white": _hsv(h, 0.04, 0.96 if e > 0.6 else 0.90),
        }

    @property
    def particle_colors(self) -> List[Tuple[int, int, int, int]]:
        p = self.palette
        base_a = int(50 + 60 * self.genes["energy"])
        return [
            p["primary"] + (base_a,),
            p["secondary"] + (int(base_a * 0.75),),
            p["accent"] + (int(base_a * 0.6),),
            p["success"] + (int(base_a * 0.5),),
        ]

    @property
    def motif_names(self) -> List[str]:
        """Which procedural motifs this video floats in its background."""
        from tools.motif_forge import MOTIF_NAMES
        rng = random.Random(self.seed + 7)
        count = 3 + int(self.genes["motif_density"] * 4)   # 3..7 kinds
        return rng.sample(MOTIF_NAMES, min(count, len(MOTIF_NAMES)))

    @property
    def motif_count(self) -> int:
        return int(4 + self.genes["motif_density"] * 10)   # 4..14 on screen

    @property
    def bgm_tempo(self) -> int:
        return int(70 + self.genes["energy"] * 50)          # 70..120 BPM

    # ─── Serialization ─────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "title": self.title,
                "seed": self.seed, "genes": self.genes}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisualDNA":
        return cls(title=d.get("title", "Untitled"),
                   seed=int(d.get("seed", 0)),
                   genes=dict(d["genes"]),
                   version=d.get("version", DNA_VERSION))

    def cache_key_material(self) -> str:
        """What Tier 0 folds into segment/mix keys: version + genes."""
        return self.version + "|" + json.dumps(self.genes, sort_keys=True)

    # ─── Evolution (Tier 4 flywheel) ───────────────────────

    def mutate(self, rng: random.Random, strength: float = 0.25) -> "VisualDNA":
        """Gaussian-perturb continuous genes, occasionally flip choices."""
        g = dict(self.genes)
        for k, spec in GENE_SPACE.items():
            if isinstance(spec, tuple) and isinstance(spec[0], float):
                lo, hi = spec
                g[k] = _clip(g[k] + rng.gauss(0, (hi - lo) * strength), lo, hi)
            elif isinstance(spec, tuple) and isinstance(spec[0], int):
                lo, hi = spec
                g[k] = int(_clip(g[k] + rng.gauss(0, (hi - lo) * strength), lo, hi))
            elif rng.random() < strength * 0.5:
                g[k] = rng.choice(spec)
        return VisualDNA(title=self.title, seed=self.seed, genes=g,
                         version=self.version)

    def describe(self) -> str:
        p = self.palette
        return (f"DNA[{self.version}] '{self.title}' hue={self.genes['hue']:.2f} "
                f"energy={self.genes['energy']:.2f} cut={self.genes['cut_rate']:.2f} "
                f"hook={self.genes['hook_style']} "
                f"bgm={self.genes['bgm_mode']}@{self.bgm_tempo}bpm "
                f"primary=#{p['primary'][0]:02x}{p['primary'][1]:02x}{p['primary'][2]:02x}")
