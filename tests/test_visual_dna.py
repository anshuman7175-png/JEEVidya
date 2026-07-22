"""Tier 2 — Visual DNA: determinism, uniqueness, serialization, evolution."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.visual_dna import GENE_SPACE, VisualDNA


def test_same_title_same_genome():
    a = VisualDNA.from_title("Escape Velocity kya hai?")
    b = VisualDNA.from_title("Escape Velocity kya hai?")
    assert a.genes == b.genes and a.seed == b.seed
    assert a.cache_key_material() == b.cache_key_material()


def test_different_titles_differ():
    a = VisualDNA.from_title("Escape Velocity kya hai?")
    b = VisualDNA.from_title("Photoelectric effect")
    assert a.genes != b.genes
    assert a.palette["primary"] != b.palette["primary"]


def test_genes_within_space():
    for title in ("A", "Doppler", "Ohm's law", "प्रकाश"):
        dna = VisualDNA.from_title(title)
        for k, spec in GENE_SPACE.items():
            v = dna.genes[k]
            if isinstance(spec, tuple) and isinstance(spec[0], (int, float)):
                assert spec[0] <= v <= spec[1], f"{k}={v} out of {spec}"
            else:
                assert v in spec


def test_palette_is_legible_dark_bg():
    dna = VisualDNA.from_title("Gravity")
    p = dna.palette
    assert max(p["bg_top"]) < 110, "background must stay dark"
    assert max(p["primary"]) > 180, "primary must pop"


def test_tuning_overrides_win():
    dna = VisualDNA.from_title("X", tuning={"energy": 0.9,
                                            "hook_style": "myth_bust"})
    assert dna.genes["energy"] == 0.9
    assert dna.genes["hook_style"] == "myth_bust"


def test_roundtrip_serialization():
    a = VisualDNA.from_title("Waves")
    b = VisualDNA.from_dict(a.to_dict())
    assert b.genes == a.genes and b.seed == a.seed
    assert b.cache_key_material() == a.cache_key_material()


def test_embedded_dna_wins_over_title_roll():
    a = VisualDNA.from_title("Waves")
    dialogue = {"title": "Completely different title", "dna": a.to_dict()}
    b = VisualDNA.from_dialogue(dialogue)
    assert b.genes == a.genes            # bit-identical re-renders


def test_mutation_stays_in_space():
    dna = VisualDNA.from_title("Momentum")
    rng = random.Random(5)
    for _ in range(50):
        dna = dna.mutate(rng, strength=0.4)
    for k, spec in GENE_SPACE.items():
        v = dna.genes[k]
        if isinstance(spec, tuple) and isinstance(spec[0], (int, float)):
            assert spec[0] <= v <= spec[1]
        else:
            assert v in spec


def test_motifs_deterministic():
    a = VisualDNA.from_title("Optics")
    b = VisualDNA.from_title("Optics")
    assert a.motif_names == b.motif_names
    assert 3 <= len(a.motif_names) <= 7
