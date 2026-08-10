"""
JEEVidya V5 — Pose Stager (Tier 1, one-time asset staging)
══════════════════════════════════════════════════════════
Turns the raw pose renders in assets/poses/ (character1*.png = Chintu,
character2*.png = Gudiya — background already removed) into the exact
directory layout the rest of Tier 1 consumes:

  assets/characters/<name>/
    body.png                 ← the neutral standing pose (rig source)
    poses/<gesture>.png      ← PoseLibrary body swaps (GESTURE_POSE_MAP names)
    visemes_src/<VISEME>.png ← real-art mouth shapes for the Rig Builder

The letter→meaning mapping below was assigned BY EYE from labeled
contact sheets of every pose's face crop (mouth articulation) and full
body (gesture) — both characters share the same letter→gesture scheme.

Viseme choices are articulation-correct, not guesses:
  • REST/BILABIAL — fully closed lips
  • OPEN_A        — jaw-dropped laugh (tongue + upper teeth visible)
  • MID_E / DENTAL / CLOSED_I / LABIODENTAL — teeth-showing spreads,
    ordered from most open to most closed
  • ROUNDED_LAX   — the surprised "O" mouth (fallback covers ROUNDED_TENSE)
Visemes not listed for a character fall back through engine.rig.VISEME_FALLBACK.

Stale-safe: poses/ and visemes_src/ are wiped of *.png before staging,
so re-running after a mapping change never leaves orphaned sprites.

Run:  python3 jvmake.py stage          (or: python3 tools/pose_stager.py)
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from engine.rig import VISEME_NAMES

POSES_DIR = os.path.join(settings.ASSETS_DIR, "poses")

# assets/poses/ file prefix → character directory name
CHARACTER_MAP: Dict[str, str] = {
    "character1": "chintu",   # boy, glasses, orange shirt
    "character2": "gudiya",   # girl, red hoodie
}

# The neutral standing pose — becomes body.png (the rig source) AND
# poses/neutral.png (the PoseLibrary default).
NEUTRAL_LETTER = "a"

# Letter → PoseLibrary gesture name. Both characters share this scheme
# (verified frame-by-frame). Names MUST match engine.gestures.GESTURE_POSE_MAP.
POSE_MAP: Dict[str, str] = {
    "a": "neutral",       # standing, arms relaxed
    "b": "presenting",    # both palms out
    "c": "pointing_up",   # index finger raised high
    "d": "thinking",      # hand on chin
    "e": "excited",       # both arms thrown up
    "f": "surprised",     # recoil, hands up, mouth open
    "g": "shrug",         # palms up shrug
    "h": "confident",     # arms crossed
    "m": "facepalm",      # hand over face
}

# Letter → viseme sprite source, PER CHARACTER (mouths differ across the
# two sets: Gudiya's only fully-closed mouth is pose n, for example).
# "" = the un-suffixed base image.
VISEME_MAP: Dict[str, Dict[str, str]] = {
    "chintu": {
        "REST": "",            # closed relaxed smile
        "BILABIAL": "g",       # pressed-closed lips
        "OPEN_A": "e",         # wide-open laugh
        "MID_E": "b",          # mid-open spread, upper teeth
        "DENTAL": "c",         # slightly-open spread, teeth together
        "CLOSED_I": "i",       # wide smile, minimal gap
        "LABIODENTAL": "k",    # upper teeth over lower lip
        "ROUNDED_LAX": "f",    # round "O" mouth
    },
    "gudiya": {
        "REST": "n",           # closed smile (only closed mouth in set)
        "BILABIAL": "n",       # same source — still a true lip closure
        "OPEN_A": "e",
        "MID_E": "k",
        "DENTAL": "h",
        "CLOSED_I": "a",
        "LABIODENTAL": "j",
        "ROUNDED_LAX": "f",
    },
}


def _src(prefix: str, letter: str) -> str:
    return os.path.join(POSES_DIR, f"{prefix}{letter}.png")


def _clean_pngs(d: str) -> None:
    """Remove previously staged sprites so mapping changes never leave
    stale files behind."""
    if not os.path.isdir(d):
        return
    for f in os.listdir(d):
        if f.lower().endswith(".png"):
            os.remove(os.path.join(d, f))


def stage_character(prefix: str, name: str) -> bool:
    char_dir = os.path.join(settings.CHARACTERS_DIR, name)
    poses_out = os.path.join(char_dir, "poses")
    visemes_out = os.path.join(char_dir, "visemes_src")
    os.makedirs(poses_out, exist_ok=True)
    os.makedirs(visemes_out, exist_ok=True)
    _clean_pngs(poses_out)
    _clean_pngs(visemes_out)

    ok = True

    # 1. body.png ← neutral standing pose
    neutral = _src(prefix, NEUTRAL_LETTER)
    if os.path.exists(neutral):
        shutil.copyfile(neutral, os.path.join(char_dir, "body.png"))
        print(f"  [Stage] {name}: body.png ← {os.path.basename(neutral)}")
    else:
        print(f"  [Stage] {name}: MISSING neutral pose {neutral}")
        ok = False

    # 2. Gesture poses for the PoseLibrary
    staged: List[str] = []
    for letter, pose_name in POSE_MAP.items():
        src = _src(prefix, letter)
        if not os.path.exists(src):
            print(f"  [Stage] {name}: pose '{pose_name}' missing ({src})")
            continue
        shutil.copyfile(src, os.path.join(poses_out, f"{pose_name}.png"))
        staged.append(pose_name)
    print(f"  [Stage] {name}: {len(staged)} poses ({', '.join(staged)})")

    # 3. Real-art viseme sources for the Rig Builder
    vmap = VISEME_MAP.get(name, {})
    baked: List[str] = []
    for viseme, letter in vmap.items():
        if viseme not in VISEME_NAMES:
            print(f"  [Stage] {name}: unknown viseme '{viseme}' — skipped")
            continue
        src = _src(prefix, letter)
        if not os.path.exists(src):
            print(f"  [Stage] {name}: viseme {viseme} missing ({src})")
            continue
        shutil.copyfile(src, os.path.join(visemes_out, f"{viseme}.png"))
        baked.append(viseme)
    print(f"  [Stage] {name}: {len(baked)} viseme sources "
          f"({', '.join(baked)})")

    return ok and bool(staged) and bool(baked)


def stage_all() -> bool:
    if not os.path.isdir(POSES_DIR):
        print(f"  [Stage] No pose directory at {POSES_DIR}")
        return False
    ok = True
    for prefix, name in CHARACTER_MAP.items():
        ok = stage_character(prefix, name) and ok
    return ok


if __name__ == "__main__":
    sys.exit(0 if stage_all() else 1)
