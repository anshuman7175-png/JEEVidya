"""
JEEVidya V5 — .jvproj Project Bundles (Tier 5)
══════════════════════════════════════════════
One file = one video's complete identity: script + DNA + seed + the
code/asset fingerprints it was rendered under. Because Tier 0 keys are
content-addressed, opening a bundle months later either re-renders
BIT-IDENTICALLY (fingerprints match → every node is a cache hit) or
tells you exactly which layer changed.

Fork-and-tweak: load, edit one line, save-as — the DAG rebuilds only
what the edit touched.
"""
from __future__ import annotations

import json
import os
import time
import zipfile
from typing import Any, Dict, Optional

from config import settings

JVPROJ_VERSION = 1
PROJECTS_DIR = os.path.join(settings.PROJECT_ROOT, "projects")


def save_project(dialogue: Dict[str, Any],
                 path: Optional[str] = None) -> str:
    """Freeze a dialogue (with its DNA) into a .jvproj bundle."""
    from engine.visual_dna import VisualDNA
    from pipeline.buildgraph import assets_fingerprint, code_fingerprint

    dna = VisualDNA.from_dialogue(dialogue)
    dialogue = dict(dialogue)
    dialogue["dna"] = dna.to_dict()            # pin the genome forever

    if path is None:
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        slug = "".join(c if c.isalnum() else "_"
                       for c in dialogue.get("title", "untitled"))[:48]
        path = os.path.join(PROJECTS_DIR, f"{slug}.jvproj")

    manifest = {
        "jvproj_version": JVPROJ_VERSION,
        "created_at": time.time(),
        "title": dialogue.get("title", "Untitled"),
        "dna_seed": dna.seed,
        "code_fingerprint": code_fingerprint(),
        "assets_fingerprint": assets_fingerprint(),
        "fps": settings.FPS,
        "canvas": [settings.WIDTH, settings.HEIGHT],
    }

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("script.json",
                   json.dumps(dialogue, indent=2, ensure_ascii=False))
        z.writestr("dna.json",
                   json.dumps(dna.to_dict(), indent=2, ensure_ascii=False))
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    print(f"  [Project] ✓ {path}")
    return path


def load_project(path: str) -> Dict[str, Any]:
    """Open a bundle; returns the dialogue (DNA embedded), and reports
    honestly whether a re-render will be bit-identical."""
    from pipeline.buildgraph import assets_fingerprint, code_fingerprint

    with zipfile.ZipFile(path, "r") as z:
        dialogue = json.loads(z.read("script.json").decode("utf-8"))
        manifest = json.loads(z.read("manifest.json").decode("utf-8"))

    drift = []
    if manifest.get("code_fingerprint") != code_fingerprint():
        drift.append("renderer code")
    if manifest.get("assets_fingerprint") != assets_fingerprint():
        drift.append("character assets")
    if drift:
        print(f"  [Project] note: {' + '.join(drift)} changed since this "
              "bundle was frozen — re-render will be equivalent, not "
              "bit-identical")
    else:
        print("  [Project] fingerprints match — re-render is bit-identical "
              "(and fully cached if the cache survives)")
    dialogue["_jvproj_manifest"] = manifest
    return dialogue
