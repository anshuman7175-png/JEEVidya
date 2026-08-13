"""
JEEVidya — Art Inventory (jvmake art)
═════════════════════════════════════
The single truth-teller for art assets. Never guesses.

  SOURCE ART   every entry in config/mouth_art_manifest.json is verified
               byte-for-byte: file exists, sha256 matches, PNG pixel
               dimensions match the recorded width/height.
  DERIVED ART  per character, reports completeness of everything the
               pipeline builds FROM the source art:
                 body        assets/characters/<c>/body.png (matted)
                 pose        assets/characters/<c>/poses/neutral.png
                 visemes_src assets/characters/<c>/visemes_src/<V>.png
                             for every mapped viseme class
                 rig plates  assets/characters/<c>/rig/visemes/<V>.png
                             (only checked when a rig exists)

Exit code is non-zero when ANY source entry fails verification, and the
report names every missing/corrupt file exactly. Derived-art gaps are
reported (with the command that rebuilds them) but only fail the run
with --strict, because a fresh clone legitimately has no derived files.

Usage:
  python3 jvmake.py art [--strict]
  python3 -m tools.art_inventory [--strict]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(REPO_ROOT, "config", "mouth_art_manifest.json")

# Viseme classes that map to real mouth shapes; SPARE/reserved letters
# are alternates and are NOT required as derived files.
_NON_DERIVED_CLASSES = {"SPARE"}


def load_manifest(path: str = MANIFEST_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_size(path: str) -> Optional[Tuple[int, int]]:
    """Read (width, height) from the PNG IHDR chunk — stdlib only."""
    try:
        with open(path, "rb") as f:
            header = f.read(24)
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if header[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", header[16:24])
    return int(w), int(h)


@dataclass
class InventoryReport:
    source_ok: List[str] = field(default_factory=list)
    source_bad: List[Tuple[str, str]] = field(default_factory=list)  # (file, why)
    derived_missing: Dict[str, List[str]] = field(default_factory=dict)
    derived_present: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def source_passed(self) -> bool:
        return not self.source_bad

    @property
    def derived_complete(self) -> bool:
        return not any(self.derived_missing.values())


def _verify_source(manifest: dict, report: InventoryReport) -> None:
    """Byte-for-byte verification of every manifest asset entry."""
    for asset in manifest.get("assets", []):
        rel = asset["file"]
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            report.source_bad.append((rel, "MISSING"))
            continue
        actual = sha256_file(path)
        if actual != asset["sha256"]:
            report.source_bad.append(
                (rel, f"sha256 mismatch (expected {asset['sha256'][:12]}…, "
                      f"got {actual[:12]}…) — re-fetch from source_url"))
            continue
        size = png_size(path)
        want = (asset.get("width"), asset.get("height"))
        if None not in want and size is not None and size != want:
            report.source_bad.append(
                (rel, f"dimensions {size[0]}x{size[1]} != manifest "
                      f"{want[0]}x{want[1]}"))
            continue
        report.source_ok.append(rel)


def _check_derived(manifest: dict, report: InventoryReport) -> None:
    """Per-character body/pose/viseme completeness — exact filenames."""
    from config import settings

    mapping = manifest.get("viseme_mapping", {})
    for character in sorted(manifest.get("characters", {})):
        char_dir = os.path.join(settings.CHARACTERS_DIR, character)
        missing: List[str] = []
        present: List[str] = []

        for rel in ("body.png", os.path.join("poses", "neutral.png")):
            p = os.path.join(char_dir, rel)
            (present if os.path.exists(p) else missing).append(p)

        classes = sorted({v for v in mapping.get(character, {}).values()
                          if v not in _NON_DERIVED_CLASSES})
        for viseme in classes:
            p = os.path.join(char_dir, "visemes_src", f"{viseme}.png")
            (present if os.path.exists(p) else missing).append(p)

        # Rig viseme plates: only meaningful once a rig has been baked.
        rig_json = os.path.join(char_dir, "rig", "rig.json")
        if os.path.exists(rig_json):
            for viseme in classes:
                p = os.path.join(char_dir, "rig", "visemes", f"{viseme}.png")
                (present if os.path.exists(p) else missing).append(p)

        report.derived_missing[character] = missing
        report.derived_present[character] = present


def run_inventory() -> InventoryReport:
    manifest = load_manifest()
    report = InventoryReport()
    _verify_source(manifest, report)
    _check_derived(manifest, report)
    return report


def print_report(report: InventoryReport) -> None:
    print("\n═══ Art Inventory ═══\n")
    print(f"  SOURCE ART ({len(report.source_ok)} verified, "
          f"{len(report.source_bad)} bad)")
    for rel in report.source_ok:
        print(f"    ✓ {rel}")
    for rel, why in report.source_bad:
        print(f"    ✗ {rel} — {why}")

    for character, missing in report.derived_missing.items():
        present = report.derived_present.get(character, [])
        print(f"\n  DERIVED ({character}): {len(present)} present, "
              f"{len(missing)} missing")
        for p in missing:
            print(f"    ✗ missing: {os.path.relpath(p, REPO_ROOT)}")
        if missing:
            print("    → rebuild: python3 jvmake.py stage && "
                  "python3 jvmake.py rig --force")

    if report.source_passed:
        print("\n  Source art: ALL VERIFIED (byte-for-byte)")
    else:
        names = ", ".join(rel for rel, _ in report.source_bad)
        print(f"\n  Source art FAILED verification: {names}")
        print("  → re-fetch: python3 -m tools.fetch_mouth_art")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify source mouth art + report derived completeness")
    parser.add_argument("--strict", action="store_true",
                        help="also fail on missing DERIVED files")
    args = parser.parse_args(argv)
    report = run_inventory()
    print_report(report)
    if not report.source_passed:
        return 1
    if args.strict and not report.derived_complete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
