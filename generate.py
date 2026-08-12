"""
JEEVidya V5 — Pipeline Entry Point
══════════════════════════════════
Thin façade over the jvmake build graph (pipeline/buildgraph.py).
The video is a content-addressed DAG: TTS → mix / per-turn frame
segments → concat+mux. Unchanged inputs are cache hits; a crash
resumes exactly where it died. Used by the CLI, jvmake, and app.py.
"""
import json
import os
import shutil
import sys
from typing import Any, Dict

from config import settings

# Configure pydub to use imageio_ffmpeg's bundled binary
try:
    import imageio_ffmpeg
    import pydub
    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    pydub.AudioSegment.converter = _ffmpeg_exe
    pydub.AudioSegment.ffprobe = _ffmpeg_exe
except ImportError:
    pass

from config.prompts import EXAMPLE_DIALOGUE  # noqa: E402


def clean_temp_dir() -> None:
    """Wipe temp directory for a fresh run (cache in .cache survives)."""
    if os.path.exists(settings.TEMP_DIR):
        shutil.rmtree(settings.TEMP_DIR)
    os.makedirs(settings.TEMP_DIR, exist_ok=True)


def run_dialogue_pipeline(dialogue: Dict[str, Any],
                          progress_callback=None,
                          preview: bool = False,
                          force: bool = False) -> str:
    """
    Render a dialogue JSON to MP4 via the jvmake DAG.

    Incremental by construction:
      • change one dialogue line → only that turn's TTS + segment rebuild
      • crash mid-render         → next run resumes at the dead segment
      • nothing changed          → the final MP4 is itself a cache hit

    Args:
        dialogue: The dialogue JSON dict (see config/prompts.py for schema)
        progress_callback: Optional callable(stage, percent, message)
        preview: Half-resolution, first-6-seconds fast path
        force: Rebuild every node even on cache hits

    Returns:
        Path to the final MP4 in output/
    """
    from pipeline.buildgraph import VideoBuild
    return VideoBuild(dialogue, preview=preview).run(
        progress_callback=progress_callback, force=force)


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Gudiya & Chintu Animated Shorts Generator")
    parser.add_argument("--input", "-i", type=str, help="Path to dialogue JSON file")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run with example dialogue for testing")
    parser.add_argument("--preview", action="store_true",
                        help="Fast preview: first 6 seconds at half resolution")

    args = parser.parse_args()

    if args.test_mode:
        dialogue = EXAMPLE_DIALOGUE
        print("\n[Test Mode] Using example dialogue.\n")
    elif args.input:
        with open(args.input, 'r') as f:
            dialogue = json.load(f)
    else:
        print("Error: Provide --input <dialogue.json> or --test-mode")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  GUDIYA & CHINTU — Animated Shorts Factory")
    print(f"  Topic: {dialogue.get('title', 'Unknown')}")
    print(f"  Turns: {len(dialogue.get('turns', []))}")
    print(f"{'=' * 60}\n")

    output = run_dialogue_pipeline(dialogue, preview=args.preview)

    print(f"\n{'=' * 60}")
    print("  ✓ SUCCESS!")
    print(f"  Output: {output}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
