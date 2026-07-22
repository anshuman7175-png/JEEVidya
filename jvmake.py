#!/usr/bin/env python3
"""
jvmake — The JEEVidya Factory CLI (V5)
══════════════════════════════════════
One command surface for the whole studio.

  jvmake doctor                    Full environment + asset health report
  jvmake rig [character] [--force] Tier 1: build skeletal puppet rig(s)
  jvmake render script.json        Incremental DAG render to MP4 (--force to rebuild)
  jvmake preview script.json       6s half-res preview (~seconds, not minutes)
  jvmake graph script.json         Show the build DAG + cache status (no build)
  jvmake dna "title"               Tier 2: show a title's Visual DNA genome
  jvmake forge [--force|--motifs]  Tier 2: synthesize the SFX/BGM library
  jvmake script "topic here"       Tier 3: Director Agent (draft→critique→revise)
  jvmake critic video.mp4          Tier 3: vision QC review → defect report
  jvmake batch [topics.txt]        Tier 4: overnight factory (resumable)
  jvmake publish [bundle_dir]      Tier 4: upload / export ready-to-post bundles
  jvmake flywheel [--pull]         Tier 4: bandit report + gene recommendation
  jvmake localize script.json      Tier 4: language variants (en/ta/te)
  jvmake bundle script.json        Tier 5: freeze a .jvproj project bundle
  jvmake test                      Render the built-in example dialogue
  jvmake clean [--cache]           Wipe .tmp (and optionally the build cache)

Tier 0 guarantees:
  • Edit one dialogue line → only that turn's TTS + one frame segment rebuild
  • Crash mid-render → the next run resumes at the segment that died
  • Nothing changed → the final MP4 is a cache hit (~1s "render")

Run with:  python3 jvmake.py <command>
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

from config import settings  # noqa: E402


# ─────────────────────────────────────────────
# doctor
# ─────────────────────────────────────────────

def cmd_doctor(_args) -> int:
    ok = True

    def check(label: str, passed: bool, detail: str = "", warn: bool = False):
        nonlocal ok
        mark = "✓" if passed else ("⚠" if warn else "✗")
        if not passed and not warn:
            ok = False
        print(f"  {mark} {label}" + (f" — {detail}" if detail else ""))

    print("\n═══ JEEVidya Doctor ═══\n")

    # Python deps
    for mod in ("PIL", "numpy", "pydub", "moviepy", "matplotlib"):
        try:
            __import__(mod)
            check(f"dep: {mod}", True)
        except ImportError as e:
            check(f"dep: {mod}", False, str(e))

    # ffmpeg
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        check("ffmpeg (bundled)", os.path.exists(exe), exe)
    except Exception as e:
        check("ffmpeg (bundled)", False, str(e))

    # edge-tts CLI (venv-aware)
    try:
        from pipeline.voice import resolve_edge_tts
        check("edge-tts CLI", True, resolve_edge_tts())
    except Exception as e:
        check("edge-tts CLI", False, str(e))

    # Devanagari captions
    from engine.render_fast import devanagari_font_path
    font = devanagari_font_path()
    check("Devanagari caption font", font is not None,
          font or "Hindi captions WILL render as boxes")

    # Character assets + Tier 1 puppet rigs
    from engine.rig import Rig, has_rig
    for name in ("gudiya", "chintu"):
        char_dir = os.path.join(settings.CHARACTERS_DIR, name)
        has_body = any(os.path.exists(os.path.join(char_dir, f"body{e}"))
                       for e in (".png", ".jpg"))
        check(f"character: {name}/body", has_body,
              "" if has_body else "run pipeline once or add original image")
        if has_rig(name):
            rig = Rig.load(name)
            n_vis = len([v for v in rig.visemes if not v.startswith("LID")])
            detail = f"{rig.generated_by}, {n_vis} visemes"
            if rig.generated_by == "heuristic":
                check(f"character: {name} puppet rig", True,
                      detail + " — verify joints in /studio", warn=True)
            else:
                check(f"character: {name} puppet rig", True, detail)
        else:
            check(f"character: {name} puppet rig", False,
                  "run: python3 jvmake.py rig — characters stay static "
                  "without it", warn=True)

    # mediapipe (auto-rigging quality)
    try:
        import mediapipe  # noqa: F401
        check("dep: mediapipe (auto-rig)", True)
    except ImportError:
        check("dep: mediapipe (auto-rig)", False,
              "rigs fall back to silhouette heuristic", warn=True)

    # Gemini key (Director Agent)
    check("GEMINI_API_KEY", bool(os.environ.get("GEMINI_API_KEY")),
          "set" if os.environ.get("GEMINI_API_KEY")
          else "Director Agent (jvmake script) unavailable", warn=True)

    # SFX library
    sfx = [f for f in os.listdir(settings.SFX_DIR)] if os.path.isdir(settings.SFX_DIR) else []
    check("SFX library", bool(sfx),
          f"{len(sfx)} files" if sfx else "empty → videos have no SFX/BGM "
          "(Tier 4 audio_forge synthesizes these)", warn=True)

    # Build cache
    from pipeline.cache import BuildCache
    stats = BuildCache().stats()
    check("build cache", True,
          f"{stats['files']} artifacts, {stats['bytes'] / 1e6:.1f} MB")

    print("\n" + ("All critical checks passed.\n" if ok
                  else "CRITICAL problems found — fix ✗ items before rendering.\n"))
    return 0 if ok else 1


# ─────────────────────────────────────────────
# render / preview / test
# ─────────────────────────────────────────────

def _render(dialogue: dict, preview: bool, force: bool = False) -> int:
    from generate import run_dialogue_pipeline
    output = run_dialogue_pipeline(dialogue, preview=preview, force=force)
    print(f"\n✓ Output: {output}\n")
    return 0


def cmd_render(args) -> int:
    with open(args.script, "r", encoding="utf-8") as f:
        dialogue = json.load(f)
    return _render(dialogue, preview=args.preview,
                   force=getattr(args, "force", False))


def cmd_preview(args) -> int:
    args.preview = True
    return cmd_render(args)


def cmd_test(args) -> int:
    from config.prompts import EXAMPLE_DIALOGUE
    return _render(EXAMPLE_DIALOGUE, preview=args.preview,
                   force=getattr(args, "force", False))


# ─────────────────────────────────────────────
# graph (Tier 0 — DAG inspection, builds nothing)
# ─────────────────────────────────────────────

def cmd_graph(args) -> int:
    with open(args.script, "r", encoding="utf-8") as f:
        dialogue = json.load(f)
    from pipeline.buildgraph import VideoBuild
    print("\n" + VideoBuild(dialogue, preview=args.preview).describe() + "\n")
    return 0


# ─────────────────────────────────────────────
# script (Tier 3 — Director Agent, multi-pass)
# ─────────────────────────────────────────────

def cmd_script(args) -> int:
    try:
        from agents.director import Director
        dialogue = Director().write(args.topic)
    except Exception as e:
        print(f"  Multi-pass Director unavailable ({e}); "
              "falling back to single-pass writer")
        from pipeline.scriptwriter import ScriptWriter
        dialogue = ScriptWriter().generate_dialogue(args.topic)

    out = args.output
    if not out:
        safe = "".join(c if c.isalnum() else "_" for c in args.topic)[:40]
        os.makedirs(os.path.join(settings.PROJECT_ROOT, "scenes"), exist_ok=True)
        out = os.path.join(settings.PROJECT_ROOT, "scenes", f"{safe}.json")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(dialogue, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Script saved: {out}")
    print(f"  Render it:  python3 jvmake.py render {out}")
    print(f"  Preview it: python3 jvmake.py preview {out}\n")
    return 0


# ─────────────────────────────────────────────
# rig (Tier 1 — Bone Engine puppets)
# ─────────────────────────────────────────────

def cmd_rig(args) -> int:
    from tools.rig_builder import build_rig, build_all
    print("\n═══ Tier 1 · Puppet Rig Builder ═══\n")
    if args.character:
        build_rig(args.character, force=args.force)
    else:
        rigs = build_all(force=args.force)
        if not rigs:
            print("  No characters found — add images under "
                  f"{settings.CHARACTERS_DIR}/<name>/body.png")
            return 1
    print("\n  Nudge joints visually:  python3 app.py → /studio\n")
    return 0


# ─────────────────────────────────────────────
# Tier 2 — dna / forge
# ─────────────────────────────────────────────

def cmd_dna(args) -> int:
    from engine.visual_dna import VisualDNA
    dna = VisualDNA.from_title(args.title)
    print("\n  " + dna.describe())
    p = dna.palette
    for name in ("bg_top", "bg_bottom", "primary", "secondary", "accent"):
        r, g, b = p[name]
        print(f"    {name:<10} #{r:02x}{g:02x}{b:02x}")
    print(f"    motifs     {', '.join(dna.motif_names)}"
          f" ×{dna.motif_count} on screen\n")
    return 0


def cmd_forge(args) -> int:
    print("\n═══ Tier 2 · Audio/Motif Forge ═══\n")
    if args.motifs:
        from tools.motif_forge import forge_sheet
        out = os.path.join(settings.OUTPUT_DIR, "motif_sheet.png")
        forge_sheet().save(out)
        print(f"  ✓ Motif contact sheet: {out}\n")
        return 0
    from tools.audio_forge import forge_library
    written = forge_library(force=args.force)
    forged = sum(1 for v in written.values() if v == "forged")
    print(f"\n  ✓ {forged} forged, {len(written) - forged} kept "
          f"→ {settings.SFX_DIR}\n")
    return 0


# ─────────────────────────────────────────────
# Tier 3 — critic
# ─────────────────────────────────────────────

def cmd_critic(args) -> int:
    from agents.critic import Critic
    report = Critic().review(args.video)
    print(f"\n  Verdict: {report.overall}")
    if report.summary:
        print(f"  {report.summary}")
    for fix in report.fixes:
        print(f"  → {fix['defect']} ({fix['severity']}): {fix['hint']}")
    print()
    return 0 if report.overall == "pass" else 1


# ─────────────────────────────────────────────
# Tier 4 — batch / publish / flywheel / localize
# ─────────────────────────────────────────────

def cmd_batch(args) -> int:
    from factory.batch import BatchFactory, read_topics
    topics = read_topics(args.topics) if args.topics else None
    result = BatchFactory(critic_enabled=not args.no_critic).run(topics)
    return 0 if result.get("failed", 0) == 0 else 1


def cmd_publish(args) -> int:
    from factory.publisher import Publisher, can_upload
    print(f"\n  Mode: {'YouTube upload' if can_upload() else 'bundle export'}")
    pub = Publisher()
    if args.bundle:
        pub.publish_bundle(args.bundle)
    else:
        pub.publish_all()
    return 0


def cmd_flywheel(args) -> int:
    from factory.flywheel import Flywheel
    fw = Flywheel()
    if args.pull:
        fw.pull_stats()
    print("\n" + fw.describe() + "\n")
    return 0


def cmd_localize(args) -> int:
    from factory.localizer import Localizer
    with open(args.script, "r", encoding="utf-8") as f:
        dialogue = json.load(f)
    langs = args.languages.split(",") if args.languages else ["en"]
    variants = Localizer().localize_all(dialogue, langs)
    base, _ = os.path.splitext(args.script)
    for lang, variant in variants.items():
        out = f"{base}.{lang}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(variant, f, indent=2, ensure_ascii=False)
        print(f"  ✓ {out}")
    return 0 if variants else 1


# ─────────────────────────────────────────────
# Tier 5 — bundle (.jvproj)
# ─────────────────────────────────────────────

def cmd_bundle(args) -> int:
    from pipeline.project import load_project, save_project
    if args.script.endswith(".jvproj"):
        dialogue = load_project(args.script)
        print(f"  Loaded: {dialogue.get('title')} "
              f"({len(dialogue.get('turns', []))} turns)")
        return 0
    with open(args.script, "r", encoding="utf-8") as f:
        dialogue = json.load(f)
    save_project(dialogue, args.output)
    return 0


# ─────────────────────────────────────────────
# clean
# ─────────────────────────────────────────────

def cmd_clean(args) -> int:
    if os.path.exists(settings.TEMP_DIR):
        shutil.rmtree(settings.TEMP_DIR)
        print(f"✓ Wiped {settings.TEMP_DIR}")
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    if args.cache:
        from pipeline.cache import BuildCache
        BuildCache().clear()
        print("✓ Wiped build cache")
    return 0


# ─────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(prog="jvmake",
                                     description="JEEVidya V5 factory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="environment + asset health report")

    p = sub.add_parser("render", help="incremental DAG render of a dialogue JSON")
    p.add_argument("script")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="rebuild every node, ignoring cache hits")

    p = sub.add_parser("preview", help="6s half-res preview render")
    p.add_argument("script")
    p.add_argument("--force", action="store_true",
                   help="rebuild every node, ignoring cache hits")

    p = sub.add_parser("graph", help="show the build DAG + cache status (no build)")
    p.add_argument("script")
    p.add_argument("--preview", action="store_true",
                   help="inspect the preview-variant graph")

    p = sub.add_parser("test", help="render the built-in example dialogue")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="rebuild every node, ignoring cache hits")

    p = sub.add_parser("script", help="Director Agent: topic → script JSON")
    p.add_argument("topic")
    p.add_argument("-o", "--output", default=None)

    p = sub.add_parser("rig", help="build skeletal puppet rig(s) (Tier 1)")
    p.add_argument("character", nargs="?", default=None)
    p.add_argument("--force", action="store_true", help="rebuild existing rigs")

    p = sub.add_parser("dna", help="show a title's Visual DNA genome (Tier 2)")
    p.add_argument("title")

    p = sub.add_parser("forge", help="synthesize the SFX/BGM library (Tier 2)")
    p.add_argument("--force", action="store_true", help="overwrite existing")
    p.add_argument("--motifs", action="store_true",
                   help="render the motif contact sheet instead")

    p = sub.add_parser("critic", help="vision QC review of a video (Tier 3)")
    p.add_argument("video")

    p = sub.add_parser("batch", help="overnight factory over topics.txt (Tier 4)")
    p.add_argument("topics", nargs="?", default=None)
    p.add_argument("--no-critic", action="store_true",
                   help="skip the vision QC loop")

    p = sub.add_parser("publish", help="upload / export bundles (Tier 4)")
    p.add_argument("bundle", nargs="?", default=None)

    p = sub.add_parser("flywheel", help="bandit report + recommendation (Tier 4)")
    p.add_argument("--pull", action="store_true", help="pull fresh YT stats first")

    p = sub.add_parser("localize", help="language variants of a script (Tier 4)")
    p.add_argument("script")
    p.add_argument("-l", "--languages", default="en",
                   help="comma list, e.g. en,ta,te")

    p = sub.add_parser("bundle", help="freeze/load a .jvproj bundle (Tier 5)")
    p.add_argument("script")
    p.add_argument("-o", "--output", default=None)

    p = sub.add_parser("clean", help="wipe temp dir")
    p.add_argument("--cache", action="store_true", help="also wipe build cache")

    args = parser.parse_args()
    return {
        "doctor": cmd_doctor,
        "render": cmd_render,
        "preview": cmd_preview,
        "graph": cmd_graph,
        "test": cmd_test,
        "script": cmd_script,
        "rig": cmd_rig,
        "dna": cmd_dna,
        "forge": cmd_forge,
        "critic": cmd_critic,
        "batch": cmd_batch,
        "publish": cmd_publish,
        "flywheel": cmd_flywheel,
        "localize": cmd_localize,
        "bundle": cmd_bundle,
        "clean": cmd_clean,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
