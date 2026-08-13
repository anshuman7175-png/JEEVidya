#!/usr/bin/env python3
"""
jvmake — The JEEVidya Factory CLI (V5)
══════════════════════════════════════
One command surface for the whole studio.

  jvmake setup [--force]           Fresh-clone bootstrap: stage+rig+font+forge+doctor
  jvmake doctor                    Full environment + asset health report
  jvmake rig [character] [--force] Tier 1: build skeletal puppet rig(s) + v3 bake
  jvmake render script.json        Incremental DAG render to MP4 (--force to rebuild)
  jvmake preview script.json       6s half-res preview (~seconds, not minutes)
  jvmake graph script.json         Show the build DAG + cache status (no build)
  jvmake dna "title"               Tier 2: show a title's Visual DNA genome
  jvmake forge [--force|--motifs]  Tier 2: synthesize the SFX/BGM library
  jvmake script "topic here"       Tier 3: Director Agent (draft→critique→revise)
  jvmake critic video.mp4          Tier 3: vision QC review → defect report
  jvmake gauntlet video.mp4        Tier 3: adversarial TEMPORAL QC gates
  jvmake batch [topics.txt]        Tier 4: overnight factory (resumable)
  jvmake publish [bundle_dir]      Tier 4: upload / export ready-to-post bundles
  jvmake flywheel [--pull]         Tier 4: bandit report + gene recommendation
  jvmake localize script.json      Tier 4: language variants (en/ta/te)
  jvmake bundle script.json        Tier 5: freeze a .jvproj project bundle
  jvmake verify-face [character]   Rendered-pixel face QC sweep (hard gate)
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

# Wire pydub to the bundled imageio-ffmpeg binary so every subcommand
# (forge, render, …) works on machines with no system ffmpeg install.
try:
    import imageio_ffmpeg as _ioff
    import pydub as _pydub
    _ffmpeg = _ioff.get_ffmpeg_exe()
    _pydub.AudioSegment.converter = _ffmpeg
    _pydub.AudioSegment.ffprobe = _ffmpeg
    os.environ.setdefault("FFMPEG_BINARY", _ffmpeg)
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
            # Rig v3 is the render gate (Part III): a pre-v3 rig is
            # refused by require_v3() rather than rendered wrong.
            if rig.is_v3():
                check(f"character: {name} rig v3", True,
                      f"{len(rig.poses)} registered pose(s), "
                      f"worst RMS {rig.worst_pose_rms():.2f}px, "
                      f"{len(rig.mouth_targets)} fitted mouth target(s)")
            else:
                check(f"character: {name} rig v3", False,
                      f"version {rig.version} — rendering is REFUSED. "
                      f"Run: python3 jvmake.py rig {name} --force")
        else:
            check(f"character: {name} puppet rig", False,
                  "run: python3 jvmake.py rig — characters stay static "
                  "without it", warn=True)

    # OpenCV variant (mediapipe drags in the GUI build, which breaks
    # headless servers with libxcb/libGL ImportErrors)
    try:
        import cv2  # noqa: F401
        check("dep: cv2 (headless-safe)", True)
    except ImportError as e:
        if "libxcb" in str(e) or "libGL" in str(e):
            check("dep: cv2 (headless-safe)", False,
                  "GUI OpenCV installed on a headless server — fix: "
                  "pip uninstall -y opencv-contrib-python && "
                  "pip install --force-reinstall opencv-contrib-python-headless")
        else:
            check("dep: cv2 (headless-safe)", False, str(e), warn=True)

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


# ───────────────────────���─────────────────────
# render / preview / test
# ─────────────────────────────────────────────

def _face_gate(skip: bool) -> int:
    """Run the rendered-pixel face verification sweep before any encode.

    'Perfect' is not a claim — it is the only state that can produce an
    MP4. Every core element (mouth, eyes, head, pose movement) is
    re-detected on rendered pixels and compared against the renderer's
    own math; any gate failure refuses the encode. `--skip-face-qc` is
    the explicit, logged escape hatch for iteration."""
    if skip:
        print("  ⚠ face QC SKIPPED (--skip-face-qc) — output is unverified")
        return 0
    from engine.rig import has_rig
    chars = [c for c in ("gudiya", "chintu") if has_rig(c)]
    if not chars:
        print("  ⚠ face QC: no rigs found — nothing to verify "
              "(run `python3 jvmake.py rig`)")
        return 0
    from tools.verify_face import run_all
    out_dir = os.path.join(settings.OUTPUT_DIR, "face_qc")
    print("  face QC: rendering verification sweep "
          f"({', '.join(chars)}) …")
    passed, reports = run_all(chars, out_dir)
    for name, rep in reports.items():
        print(f"\n  ─ {name} ─")
        print("  " + rep.summary().replace("\n", "\n  "))
    print(f"\n  → report: {os.path.join(out_dir, 'face_qc_report.json')}")
    if not passed:
        print("\n  ✗ FACE QC FAILED — encode refused. Inspect the failure "
              "strips under output/face_qc/<character>/failures/ and fix "
              "the rig/art, or re-run with --skip-face-qc to iterate.\n")
        return 1
    print("  ✓ ALL FACE GATES PASS\n")
    return 0


def _render(dialogue: dict, preview: bool, force: bool = False,
            skip_face_qc: bool = False) -> int:
    rc = _face_gate(skip_face_qc)
    if rc != 0:
        return rc
    from generate import run_dialogue_pipeline
    output = run_dialogue_pipeline(dialogue, preview=preview, force=force)
    print(f"\n✓ Output: {output}\n")
    return 0


def cmd_render(args) -> int:
    with open(args.script, "r", encoding="utf-8") as f:
        dialogue = json.load(f)
    return _render(dialogue, preview=args.preview,
                   force=getattr(args, "force", False),
                   skip_face_qc=getattr(args, "skip_face_qc", False))


def cmd_preview(args) -> int:
    args.preview = True
    return cmd_render(args)


def cmd_test(args) -> int:
    from config.prompts import EXAMPLE_DIALOGUE
    return _render(EXAMPLE_DIALOGUE, preview=args.preview,
                   force=getattr(args, "force", False),
                   skip_face_qc=getattr(args, "skip_face_qc", False))


def cmd_verify_face(args) -> int:
    """Deterministic verification sweep on the real rig(s): every baked
    viseme class held, a full blink, one pose transition with per-frame
    mouth lock, and a synthetic speech line — with the FULL face_qc gate
    suite run on the rendered frames. Non-zero exit on any failure."""
    from engine.rig import has_rig
    if args.character:
        chars = [args.character]
    else:
        chars = [c for c in ("gudiya", "chintu") if has_rig(c)]
    if not chars:
        print("\n  No rigs found — run `python3 jvmake.py rig` first.\n")
        return 1
    from tools.verify_face import run_all
    out_dir = args.out or os.path.join(settings.OUTPUT_DIR, "face_qc")
    print("\n═══ Face Verification Sweep (rendered-pixel gates) ═══")
    passed, reports = run_all(chars, out_dir, speech_s=args.speech_s)
    for name, rep in reports.items():
        print(f"\n─ {name} ─")
        print(rep.summary())
    print(f"\n→ report: {os.path.join(out_dir, 'face_qc_report.json')}")
    if not passed:
        print("✗ FACE QC FAILED — failure strips under "
              f"{out_dir}/<character>/failures/\n")
        return 1
    print("✓ ALL FACE GATES PASS — every core element measured on "
          "rendered pixels.\n")
    return 0


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
# setup (Tier 0 — one-shot fresh-clone bootstrap)
# ─────────────────────────────────────────────

# Bundled caption font: assets/ is gitignored, so every fresh clone loses
# it. Candidates are tried in order; all serve the same OFL-licensed file.
_DEVANAGARI_FONT_URLS = (
    "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
    "NotoSansDevanagari/NotoSansDevanagari-Regular.ttf",
    "https://github.com/notofonts/devanagari/raw/main/fonts/"
    "NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf",
)


def _ensure_devanagari_font(force: bool = False) -> bool:
    """Download the bundled Noto Sans Devanagari if no capable font is
    resolvable. Returns True when a usable font is present afterwards."""
    from engine.render_fast import devanagari_font_path
    if devanagari_font_path() and not force:
        return True
    dest = os.path.join(settings.FONTS_DIR, "NotoSansDevanagari-Regular.ttf")
    import urllib.request
    for url in _DEVANAGARI_FONT_URLS:
        try:
            print(f"  ↓ font: {url.split('/')[-1]}")
            urllib.request.urlretrieve(url, dest)
            if os.path.getsize(dest) > 50_000:  # sanity: not an error page
                return True
        except Exception as e:
            print(f"    failed ({e}); trying next mirror")
    if os.path.exists(dest) and os.path.getsize(dest) <= 50_000:
        os.remove(dest)
    return devanagari_font_path() is not None


def _repair_opencv_headless() -> bool:
    """mediapipe pins the GUI opencv-contrib-python, which dies on headless
    servers ('libxcb.so.1: cannot open shared object file'). Detect that
    exact failure and swap in the headless build automatically."""
    try:
        import cv2  # noqa: F401
        return True
    except ImportError as e:
        if "libxcb" not in str(e) and "libGL" not in str(e):
            print(f"    ✗ cv2 import failed for another reason: {e}")
            return False
    import subprocess
    print("    GUI OpenCV on a headless server — swapping to headless build")
    pip = [sys.executable, "-m", "pip"]
    subprocess.run(pip + ["uninstall", "-q", "-y", "opencv-contrib-python"],
                   check=False)
    subprocess.run(pip + ["install", "-q", "--force-reinstall",
                          "opencv-contrib-python-headless"], check=False)
    # cv2 was already partially imported above; a clean check needs a
    # fresh interpreter.
    rc = subprocess.run([sys.executable, "-c", "import cv2"],
                        capture_output=True).returncode
    return rc == 0


def cmd_setup(args) -> int:
    """Idempotent fresh-clone bootstrap: everything under assets/ except
    assets/poses/ is gitignored, so a new checkout has no staged
    characters, rigs, caption font, or SFX. One command rebuilds all of
    it, then runs doctor as the gate."""
    force = getattr(args, "force", False)
    failures = []
    print("\n═══ Tier 0 · Setup (fresh-clone bootstrap) ═══\n")

    # 0 · OpenCV headless repair (must run before the rig builder, which
    # imports cv2 via mediapipe)
    print("[0/4] opencv variant check")
    if _repair_opencv_headless():
        print("    ✓ cv2 imports cleanly")
    else:
        failures.append("opencv")
        print("    ✗ cv2 broken — rigs will fall back to the heuristic")

    # 1 · Stage pose assets → character dirs (skip if already staged)
    need_stage = force or not all(
        os.path.exists(os.path.join(settings.CHARACTERS_DIR, c, "body.png"))
        for c in ("gudiya", "chintu"))
    if need_stage:
        from tools.pose_stager import stage_all
        print("[1/4] staging pose assets")
        if not stage_all():
            failures.append("stage")
    else:
        print("[1/4] characters already staged — skip")

    # 2 · Puppet rigs (mediapipe face landmarks; heuristic fallback)
    from engine.rig import has_rig
    need_rig = force or not all(has_rig(c) for c in ("gudiya", "chintu"))
    if need_rig:
        from tools.rig_builder import build_all
        print("[2/4] building puppet rigs")
        if not build_all(force=True):
            failures.append("rig")
    else:
        print("[2/4] rigs already built — skip")

    # 3 · Devanagari caption font
    print("[3/4] caption font")
    if _ensure_devanagari_font(force=False):
        print("    ✓ Devanagari font available")
    else:
        failures.append("font")
        print("    ✗ no Devanagari font — Hindi captions will render as boxes")

    # 4 · SFX/BGM library
    sfx_files = (os.listdir(settings.SFX_DIR)
                 if os.path.isdir(settings.SFX_DIR) else [])
    if force or not sfx_files:
        from tools.audio_forge import forge_library
        print("[4/4] forging SFX/BGM library")
        try:
            forge_library(force=force)
        except Exception as e:
            failures.append("forge")
            print(f"    ✗ forge failed: {e}")
    else:
        print(f"[4/4] SFX library present ({len(sfx_files)} files) — skip")

    print()
    doctor_rc = cmd_doctor(args)
    if failures:
        print(f"Setup finished with failures: {', '.join(failures)}\n")
        return 1
    return doctor_rc


# ───────────────────���─────────────────────────
# stage (Tier 1 — pose asset staging)
# ───────────────────────────────────��─────────

def cmd_stage(_args) -> int:
    from tools.pose_stager import stage_all
    print("\n═══ Tier 1 · Pose Stager ═══\n")
    ok = stage_all()
    if ok:
        print("\n  Next:  python3 jvmake.py rig --force\n")
    return 0 if ok else 1


# ─────────────────────────────────────────────
# art (source-art integrity + derived completeness)
# ─────────────────────────────────────────────

def cmd_art(args) -> int:
    """sha256-verify every committed source-art file against the
    manifest and report derived body/pose/viseme completeness per
    character. Exits non-zero with exact filenames — never guesses."""
    from tools.art_inventory import main as art_main
    argv = ["--strict"] if getattr(args, "strict", False) else []
    return art_main(argv)


# ─────────────────────────────────────────────
# rig (Tier 1 — Bone Engine puppets)
# ─────────────────────────────────────────────

def cmd_rig(args) -> int:
    from tools.rig_builder import build_rig, build_all
    print("\n═══ Tier 1 · Puppet Rig Builder ═══\n")
    v3 = getattr(args, "v3", True)
    if args.character:
        rigs = [build_rig(args.character, force=args.force, v3=v3)]
    else:
        rigs = build_all(force=args.force, v3=v3)
        if not rigs:
            print("  No characters found — add images under "
                  f"{settings.CHARACTERS_DIR}/<name>/body.png")
            return 1
    stale = [r.character for r in rigs if not r.is_v3()]
    if stale:
        print(f"\n  NOT RENDERABLE (pre-v3): {', '.join(stale)}")
        print("  The render path refuses a pre-v3 rig rather than placing "
              "the face with body.png's boxes. Install mediapipe, confirm "
              "each body.png shows a detectable face, then re-run "
              "`python3 jvmake.py rig --force`.")
    print("\n  Nudge joints visually:  python3 app.py → /studio\n")
    return 1 if stale else 0


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

def cmd_gauntlet(args) -> int:
    """Adversarial temporal QC (Part XXI) — the gates that only exist
    between adjacent frames: flicker, freeze, teleport, jitter,
    letterbox, chroma drift."""
    from tools.gauntlet import BURST_FRAMES, BURSTS, run
    print("\n═══ Tier 3 · Adversarial Gauntlet ═══\n")
    report = run(args.video,
                 bursts=args.bursts if args.bursts is not None else BURSTS,
                 burst_frames=(args.burst_frames if args.burst_frames
                               is not None else BURST_FRAMES))
    print(report.summary() + "\n")
    if args.report:
        report.save(args.report)
        print(f"  → {args.report}\n")
    return 0 if report.passed else 1


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

    p = sub.add_parser("setup", help="fresh-clone bootstrap: stage + rig + "
                                     "font + forge + doctor (idempotent)")
    p.add_argument("--force", action="store_true",
                   help="rebuild even if assets already exist")

    p = sub.add_parser("render", help="incremental DAG render of a dialogue JSON")
    p.add_argument("script")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="rebuild every node, ignoring cache hits")
    p.add_argument("--skip-face-qc", action="store_true",
                   help="skip the rendered-pixel face gate (unverified output)")

    p = sub.add_parser("preview", help="6s half-res preview render")
    p.add_argument("script")
    p.add_argument("--force", action="store_true",
                   help="rebuild every node, ignoring cache hits")
    p.add_argument("--skip-face-qc", action="store_true",
                   help="skip the rendered-pixel face gate (unverified output)")

    p = sub.add_parser("verify-face",
                       help="rendered-pixel face QC sweep: every gate in "
                            "tools/face_qc.py run on real rendered frames")
    p.add_argument("character", nargs="?", default=None)
    p.add_argument("--out", default=None,
                   help="report directory (default: output/face_qc)")
    p.add_argument("--speech-s", type=float, default=6.0,
                   help="synthetic speech line length in seconds")

    p = sub.add_parser("graph", help="show the build DAG + cache status (no build)")
    p.add_argument("script")
    p.add_argument("--preview", action="store_true",
                   help="inspect the preview-variant graph")

    p = sub.add_parser("test", help="render the built-in example dialogue")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="rebuild every node, ignoring cache hits")
    p.add_argument("--skip-face-qc", action="store_true",
                   help="skip the rendered-pixel face gate (unverified output)")

    p = sub.add_parser("script", help="Director Agent: topic → script JSON")
    p.add_argument("topic")
    p.add_argument("-o", "--output", default=None)

    sub.add_parser("stage", help="stage assets/poses into character dirs (Tier 1)")

    p = sub.add_parser("art", help="verify source mouth art (sha256) + "
                                   "derived asset completeness")
    p.add_argument("--strict", action="store_true",
                   help="also fail on missing derived files")

    p = sub.add_parser("rig", help="build skeletal puppet rig(s) (Tier 1)")
    p.add_argument("character", nargs="?", default=None)
    p.add_argument("--force", action="store_true", help="rebuild existing rigs")
    p.add_argument("--no-v3", dest="v3", action="store_false",
                   help="skip the v3 bake (head plate, pose registration, "
                        "mouth targets) — the result is NOT renderable")

    p = sub.add_parser("dna", help="show a title's Visual DNA genome (Tier 2)")
    p.add_argument("title")

    p = sub.add_parser("forge", help="synthesize the SFX/BGM library (Tier 2)")
    p.add_argument("--force", action="store_true", help="overwrite existing")
    p.add_argument("--motifs", action="store_true",
                   help="render the motif contact sheet instead")

    p = sub.add_parser("critic", help="vision QC review of a video (Tier 3)")
    p.add_argument("video")

    p = sub.add_parser("gauntlet",
                       help="adversarial temporal QC of a video (Tier 3)")
    p.add_argument("video")
    p.add_argument("--bursts", type=int, default=None,
                   help="contiguous-frame bursts sampled across the runtime")
    p.add_argument("--burst-frames", type=int, default=None,
                   help="frames decoded per burst")
    p.add_argument("--report", default=None, help="write the report JSON here")

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
        "setup": cmd_setup,
        "render": cmd_render,
        "preview": cmd_preview,
        "graph": cmd_graph,
        "test": cmd_test,
        "verify-face": cmd_verify_face,
        "script": cmd_script,
        "stage": cmd_stage,
        "art": cmd_art,
        "rig": cmd_rig,
        "dna": cmd_dna,
        "forge": cmd_forge,
        "critic": cmd_critic,
        "gauntlet": cmd_gauntlet,
        "batch": cmd_batch,
        "publish": cmd_publish,
        "flywheel": cmd_flywheel,
        "localize": cmd_localize,
        "bundle": cmd_bundle,
        "clean": cmd_clean,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
