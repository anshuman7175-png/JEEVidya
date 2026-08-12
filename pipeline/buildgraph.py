"""
JEEVidya V5 — The Video Build Graph (Tier 0)
════════════════════════════════════════════
The whole video expressed as a content-addressed DAG:

    tts:NN (parallel) ──┐
                        ├──→ mix ─────────────┐
    tts:NN ─────────────┘                     ├──→ final ──→ output/*.mp4
    seg:NN (per-turn frames, deterministic) ──┘

Guarantees this module exists to provide:
  • Edit one dialogue line   → that line's TTS + that ONE segment rebuild.
    Every other artifact is a byte-identical cache hit. (Turn durations are
    quantized to exact frame multiples in voice.py, so neighbouring
    segments' frame counts never shift; segment motion is seeded from the
    content key, never the absolute frame position.)
  • Crash mid-render         → finished segments already live in .cache;
    the next run resumes at the exact segment that died.
  • Nothing changed          → even the final MP4 is a cache hit; a full
    "render" completes in about a second.
  • Edit the renderer code   → code_fingerprint() changes, every stale
    segment honestly invalidates. No haunted caches, ever.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from pipeline.cache import BuildCache, key_of
from pipeline.dag import Graph, Node

# ═══════════════════════════════════════════
# FINGERPRINTS — the honesty of the cache
# ═══════════════════════════════════════════

# Every source file that can change a rendered pixel or an encoded byte.
_CODE_DIRS = ("engine",)
_CODE_FILES = (
    "pipeline/camera.py",
    "pipeline/compositor.py",
    "pipeline/compositor_v5.py",
    "pipeline/lipsync.py",
    "pipeline/puppet.py",
    "pipeline/timeline.py",
    "pipeline/encoder.py",
    "config/brand.py",
    "config/settings.py",
    "tools/motif_forge.py",       # Tier 2: motifs paint pixels
)


def code_fingerprint() -> str:
    """sha256 over every renderer source file. Editing the engine safely
    invalidates every segment that the edit could have repainted."""
    h = hashlib.sha256()
    root = settings.PROJECT_ROOT
    paths: List[str] = []
    for d in _CODE_DIRS:
        full = os.path.join(root, d)
        if os.path.isdir(full):
            paths.extend(os.path.join(full, n) for n in sorted(os.listdir(full))
                         if n.endswith(".py"))
    paths.extend(os.path.join(root, f) for f in _CODE_FILES)
    for p in sorted(paths):
        if not os.path.exists(p):
            continue
        h.update(os.path.relpath(p, root).encode("utf-8"))
        with open(p, "rb") as f:
            h.update(f.read())
        h.update(b"\x1f")
    return h.hexdigest()


def _dir_fingerprint(base: str) -> str:
    """Cheap recursive fingerprint (relpath|size|mtime) of an asset tree."""
    h = hashlib.sha256()
    if os.path.isdir(base):
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames.sort()
            for name in sorted(filenames):
                p = os.path.join(dirpath, name)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                h.update(f"{os.path.relpath(p, base)}|{st.st_size}|"
                         f"{st.st_mtime_ns}".encode("utf-8"))
    return h.hexdigest()


def assets_fingerprint() -> str:
    """Character art + puppet rigs — anything pasted onto a frame."""
    return _dir_fingerprint(settings.CHARACTERS_DIR)


def sfx_fingerprint() -> str:
    """SFX/BGM library — anything mixed into the audio track."""
    return _dir_fingerprint(settings.SFX_DIR)


# ═══════════════════════════════════════════
# NODE BUILDERS
# ═══════════════════════════════════════════

def build_mixed_audio(turn_data: List[Dict[str, Any]], out_path: str,
                      dna=None) -> str:
    """
    Concatenate per-turn audio with exact-duration silences, overlay BGM.
    Silent turns (explanations, empty lines) contribute EXACTLY their
    timeline duration, so audio and frames stay in lockstep by design.

    BGM priority: assets/sfx/bgm_lofi.mp3 if present, else a per-DNA
    generative bed (Tier 2 audio_forge) sidechain-ducked under the voice.
    """
    from pipeline.lipsync import _load_audio
    from pipeline.sfx import SFXManager, generate_silence, mix_audio_layers

    segments = []
    for turn in turn_data:
        audio = turn.get("audio")
        if audio and os.path.exists(audio):
            segments.append(_load_audio(audio))
            padding_ms = turn.get("padding_ms", 0)
            if padding_ms > 0:
                segments.append(generate_silence(padding_ms))
        elif turn.get("duration_ms", 0) > 0:
            segments.append(generate_silence(turn["duration_ms"]))

    if not segments:
        segments = [generate_silence(1000)]

    combined = segments[0]
    for seg in segments[1:]:
        combined += seg

    bgm = SFXManager().get("bgm")
    if bgm is None and dna is not None:
        # Tier 2: synthesize this video's OWN bed from its genes
        try:
            from tools.audio_forge import (_to_segment, forge_bgm,
                                           sidechain_duck)
            bed = _to_segment(forge_bgm(dna, seconds=len(combined) / 1000 + 2))
            bgm = sidechain_duck(bed[:len(combined)], combined)
            combined = combined.overlay(bgm)
            bgm = None                        # already mixed in
        except Exception as e:                # noqa: BLE001 — BGM is optional
            print(f"  [Mix] generative BGM skipped: {e}")
    if bgm is not None:
        combined = mix_audio_layers(combined, bgm=bgm)

    combined.export(out_path, format="mp3")
    return out_path


# ═══════════════════════════════════════════
# THE BUILD
# ═══════════════════════════════════════════

class VideoBuild:
    """Plans and executes the jvmake DAG for one dialogue JSON."""

    PREVIEW_SECONDS = 6

    def __init__(self, dialogue: Dict[str, Any], preview: bool = False):
        self.dialogue = dialogue
        self.preview = preview
        self.res_scale = 0.5 if preview else 1.0
        self.fps = settings.FPS
        # Mirror StreamingCompositor's even-dimension rule (yuv420p)
        self.width = int(settings.WIDTH * self.res_scale) // 2 * 2
        self.height = int(settings.HEIGHT * self.res_scale) // 2 * 2
        self.title = dialogue.get("title", "Untitled")
        # Tier 2: this video's genome + Tier 3 critic overrides — both are
        # cache-key material, so gene/fix changes rebuild exactly the
        # artifacts they repaint.
        from engine.visual_dna import VisualDNA
        self.dna = VisualDNA.from_dialogue(dialogue)
        self.overrides = dict(dialogue.get("render_overrides", {}))

    # ─── Public API ────────────────────────────────────────

    @staticmethod
    def _ensure_character_assets(report) -> None:
        """
        Fail loudly (or auto-repair) if character art is missing.

        A blank video with only voice + captions means the compositor found
        neither a puppet rig nor expression PNGs. If assets/poses has source
        art, auto-run pose staging + rig build; otherwise raise so the render
        never silently produces an empty frame.
        """
        def _chars_ok() -> bool:
            for name in ("gudiya", "chintu"):
                char_dir = os.path.join(settings.CHARACTERS_DIR, name)
                has_body = any(
                    os.path.exists(os.path.join(char_dir, f"body{e}"))
                    for e in (".png", ".jpg"))
                if not has_body:
                    return False
            return True

        if _chars_ok():
            return

        report("ASSETS", 1, "Character assets missing — auto-staging "
                            "from assets/poses...")
        try:
            from tools.pose_stager import stage_all
            stage_all()
        except Exception:
            # Older stager exposes main(); fall back to module execution
            import subprocess, sys as _sys
            r = subprocess.run(
                [_sys.executable, os.path.join(settings.BASE_DIR
                 if hasattr(settings, "BASE_DIR") else ".",
                 "tools", "pose_stager.py")], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(
                    "Character assets are missing and auto-staging failed.\n"
                    f"{r.stderr[-500:]}\n"
                    "Run: python3 tools/pose_stager.py && "
                    "python3 jvmake.py rig") from None

        # Build puppet rigs so lip-sync uses the real mouth art
        try:
            from tools.rig_builder import build_all
            build_all()
        except Exception as e:
            report("ASSETS", 2, f"Rig build failed ({e}) — expression "
                                "fallback will be used.")

        if not _chars_ok():
            raise RuntimeError(
                "Character assets are missing (assets/characters/ is empty) "
                "and could not be auto-staged from assets/poses/. The video "
                "would render BLANK. Run: python3 tools/pose_stager.py && "
                "python3 jvmake.py rig, then re-render.")
        report("ASSETS", 2, "Characters auto-staged + rigged ✓")

    def run(self, progress_callback=None, force: bool = False) -> str:
        """Full build: TTS → timeline → DAG (mix + segments + final) → MP4."""
        def report(stage: str, pct: float, msg: str):
            print(f"  [{stage}] {msg}")
            if progress_callback:
                progress_callback(stage, pct, msg)

        mode = "preview 540p/6s" if self.preview else "full 1080p"
        report("INIT", 0, f"jvmake DAG build ({mode}): {self.title}")
        self._reset_temp()

        # Guard: characters MUST exist before rendering, otherwise the
        # compositor silently pastes nothing and the video comes out blank
        # (voice + captions only). Auto-stage from assets/poses if possible.
        self._ensure_character_assets(report)

        # Asset prep mutates assets/characters — MUST precede fingerprinting
        try:
            from pipeline.bg_remove import prepare_all_characters
            prepare_all_characters()
        except Exception as e:
            report("ASSETS", 3, f"Asset prep skipped ({e}). Using originals.")

        # Phase A: TTS nodes (VoiceEngine = parallel + content-addressed)
        report("VOICE", 8, "TTS nodes (parallel, content-addressed)...")
        from pipeline.voice import VoiceEngine
        turn_data = VoiceEngine().generate_dialogue(self.dialogue,
                                                    settings.TEMP_DIR)
        total_s = sum(t["duration_ms"] for t in turn_data) / 1000
        report("VOICE", 28, f"{len(turn_data)} turns, {total_s:.1f}s total")

        # Tier 2: plan the camera (deterministic, DNA-seeded; explicit
        # shot_type in the script always wins)
        from engine.shot_sequencer import ShotSequencer
        ShotSequencer(self.dna).apply(turn_data)

        # Phase B: the timeline durations are now known → plan the DAG
        from pipeline.timeline import Timeline
        timeline = Timeline(turn_data, fps=self.fps)
        report("TIMELINE", 32, timeline.describe())

        graph, final = self._plan(timeline, turn_data)

        # ── Multi-core prepass: render every MISSING segment across all
        #    cores. Segments are pure functions of their keys, so the
        #    parallel output is bit-identical to serial. The DAG build
        #    afterwards sees pure cache hits. ──
        from pipeline.parallel import default_workers, render_segments_parallel
        missing = [payload for key, payload in self._seg_payloads.items()
                   if force or graph.cache.get(key, "mp4") is None]
        workers = default_workers(len(missing))
        if missing and workers > 1:
            report("RENDER", 36,
                   f"parallel prepass: {len(missing)} segments on "
                   f"{workers} cores")
            render_segments_parallel(
                missing, graph.cache,
                on_done=lambda d, t, name: report(
                    "RENDER", 36 + 55 * d / t, f"{name} rendered ({d}/{t})"),
                workers=workers)
            if force:
                # prepass results ARE the segment rebuild; force the
                # rest of the graph honestly by evicting downstream
                graph.cache.evict(self._mix_key, "mp3")
                graph.cache.evict(self._final_key, "mp4")
            force = False

        def on_progress(done: int, total: int, node: Node):
            verb = "cache hit" if node.cached else f"built in {node.seconds:.1f}s"
            report("RENDER", 35 + 60 * done / max(1, total),
                   f"{node.name} [{node.short_key}] {verb}")

        graph.build(final, force=force, on_progress=on_progress)
        report("RENDER", 95, f"DAG complete: {graph.summary(final)}")

        # Publish the cached artifact under a human-readable name
        clean_title = "".join(c if c.isalnum() else "_" for c in self.title)[:50]
        suffix = "_preview" if self.preview else ""
        out = os.path.join(settings.OUTPUT_DIR,
                           f"{clean_title}{suffix}_{int(time.time())}.mp4")
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        shutil.copy2(final.path, out)

        # ── Post-mux delivery stage (Part XI + Part XIX): the beat ledger
        #    and the QC-pass manifest are what the publisher demands. A
        #    preview render skips it (it is deliberately 6 s of 540p).
        if not self.preview:
            self.deliver(out, timeline, report)

        report("DONE", 100, f"✓ Video saved to: {out}")
        return out

    # ─── Delivery: beat ledger + decoded-pixel/loudness QC ────

    def deliver(self, video_path: str, timeline,
                report=None, strict: bool = False) -> Dict[str, Any]:
        """Emit `<video>.beats.json` + `<video>.qc-manifest.json`.

        Nothing downstream trusts a render without these two artifacts:
        `factory/publisher.admission_check` refuses to upload without a
        green manifest AND a checksum-valid ledger. Failures are reported
        (and raised under `strict`), never silently swallowed.
        """
        def say(stage: str, pct: float, msg: str):
            if report:
                report(stage, pct, msg)
            else:
                print(f"  [{stage}] {msg}")

        result: Dict[str, Any] = {"video": video_path}

        # 1 · beat ledger — the learning loop's ground truth
        try:
            from factory.beats import emit as emit_beats
            result["beats"] = emit_beats(self.dialogue, timeline,
                                         video_path, self.fps)
            say("BEATS", 96, f"beat ledger → {os.path.basename(result['beats'])}")
        except Exception as e:                   # noqa: BLE001
            result["beats_error"] = str(e)
            say("BEATS", 96, f"beat ledger FAILED: {e}")
            if strict:
                raise

        # 2 · delivery QC on the MUXED file (decoded pixels, loudness,
        #     colour metadata, A/V start offset, phone-scale legibility)
        try:
            from pipeline.delivery_qc import audit, write_manifest
            qc = audit(video_path)
            # 2b · the adversarial gauntlet (Part XXI): flicker, freeze,
            #      teleport, jitter, letterbox, chroma drift. Delivery QC
            #      asks "is this frame shippable?"; the gauntlet asks the
            #      harder question, "given the frame before it?" — and its
            #      gates land in the SAME manifest the publisher checks.
            try:
                from tools.gauntlet import run as gauntlet_run
                temporal = gauntlet_run(video_path)
                for gate in temporal.gates:
                    qc.add(gate)
                say("QC", 98, f"gauntlet: "
                              f"{'PASS' if temporal.passed else 'FAIL'}")
            except Exception as ge:              # noqa: BLE001
                result["gauntlet_error"] = str(ge)
                say("QC", 98, f"temporal gauntlet FAILED: {ge}")
                if strict:
                    raise
            extra: Dict[str, Any] = {"fps": self.fps, "title": self.title}
            ledger_path = result.get("beats")
            if ledger_path and os.path.exists(ledger_path):
                from factory.beats import BeatLedger
                extra["beats_checksum"] = BeatLedger.load(
                    ledger_path).checksum
            write_manifest(video_path, qc, extra)
            result["qc_passed"] = qc.passed
            say("QC", 98, qc.summary())
            if strict and not qc.passed:
                raise RuntimeError("delivery QC failed — see the manifest")
        except Exception as e:                   # noqa: BLE001
            result["qc_error"] = str(e)
            say("QC", 98, f"delivery QC FAILED: {e}")
            if strict:
                raise
        return result

    def describe(self) -> str:
        """`jvmake graph`: every node's cache status, building nothing."""
        from pipeline.voice import VoiceEngine

        lines = [f"═══ jvmake graph · {self.title}"
                 f"{' · preview' if self.preview else ''} ═══", ""]

        plan = VoiceEngine().plan_dialogue(self.dialogue)
        tts_hits = tts_misses = 0
        durations_known = True
        for p in plan:
            if p["tts_key"] is None:
                lines.append(f"  tts:{p['turn_id']:02d}   ········  "
                             f"(explanation — no audio)")
                continue
            if p["cached"] and p["duration_ms"] is not None:
                tts_hits += 1
                mark = "HIT "
            else:
                tts_misses += 1
                durations_known = False
                mark = "MISS"
            lines.append(f"  tts:{p['turn_id']:02d}   {p['tts_key'][:8]}  {mark}")

        if durations_known:
            # Mirror run() exactly: sequencer plans shots before keying
            from engine.shot_sequencer import ShotSequencer
            ShotSequencer(self.dna).apply(plan)
            from pipeline.timeline import Timeline
            timeline = Timeline(plan, fps=self.fps)
            graph, final = self._plan(timeline, plan)
            statuses = graph.status(final)
            for node, hit in statuses:
                lines.append(f"  {node.name:<8} {node.short_key}  "
                             f"{'HIT ' if hit else 'MISS'}")
            hits = tts_hits + sum(1 for _, h in statuses if h)
            misses = tts_misses + sum(1 for _, h in statuses if not h)
        else:
            lines.append("")
            lines.append("  (mix/segment/final keys unknown until TTS is "
                         "built — run: jvmake render)")
            hits, misses = tts_hits, tts_misses

        lines += ["", f"  Plan: {hits} cached · {misses} to build"]
        return "\n".join(lines)

    # ─── DAG construction ──────────────────────────────────

    def _plan(self, timeline, turn_data: List[Dict[str, Any]]
              ) -> Tuple[Graph, Node]:
        graph = Graph(BuildCache())
        self._seg_payloads: Dict[str, Dict[str, Any]] = {}
        code_fp = code_fingerprint()
        asset_fp = assets_fingerprint()
        sfx_fp = sfx_fingerprint()

        # ── mix: the FULL CONSOLE (voice bus → sound design → ducked
        #    stereo bed → −14 LUFS master). Keyed by voice identities,
        #    exact silences, SHOTS (they place whoosh/riser events) and
        #    only the AUDIO genes — visual gene edits never remix ──
        from pipeline.mixdown import MIX_VERSION, mixdown
        audio_genes = {k: self.dna.genes[k]
                       for k in ("bgm_mode", "bgm_root", "energy")}
        mix_recipe = [(t.get("tts_key") or "silence",
                       t.get("duration_ms"), t.get("padding_ms") or 0,
                       t.get("shot_type"))
                      for t in turn_data]
        mix_key = key_of(MIX_VERSION, json.dumps(mix_recipe), sfx_fp,
                         json.dumps(audio_genes, sort_keys=True))
        self._mix_key = mix_key
        mix = graph.node(
            "mix", mix_key, "mp3",
            lambda node: mixdown(
                turn_data, os.path.join(settings.TEMP_DIR, "mixed_audio.mp3"),
                dna=self.dna))

        # ── one segment node per turn ──
        holder: Dict[str, Any] = {}

        def compositor():
            # Lazy, shared: character art / rigs load once for all segments
            if "c" not in holder:
                from pipeline.compositor_v5 import StreamingCompositor
                holder["c"] = StreamingCompositor(res_scale=self.res_scale,
                                                  dna=self.dna,
                                                  overrides=self.overrides)
            return holder["c"]

        seg_nodes: List[Node] = []
        budget = (int(self.PREVIEW_SECONDS * self.fps)
                  if self.preview else None)
        prev_shot = "two_shot"

        for i, span in enumerate(timeline.spans):
            n_frames = span.end_frame - span.start_frame
            if n_frames <= 0:
                continue
            cap: Optional[int] = None
            if budget is not None:
                if budget <= 0:
                    break
                cap = min(n_frames, budget)
                budget -= cap

            turn = span.turn
            default_shot = ("fullscreen_explain"
                            if turn["speaker"] == "explanation" else "two_shot")
            shot = turn.get("shot_type", default_shot)

            # EVERYTHING that can influence this segment's pixels:
            recipe = {
                "speaker": turn["speaker"],
                "text": turn.get("text", ""),
                "emotion": turn.get("emotion", "neutral"),
                "shot": shot,
                "prev_shot": prev_shot,          # camera transition source
                "visuals": turn.get("visual_elements", []),
                "gesture": turn.get("gesture"),  # Tier 1/3 bone triggers
                "tts": turn.get("tts_key") or "silence",  # audio→lipsync+captions
                "frames": n_frames,
                "cap": cap,
                "canvas": [self.width, self.height, self.fps, self.res_scale],
                "dna": self.dna.cache_key_material(),      # Tier 2 genome
                "overrides": self.overrides,               # Tier 3 critic fixes
            }
            seg_key = key_of("seg-v1", code_fp, asset_fp,
                             json.dumps(recipe, sort_keys=True,
                                        ensure_ascii=False))

            def make_build(idx: int = i, skey: str = seg_key,
                           pshot: str = prev_shot, scap: Optional[int] = cap):
                def _build(node: Node) -> str:
                    from pipeline.encoder import StreamEncoder
                    comp = compositor()
                    out = os.path.join(settings.TEMP_DIR, f"seg_{idx:03d}.mp4")
                    with StreamEncoder(out, comp.width, comp.height,
                                       self.fps) as enc:
                        comp.render_segment(timeline, idx, enc, skey,
                                            prev_shot=pshot, max_frames=scap)
                    return out
                return _build

            seg_nodes.append(graph.node(f"seg:{i:02d}", seg_key, "mp4",
                                        make_build()))
            # Picklable payload for the multi-core prepass
            self._seg_payloads[seg_key] = {
                "name": f"seg:{i:02d}",
                "project_root": settings.PROJECT_ROOT,
                "turn_data": turn_data,
                "index": i,
                "seg_key": seg_key,
                "seed_key": seg_key,
                "prev_shot": prev_shot,
                "cap": cap,
                "res_scale": self.res_scale,
                "dna": self.dna.to_dict(),
                "overrides": self.overrides,
            }
            prev_shot = shot

        if not seg_nodes:
            raise ValueError("dialogue produced no renderable turns")

        # ── final: lossless concat of segments + audio mux ──
        final_key = key_of("final-v1", mix_key, *[n.key for n in seg_nodes])
        self._final_key = final_key

        def build_final(node: Node) -> str:
            from pipeline.encoder import concat_and_mux
            out = os.path.join(settings.TEMP_DIR, "final.mp4")
            return concat_and_mux([n.path for n in seg_nodes], mix.path, out)

        final = graph.node("final", final_key, "mp4", build_final,
                           deps=[*seg_nodes, mix])
        return graph, final

    # ─── Housekeeping ──────────────────────────────────────

    @staticmethod
    def _reset_temp() -> None:
        """Scratch space only — every artifact of value lives in .cache."""
        if os.path.exists(settings.TEMP_DIR):
            shutil.rmtree(settings.TEMP_DIR)
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
