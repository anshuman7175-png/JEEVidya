"""
Gudiya & Chintu — Dual Voice Engine V2
Generates per-turn audio with character-specific voice profiles.
Girl (Gudiya): hi-IN-SwaraNeural with pitch shift for younger sound.
Boy (Chintu): hi-IN-MadhurNeural with pitch shift for younger sound.
"""
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from pydub import AudioSegment

from config import settings
from pipeline.cache import BuildCache, key_of

# Parallel TTS workers (edge-tts is network-bound; 4 overlaps latency well)
TTS_WORKERS = 4

# ─── Frame quantization (the drift killer) ─────────────────────────────
# Smallest millisecond quantum that is an exact whole number of frames.
# At 30 fps this is 100 ms (= 3 frames). Every turn's total duration
# (audio + padding) is stretched to a multiple of this, so EVERY turn
# boundary lands on an exact frame. Consequences:
#   • cumulative A/V drift is structurally 0 ms, and
#   • a turn's frame count depends only on its OWN audio — editing one
#     line never shifts a neighbouring segment (jvmake cache stays hot).
FRAME_QUANTUM_MS = 1000 // math.gcd(1000, settings.FPS)


def quantize_up(ms: int) -> int:
    """Round a duration UP to the next exact-frame boundary."""
    return ms + (-ms) % FRAME_QUANTUM_MS


def quantize_padding(audio_ms: int, base_padding_ms: int) -> int:
    """
    Padding ≥ base such that audio+padding is an exact frame multiple.
    The stretch (< FRAME_QUANTUM_MS) hides inside the inter-turn silence.
    """
    return base_padding_ms + (-(audio_ms + base_padding_ms)) % FRAME_QUANTUM_MS


def tts_cache_key(text: str, speaker: str,
                  voice: Optional[str] = None) -> str:
    """Content key for one TTS synthesis — THE identity of a spoken line.
    `voice` overrides the speaker's default (Tier 4 localizer variants)."""
    profile = VOICE_PROFILES.get(speaker, None)
    if profile is None:
        profile = {"voice": settings.VOICE_BOY, "rate": settings.VOICE_BOY_RATE,
                   "pitch": settings.VOICE_BOY_PITCH}
    return key_of("tts-v2-wordvtt", text.strip(), voice or profile["voice"],
                  profile["rate"], profile["pitch"])


def resolve_edge_tts() -> str:
    """
    Locate the edge-tts CLI. Prefers the copy inside the running Python's
    environment (venv), so the pipeline works even when the venv isn't
    activated in the caller's shell.
    """
    candidate = os.path.join(os.path.dirname(sys.executable), "edge-tts")
    if os.path.exists(candidate):
        return candidate
    found = shutil.which("edge-tts")
    if found:
        return found
    raise RuntimeError(
        "edge-tts CLI not found. Install with: pip install edge-tts")


def _ms_to_vtt(ms: int) -> str:
    """Milliseconds → VTT timestamp (HH:MM:SS.mmm)."""
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{milli:03d}"


def synthesize_with_word_boundaries(text: str, voice: str, rate: str,
                                    pitch: str, audio_path: str,
                                    vtt_path: str) -> None:
    """
    V5: synthesize via the edge-tts PYTHON API, capturing WordBoundary
    events into a word-level VTT. (The CLI in edge-tts >= 7 only writes
    sentence-level SRT, which kills karaoke captions and viseme timing.)

    WordBoundary offsets arrive in 100-nanosecond ticks.
    """
    import asyncio
    import edge_tts

    async def _run():
        # boundary="WordBoundary" is REQUIRED in edge-tts >= 7
        # (default SentenceBoundary yields no word events)
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch,
                                           boundary="WordBoundary")
        words = []
        with open(audio_path, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    start_ms = int(chunk["offset"] / 10_000)
                    end_ms = int((chunk["offset"] + chunk["duration"]) / 10_000)
                    words.append((start_ms, end_ms, str(chunk["text"]).strip()))

        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for start_ms, end_ms, word in words:
                if word:
                    f.write(f"{_ms_to_vtt(start_ms)} --> {_ms_to_vtt(end_ms)}\n")
                    f.write(f"{word}\n\n")

    # Each worker thread gets its own event loop via asyncio.run
    asyncio.run(_run())


def _get_audio_duration_ms(audio_path: str) -> int:
    """Get audio file duration in milliseconds.

    Decodes via ffmpeg subprocess (imageio_ffmpeg bundles ffmpeg but NOT
    ffprobe, and moviepy 2.x removed the `moviepy.editor` module, so both
    of the old probes silently failed and every turn fell back to 2000ms,
    corrupting the whole timeline).
    """
    import re
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = "ffmpeg"

    # Decode to null muxer — the LAST "time=" ffmpeg reports is the true
    # decoded duration (container "Duration:" headers can lie for VBR mp3).
    try:
        proc = subprocess.run(
            [ffmpeg_exe, "-v", "info", "-i", audio_path, "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        matches = re.findall(
            r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
        if matches:
            h, m, s = matches[-1]
            return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)
        # Fallback: container header
        header = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
        if header:
            h, m, s = header.groups()
            return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)
    except Exception:
        pass

    # Fallback: moviepy 2.x API (no moviepy.editor)
    try:
        from moviepy import AudioFileClip
        clip = AudioFileClip(audio_path)
        dur = int(clip.duration * 1000)
        clip.close()
        return dur
    except Exception:
        return 2000  # Default 2 seconds if all probes fail


# Voice profiles per character
VOICE_PROFILES = {
    "girl": {
        "voice": settings.VOICE_GIRL,
        "rate": settings.VOICE_GIRL_RATE,
        "pitch": settings.VOICE_GIRL_PITCH,
    },
    "boy": {
        "voice": settings.VOICE_BOY,
        "rate": settings.VOICE_BOY_RATE,
        "pitch": settings.VOICE_BOY_PITCH,
    },
}


class VoiceEngine:
    """Generates character-specific audio and subtitles from dialogue turns."""

    def generate_turn_audio(self, turn_id: int, text: str, speaker: str,
                            output_dir: str,
                            voice: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate audio and VTT subtitles for a single dialogue turn.
        Uses the correct voice profile based on speaker.
        
        Returns: {"audio": path, "vtt": path, "duration_ms": int}
        """
        os.makedirs(output_dir, exist_ok=True)

        audio_path = os.path.join(output_dir, f"turn_{turn_id:03d}.mp3")
        vtt_path = os.path.join(output_dir, f"turn_{turn_id:03d}.vtt")

        if not text or not text.strip():
            return {"audio": None, "vtt": None, "duration_ms": 0}

        # Get voice profile for this speaker (optionally overridden — the
        # Tier 4 localizer swaps voices per language variant)
        profile = dict(VOICE_PROFILES.get(speaker, VOICE_PROFILES["boy"]))
        if voice:
            profile["voice"] = voice

        # ─── Content-addressed cache: identical text+voice never hits TTS twice ───
        cache = BuildCache()
        cache_key = tts_cache_key(text, speaker, voice)
        if cache.fetch(cache_key, "mp3", audio_path) and cache.fetch(cache_key, "vtt", vtt_path):
            # Duration sidecar: a hit skips audio decoding entirely
            ms_text = cache.get_text(cache_key, "ms")
            if ms_text and ms_text.strip().isdigit():
                duration_ms = int(ms_text.strip())
            else:
                duration_ms = _get_audio_duration_ms(audio_path)
                cache.put_text(cache_key, "ms", str(duration_ms))
            print(f"  [Voice] Turn {turn_id} ({speaker}): cache hit ({duration_ms}ms)")
            return {"audio": audio_path, "vtt": vtt_path,
                    "duration_ms": duration_ms, "key": cache_key}

        print(f"  [Voice] Turn {turn_id} ({speaker}): generating audio...")

        try:
            synthesize_with_word_boundaries(
                text.strip(), profile["voice"], profile["rate"],
                profile["pitch"], audio_path, vtt_path)
        except Exception as e:
            print(f"  [Voice] Error on turn {turn_id}: {e}")
            raise RuntimeError(f"edge-tts failed for turn {turn_id}: {e}")

        # Get actual duration from the generated file
        duration_ms = 0
        if os.path.exists(audio_path):
            duration_ms = _get_audio_duration_ms(audio_path)

        # Store in cache for instant re-renders
        cache.put(cache_key, "mp3", audio_path)
        cache.put(cache_key, "vtt", vtt_path)
        cache.put_text(cache_key, "ms", str(duration_ms))

        print(f"  [Voice] Turn {turn_id}: {duration_ms}ms")
        return {
            "audio": audio_path,
            "vtt": vtt_path,
            "duration_ms": duration_ms,
            "key": cache_key,
        }

    def generate_dialogue(self, dialogue: Dict[str, Any],
                          output_dir: str) -> List[Dict[str, Any]]:
        """
        Generate audio for all dialogue turns.
        Explanation turns get no audio (they use the pause_after duration).
        
        Returns list of turn results with timing information.
        """
        voice_dir = os.path.join(output_dir, "voice")
        os.makedirs(voice_dir, exist_ok=True)

        turns = dialogue.get("turns", [])
        voices = dialogue.get("voices") or {}      # localizer overrides

        # ─── Fire all speaking-turn TTS jobs in parallel (order preserved) ───
        futures: Dict[int, Any] = {}
        with ThreadPoolExecutor(max_workers=TTS_WORKERS) as pool:
            for turn in turns:
                if turn["speaker"] != "explanation":
                    futures[turn["turn_id"]] = pool.submit(
                        self.generate_turn_audio, turn["turn_id"],
                        turn.get("text", ""), turn["speaker"], voice_dir,
                        voices.get(turn["speaker"]))

            results = []
            for turn in turns:
                turn_id = turn["turn_id"]
                speaker = turn["speaker"]

                if speaker == "explanation":
                    # No audio for explanation turns — use specified duration
                    # (quantized UP to an exact frame boundary: drift = 0)
                    duration_s = turn.get("duration_seconds", 4.0)
                    pause_after = turn.get("pause_after", 1.0)
                    results.append({
                        "turn_id": turn_id,
                        "speaker": speaker,
                        "audio": None,
                        "vtt": None,
                        "tts_key": None,
                        "duration_ms": quantize_up(
                            int((duration_s + pause_after) * 1000)),
                        "emotion": turn.get("emotion", "neutral"),
                        # raw pass-through: None lets the shot sequencer plan
                        "shot_type": turn.get("shot_type"),
                        "visual_elements": turn.get("visual_elements", []),
                    })
                    continue

                result = futures[turn_id].result()  # raises on TTS failure
                # Frame-quantized padding: audio+padding is an exact frame
                # multiple, so this turn's frame count is position-independent
                padding_ms = quantize_padding(result["duration_ms"],
                                              settings.INTER_TURN_PADDING_MS)

                results.append({
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "text": turn.get("text", ""),
                    "audio": result["audio"],
                    "vtt": result["vtt"],
                    "tts_key": result.get("key"),
                    "duration_ms": result["duration_ms"] + padding_ms,
                    "audio_duration_ms": result["duration_ms"],
                    "padding_ms": padding_ms,
                    "emotion": turn.get("emotion", "neutral"),
                    # raw pass-through: None lets the shot sequencer plan
                    "shot_type": turn.get("shot_type"),
                    "gesture": turn.get("gesture"),   # Tier 1/3 bone trigger
                })

        return results

    def plan_dialogue(self, dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        DRY RUN for `jvmake graph`: compute every turn's TTS key and cache
        status WITHOUT synthesizing anything. Durations come from the "ms"
        sidecar; None means the turn must be synthesized before the full
        downstream DAG (segments/mix/final) can be keyed.
        """
        cache = BuildCache()
        voices = dialogue.get("voices") or {}
        plan: List[Dict[str, Any]] = []
        for turn in dialogue.get("turns", []):
            # These dicts MUST mirror generate_dialogue() field-for-field:
            # buildgraph derives segment keys from them, and a dry-run key
            # must equal the real-build key byte-for-byte.
            base = {
                "turn_id": turn["turn_id"],
                "speaker": turn["speaker"],
                "emotion": turn.get("emotion", "neutral"),
                "audio": None,
                "vtt": None,
            }
            if turn["speaker"] == "explanation":
                base.update({
                    "tts_key": None,
                    "cached": True,
                    "shot_type": turn.get("shot_type"),
                    "visual_elements": turn.get("visual_elements", []),
                    "duration_ms": quantize_up(int(
                        (turn.get("duration_seconds", 4.0)
                         + turn.get("pause_after", 1.0)) * 1000)),
                })
            else:
                key = tts_cache_key(turn.get("text", ""), turn["speaker"],
                                    voices.get(turn["speaker"]))
                hit = (cache.get(key, "mp3") is not None
                       and cache.get(key, "vtt") is not None)
                ms_text = cache.get_text(key, "ms")
                duration_ms = None
                padding_ms = None
                if hit and ms_text and ms_text.strip().isdigit():
                    audio_ms = int(ms_text.strip())
                    padding_ms = quantize_padding(
                        audio_ms, settings.INTER_TURN_PADDING_MS)
                    duration_ms = audio_ms + padding_ms
                base.update({
                    "text": turn.get("text", ""),
                    "tts_key": key,
                    "cached": hit,
                    "shot_type": turn.get("shot_type"),
                    "gesture": turn.get("gesture"),
                    "duration_ms": duration_ms,
                    "padding_ms": padding_ms,
                })
            plan.append(base)
        return plan

    # Legacy compatibility
    def generate_scene_audio(self, scene_id: int, narration: str,
                             output_dir: str) -> Dict[str, Any]:
        """Legacy method — routes to generate_turn_audio."""
        result = self.generate_turn_audio(scene_id, narration, "boy", output_dir)
        return {"audio": result["audio"], "vtt": result["vtt"]}

    def generate_all(self, storyboard: Dict[str, Any],
                     output_dir: str) -> List[Dict[str, Any]]:
        """Legacy method — generates audio for old-format storyboards."""
        results = []
        for scene in storyboard.get("scenes", []):
            scene_id = scene.get("scene_id")
            narration = scene.get("narration", "")
            paths = self.generate_scene_audio(scene_id, narration, output_dir)
            results.append({
                "scene_id": scene_id,
                "audio_path": paths["audio"],
                "vtt_path": paths["vtt"],
                "duration_target": scene.get("duration_seconds", settings.QUESTION_DURATION),
            })
        return results
