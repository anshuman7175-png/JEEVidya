"""
Gudiya & Chintu — Dual Voice Engine V2
Generates per-turn audio with character-specific voice profiles.
Girl (Gudiya): hi-IN-SwaraNeural with pitch shift for younger sound.
Boy (Chintu): hi-IN-MadhurNeural with pitch shift for younger sound.
"""
import hashlib
import math
import os
import re
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


class AudioDurationError(RuntimeError):
    """Every duration probe failed for a real audio file. A WRONG
    duration is worse than a failed build: a fake fallback silently
    desynchronises the entire viseme timeline from the audio."""


def _mp3_duration_ms(audio_path: str) -> Optional[int]:
    """Stdlib MP3 duration probe: walk every MPEG audio frame header and
    sum samples/sample-rate. No ffmpeg, no moviepy — pure byte parsing,
    so it works on a bare install and is immune to lying VBR container
    headers (each frame is measured individually).

    Returns None when the file is not parseable MPEG audio.
    """
    BITRATES = {  # kbps, indexed [version_key][layer_key][bitrate_index]
        # MPEG1 Layer III
        (1, 3): (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192,
                 224, 256, 320),
        # MPEG2/2.5 Layer III
        (2, 3): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112,
                 128, 144, 160),
    }
    SAMPLE_RATES = {3: (44100, 48000, 32000),    # MPEG1
                    2: (22050, 24000, 16000),    # MPEG2
                    0: (11025, 12000, 8000)}     # MPEG2.5
    try:
        with open(audio_path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if len(data) < 4:
        return None
    pos = 0
    # Skip a leading ID3v2 tag (synchsafe 28-bit size).
    if data[:3] == b"ID3" and len(data) >= 10:
        size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) | \
               ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
        pos = 10 + size
    total_seconds = 0.0
    frames = 0
    n = len(data)
    while pos + 4 <= n:
        if data[pos] != 0xFF or (data[pos + 1] & 0xE0) != 0xE0:
            pos += 1
            continue
        b1, b2 = data[pos + 1], data[pos + 2]
        version_id = (b1 >> 3) & 0x03      # 3=MPEG1, 2=MPEG2, 0=MPEG2.5
        layer_id = (b1 >> 1) & 0x03        # 1=III, 2=II, 3=I
        bitrate_idx = (b2 >> 4) & 0x0F
        sr_idx = (b2 >> 2) & 0x03
        padding = (b2 >> 1) & 0x01
        if (version_id == 1 or layer_id == 0 or bitrate_idx in (0, 15)
                or sr_idx == 3):
            pos += 1
            continue
        sample_rate = SAMPLE_RATES[version_id][sr_idx]
        vkey = 1 if version_id == 3 else 2
        layer = {3: 1, 2: 2, 1: 3}[layer_id]
        if layer != 3:
            return None  # only Layer III (mp3) is supported here
        bitrate = BITRATES[(vkey, 3)][bitrate_idx] * 1000
        if bitrate == 0:
            pos += 1
            continue
        samples = 1152 if vkey == 1 else 576
        frame_len = samples * bitrate // (8 * sample_rate) + padding
        if frame_len <= 0:
            pos += 1
            continue
        total_seconds += samples / sample_rate
        frames += 1
        pos += frame_len
    if frames < 2:
        return None
    return int(total_seconds * 1000)


def _get_audio_duration_ms(audio_path: str) -> int:
    """Get audio file duration in milliseconds.

    Decodes via ffmpeg subprocess (imageio_ffmpeg bundles ffmpeg but NOT
    ffprobe, and moviepy 2.x removed the `moviepy.editor` module, so both
    of the old probes silently failed and every turn fell back to 2000ms,
    corrupting the whole timeline). Falls back to a stdlib MP3
    frame-header walk; if EVERY probe fails, raises AudioDurationError
    naming the file — there is no fake-duration fallback.
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
        if dur > 0:
            return dur
    except Exception:
        pass

    # Fallback: stdlib MP3 frame-header walk (no dependencies at all).
    dur = _mp3_duration_ms(audio_path)
    if dur is not None and dur > 0:
        return dur

    raise AudioDurationError(
        f"could not determine the duration of '{audio_path}': ffmpeg, "
        f"moviepy and the stdlib MP3 frame probe all failed. Refusing to "
        f"guess — a wrong duration desynchronises every viseme in the "
        f"timeline from the audio, which is worse than a failed build.")


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


KNOWN_EMOTION_TAGS = {
    "curious", "confidently", "deadpan", "surprised", "excited", "confused",
    "cheerfully", "thoughtful", "elated", "proudly", "mischievously", "sighs",
    "sarcastic", "laughs", "gasps", "dramatic tone", "shocked", "intrigued",
    "whispers", "shouting", "puzzled", "annoyed", "happy", "wonder"
}


def sanitize_spoken_text(text: str) -> str:
    """
    Sanitize dialogue line so speaker labels, metadata lines, turn numbers,
    and stage direction / emotion tags are NEVER spoken by TTS or rendered into captions.
    """
    if not text:
        return ""
    t = text.strip()
    # 1. Detect & drop metadata lines
    if re.match(r"^(?:tags|hashtags|keywords|title|topic|description|subject|scene|setting|sfx|sound effect|bgm|music)\s*[:：\-–]", t, re.I):
        return ""
    if re.match(r"^(?:#[\w\-]+\s*)+$", t):
        return ""
    if re.match(r"^\[(?:scene|setting|sfx|sound effect|bgm|music)[:\s].*\]$", t, re.I):
        return ""
    if re.match(r"^\((?:scene|setting|sfx|sound effect|bgm|music)[:\s].*\)$", t, re.I):
        return ""

    # 2. Strip leading turn numbers: '1. ', 'Turn 1: ', '1) '
    t = re.sub(r"^(?:turn\s*\d+[\.\:\-–\s]*|\d+[\.\)\-–]\s*)", "", t, flags=re.I)

    # 3. Strip speaker prefixes: '**Gudiya:**', 'Speaker 1:', 'Gudiya (curious):', '[Chintu]:'
    prefix_pattern = r"^(?:\*{1,2}|\[)?(?:speaker\s*\d+|gudiya|chintu|girl|boy|didi|bhai|character\s*\d+|person\s*\d+|s[12])(?:\*{1,2}|\])?\s*(?:\([^\)]+\)|\[[^\]]+\])?\s*[:：\-–]\s*"
    t = re.sub(prefix_pattern, "", t, flags=re.I)

    # 4. Strip outer quotes
    t = t.strip('"\'*` ')

    # 5. Strip ALL bracketed or parenthesized emotion tags / stage directions anywhere in text
    # e.g., '[curious]', '[gasps]', '(laughs)', '[deadpan]', '[dramatic tone]'
    def _strip_emotion(m: re.Match) -> str:
        content = m.group(1).strip().lower()
        if content in KNOWN_EMOTION_TAGS or any(em in content for em in KNOWN_EMOTION_TAGS):
            return ""
        # Also strip short single/double word stage cues
        if len(content.split()) <= 3 and not any(c in content for c in "=+-*/^0123456789"):
            return ""
        return m.group(0)

    t = re.sub(r"\[([a-zA-Z\s,]+)\]", _strip_emotion, t)
    t = re.sub(r"\(([a-zA-Z\s,]+)\)", _strip_emotion, t)

    # Clean up any leftover double spaces
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def synthesize_elevenlabs(text: str, speaker: str, output_path: str,
                          api_key: Optional[str] = None) -> bool:
    """
    Synthesize audio using ElevenLabs API:
      - Gudiya: Chutki (Cute Young Kids Storyteller)
      - Chintu: Bhola (Soft, Innocent and Cheerful)
    Returns True if synthesis succeeded and output_path exists.
    """
    import json
    import urllib.request
    import urllib.error

    key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        return False

    is_girl = speaker.lower() in ("girl", "gudiya", "s1", "chutki")
    target_voice_name = settings.ELEVENLABS_VOICE_GUDIYA if is_girl else settings.ELEVENLABS_VOICE_CHINTU

    # Cache for voice IDs to avoid repeated /v1/voices lookups
    voice_cache_file = os.path.join(settings.PROJECT_ROOT, "uploads", "elevenlabs_voices.json")
    voice_id = None

    if os.path.exists(voice_cache_file):
        try:
            with open(voice_cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                voice_id = cached.get(target_voice_name)
        except Exception:
            pass

    if not voice_id:
        # Query ElevenLabs voices
        try:
            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": key, "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                voices = data.get("voices", [])
                cache_dict = {}
                for v in voices:
                    v_name = v.get("name", "")
                    v_id = v.get("voice_id", "")
                    cache_dict[v_name] = v_id
                    if target_voice_name.lower() in v_name.lower() or v_name.lower() in target_voice_name.lower():
                        voice_id = v_id

                os.makedirs(os.path.dirname(voice_cache_file), exist_ok=True)
                with open(voice_cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_dict, f, indent=2)
        except Exception as e:
            print(f"  [Voice] ElevenLabs voice lookup failed: {e}")

    if not voice_id:
        print(f"  [Voice] ElevenLabs voice '{target_voice_name}' not found; falling back")
        return False

    # Perform synthesis
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = {
            "text": text,
            "model_id": getattr(settings, "ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
            "voice_settings": {
                "stability": 0.55,
                "similarity_boost": 0.80,
                "style": 0.20,
                "use_speaker_boost": True
            }
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "xi-api-key": key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg"
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            audio_data = resp.read()
            with open(output_path, "wb") as f:
                f.write(audio_data)
        return True
    except Exception as e:
        print(f"  [Voice] ElevenLabs TTS synthesis error: {e}")
        return False


class VoiceEngine:
    """Generates character-specific audio and subtitles from dialogue turns."""

    def generate_turn_audio(self, turn_id: int, text: str, speaker: str,
                            output_dir: str,
                            voice: Optional[str] = None,
                            custom_audio: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate audio and VTT subtitles for a single dialogue turn.
        If custom_audio is provided and exists on disk, standardizes the MP3
        and aligns word cues via engine.align. Otherwise uses edge-tts.
        
        Returns: {"audio": path, "vtt": path, "duration_ms": int, "key": str}
        """
        os.makedirs(output_dir, exist_ok=True)

        audio_path = os.path.join(output_dir, f"turn_{turn_id:03d}.mp3")
        vtt_path = os.path.join(output_dir, f"turn_{turn_id:03d}.vtt")

        # Sanitize text so speaker labels or tags never leak into TTS or captions
        text = sanitize_spoken_text(text)

        # ─── Custom audio path (user-supplied line-by-line MP3) ───
        resolved_custom = None
        if custom_audio:
            if os.path.isabs(custom_audio) and os.path.exists(custom_audio):
                resolved_custom = custom_audio
            else:
                # Try relative to uploads/audio
                bn = os.path.basename(custom_audio)
                cand = os.path.join(settings.PROJECT_ROOT, "uploads", "audio", bn)
                if os.path.exists(cand):
                    resolved_custom = cand
                elif os.path.exists(os.path.join(settings.PROJECT_ROOT, custom_audio)):
                    resolved_custom = os.path.join(settings.PROJECT_ROOT, custom_audio)

        if resolved_custom:
            return self._process_custom_audio(turn_id, text, resolved_custom, audio_path, vtt_path)

        if not text or not text.strip():
            return {"audio": None, "vtt": None, "duration_ms": 0}

        # Get voice profile for this speaker (optionally overridden — the
        # Tier 4 localizer swaps voices per language variant)
        profile = dict(VOICE_PROFILES.get(speaker, VOICE_PROFILES["boy"]))
        if voice:
            profile["voice"] = voice

        # ─── Content-addressed cache: identical text+voice never hits TTS twice ───
        use_elevenlabs = bool(os.environ.get("ELEVENLABS_API_KEY") or voice == "elevenlabs")
        voice_id_str = "elevenlabs-bhola-chutki" if use_elevenlabs else (voice or profile["voice"])
        cache = BuildCache()
        cache_key = key_of("tts-v3", text.strip(), speaker, voice_id_str)
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

        generated_via_eleven = False
        if use_elevenlabs:
            print(f"  [Voice] Synthesizing turn {turn_id} via ElevenLabs ({speaker})...")
            if synthesize_elevenlabs(text.strip(), speaker, audio_path):
                # Run forced alignment on ElevenLabs audio
                try:
                    tmp_wav = audio_path + ".align.wav"
                    subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-vn", "-ar", "16000",
                                    "-ac", "1", tmp_wav], capture_output=True, check=True)
                    from engine.align import align_turn
                    aligned = align_turn(tmp_wav, text.strip() or "shabd")
                    words = aligned.words
                    with open(vtt_path, "w", encoding="utf-8") as vf:
                        vf.write("WEBVTT\n\n")
                        for w in words:
                            start_str = _ms_to_vtt(int(round(w.start_ms)))
                            end_str = _ms_to_vtt(int(round(w.end_ms)))
                            vf.write(f"{start_str} --> {end_str}\n{w.text}\n\n")
                    generated_via_eleven = True
                    if os.path.exists(tmp_wav):
                        os.remove(tmp_wav)
                except Exception as e:
                    print(f"  [Voice] ElevenLabs alignment fallback for turn {turn_id}: {e}")
                    generated_via_eleven = True  # audio exists, fallback VTT if needed

        if not generated_via_eleven:
            try:
                synthesize_with_word_boundaries(
                    text.strip(), profile["voice"], profile["rate"],
                    profile["pitch"], audio_path, vtt_path)
            except Exception as e:
                print(f"  [Voice] Error on turn {turn_id}: {e}")
                raise RuntimeError(f"TTS synthesis failed for turn {turn_id}: {e}") from e

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

    def _process_custom_audio(self, turn_id: int, text: str,
                              custom_audio: str, audio_path: str,
                              vtt_path: str) -> Dict[str, Any]:
        """Process user-supplied audio: transcode to standard MP3, measure duration,
        and generate word-level VTT cues via waveform alignment."""
        # Convert / standardize to 44.1kHz stereo MP3
        if os.path.abspath(custom_audio) != os.path.abspath(audio_path):
            cmd = ["ffmpeg", "-y", "-i", custom_audio, "-vn", "-ar", "44100",
                   "-ac", "2", "-b:a", "192k", audio_path]
            try:
                subprocess.run(cmd, capture_output=True, check=True)
            except Exception as e:
                print(f"  [Voice] FFmpeg transcode warning for turn {turn_id}: {e}; copying directly")
                shutil.copy2(custom_audio, audio_path)

        duration_ms = _get_audio_duration_ms(audio_path)

        # Cache key based on audio file hash and text
        with open(audio_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()[:24]
        cache_key = key_of("custom-audio-v3", h, text.strip())

        # Align words to waveform using engine.align
        tmp_wav = audio_path + ".align.wav"
        try:
            subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-vn", "-ar", "16000",
                            "-ac", "1", tmp_wav], capture_output=True, check=True)
            from engine.align import align_turn
            aligned = align_turn(tmp_wav, text.strip() or "shabd")
            words = aligned.words
        except Exception as e:
            print(f"  [Voice] Alignment warning for turn {turn_id}: {e}; using proportional word spacing")
            tokens = [t for t in text.strip().split() if t]
            if not tokens:
                tokens = ["..."]
            step = max(50.0, float(duration_ms) / len(tokens))
            from engine.align import AlignedWord
            words = [AlignedWord(text=tok, start_ms=i * step, end_ms=(i + 1) * step)
                     for i, tok in enumerate(tokens)]
        finally:
            if os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass

        with open(vtt_path, "w", encoding="utf-8") as vf:
            vf.write("WEBVTT\n\n")
            for w in words:
                start_str = _ms_to_vtt(int(round(w.start_ms)))
                end_str = _ms_to_vtt(int(round(w.end_ms)))
                vf.write(f"{start_str} --> {end_str}\n{w.text}\n\n")

        print(f"  [Voice] Turn {turn_id} (custom audio): {duration_ms}ms, {len(words)} aligned words")
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

        # ─── Fire all speaking-turn TTS / custom audio jobs in parallel ───
        futures: Dict[int, Any] = {}
        with ThreadPoolExecutor(max_workers=TTS_WORKERS) as pool:
            for turn in turns:
                if turn["speaker"] != "explanation":
                    custom_aud = turn.get("custom_audio") or turn.get("audio_file") or turn.get("audio")
                    futures[turn["turn_id"]] = pool.submit(
                        self.generate_turn_audio, turn["turn_id"],
                        turn.get("text", ""), turn["speaker"], voice_dir,
                        voices.get(turn["speaker"]), custom_audio=custom_aud)

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
                custom_aud = turn.get("custom_audio") or turn.get("audio_file")
                if custom_aud and os.path.exists(custom_aud):
                    with open(custom_aud, "rb") as f:
                        h = hashlib.sha256(f.read()).hexdigest()[:24]
                    key = key_of("custom-audio-v3", h, turn.get("text", "").strip())
                    audio_ms = _get_audio_duration_ms(custom_aud)
                    padding_ms = quantize_padding(audio_ms, settings.INTER_TURN_PADDING_MS)
                    duration_ms = audio_ms + padding_ms
                    base.update({
                        "tts_key": key,
                        "cached": True,
                        "duration_ms": duration_ms,
                        "audio_duration_ms": audio_ms,
                        "padding_ms": padding_ms,
                        "shot_type": turn.get("shot_type"),
                        "gesture": turn.get("gesture"),
                    })
                    plan.append(base)
                    continue

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
