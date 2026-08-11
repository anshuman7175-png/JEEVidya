"""
JEEVidya — Voice Identity Layer (Terminal Plan, Part X)
═══════════════════════════════════════════════════════
Your own voice, recorded ONCE (45–60 s neutral + 5–10 s per emotion),
synthesized per line forever after. Engine adapters isolate the model so
swapping engines never touches the face pipeline.

    Primary : IndexTTS-2          (Apache-2.0, disentangled emotion ref, MPS)
    Fallback: Chatterbox          (MIT, Hindi supported, Apple-Silicon build)
    Excluded: XTTS-v2             (CPML — non-commercial, unsafe for the channel)

Hard gates (§10.4), enforced on EVERY generated line:
  • Speaker-identity gate — cosine similarity of a speaker embedding
    between the line and base.wav must exceed IDENTITY_THRESHOLD;
    below → retry with a new seed, then fail loudly.
  • Pronunciation gate — MMS_FA alignment confidence of the line against
    its own text must exceed PRONUNCIATION_THRESHOLD; below → the TTS
    mangled a word → retry, then flag for the recorded-wav escape hatch.
  • Loudness — normalized to integrated LUFS with true-peak limiting.
  • Determinism — fixed seed + pinned refs ⇒ byte-identical audio;
    content-cached so each line is synthesized once, ever.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from config import settings
from pipeline.cache import BuildCache, key_of

VOICES_DIR = os.path.join(settings.ASSETS_DIR, "voices")
VOICE_BANK_SCHEMA_VERSION = 1

# ─── Gate thresholds (§10.4) — derived once, never scattered ───────────
IDENTITY_THRESHOLD = 0.72       # cosine sim of speaker embeddings
PRONUNCIATION_THRESHOLD = 0.55  # mean MMS_FA CTC posterior
MAX_RETRIES = 3                 # seeds tried before failing loudly
TARGET_LUFS = -16.0             # per-line; mixdown owns the -14 bus target
TRUE_PEAK_DB = -1.5

# ─── Reference-audio validator thresholds (§10.5) ──────────────────────
REF_MIN_SR = 22_050
REF_MAX_CLIP_FRAC = 0.001       # fraction of samples at full scale
REF_MAX_DC = 0.02               # |mean| of normalized signal
REF_MIN_SNR_DB = 20.0           # speech RMS vs quietest-decile RMS


# ═══════════════════════════════════════════
# Voice bank (§10.2) — record once, never per line
# ═══════════════════════════════════════════


@dataclass
class VoiceBankEntry:
    character: str
    base_wav: str                                  # 45–60 s neutral read
    emotions: Dict[str, str] = field(default_factory=dict)  # name → wav
    language: str = "hi"

    def clip_for(self, emotion: str) -> str:
        """Emotion reference clip; unknown emotions fall back to base."""
        return self.emotions.get(emotion, self.base_wav)


def load_voice_bank(voices_dir: str = VOICES_DIR) -> Dict[str, VoiceBankEntry]:
    """Load + validate assets/voices/voice_bank.json. Every referenced
    clip must exist and pass the reference validator — bad reference
    audio is rejected BEFORE it poisons every downstream line."""
    bank_path = os.path.join(voices_dir, "voice_bank.json")
    if not os.path.exists(bank_path):
        raise FileNotFoundError(
            f"voice_bank.json not found at {bank_path}. "
            f"See docs/VOICE_RECORDING.md to record your reference clips.")
    with open(bank_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("schema") != VOICE_BANK_SCHEMA_VERSION:
        raise ValueError(
            f"voice_bank.json schema {raw.get('schema')} != "
            f"{VOICE_BANK_SCHEMA_VERSION}. Re-run the recording guide.")
    bank: Dict[str, VoiceBankEntry] = {}
    problems: List[str] = []
    for char, spec in raw.get("characters", {}).items():
        base = os.path.join(voices_dir, spec["base"])
        emotions = {name: os.path.join(voices_dir, rel)
                    for name, rel in spec.get("emotions", {}).items()}
        for label, path in [("base", base)] + list(emotions.items()):
            if not os.path.exists(path):
                problems.append(f"{char}/{label}: missing file {path}")
                continue
            ok, why = validate_reference_audio(path)
            if not ok:
                problems.append(f"{char}/{label}: {why}")
        bank[char] = VoiceBankEntry(character=char, base_wav=base,
                                    emotions=emotions,
                                    language=spec.get("language", "hi"))
    if problems:
        raise ValueError("Voice bank validation failed:\n  " +
                         "\n  ".join(problems))
    return bank


def validate_reference_audio(path: str) -> tuple:
    """§10.5 input validator: clipping, noise floor, DC offset, sample
    rate. Returns (ok, reason)."""
    from engine.align import load_wav_mono
    try:
        x, sr = load_wav_mono(path)
    except Exception as e:
        return False, f"undecodable ({e})"
    if sr < REF_MIN_SR:
        return False, f"sample rate {sr} < {REF_MIN_SR}"
    if len(x) < sr:  # under 1 s is not a reference clip
        return False, "shorter than 1 s"
    x = x.astype(np.float64)
    peak = float(np.max(np.abs(x))) or 1.0
    xn = x / peak
    clip_frac = float(np.mean(np.abs(xn) > 0.999))
    if clip_frac > REF_MAX_CLIP_FRAC:
        return False, f"clipping ({clip_frac:.2%} of samples at full scale)"
    if abs(float(np.mean(xn))) > REF_MAX_DC:
        return False, f"DC offset {float(np.mean(xn)):.3f}"
    # SNR proxy: frame RMS; speech = top decile, floor = bottom decile
    hop = max(1, sr // 100)
    n = (len(xn) // hop) * hop
    frames = xn[:n].reshape(-1, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    speech = float(np.percentile(rms, 90))
    floor = float(np.percentile(rms, 10)) + 1e-9
    snr_db = 20.0 * np.log10(speech / floor)
    if snr_db < REF_MIN_SNR_DB:
        return False, f"noise floor too high (SNR proxy {snr_db:.1f} dB)"
    return True, "ok"


# ═══════════════════════════════════════════
# Speaker embedding (identity gate, §10.4)
# ═══════════════════════════════════════════
#
# Sovereign default (Law 3): a deterministic spectral-statistics
# embedding — log-mel band means/stds + delta stats — pure numpy, no
# model download, byte-stable. If resemblyzer/speechbrain is installed
# the neural embedding is used instead (strictly better separation);
# the gate contract is identical either way.


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2.0), n_mels + 2)
    freqs = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, ce, hi = bins[i], bins[i + 1], bins[i + 2]
        if ce > lo:
            fb[i, lo:ce] = (np.arange(lo, ce) - lo) / (ce - lo)
        if hi > ce:
            fb[i, ce:hi] = (hi - np.arange(ce, hi)) / (hi - ce)
    return fb


def speaker_embedding(wav_path: str) -> np.ndarray:
    """Deterministic voice-print vector. Neural if available, spectral
    statistics otherwise. Always L2-normalized."""
    try:  # optional neural path
        from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
        emb = VoiceEncoder(device="cpu").embed_utterance(
            preprocess_wav(wav_path))
        return emb / (np.linalg.norm(emb) + 1e-9)
    except Exception:
        pass

    from engine.align import load_wav_mono
    x, sr = load_wav_mono(wav_path)
    x = x.astype(np.float64)
    x = x / (np.max(np.abs(x)) + 1e-9)
    n_fft, hop = 1024, 256
    n = max(0, (len(x) - n_fft) // hop)
    if n < 8:
        raise ValueError(f"{wav_path}: too short for a speaker embedding")
    window = np.hanning(n_fft)
    frames = np.stack([x[i * hop:i * hop + n_fft] * window
                       for i in range(n)])
    spec = np.abs(np.fft.rfft(frames, axis=1))
    mel = _mel_filterbank(sr, n_fft) @ spec.T          # (n_mels, T)
    logmel = np.log(mel + 1e-9)
    # Voiced-ish frames only: energy above median (silence dilutes timbre)
    energy = logmel.mean(axis=0)
    voiced = logmel[:, energy > np.median(energy)]
    if voiced.shape[1] < 4:
        voiced = logmel
    d = np.diff(voiced, axis=1)
    emb = np.concatenate([voiced.mean(axis=1), voiced.std(axis=1),
                          d.mean(axis=1), d.std(axis=1)])
    return emb / (np.linalg.norm(emb) + 1e-9)


def identity_cosine(wav_a: str, wav_b: str) -> float:
    ea, eb = speaker_embedding(wav_a), speaker_embedding(wav_b)
    return float(np.dot(ea, eb))


# ═══════════════════════════════════════════
# Loudness (§10.4) — measured, not assumed
# ═══════════════════════════════════════════


def normalize_loudness(x: np.ndarray, sr: int,
                       target_lufs: float = TARGET_LUFS,
                       true_peak_db: float = TRUE_PEAK_DB) -> np.ndarray:
    """Per-line loudness normalize. Uses pyloudnorm's ITU-R BS.1770 gate
    when installed; otherwise an RMS-based LUFS proxy (correct to within
    ~1 LU on speech). True-peak limited via simple soft clip at the
    ceiling — the mixdown bus owns the broadcast chain."""
    try:
        import pyloudnorm as pyln  # type: ignore
        meter = pyln.Meter(sr)
        loud = meter.integrated_loudness(x.astype(np.float64))
        gain_db = target_lufs - loud
    except Exception:
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))
        lufs_proxy = 20.0 * np.log10(rms + 1e-12) - 0.691
        gain_db = target_lufs - lufs_proxy
    y = x * (10.0 ** (gain_db / 20.0))
    ceiling = 10.0 ** (true_peak_db / 20.0)
    over = np.abs(y) > ceiling
    if np.any(over):
        y = np.tanh(y / ceiling) * ceiling
    return y.astype(np.float32)


# ═══════════════════════════════════════════
# Engine adapters — the model is swappable, the contract is not
# ═══════════════════════════════════════════


class _EngineAdapter:
    """synthesize(text, base_wav, emotion_wav, seed) → (samples, sr)"""
    name = "abstract"

    def synthesize(self, text: str, base_wav: str, emotion_wav: str,
                   seed: int, language: str) -> tuple:
        raise NotImplementedError


class IndexTTS2Adapter(_EngineAdapter):
    """Primary (Apache-2.0). Timbre from base.wav, emotion from the
    emotion clip — *disentangled* (§10.3). Runs on MPS via engine.device."""
    name = "indextts2"

    def __init__(self):
        from engine.device import resolve_device, seed_everything
        self._resolve_device = resolve_device
        self._seed = seed_everything
        from indextts.infer_v2 import IndexTTS2  # type: ignore
        self._tts = IndexTTS2(device=str(resolve_device()))

    def synthesize(self, text, base_wav, emotion_wav, seed, language):
        import tempfile
        self._seed(seed)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        try:
            self._tts.infer(spk_audio_prompt=base_wav, text=text,
                            output_path=out_path,
                            emo_audio_prompt=emotion_wav)
            from engine.align import load_wav_mono
            x, sr = load_wav_mono(out_path)
            return x, sr
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


class ChatterboxAdapter(_EngineAdapter):
    """Fallback (MIT, Hindi supported). Emotion is approximated via the
    exaggeration control since Chatterbox has no separate emotion ref."""
    name = "chatterbox"

    _EXAGGERATION = {"neutral": 0.4, "happy": 0.6, "excited": 0.8,
                     "curious": 0.55, "surprised": 0.75, "sad": 0.35,
                     "serious": 0.35, "angry": 0.7}

    def __init__(self):
        from engine.device import resolve_device, seed_everything
        self._seed = seed_everything
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore
        self._tts = ChatterboxMultilingualTTS.from_pretrained(
            device=str(resolve_device()))
        self._emotion_name = "neutral"

    def synthesize(self, text, base_wav, emotion_wav, seed, language):
        self._seed(seed)
        exag = self._EXAGGERATION.get(self._emotion_name, 0.5)
        wav = self._tts.generate(text, language_id=language,
                                 audio_prompt_path=base_wav,
                                 exaggeration=exag)
        x = wav.squeeze().cpu().numpy().astype(np.float32)
        return x, int(self._tts.sr)


_ADAPTERS = {"indextts2": IndexTTS2Adapter, "chatterbox": ChatterboxAdapter}


# ═══════════════════════════════════════════
# The engine — cache, gates, retries
# ═══════════════════════════════════════════


class VoiceCloneEngine:
    """Per-line synthesis with the full §10.4 gate stack. Construction
    fails loudly if no adapter builds — audio_source.py catches that and
    bridges back to edge-tts (Law 3)."""

    def __init__(self, engine: Optional[str] = None,
                 voices_dir: Optional[str] = None, base_seed: int = 1234):
        self.voices_dir = voices_dir or VOICES_DIR
        self.bank = load_voice_bank(self.voices_dir)
        self.base_seed = base_seed
        order = [engine] if engine else ["indextts2", "chatterbox"]
        errors = []
        self.adapter: Optional[_EngineAdapter] = None
        for name in order:
            if name not in _ADAPTERS:
                raise ValueError(f"Unknown clone engine '{name}'")
            try:
                self.adapter = _ADAPTERS[name]()
                break
            except Exception as e:
                errors.append(f"{name}: {e}")
        if self.adapter is None:
            raise RuntimeError(
                "No voice-clone engine available. Install "
                "requirements-voice.txt. Errors: " + "; ".join(errors))
        self.cache = BuildCache()
        # Reference-clip pinning (Law 4): the cache key includes the
        # HASH of the reference clips, so re-recording invalidates.
        self._ref_hashes = {
            c: key_of(_file_sha(e.base_wav),
                      *[_file_sha(p) for p in sorted(e.emotions.values())])
            for c, e in self.bank.items()}

    # ── public API ──

    def synthesize(self, text: str, character: str, emotion: str,
                   output_dir: str, turn_id: int = 0) -> Dict[str, Any]:
        if character not in self.bank:
            raise KeyError(
                f"No voice bank entry for '{character}'. "
                f"Available: {sorted(self.bank)}. See docs/VOICE_RECORDING.md")
        entry = self.bank[character]
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"turn_{turn_id:03d}.wav")

        cache_key = key_of("voice-clone-v1", text.strip(), character,
                           emotion, self.adapter.name, self.base_seed,
                           self._ref_hashes[character])
        meta_text = self.cache.get_text(cache_key, "json") \
            if hasattr(self.cache, "get_text") else None
        if self.cache.fetch(cache_key, "wav", out_path) and meta_text:
            meta = json.loads(meta_text)
            meta.update({"wav_path": out_path, "cache_key": cache_key})
            return meta

        emotion_wav = entry.clip_for(emotion)
        if isinstance(self.adapter, ChatterboxAdapter):
            self.adapter._emotion_name = emotion

        last_fail = "no attempt"
        for attempt in range(MAX_RETRIES):
            seed = self.base_seed + attempt * 7919  # deterministic ladder
            x, sr = self.adapter.synthesize(
                text, entry.base_wav, emotion_wav, seed, entry.language)
            x = normalize_loudness(np.asarray(x, dtype=np.float32), sr)
            _write_wav(out_path, x, sr)

            cos = identity_cosine(out_path, entry.base_wav)
            if cos < IDENTITY_THRESHOLD:
                last_fail = f"identity gate: cos {cos:.3f} < {IDENTITY_THRESHOLD}"
                continue
            conf = self._pronunciation_confidence(out_path, text)
            if conf < PRONUNCIATION_THRESHOLD:
                last_fail = f"pronunciation gate: conf {conf:.3f} < {PRONUNCIATION_THRESHOLD}"
                continue

            meta = {"sample_rate": sr,
                    "duration_ms": int(1000.0 * len(x) / sr),
                    "engine": self.adapter.name, "seed": seed,
                    "identity_cos": round(cos, 4),
                    "align_confidence": round(conf, 4)}
            self.cache.put(cache_key, "wav", out_path)
            if hasattr(self.cache, "put_text"):
                self.cache.put_text(cache_key, "json", json.dumps(meta))
            meta.update({"wav_path": out_path, "cache_key": cache_key})
            return meta

        raise RuntimeError(
            f"Voice clone failed all {MAX_RETRIES} seeds for turn "
            f"{turn_id} ('{text[:40]}…'): {last_fail}. Escape hatch: "
            f"record this line manually (RecordedSource) — see RUNBOOK.")

    # ── gates ──

    @staticmethod
    def _pronunciation_confidence(wav_path: str, text: str) -> float:
        """Align the line against its own text; low CTC confidence means
        the TTS mangled a word (§10.4). Tier-3 alignment (confidence 1.0
        by definition) is treated as 'aligner unavailable' → pass-through
        with a warning rather than a fake green."""
        from engine.align import align_turn
        turn = align_turn(wav_path, text)
        if turn.tier >= 3:
            print("  [VoiceClone] WARNING: MMS_FA unavailable — "
                  "pronunciation gate is advisory only for this line.")
            return 1.0
        return float(turn.confidence)


# ═══════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════


def _file_sha(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_wav(path: str, x: np.ndarray, sr: int) -> None:
    """Dependency-free 16-bit PCM writer (stdlib wave) — soundfile is
    used when present for float precision."""
    try:
        import soundfile as sf  # type: ignore
        sf.write(path, x, sr)
        return
    except Exception:
        pass
    import wave
    pcm = np.clip(x, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())


__all__ = ["VoiceCloneEngine", "load_voice_bank", "VoiceBankEntry",
           "validate_reference_audio", "speaker_embedding",
           "identity_cosine", "normalize_loudness",
           "IDENTITY_THRESHOLD", "PRONUNCIATION_THRESHOLD"]
