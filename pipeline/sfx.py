"""
Gudiya & Chintu — Sound Effects Manager
Loads, triggers, and mixes sound effects and background music.
All SFX are CC0 (no attribution required) from Mixkit.
"""
import os
from typing import Dict, Optional

from pydub import AudioSegment

from config import settings, brand


class SFXManager:
    """Manages the sound effect library and audio mixing."""

    # Expected SFX files in assets/sfx/
    EXPECTED_FILES = {
        "whoosh": "whoosh.mp3",
        "pop": "pop.mp3",
        "bass_drop": "bass_drop.mp3",
        "achievement": "achievement.mp3",
        "bgm": "bgm_lofi.mp3",
    }

    def __init__(self):
        self._cache: Dict[str, AudioSegment] = {}
        self._load_available()

    def _load_available(self) -> None:
        """Load all available SFX files."""
        for name, filename in self.EXPECTED_FILES.items():
            path = os.path.join(settings.SFX_DIR, filename)
            if os.path.exists(path):
                try:
                    self._cache[name] = AudioSegment.from_file(path)
                except Exception as e:
                    print(f"  [SFX] Warning: Could not load {filename}: {e}")

    def get(self, name: str) -> Optional[AudioSegment]:
        """Get an SFX AudioSegment by name."""
        return self._cache.get(name)

    def get_adjusted(self, name: str, db_adjust: float = None) -> Optional[AudioSegment]:
        """Get SFX with volume adjustment."""
        sfx = self.get(name)
        if sfx is None:
            return None
        if db_adjust is None:
            db_adjust = brand.AUDIO_SFX_DB
        return sfx + db_adjust

    @property
    def available(self) -> list:
        return list(self._cache.keys())


def generate_silence(duration_ms: int) -> AudioSegment:
    """Generate silence for padding between turns."""
    return AudioSegment.silent(duration=duration_ms)


def mix_audio_layers(voice: AudioSegment,
                     sfx_events: list = None,
                     bgm: AudioSegment = None) -> AudioSegment:
    """
    Mix multiple audio layers together.
    
    Args:
        voice: The dialogue audio track (already sequenced)
        sfx_events: List of (time_ms, sfx_segment) tuples
        bgm: Background music (will be looped and ducked)
    
    Returns:
        Mixed AudioSegment
    """
    total_length = len(voice)

    # Start with voice at the right level
    mixed = voice + brand.AUDIO_VOICE_DB

    # Overlay SFX events at their timestamps
    if sfx_events:
        for time_ms, sfx in sfx_events:
            if time_ms < total_length:
                mixed = mixed.overlay(sfx + brand.AUDIO_SFX_DB, position=time_ms)

    # Add background music (looped, very quiet)
    if bgm is not None:
        # Loop BGM to cover the entire video
        bgm_ducked = bgm + brand.AUDIO_BGM_DB
        loops_needed = (total_length // len(bgm_ducked)) + 1
        bgm_loop = bgm_ducked * loops_needed
        bgm_loop = bgm_loop[:total_length]

        # Fade in/out the BGM
        bgm_loop = bgm_loop.fade_in(2000).fade_out(2000)

        mixed = mixed.overlay(bgm_loop)

    return mixed
