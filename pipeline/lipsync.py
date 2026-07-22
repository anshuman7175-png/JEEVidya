"""
Gudiya & Chintu — Amplitude-Driven Expression Engine
Analyzes audio amplitude per frame and selects character expression states.
Also computes body animation (breathing, bounce, sway).
This replaces SadTalker — renders in milliseconds, full 1080p quality.
"""
import math
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image
from pydub import AudioSegment
from pydub.utils import make_chunks

from config import settings, brand


# Configure pydub to use imageio_ffmpeg's bundled binary
try:
    import imageio_ffmpeg as _ioff
    AudioSegment.converter = _ioff.get_ffmpeg_exe()
    AudioSegment.ffprobe = _ioff.get_ffmpeg_exe()
except ImportError:
    pass


# ═══════════════════════════════════════════
# AUDIO AMPLITUDE ANALYSIS
# ═══════════════════════════════════════════

def _load_audio(audio_path: str) -> AudioSegment:
    """
    Load audio file into pydub AudioSegment.
    Converts to WAV first via ffmpeg subprocess to avoid ffprobe dependency.
    """
    import subprocess
    import tempfile

    # Get ffmpeg path from imageio_ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = "ffmpeg"

    # Convert to WAV in temp file
    wav_path = audio_path.rsplit(".", 1)[0] + "_tmp.wav"
    try:
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", audio_path, "-ar", "22050", "-ac", "1", wav_path],
            capture_output=True, check=True,
        )
        audio = AudioSegment.from_wav(wav_path)
        os.remove(wav_path)
        return audio
    except Exception:
        # Fallback: try direct pydub loading
        try:
            return AudioSegment.from_file(audio_path)
        except Exception:
            # Last resort: return silence
            return AudioSegment.silent(duration=2000)

def analyze_audio(audio_path: str, fps: int = settings.FPS) -> List[Dict]:
    """
    Extract per-frame amplitude data from an audio file.
    Returns a list of frame dicts with mouth_state and speaking flag.
    """
    audio = _load_audio(audio_path)
    frame_ms = 1000 / fps
    chunks = make_chunks(audio, frame_ms)

    raw_frames = []
    for chunk in chunks:
        db = chunk.dBFS if chunk.dBFS > -80 else -80

        if db > settings.AMPLITUDE_THRESHOLD_HIGH:
            mouth_state = 2    # Wide open
        elif db > settings.AMPLITUDE_THRESHOLD_MED:
            mouth_state = 1    # Slightly open
        else:
            mouth_state = 0    # Closed

        raw_frames.append({
            "db": db,
            "mouth_state": mouth_state,
            "is_speaking": db > settings.AMPLITUDE_THRESHOLD_SILENT,
        })

    # Apply smoothing to prevent jittery rapid switching
    smoothed = _smooth_mouth_states(raw_frames, settings.AMPLITUDE_SMOOTHING_FRAMES)
    return smoothed


def _smooth_mouth_states(frames: List[Dict], window: int) -> List[Dict]:
    """
    Debounce rapid mouth state changes.
    A state must persist for `window` frames to be applied.
    """
    if window <= 1 or len(frames) < 2:
        return frames

    smoothed = [frames[0].copy()]
    current_state = frames[0]["mouth_state"]
    state_count = 1

    for i in range(1, len(frames)):
        frame = frames[i].copy()
        if frame["mouth_state"] != current_state:
            state_count += 1
            if state_count >= window:
                current_state = frame["mouth_state"]
                state_count = 1
            else:
                frame["mouth_state"] = current_state
        else:
            state_count = 1
        smoothed.append(frame)

    return smoothed


# ═══════════════════════════════════════════
# EXPRESSION SELECTION
# ═══════════════════════════════════════════

def select_expression(is_active_speaker: bool, frame_data: Dict,
                      emotion: str = "neutral") -> str:
    """
    Select which expression image file to use for this frame.
    Returns the expression name (maps to filename in assets/characters/<name>/).
    """
    if not is_active_speaker:
        # Character is listening — use emotion-based reaction
        emotion_map = {
            "curious": "thinking",
            "amazed": "surprised",
            "happy": "happy",
            "enthusiastic": "happy",
            "dramatic": "surprised",
        }
        return emotion_map.get(emotion, "neutral")

    # Active speaker — map audio amplitude to mouth state
    mouth = frame_data.get("mouth_state", 0)
    if mouth == 2:
        return "talk_2"
    elif mouth == 1:
        return "talk_1"
    return "neutral"


def select_reaction_expression(emotion: str) -> str:
    """Select expression for reaction cut shots."""
    mapping = {
        "curious": "thinking",
        "amazed": "surprised",
        "happy": "happy",
        "enthusiastic": "surprised",
        "confident": "happy",
        "thinking": "thinking",
        "explaining": "neutral",
        "dramatic": "surprised",
    }
    return mapping.get(emotion, "surprised")


# ═══════════════════════════════════════════
# BODY ANIMATION
# ═══════════════════════════════════════════

def compute_body_animation(frame_num: int, is_speaking: bool,
                            base_x: float, base_y: float) -> Tuple[float, float, float]:
    """
    Compute animated position and scale for a character.
    Returns (x, y, scale_multiplier).
    
    Always applies gentle breathing oscillation.
    When speaking, adds bouncing and sway.
    """
    x, y = base_x, base_y

    # Breathing (always active) — slow vertical oscillation
    y += math.sin(frame_num * settings.BODY_BREATHE_SPEED) * settings.BODY_BREATHE_AMPLITUDE

    # Scale pulse
    scale = 1.0

    if is_speaking:
        # Speaking bounce — faster, larger vertical motion
        y += math.sin(frame_num * settings.BODY_SPEAK_SPEED) * settings.BODY_SPEAK_BOUNCE
        # Horizontal sway
        x += math.sin(frame_num * 0.25) * settings.BODY_SPEAK_SWAY
        # Scale pulse
        scale = 1.0 + math.sin(frame_num * 0.3) * settings.BODY_SPEAK_SCALE_PULSE

    return x, y, scale


# ═══════════════════════════════════════════
# EXPRESSION IMAGE LOADER
# ═══════════════════════════════════════════

class ExpressionLibrary:
    """
    Loads and caches expression images for a character.
    Images are expected at: assets/characters/<name>/<expression>.png
    Falls back to body.png / original if specific expressions aren't found.
    """

    def __init__(self, character_name: str):
        self.name = character_name
        self.char_dir = os.path.join(settings.CHARACTERS_DIR, character_name)
        self._cache: Dict[str, Image.Image] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all available expression images."""
        for expr in brand.EXPRESSIONS:
            path = os.path.join(self.char_dir, f"{expr}.png")
            if os.path.exists(path):
                self._cache[expr] = Image.open(path).convert('RGBA')

        # Try to load body image
        for ext in ['.png', '.jpg', '.jpeg']:
            body_path = os.path.join(self.char_dir, f"body{ext}")
            if os.path.exists(body_path):
                self._cache["body"] = Image.open(body_path).convert('RGBA')
                break

        # Fallback: load original image as both body and neutral
        if "body" not in self._cache:
            for ext in ['.png', '.jpg', '.jpeg']:
                orig = os.path.join(self.char_dir, f"original{ext}")
                if os.path.exists(orig):
                    self._cache["body"] = Image.open(orig).convert('RGBA')
                    if "neutral" not in self._cache:
                        self._cache["neutral"] = self._cache["body"]
                    break

    def get(self, expression: str) -> Optional[Image.Image]:
        """Get expression image, falling back to neutral → body."""
        if expression in self._cache:
            return self._cache[expression]
        if "neutral" in self._cache:
            return self._cache["neutral"]
        if "body" in self._cache:
            return self._cache["body"]
        return None

    def get_blended(self, prev_expr: str, next_expr: str,
                    progress: float) -> Optional[Image.Image]:
        """Cross-dissolve between two expression images."""
        prev_img = self.get(prev_expr)
        next_img = self.get(next_expr)

        if prev_img is None:
            return next_img
        if next_img is None:
            return prev_img
        if prev_expr == next_expr:
            return prev_img

        # Ensure same size
        if prev_img.size != next_img.size:
            next_img = next_img.resize(prev_img.size, Image.Resampling.LANCZOS)

        return Image.blend(prev_img, next_img, alpha=min(1.0, max(0.0, progress)))

    @property
    def available_expressions(self) -> List[str]:
        return list(self._cache.keys())
