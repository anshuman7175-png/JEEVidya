"""
JEEVidya — Global Settings (Terminal Plan)
══════════════════════════════════════════
Resolution, FPS, paths, voice config, amplitude thresholds, particles.

FPS DOCTRINE (Terminal Plan, Part II §2.2)
------------------------------------------
FPS is 60. Every time-dependent quantity in this file is stored as a
PER-SECOND (or per-millisecond) quantity and converted to per-frame at
USE TIME via the derivation helpers below. A per-frame literal anywhere
in config or engine code is a lint violation (tools/lint_constants.py):
that class of bug (constants silently running 2× fast at 60 fps) is
unrepresentable by construction.

    frame_ms()            → duration of one frame in ms   (1000 / FPS)
    frames(seconds)       → integer frame count for a wall-clock duration
    per_frame(per_second) → convert a rate/speed to this FPS's step size
"""
import os

# === Project Paths ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
CHARACTERS_DIR = os.path.join(ASSETS_DIR, "characters")
SFX_DIR = os.path.join(ASSETS_DIR, "sfx")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")
TEMP_DIR = os.path.join(PROJECT_ROOT, ".tmp")

# Ensure directories exist
for d in [OUTPUT_DIR, ASSETS_DIR, CHARACTERS_DIR, SFX_DIR, FONTS_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# === Video Settings ===
WIDTH: int = 1080
HEIGHT: int = 1920
# 60 fps: a frame is 16.7 ms — short enough to sample a 15–25 ms plosive
# closure. 30 fps (33 ms frames) structurally cannot. Draft renders may
# pass --fps 30; every derived quantity below scales automatically.
FPS: int = 60
_REFERENCE_FPS: int = 30   # the FPS the legacy hand-tuned constants assumed
TARGET_DURATION_MIN: float = 35.0
TARGET_DURATION_MAX: float = 50.0


# === FPS derivation helpers (the ONLY sanctioned frame/time conversions) ===

def frame_ms(fps: int | None = None) -> float:
    """Duration of one frame in milliseconds. NEVER hardcode 33 or 16.7."""
    return 1000.0 / float(fps or FPS)


def frames(seconds: float, fps: int | None = None) -> int:
    """Wall-clock seconds → integer frame count at the given FPS."""
    return max(1, round(seconds * (fps or FPS)))


def per_frame(per_second: float, fps: int | None = None) -> float:
    """A per-second rate → the per-frame step at the given FPS."""
    return per_second / float(fps or FPS)


def set_fps(fps: int) -> None:
    """CLI --fps entry point (draft 30 / publish 60). Updates the single
    module-level FPS that every derivation reads."""
    global FPS
    FPS = int(fps)


# === Voice Profiles ===
# Girl (Gudiya): young, curious, high-pitched
VOICE_GIRL: str = "hi-IN-SwaraNeural"
VOICE_GIRL_RATE: str = "+8%"
VOICE_GIRL_PITCH: str = "+15Hz"

# Boy (Chintu): young, confident, energetic
VOICE_BOY: str = "hi-IN-MadhurNeural"
VOICE_BOY_RATE: str = "+5%"
VOICE_BOY_PITCH: str = "+20Hz"

# Legacy (keep for backward compat)
DEFAULT_VOICE: str = VOICE_BOY
VOICE_RATE: str = "+5%"
VOICE_PITCH: str = "+0Hz"

# === Amplitude Detection (for mouth animation) ===
# Analysis granularity = exactly one frame at the CURRENT fps, derived.


def amplitude_frame_ms() -> float:
    return frame_ms()


AMPLITUDE_THRESHOLD_HIGH: float = -15.0   # dBFS above this → mouth wide open
AMPLITUDE_THRESHOLD_MED: float = -30.0    # dBFS above this → slightly open
AMPLITUDE_THRESHOLD_SILENT: float = -35.0 # dBFS below this → not speaking
AMPLITUDE_SMOOTHING_S: float = 2.0 / _REFERENCE_FPS  # debounce window, seconds


def amplitude_smoothing_frames(fps: int | None = None) -> int:
    return frames(AMPLITUDE_SMOOTHING_S, fps)


# === Particle System ===
PARTICLE_COUNT: int = 25
# Speeds are PER SECOND (px/s). Use settings.per_frame() at the call site.
PARTICLE_SPEED_MIN_PS: float = 0.3 * _REFERENCE_FPS   # px/s upward drift
PARTICLE_SPEED_MAX_PS: float = 1.2 * _REFERENCE_FPS
PARTICLE_SIZE_MIN: int = 2
PARTICLE_SIZE_MAX: int = 6
PARTICLE_GLOW_RADIUS: int = 8

# === Body Animation ===
BODY_BREATHE_AMPLITUDE: float = 3.0            # px of vertical oscillation
# Angular speeds are PER SECOND (rad/s). Legacy values were rad/frame @30.
BODY_BREATHE_SPEED_PS: float = 0.08 * _REFERENCE_FPS   # rad/s
BODY_SPEAK_BOUNCE: float = 5.0                 # px extra bounce when speaking
BODY_SPEAK_SPEED_PS: float = 0.4 * _REFERENCE_FPS      # rad/s
BODY_SPEAK_SWAY: float = 2.0                   # px horizontal sway
BODY_SPEAK_SWAY_SPEED_PS: float = 0.25 * _REFERENCE_FPS
BODY_SPEAK_SCALE_PULSE: float = 0.012          # scale pulse amplitude (1.2%)
BODY_SPEAK_PULSE_SPEED_PS: float = 0.3 * _REFERENCE_FPS
BODY_BREATHE_SWELL_SPEED_PS: float = 0.007 * _REFERENCE_FPS


def body_breathe_speed(fps: int | None = None) -> float:
    """rad/frame at the current fps."""
    return per_frame(BODY_BREATHE_SPEED_PS, fps)


def body_speak_speed(fps: int | None = None) -> float:
    return per_frame(BODY_SPEAK_SPEED_PS, fps)


# Legacy aliases so untouched call sites keep working at their original
# 30 fps tuning until they migrate. These are @property-like shims that
# derive from FPS — they are NOT independent constants.
def _legacy_rate(per_second: float) -> float:
    return per_frame(per_second)


# === Scene Timing (seconds) ===
HOOK_DURATION: float = 2.0
QUESTION_DURATION: float = 5.0
ARRIVAL_DURATION: float = 2.0
REVEAL_HOLD: float = 3.0
CTA_DURATION: float = 4.0
SCENE_TRANSITION_S: float = 8.0 / _REFERENCE_FPS   # was 8 frames @30 ≈ 267 ms


def scene_transition_frames(fps: int | None = None) -> int:
    return frames(SCENE_TRANSITION_S, fps)


PAUSE_AFTER_REVEAL: float = 1.5
REACTION_CUT_DURATION: float = 0.5     # Quick reaction shots
INTER_TURN_PADDING_MS: int = 300       # Silence between dialogue turns

# === Caption Settings ===
CAPTION_FONT_SIZE: int = 60
CAPTION_STROKE_WIDTH: int = 6
CAPTION_Y_POSITION: float = 0.63  # Lower safe zone (y=1210px): 240px above Shorts title, between characters
CAPTION_MAX_CHARS_PER_LINE: int = 18   # Punchy 2-3 words per line for Shorts

# === rembg Settings ===
REMBG_MODEL: str = "isnet-general-use"
REMBG_ALPHA_MATTING: bool = True
REMBG_FOREGROUND_THRESHOLD: int = 220
REMBG_BACKGROUND_THRESHOLD: int = 20
REMBG_ERODE_SIZE: int = 8


# === Backward-compat derived constants ===============================
# These names are consumed across the legacy pipeline. They now DERIVE
# from FPS instead of being independent literals — the duplicated-
# constant bug class is dead. New code must call the functions above.
AMPLITUDE_FRAME_MS: float = frame_ms()
AMPLITUDE_SMOOTHING_FRAMES: int = amplitude_smoothing_frames()
SCENE_TRANSITION_FRAMES: int = scene_transition_frames()
PARTICLE_SPEED_MIN: float = per_frame(PARTICLE_SPEED_MIN_PS)
PARTICLE_SPEED_MAX: float = per_frame(PARTICLE_SPEED_MAX_PS)
BODY_BREATHE_SPEED: float = per_frame(BODY_BREATHE_SPEED_PS)
BODY_SPEAK_SPEED: float = per_frame(BODY_SPEAK_SPEED_PS)
