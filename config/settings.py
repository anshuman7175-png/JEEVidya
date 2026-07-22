"""
Gudiya & Chintu — Global Settings V2
Resolution, FPS, paths, voice config, amplitude thresholds, particle settings.
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
FPS: int = 30
TARGET_DURATION_MIN: float = 35.0
TARGET_DURATION_MAX: float = 50.0

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
AMPLITUDE_FRAME_MS: int = 33       # ~30fps granularity
AMPLITUDE_THRESHOLD_HIGH: float = -15.0   # dBFS above this → mouth_state=2 (wide open)
AMPLITUDE_THRESHOLD_MED: float = -30.0    # dBFS above this → mouth_state=1 (slightly open)
AMPLITUDE_THRESHOLD_SILENT: float = -35.0 # dBFS below this → not speaking
AMPLITUDE_SMOOTHING_FRAMES: int = 2       # Debounce rapid state changes

# === Particle System ===
PARTICLE_COUNT: int = 25           # Number of floating particles
PARTICLE_SPEED_MIN: float = 0.3   # Pixels per frame upward drift
PARTICLE_SPEED_MAX: float = 1.2
PARTICLE_SIZE_MIN: int = 2
PARTICLE_SIZE_MAX: int = 6
PARTICLE_GLOW_RADIUS: int = 8

# === Body Animation ===
BODY_BREATHE_AMPLITUDE: float = 3.0    # Pixels of vertical oscillation
BODY_BREATHE_SPEED: float = 0.08       # Oscillation frequency
BODY_SPEAK_BOUNCE: float = 5.0         # Extra bounce when speaking
BODY_SPEAK_SPEED: float = 0.4
BODY_SPEAK_SWAY: float = 2.0           # Horizontal sway
BODY_SPEAK_SCALE_PULSE: float = 0.012  # Scale pulse amplitude (1.2%)

# === Scene Timing (seconds) ===
HOOK_DURATION: float = 2.0
QUESTION_DURATION: float = 5.0
ARRIVAL_DURATION: float = 2.0
REVEAL_HOLD: float = 3.0
CTA_DURATION: float = 4.0
SCENE_TRANSITION_FRAMES: int = 8       # Frames for camera transitions
PAUSE_AFTER_REVEAL: float = 1.5
REACTION_CUT_DURATION: float = 0.5     # Quick reaction shots
INTER_TURN_PADDING_MS: int = 300       # Silence between dialogue turns

# === Caption Settings ===
CAPTION_FONT_SIZE: int = 48
CAPTION_STROKE_WIDTH: int = 3
CAPTION_Y_POSITION: float = 0.80
CAPTION_MAX_CHARS_PER_LINE: int = 24   # Shorter lines for mobile

# === rembg Settings ===
REMBG_MODEL: str = "isnet-general-use"
REMBG_ALPHA_MATTING: bool = True
REMBG_FOREGROUND_THRESHOLD: int = 220
REMBG_BACKGROUND_THRESHOLD: int = 20
REMBG_ERODE_SIZE: int = 8
