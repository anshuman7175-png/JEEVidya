"""
Gudiya & Chintu — Brand Visual System V2
The complete cinematic design language for the animated shorts factory.
Every pixel in every video references ONLY these constants.
"""
from typing import Tuple, Dict

# ═══════════════════════════════════════════
# COLOR PALETTE (Dark Premium Cinematic)
# ═══════════════════════════════════════════

# Background gradient (animated, shifts slowly)
BG_TOP: Tuple[int, int, int] = (10, 10, 46)       # Deep navy #0A0A2E
BG_BOTTOM: Tuple[int, int, int] = (26, 26, 78)    # Indigo #1A1A4E

# Primary — main elements, speaker highlight
PRIMARY: Tuple[int, int, int] = (0, 212, 255)      # Electric cyan #00D4FF
PRIMARY_HEX: str = "#00D4FF"

# Secondary — labels, annotations, active word highlight
SECONDARY: Tuple[int, int, int] = (255, 215, 0)    # Gold #FFD700
SECONDARY_HEX: str = "#FFD700"

# Accent — urgency, CTA, emphasis
ACCENT: Tuple[int, int, int] = (255, 51, 102)      # Neon pink #FF3366
ACCENT_HEX: str = "#FF3366"

# Success — final answers, reveals
SUCCESS: Tuple[int, int, int] = (0, 230, 118)      # Bright green #00E676
SUCCESS_HEX: str = "#00E676"

# Text
TEXT_WHITE: Tuple[int, int, int] = (240, 240, 255)
TEXT_DIM: Tuple[int, int, int] = (140, 140, 170)
TEXT_CAPTION: Tuple[int, int, int] = (255, 255, 255)
TEXT_CAPTION_ACTIVE: Tuple[int, int, int] = (255, 215, 0)  # Gold highlight

# Chalkboard overlay (for explanation scenes)
CHALKBOARD_BG: Tuple[int, int, int] = (20, 35, 30)         # Dark green
CHALKBOARD_GRID: Tuple[int, int, int] = (40, 60, 50)       # Faint grid lines
CHALKBOARD_TEXT: Tuple[int, int, int] = (220, 230, 210)     # Chalk white

# Structural
GRID_COLOR: Tuple[int, int, int] = (255, 255, 255)
AXIS_COLOR: Tuple[int, int, int] = (80, 80, 120)

# ═══════════════════════════════════════════
# PARTICLE SYSTEM COLORS
# ═══════════════════════════════════════════
PARTICLE_COLORS = [
    (0, 212, 255, 80),     # Cyan, 30% opacity
    (255, 215, 0, 60),     # Gold, 25% opacity
    (255, 51, 102, 50),    # Pink, 20% opacity
    (0, 230, 118, 40),     # Green, 15% opacity
]

# ═══════════════════════════════════════════
# OPACITY LAYERS (Visual hierarchy)
# ═══════════════════════════════════════════
OPACITY_FOCUS: float = 1.0
OPACITY_CONTEXT: float = 0.45
OPACITY_STRUCTURE: float = 0.12
OPACITY_DIM: float = 0.25
OPACITY_GLOW: float = 0.40
OPACITY_LISTENER: float = 0.5       # Dimmed non-speaking character
OPACITY_CORNER_CHAR: float = 0.4    # Characters in corners during explanation

# ═══════════════════════════════════════════
# TYPOGRAPHY
# ═══════════════════════════════════════════
FONT_MAIN: str = "Menlo"
FONT_HINDI: str = "Devanagari MT"
FONT_BOLD: str = "Menlo-Bold"
FONT_CAPTION: str = "Arial-Bold"     # Bold sans-serif for captions
FONT_FALLBACK: str = "Arial"

# Font sizes (1080x1920 canvas)
FONT_SIZE_TITLE: int = 72
FONT_SIZE_HEADING: int = 56
FONT_SIZE_BODY: int = 44
FONT_SIZE_LABEL: int = 36
FONT_SIZE_SMALL: int = 28
FONT_SIZE_FORMULA: int = 52
FONT_SIZE_CAPTION: int = 48        # Larger for mobile readability
FONT_SIZE_HOOK_TEXT: int = 64       # Hook text — massive

# ═══════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════
MARGIN_X: float = 0.08
MARGIN_Y: float = 0.05
CONTENT_TOP: float = 0.08
CONTENT_BOTTOM: float = 0.75
CONTENT_CENTER_Y: float = 0.42
CAPTION_TOP: float = 0.78
CAPTION_BOTTOM: float = 0.92

# ═══════════════════════════════════════════
# ANIMATION TIMING
# ═══════════════════════════════════════════
DEFAULT_FADE_DURATION: float = 0.4
DEFAULT_DRAW_DURATION: float = 0.8
DEFAULT_WRITE_DURATION: float = 1.2
DEFAULT_SCALE_DURATION: float = 0.5
EXPRESSION_BLEND_FRAMES: int = 3     # Cross-dissolve between face states
INTER_TURN_PADDING_MS: int = 300     # Silence between dialogue turns

# ═══════════════════════════════════════════
# GLOW EFFECT
# ═══════════════════════════════════════════
GLOW_RADIUS: int = 15
GLOW_COLOR: Tuple[int, int, int] = PRIMARY
GLOW_INTENSITY: float = 0.4

# ═══════════════════════════════════════════
# AUDIO MIX LEVELS (dB adjustments)
# ═══════════════════════════════════════════
AUDIO_VOICE_DB: float = -3.0       # Voice is dominant
AUDIO_SFX_DB: float = -8.0         # Noticeable but not distracting
AUDIO_BGM_DB: float = -20.0        # Barely audible warmth

# ═══════════════════════════════════════════
# CAMERA SHOT PRESETS
# Coordinates for 1080x1920 canvas
# (char_x, char_y, char_scale, char_opacity) per character
# ═══════════════════════════════════════════

SHOT_PRESETS: Dict[str, Dict] = {
    # Normal dialogue: character anchored to dedicated side (Gudiya Left, Chintu Right).
    # Active speaker steps forward into foreground (scale 0.98, y 1735).
    # Inactive listener steps back into background (scale 0.76, y 1680).
    # Center corridor [360, 720] strictly reserved for dynamic captions.
    "two_shot": {
        "girl_active":   {"x": 230, "y": 1735, "scale": 0.98, "opacity": 1.0},
        "girl_inactive": {"x": 180, "y": 1680, "scale": 0.76, "opacity": 1.0},
        "boy_active":    {"x": 850, "y": 1735, "scale": 0.98, "opacity": 1.0},
        "boy_inactive":  {"x": 900, "y": 1680, "scale": 0.76, "opacity": 1.0},
        "active":        {"x": 230, "y": 1735, "scale": 0.98, "opacity": 1.0},
        "inactive":      {"x": 900, "y": 1680, "scale": 0.76, "opacity": 1.0},
    },
    # Speaker solo medium shot: side-framed on character's native side
    "medium": {
        "girl_active":   {"x": 240, "y": 1730, "scale": 0.95, "opacity": 1.0},
        "girl_inactive": {"x": 180, "y": 1720, "scale": 0.0,  "opacity": 0.0},
        "boy_active":    {"x": 840, "y": 1730, "scale": 0.95, "opacity": 1.0},
        "boy_inactive":  {"x": 900, "y": 1720, "scale": 0.0,  "opacity": 0.0},
        "active":        {"x": 240, "y": 1730, "scale": 0.95, "opacity": 1.0},
        "inactive":      {"x": 900, "y": 1720, "scale": 0.0,  "opacity": 0.0},
    },
    # Hook / Dramatic closeup: speaker extra prominent on native side
    "extreme_closeup": {
        "girl_active":   {"x": 260, "y": 1745, "scale": 1.10, "opacity": 1.0},
        "girl_inactive": {"x": 180, "y": 1720, "scale": 0.0,  "opacity": 0.0},
        "boy_active":    {"x": 820, "y": 1745, "scale": 1.10, "opacity": 1.0},
        "boy_inactive":  {"x": 900, "y": 1720, "scale": 0.0,  "opacity": 0.0},
        "active":        {"x": 260, "y": 1745, "scale": 1.10, "opacity": 1.0},
        "inactive":      {"x": 900, "y": 1720, "scale": 0.0,  "opacity": 0.0},
    },
    # Full-screen explanation: characters tiny in corners
    "fullscreen_explain": {
        "girl_active":   {"x": 150, "y": 1850, "scale": 0.30, "opacity": 0.4},
        "girl_inactive": {"x": 150, "y": 1850, "scale": 0.30, "opacity": 0.4},
        "boy_active":    {"x": 930, "y": 1850, "scale": 0.30, "opacity": 0.4},
        "boy_inactive":  {"x": 930, "y": 1850, "scale": 0.30, "opacity": 0.4},
        "active":        {"x": 150, "y": 1850, "scale": 0.30, "opacity": 0.4},
        "inactive":      {"x": 930, "y": 1850, "scale": 0.30, "opacity": 0.4},
    },
    # Reaction cut: reacting listener prominent on their native side
    "reaction_cut": {
        "girl_active":   {"x": 240, "y": 1730, "scale": 0.95, "opacity": 1.0},
        "girl_inactive": {"x": 180, "y": 1720, "scale": 0.0,  "opacity": 0.0},
        "boy_active":    {"x": 840, "y": 1730, "scale": 0.95, "opacity": 1.0},
        "boy_inactive":  {"x": 900, "y": 1720, "scale": 0.0,  "opacity": 0.0},
        "active":        {"x": 840, "y": 1730, "scale": 0.95, "opacity": 1.0},
        "inactive":      {"x": 210, "y": 1720, "scale": 0.0,  "opacity": 0.0},
    },
    # Reveal: both characters small at bottom, content big above
    "reveal": {
        "girl_active":   {"x": 350, "y": 1750, "scale": 0.50, "opacity": 1.0},
        "girl_inactive": {"x": 350, "y": 1750, "scale": 0.50, "opacity": 1.0},
        "boy_active":    {"x": 730, "y": 1750, "scale": 0.50, "opacity": 1.0},
        "boy_inactive":  {"x": 730, "y": 1750, "scale": 0.50, "opacity": 1.0},
        "active":        {"x": 350, "y": 1750, "scale": 0.50, "opacity": 1.0},
        "inactive":      {"x": 730, "y": 1750, "scale": 0.50, "opacity": 1.0},
    },
}

# ═══════════════════════════════════════════
# EXPRESSION NAMES (files expected in assets/characters/<name>/)
# ═══════════════════════════════════════════
EXPRESSIONS = ["neutral", "talk_1", "talk_2", "surprised", "thinking", "happy", "pointing"]

# ═══════════════════════════════════════════
# GRADIENT PRESETS
# ═══════════════════════════════════════════
def bg_gradient_colors() -> list:
    """Background gradient stops."""
    return [
        (0.0, BG_TOP),
        (0.3, (14, 14, 56)),
        (0.7, (20, 20, 66)),
        (1.0, BG_BOTTOM),
    ]

def accent_gradient() -> list:
    """Primary to accent gradient for emphasis."""
    return [
        (0.0, PRIMARY),
        (1.0, ACCENT),
    ]
