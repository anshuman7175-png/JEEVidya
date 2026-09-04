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

#
# SHORTS SAFE-ZONE FRAMING (1080×1920, character height h = 0.55·H·scale)
#   YouTube Shorts player UI — conservative end of the published 2025–26
#   safe-zone ranges, since the overlay shifts by device/app version:
#     top bar        y <250
#     ACTION RAIL    x ≥900 @ y 900–1560  (like / dislike / comment / share / remix)
#     metadata row   y ≥1500             (@channel · Subscribe · title · audio)
#     nav bar        y ≥1810
#   caption band (settings.CAPTION_Y_POSITION 0.19, 2 lines + stroke/shadow
#   padding, as compositor_v5 computes it)                     : y 266–464
#   head clear line (band bottom + 30 px clearance)            : y 493+
#
# The safe window (y 250 → 1500) is filled edge to edge: caption right
# under the header, characters directly under the caption, FEET at 1480 —
# just above the title row, so the legs are never hidden behind the
# @channel / Subscribe / title block. Nothing empty above, nothing under
# a button. Faces land in the 26–38 % band (upper-middle: peak attention
# in vertical-video eye-tracking).
#
# Three constraints solve every talking preset below:
#   (1) head top never under the text:   y − 1056·scale ≥ 493
#       two_shot active   : 1480 − 1056·0.93 = 498 ✓   (largest that fits)
#       two_shot listener : 1468 − 1056·0.85 = 571 ✓   (one plane back)
#   (2) feet above the metadata row:     y ≤ 1500   (all presets: 1480)
#       — this is what caps the speaker at 0.93: header → caption → head
#       → feet → title leaves exactly ~990 px of body.
#   (3) Chintu's OPAQUE body never under the action rail:
#       x + 0.1475·h ≤ 900   (his body is 0.295·h wide — measured from
#       assets/characters/chintu/body.png; Gudiya's is 0.485·h)
#       two_shot active   : 750 + 145 = 895 ✓   listener : 765 + 132 = 897 ✓
#       extreme_closeup   : 720 + 145 = 865 ✓
#     Gudiya has no rail on her side, so she keeps the mirror-ish offset;
#     the pair's centre (≈500) sits slightly LEFT of frame centre — the
#     "shift left of the rail" convention every safe-zone guide recommends.
#     Arm gestures may still flick past x=900 — transient, and the rail
#     only has icons at discrete points, so that is acceptable.
#   Gudiya's anchor (310 active, 285 listener) ensures her maximum arm
#   reach (0.310·h = 304 px / 278 px) stays strictly inside the frame
#   (x ≥ 0) even under dynamic camera push-in with EDGE_SLACK_FRAC = 0.00.
# pipeline/compositor_v5._below_caption + compositor._safe_anchor enforce
# (1) and on-screen bounds at run time for camera push-ins.
SHOT_PRESETS: Dict[str, Dict] = {
    # Normal dialogue: Gudiya left, Chintu right. Speaker forward & large,
    # listener one plane back (smaller, feet slightly higher = further).
    "two_shot": {
        "girl_active":   {"x": 310, "y": 1480, "scale": 0.93, "opacity": 1.0},
        "girl_inactive": {"x": 285, "y": 1468, "scale": 0.85, "opacity": 1.0},
        "boy_active":    {"x": 750, "y": 1480, "scale": 0.93, "opacity": 1.0},
        "boy_inactive":  {"x": 765, "y": 1468, "scale": 0.85, "opacity": 1.0},
        "active":        {"x": 310, "y": 1480, "scale": 0.93, "opacity": 1.0},
        "inactive":      {"x": 765, "y": 1468, "scale": 0.85, "opacity": 1.0},
    },
    # Speaker solo medium shot: side-framed on the character's native side
    "medium": {
        "girl_active":   {"x": 310, "y": 1480, "scale": 0.93, "opacity": 1.0},
        "girl_inactive": {"x": 285, "y": 1468, "scale": 0.0,  "opacity": 0.0},
        "boy_active":    {"x": 750, "y": 1480, "scale": 0.93, "opacity": 1.0},
        "boy_inactive":  {"x": 765, "y": 1468, "scale": 0.0,  "opacity": 0.0},
        "active":        {"x": 310, "y": 1480, "scale": 0.93, "opacity": 1.0},
        "inactive":      {"x": 765, "y": 1468, "scale": 0.0,  "opacity": 0.0},
    },
    # Hook / dramatic close-up: the speaker alone, pulled toward centre
    # frame at full size. Legs stay visible (feet 1480) — the intimacy
    # comes from the centred framing plus the cinematics rack-focus /
    # push-in that engine.cinematics applies to this shot type.
    "extreme_closeup": {
        "girl_active":   {"x": 340, "y": 1480, "scale": 0.93, "opacity": 1.0},
        "girl_inactive": {"x": 285, "y": 1468, "scale": 0.0,  "opacity": 0.0},
        "boy_active":    {"x": 720, "y": 1480, "scale": 0.93, "opacity": 1.0},
        "boy_inactive":  {"x": 765, "y": 1468, "scale": 0.0,  "opacity": 0.0},
        "active":        {"x": 340, "y": 1480, "scale": 0.93, "opacity": 1.0},
        "inactive":      {"x": 765, "y": 1468, "scale": 0.0,  "opacity": 0.0},
    },
    # Full-screen explanation: characters tiny in the bottom corners, feet
    # just ABOVE the metadata row (y 1500) and Chintu left of the rail.
    # h = 316 → heads at y 1164, well below the board content (which the
    # renderer centres around y 860 and never draws past ≈1300).
    "fullscreen_explain": {
        "girl_active":   {"x": 150, "y": 1480, "scale": 0.30, "opacity": 0.4},
        "girl_inactive": {"x": 150, "y": 1480, "scale": 0.30, "opacity": 0.4},
        "boy_active":    {"x": 850, "y": 1480, "scale": 0.30, "opacity": 0.4},
        "boy_inactive":  {"x": 850, "y": 1480, "scale": 0.30, "opacity": 0.4},
        "active":        {"x": 150, "y": 1480, "scale": 0.30, "opacity": 0.4},
        "inactive":      {"x": 850, "y": 1480, "scale": 0.30, "opacity": 0.4},
    },
    # Reaction cut: reacting listener prominent on their native side
    "reaction_cut": {
        "girl_active":   {"x": 310, "y": 1480, "scale": 0.93, "opacity": 1.0},
        "girl_inactive": {"x": 285, "y": 1468, "scale": 0.0,  "opacity": 0.0},
        "boy_active":    {"x": 750, "y": 1480, "scale": 0.93, "opacity": 1.0},
        "boy_inactive":  {"x": 765, "y": 1468, "scale": 0.0,  "opacity": 0.0},
        "active":        {"x": 750, "y": 1480, "scale": 0.93, "opacity": 1.0},
        "inactive":      {"x": 310, "y": 1468, "scale": 0.0,  "opacity": 0.0},
    },
    # Reveal: both characters small, content big above. h = 528 → heads at
    # y 952, faces to ≈1079; feet at 1480 keep the legs clear of the title.
    "reveal": {
        "girl_active":   {"x": 350, "y": 1480, "scale": 0.50, "opacity": 1.0},
        "girl_inactive": {"x": 350, "y": 1480, "scale": 0.50, "opacity": 1.0},
        "boy_active":    {"x": 730, "y": 1480, "scale": 0.50, "opacity": 1.0},
        "boy_inactive":  {"x": 730, "y": 1480, "scale": 0.50, "opacity": 1.0},
        "active":        {"x": 350, "y": 1480, "scale": 0.50, "opacity": 1.0},
        "inactive":      {"x": 730, "y": 1480, "scale": 0.50, "opacity": 1.0},
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
