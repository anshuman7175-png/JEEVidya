"""
Gudiya & Chintu — Cinematic Camera System
Defines shot types and manages smooth transitions between camera presets.
Each shot type is a set of (x, y, scale, opacity) values per character.
"""
from typing import Dict, Optional, Tuple

from config import brand, settings
from engine.transitions import camera_transition


class CameraSystem:
    """
    Manages the virtual camera: which shot type is active,
    and smoothly transitions between presets.
    """

    def __init__(self):
        self.current_shot: str = "two_shot"
        self.prev_shot: str = "two_shot"
        self.transition_frame: int = 0
        self.transition_total: int = settings.SCENE_TRANSITION_FRAMES
        self.is_transitioning: bool = False

    def cut_to(self, shot_name: str) -> None:
        """
        Initiate a camera cut to a new shot type.
        Camera shot changes are clean cinematic hard cuts so inactive
        characters never linger as ghost overlays across shots.
        """
        if shot_name not in brand.SHOT_PRESETS:
            return
        if shot_name == self.current_shot:
            return

        self.prev_shot = self.current_shot
        self.current_shot = shot_name
        self.transition_frame = 0
        self.is_transitioning = False

    def update(self) -> None:
        """Advance the frame state."""
        if self.is_transitioning:
            self.transition_frame += 1
            if self.transition_frame >= self.transition_total:
                self.is_transitioning = False

    def get_character_params(self, role: str, char_name: Optional[str] = None) -> Dict:
        """
        Get the current (x, y, scale, opacity) for a character role.
        role: "active" or "inactive"
        char_name: "girl" or "boy" (optional, enables side-anchored depth presets)
        """
        current_preset = brand.SHOT_PRESETS.get(self.current_shot, brand.SHOT_PRESETS["two_shot"])
        if char_name:
            char_key = f"{char_name}_{role}"
            if char_key in current_preset:
                current_params = current_preset[char_key]
            elif char_name in current_preset:
                current_params = current_preset[char_name]
            else:
                current_params = current_preset.get(role, current_preset.get("active", {}))
        else:
            current_params = current_preset.get(role, current_preset.get("active", {}))

        if not self.is_transitioning or current_params.get("opacity", 1.0) <= 0.01:
            return current_params.copy()

        # Interpolate from previous to current only if both presets keep character visible
        prev_preset = brand.SHOT_PRESETS.get(self.prev_shot, brand.SHOT_PRESETS["two_shot"])
        if char_name:
            char_key = f"{char_name}_{role}"
            if char_key in prev_preset:
                prev_params = prev_preset[char_key]
            elif char_name in prev_preset:
                prev_params = prev_preset[char_name]
            else:
                prev_params = prev_preset.get(role, prev_preset.get("active", {}))
        else:
            prev_params = prev_preset.get(role, prev_preset.get("active", {}))

        if prev_params.get("opacity", 1.0) <= 0.01:
            return current_params.copy()

        return camera_transition(
            self.transition_frame,
            self.transition_total,
            prev_params,
            current_params,
        )

    def get_both_characters(self, active_speaker: str) -> Tuple[Dict, Dict]:
        """
        Get params for both Gudiya and Chintu based on who's speaking.
        Preserves side anchors (Gudiya Left, Chintu Right) while dynamically
        moving the active speaker to the foreground (coming front).
        
        Args:
            active_speaker: "girl" or "boy"
            
        Returns:
            (gudiya_params, chintu_params)
        """
        if active_speaker == "girl":
            return (self.get_character_params("active", char_name="girl"),
                    self.get_character_params("inactive", char_name="boy"))
        elif active_speaker == "boy":
            return (self.get_character_params("inactive", char_name="girl"),
                    self.get_character_params("active", char_name="boy"))
        else:
            # Explanation / ambient: both get their native inactive/background positions
            return (self.get_character_params("inactive", char_name="girl"),
                    self.get_character_params("inactive", char_name="boy"))
