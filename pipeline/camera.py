"""
Gudiya & Chintu — Cinematic Camera System
Defines shot types and manages smooth transitions between camera presets.
Each shot type is a set of (x, y, scale, opacity) values per character.
"""
from typing import Dict, Tuple

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
        Initiate a camera transition to a new shot type.
        If the shot is the same, no transition occurs.
        """
        if shot_name not in brand.SHOT_PRESETS:
            return
        if shot_name == self.current_shot and not self.is_transitioning:
            return

        self.prev_shot = self.current_shot
        self.current_shot = shot_name
        self.transition_frame = 0
        self.is_transitioning = True

    def update(self) -> None:
        """Advance the transition by one frame."""
        if self.is_transitioning:
            self.transition_frame += 1
            if self.transition_frame >= self.transition_total:
                self.is_transitioning = False

    def get_character_params(self, role: str) -> Dict:
        """
        Get the current (x, y, scale, opacity) for a character role.
        role: "active" or "inactive"
        
        During transitions, interpolates between previous and current presets.
        """
        current_preset = brand.SHOT_PRESETS.get(self.current_shot, brand.SHOT_PRESETS["two_shot"])
        current_params = current_preset.get(role, current_preset["active"])

        if not self.is_transitioning:
            return current_params.copy()

        # Interpolate from previous to current
        prev_preset = brand.SHOT_PRESETS.get(self.prev_shot, brand.SHOT_PRESETS["two_shot"])
        prev_params = prev_preset.get(role, prev_preset["active"])

        return camera_transition(
            self.transition_frame,
            self.transition_total,
            prev_params,
            current_params,
        )

    def get_both_characters(self, active_speaker: str) -> Tuple[Dict, Dict]:
        """
        Get params for both Gudiya and Chintu based on who's speaking.
        
        Args:
            active_speaker: "girl" or "boy"
            
        Returns:
            (gudiya_params, chintu_params)
        """
        if active_speaker == "girl":
            return self.get_character_params("active"), self.get_character_params("inactive")
        elif active_speaker == "boy":
            return self.get_character_params("inactive"), self.get_character_params("active")
        else:
            # Explanation: both get "active" (which maps to corner positions)
            return self.get_character_params("active"), self.get_character_params("inactive")
