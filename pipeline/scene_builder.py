"""
JEEVidya — Scene Builder
Converts JSON storyboard scenes into executable AnimationTimelines.
Maps string commands ("draw_circle", "show_formula") to Engine API calls.
"""
from typing import Dict, Any

from engine.animator import AnimationTimeline, fade_in, fade_out, draw_progressive, write_on, scale_to, highlight_pulse
from config import brand


class SceneBuilder:
    """Builds animation timelines from storyboard JSON data."""

    def __init__(self):
        pass

    def build_timeline(self, scene_data: Dict[str, Any], actual_audio_duration: float) -> AnimationTimeline:
        """
        Build an AnimationTimeline for a single scene.
        Adjusts timing based on the actual audio duration (so visuals sync with voice).
        """
        # We add some padding for the pause at the end of the scene
        pause = scene_data.get("pause_after", 1.5)
        total_duration = actual_audio_duration + pause
        
        timeline = AnimationTimeline(duration=total_duration)
        
        visual_elements = scene_data.get("visual_elements", [])
        
        for i, el in enumerate(visual_elements):
            action = el.get("action")
            params = el.get("params", {})
            # Use provided delay, or space things out linearly if not provided
            delay = el.get("delay", (actual_audio_duration / max(1, len(visual_elements))) * i)
            
            # Create unique ID
            el_id = f"el_{i}_{action}"
            
            self._apply_action(timeline, action, params, delay, el_id)
            
        # If there's a formula for the whole scene, show it prominently
        formula = scene_data.get("formula")
        if formula:
            # Show formula midway through narration
            formula_delay = actual_audio_duration * 0.4
            self._apply_action(timeline, "show_formula", {"latex": formula, "y": 400}, formula_delay, "scene_formula")

        return timeline

    def _apply_action(self, timeline: AnimationTimeline, action: str, params: Dict[str, Any], start_time: float, el_id: str):
        """Map a JSON action string to engine timeline elements."""
        
        # Color mapping helper
        def get_color(color_name):
            if color_name == "primary": return brand.PRIMARY
            if color_name == "secondary": return brand.SECONDARY
            if color_name == "accent": return brand.ACCENT
            if color_name == "success": return brand.SUCCESS
            return brand.PRIMARY

        if action == "draw_circle":
            elem = timeline.add_element(
                "circle",
                element_id=el_id,
                x=params.get("x", 0),
                y=params.get("y", 0),
                radius=params.get("radius", 200),
                color=get_color(params.get("color", "primary")),
                layer=1
            )
            draw_progressive(elem, start_time, 1.0)
            
        elif action == "draw_line":
            elem = timeline.add_element(
                "line",
                element_id=el_id,
                x=params.get("x1", 0),
                y=params.get("y1", 0),
                x2=params.get("x2", 100),
                y2=params.get("y2", 100),
                color=get_color(params.get("color", "primary")),
                layer=2
            )
            draw_progressive(elem, start_time, 0.8)
            
        elif action == "draw_arrow":
            elem = timeline.add_element(
                "arrow",
                element_id=el_id,
                x=params.get("x1", 0),
                y=params.get("y1", 0),
                x2=params.get("x2", 100),
                y2=params.get("y2", 100),
                color=get_color(params.get("color", "accent")),
                layer=3
            )
            draw_progressive(elem, start_time, 0.6)
            
        elif action == "show_text":
            elem = timeline.add_element(
                "text",
                element_id=el_id,
                text=params.get("text", ""),
                x=params.get("x", 0),
                y=params.get("y", 0),
                color=brand.TEXT_WHITE,
                font_size=params.get("font_size", brand.FONT_SIZE_BODY),
                layer=5
            )
            write_on(elem, start_time, 1.2)
            
        elif action == "show_formula":
            elem = timeline.add_element(
                "formula",
                element_id=el_id,
                latex=params.get("latex", ""),
                x=params.get("x", 0),
                y=params.get("y", 0),
                layer=5
            )
            # Formulas pop in
            elem.set_property('opacity', start_time, 0.0)
            elem.set_property('opacity', start_time + 0.3, 1.0)
            elem.set_property('scale', start_time, 0.5)
            elem.set_property('scale', start_time + 0.4, 1.0)
            
        elif action == "highlight":
            target_id = params.get("target_id")
            if target_id in timeline.elements:
                target = timeline.elements[target_id]
                highlight_pulse(target, start_time)
                
        elif action == "show_angle":
            elem = timeline.add_element(
                "angle_marker",
                element_id=el_id,
                x=params.get("x", 0),
                y=params.get("y", 0),
                angle1=params.get("angle1", 0),
                angle2=params.get("angle2", 90),
                label=params.get("label", ""),
                radius=params.get("radius", 40),
                layer=4
            )
            fade_in(elem, start_time, 0.5)

        # Fallback for unrecognized action
        else:
            print(f"Warning: Unknown action '{action}' in scene builder.")
