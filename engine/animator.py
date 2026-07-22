"""
JEEVidya — Animation System
Keyframe-based animation engine that drives smooth, timed visual transitions.
Each animation modifies element properties over time using easing functions.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from engine.easing import ease_out_cubic, EasingFunc, clamp


@dataclass
class Keyframe:
    """A single keyframe defining a property value at a point in time."""
    time: float          # Time in seconds from scene start
    value: Any           # Target value at this time
    easing: EasingFunc = ease_out_cubic


@dataclass
class AnimatedProperty:
    """
    An animated property that interpolates between keyframes over time.
    Supports float, tuple (for colors/positions), and string (discrete).
    """
    name: str
    keyframes: List[Keyframe] = field(default_factory=list)

    def add_keyframe(self, time: float, value: Any, easing: EasingFunc = ease_out_cubic) -> None:
        """Add a keyframe at the given time."""
        self.keyframes.append(Keyframe(time=time, value=value, easing=easing))
        self.keyframes.sort(key=lambda k: k.time)

    def get_value(self, time: float) -> Any:
        """Get the interpolated value at the given time."""
        if not self.keyframes:
            return None

        # Before first keyframe
        if time <= self.keyframes[0].time:
            return self.keyframes[0].value

        # After last keyframe
        if time >= self.keyframes[-1].time:
            return self.keyframes[-1].value

        # Find surrounding keyframes
        for i in range(len(self.keyframes) - 1):
            k1 = self.keyframes[i]
            k2 = self.keyframes[i + 1]

            if k1.time <= time <= k2.time:
                # Calculate local t
                duration = k2.time - k1.time
                if duration == 0:
                    return k2.value
                local_t = (time - k1.time) / duration

                # Apply easing
                eased_t = k2.easing(clamp(local_t))

                # Interpolate based on type
                return self._interpolate(k1.value, k2.value, eased_t)

        return self.keyframes[-1].value

    @staticmethod
    def _interpolate(start: Any, end: Any, t: float) -> Any:
        """Interpolate between two values based on their type."""
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            return start + (end - start) * t

        if isinstance(start, tuple) and isinstance(end, tuple):
            return tuple(
                s + (e - s) * t if isinstance(s, (int, float)) else s
                for s, e in zip(start, end)
            )

        # For non-numeric types, snap at 50%
        return end if t >= 0.5 else start


@dataclass
class SceneElement:
    """
    A visual element in a scene with animated properties.
    The scene renderer reads these properties each frame to draw the element.
    """
    element_id: str
    element_type: str    # "circle", "line", "arrow", "text", "formula", etc.
    properties: Dict[str, AnimatedProperty] = field(default_factory=dict)
    draw_params: Dict[str, Any] = field(default_factory=dict)  # Static params
    visible: bool = True
    layer: int = 0       # Drawing order (higher = on top)

    def set_property(self, name: str, time: float, value: Any,
                     easing: EasingFunc = ease_out_cubic) -> None:
        """Set an animated property value at a specific time."""
        if name not in self.properties:
            self.properties[name] = AnimatedProperty(name=name)
        self.properties[name].add_keyframe(time, value, easing)

    def get_property(self, name: str, time: float, default: Any = None) -> Any:
        """Get the current value of an animated property."""
        if name in self.properties:
            return self.properties[name].get_value(time)
        return self.draw_params.get(name, default)

    def set_static(self, name: str, value: Any) -> None:
        """Set a non-animated (static) property."""
        self.draw_params[name] = value


class AnimationTimeline:
    """
    Manages the complete timeline of a scene, tracking all elements
    and their animated properties. The scene renderer queries this
    each frame to know what to draw.
    """

    def __init__(self, duration: float):
        self.duration = duration
        self.elements: Dict[str, SceneElement] = {}
        self._next_id: int = 0

    def add_element(
        self, element_type: str,
        element_id: str = None,
        layer: int = 0,
        **static_params
    ) -> SceneElement:
        """
        Add a new visual element to the timeline.
        Returns the SceneElement for further animation configuration.
        """
        if element_id is None:
            element_id = f"{element_type}_{self._next_id}"
            self._next_id += 1

        element = SceneElement(
            element_id=element_id,
            element_type=element_type,
            layer=layer,
        )
        element.draw_params = static_params
        self.elements[element_id] = element
        return element

    def get_elements_at(self, time: float) -> List[SceneElement]:
        """Get all visible elements at the given time, sorted by layer."""
        visible = []
        for elem in self.elements.values():
            if not elem.visible:
                continue
            # Check if element has opacity — if 0, skip
            opacity = elem.get_property('opacity', time, 1.0)
            if opacity > 0.01:
                visible.append(elem)

        return sorted(visible, key=lambda e: e.layer)

    def remove_element(self, element_id: str) -> None:
        """Remove an element from the timeline."""
        self.elements.pop(element_id, None)


# === Convenience Animation Builders ===

def fade_in(element: SceneElement, start: float, duration: float = 0.4,
            easing: EasingFunc = ease_out_cubic) -> None:
    """Animate element fading in."""
    element.set_property('opacity', start, 0.0)
    element.set_property('opacity', start + duration, 1.0, easing)


def fade_out(element: SceneElement, start: float, duration: float = 0.4,
             easing: EasingFunc = ease_out_cubic) -> None:
    """Animate element fading out."""
    element.set_property('opacity', start, 1.0)
    element.set_property('opacity', start + duration, 0.0, easing)


def draw_progressive(element: SceneElement, start: float, duration: float = 0.8,
                      easing: EasingFunc = ease_out_cubic) -> None:
    """Animate element being drawn progressively (lines, circles)."""
    element.set_property('progress', start, 0.0)
    element.set_property('progress', start + duration, 1.0, easing)
    fade_in(element, start, duration * 0.3)


def write_on(element: SceneElement, start: float, duration: float = 1.0,
             easing: EasingFunc = ease_out_cubic) -> None:
    """Animate text appearing character by character."""
    element.set_property('progress', start, 0.0)
    element.set_property('progress', start + duration, 1.0, easing)
    fade_in(element, start, 0.2)


def move_to(element: SceneElement, start: float, duration: float,
            x: float = None, y: float = None,
            easing: EasingFunc = ease_out_cubic) -> None:
    """Animate element moving to a new position."""
    if x is not None:
        element.set_property('x', start + duration, x, easing)
    if y is not None:
        element.set_property('y', start + duration, y, easing)


def scale_to(element: SceneElement, start: float, duration: float,
             target_scale: float, easing: EasingFunc = ease_out_cubic) -> None:
    """Animate element scaling."""
    element.set_property('scale', start + duration, target_scale, easing)


def highlight_pulse(element: SceneElement, start: float, duration: float = 0.6) -> None:
    """Create a highlight pulse effect (scale up then back)."""
    element.set_property('scale', start, 1.0)
    element.set_property('scale', start + duration * 0.4, 1.15, ease_out_cubic)
    element.set_property('scale', start + duration, 1.0, ease_out_cubic)


def dim(element: SceneElement, start: float, duration: float = 0.3,
        target_opacity: float = 0.3) -> None:
    """Dim an element to background status."""
    element.set_property('opacity', start + duration, target_opacity, ease_out_cubic)
