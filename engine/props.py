"""Prop layer system (3.1): pose-anchored props (pencil, chalk, book) that may
only appear or vanish exactly on scene cuts -- never during cross-fades, so a
prop can never pop into existence mid-shot."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from PIL import Image


@dataclass
class PropDefinition:
    name: str
    image: Image.Image                                # RGBA, transparent bg
    anchors: Dict[str, Tuple[int, int, float]]        # pose_name -> (x, y, rot deg)


class PropLayer:
    """Cut-gated prop overlay. Visibility changes are latched and only
    applied on the next hard cut, preventing mid-transition ghost props."""

    def __init__(self, prop: PropDefinition):
        self.prop = prop
        self._visible = False
        self._pending: Optional[bool] = None
        self._rot_cache: Dict[float, Image.Image] = {}

    def set_visible(self, show: bool) -> None:
        """Latched: state change only applies at the next cut frame."""
        self._pending = show

    def on_cut(self) -> None:
        """Commit pending visibility at a hard cut boundary."""
        if self._pending is not None:
            self._visible = self._pending
            self._pending = None

    def _rotated(self, rot: float) -> Image.Image:
        key = round(rot, 1)
        if key not in self._rot_cache:
            self._rot_cache[key] = self.prop.image.rotate(
                rot, resample=Image.BICUBIC, expand=True)
        return self._rot_cache[key]

    def composite(self, canvas: Image.Image, pose_name: str,
                  offset: Tuple[int, int] = (0, 0)) -> Image.Image:
        """Overlay the prop onto canvas if visible and pose has an anchor."""
        if not self._visible or pose_name not in self.prop.anchors:
            return canvas
        x, y, rot = self.prop.anchors[pose_name]
        img = self._rotated(rot)
        canvas.alpha_composite(img, dest=(x + offset[0] - img.width // 2,
                                          y + offset[1] - img.height // 2))
        return canvas
