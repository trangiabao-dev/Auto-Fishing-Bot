"""
Value objects for screen geometry.

Why this exists (audit finding): the engine mixes two coordinate spaces. `_find_repair_button`
receives GLOBAL match coordinates and uses them as LOCAL indices into the frame, and its branches
return a mix of local and global results. It only works while the capture origin is (0, 0).

The fix is to put the coordinate space into the type:
  * `Rect`  - an immutable x/y/w/h box. Pure arithmetic, no OpenCV.
  * `Frame` - a pixel array TOGETHER with the screen position of its top-left pixel (`origin`).
              Converting between "inside this frame" and "on the screen" is a method, not a
              hand-written `+ offset` scattered through the code.

Nothing in the existing code is forced to use these yet (strangler approach): new modules use them,
old modules are migrated one at a time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import numpy as np


class Point(NamedTuple):
    x: int
    y: int


@dataclass(frozen=True)
class Rect:
    """Axis-aligned box. Edges are half-open: x in [x, x + w), y in [y, y + h)."""
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w < 0 or self.h < 0:
            raise ValueError(f"Rect cannot have a negative size: {self!r}")

    @classmethod
    def from_tuple(cls, box: Tuple[int, int, int, int]) -> "Rect":
        x, y, w, h = box
        return cls(int(x), int(y), int(w), int(h))

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> Point:
        return Point(self.x + self.w // 2, self.y + self.h // 2)

    @property
    def is_empty(self) -> bool:
        return self.w == 0 or self.h == 0

    def translate(self, dx: int, dy: int) -> "Rect":
        return Rect(self.x + dx, self.y + dy, self.w, self.h)

    def inflate(self, dx: int, dy: Optional[int] = None) -> "Rect":
        """Grows the box by dx/dy on EACH side (negative values shrink it; size never goes below 0)."""
        dy = dx if dy is None else dy
        return Rect(self.x - dx, self.y - dy, max(0, self.w + 2 * dx), max(0, self.h + 2 * dy))

    def intersect(self, other: "Rect") -> Optional["Rect"]:
        """Overlapping area, or None when the boxes do not overlap (touching edges do not count)."""
        x0, y0 = max(self.x, other.x), max(self.y, other.y)
        x1, y1 = min(self.right, other.right), min(self.bottom, other.bottom)
        if x1 <= x0 or y1 <= y0:
            return None
        return Rect(x0, y0, x1 - x0, y1 - y0)

    def contains(self, point: Point) -> bool:
        px, py = point
        return self.x <= px < self.right and self.y <= py < self.bottom


@dataclass(frozen=True, eq=False)  # eq=False: comparing numpy arrays with == is ambiguous
class Frame:
    """A captured image plus the SCREEN position of its top-left pixel."""
    image: np.ndarray
    origin: Point = Point(0, 0)

    def __post_init__(self) -> None:
        if self.image.ndim not in (2, 3):
            raise ValueError(f"Frame image must be 2-D or 3-D, got shape {self.image.shape}")

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def bounds(self) -> Rect:
        """The area this frame covers, in SCREEN coordinates."""
        return Rect(self.origin.x, self.origin.y, self.width, self.height)

    def to_global(self, local: Rect) -> Rect:
        """A box measured inside this frame -> the same box on the screen."""
        return local.translate(self.origin.x, self.origin.y)

    def to_local(self, global_rect: Rect) -> Rect:
        """A box on the screen -> the same box measured inside this frame."""
        return global_rect.translate(-self.origin.x, -self.origin.y)

    def crop(self, region: Rect) -> Optional["Frame"]:
        """
        Sub-frame for `region` (SCREEN coordinates), clipped to what this frame covers.
        Returns None when there is no overlap. The pixel data is a view, not a copy.
        """
        inter = region.intersect(self.bounds)
        if inter is None:
            return None
        local = self.to_local(inter)
        view = self.image[local.y:local.bottom, local.x:local.right]
        return Frame(view, Point(inter.x, inter.y))
