"""
Click guard: WHERE the bot is allowed to click, decided from the pixels, not from the template box.

Two real mis-click hazards (measured on 1917x1078 screenshots):

1) After a catch the popup shows  [Bảo quản] [Khoe]  side by side:
     Bảo quản  blue   x 1395-1625  y 779-872   (230x93, HSV ~ 98,187,243)
     Khoe      yellow x 1639-1870  same row    (14 px GAP, HSV ~ 23,226,255)
     'Thành viên Bán tức thì' orange coin icon, x 1217-1329 (a floating shortcut left of the popup)
   A template box that is even slightly too big (or a match one button off) makes a random click land on Khoe.
   -> keep_button_core(): find the BLUE pill that contains the match, return its inner core (inset 14 % / 22 %),
      and refuse (None) when yellow pixels are close to that core or the pill is not found. The bot never clicks
      "somewhere near" Bảo quản: it clicks inside the blue pill or not at all.

2) Before casting, the UI is a cluster (Hough-measured, 1080p):  bobber/cast r=52, X r=36, '+' r=39, big arrow r=89.
   Edges of the other buttons are 89-102 px from the cast button's centre; the cast button's own edge is 52 px.
   -> cast_button_core(): a small square at the centre of the matched bobber icon (side = half the button diameter),
      so even a wrong template box can only produce a click within ~20 px of the icon centre. The X button (the one
      that throws the rod away) is physically unreachable.

Keyboard: NOT used. In the cast cluster the Space badge is on the big arrow button, the mouse badge on the bobber,
and the popup shows no key hint for Bảo quản, so Space would trigger a different action.

All sizes scale with the SCREEN height (REF_H = 1078); pass the full screen height, not a crop's height.
Boxes are (x, y, w, h) in SCREEN coordinates; the returned core is too and is meant to be fed to
TemplateMatcher.get_random_click_point (anti-clustering and the circle/capsule geometry still apply).
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]

REF_H = 1078.0
CAST_BUTTON_DIAMETER = 104.0                 # px at REF_H
BLUE_LO, BLUE_HI = (90, 120, 160), (112, 255, 255)        # Bảo quản face (measured 98,187,243)
YELLOW_LO, YELLOW_HI = (12, 140, 190), (34, 255, 255)     # Khoe (23,226,255) and the orange 'Bán tức thì' icon


def cast_button_core(box: Box, screen_h: float) -> Box:
    """Square around the CENTRE of the matched bobber icon; side = half the button diameter (~26 px radius)."""
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(8, int(round(0.5 * CAST_BUTTON_DIAMETER * screen_h / REF_H)))
    return (int(round(cx - side / 2.0)), int(round(cy - side / 2.0)), side, side)


def keep_button_core(frame_bgr: np.ndarray, origin: Tuple[int, int], box: Box, screen_h: float,
                     inset_x: float = 0.14, inset_y: float = 0.22) -> Optional[Box]:
    """Inner click area of the blue Bảo quản pill that contains `box`; None = do not click."""
    fh, fw = frame_bgr.shape[:2]
    s = screen_h / REF_H
    ox, oy = origin
    x, y, w, h = box
    pad = int(round(40 * s))
    lx0, ly0 = max(0, x - ox - pad), max(0, y - oy - pad)
    lx1, ly1 = min(fw, x - ox + w + pad), min(fh, y - oy + h + pad)
    if lx1 - lx0 < 10 or ly1 - ly0 < 10:
        return None
    hsv = cv2.cvtColor(frame_bgr[ly0:ly1, lx0:lx1], cv2.COLOR_BGR2HSV)
    blue = cv2.morphologyEx(cv2.inRange(hsv, BLUE_LO, BLUE_HI), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blue)
    if n <= 1:
        return None

    cx, cy = x - ox + w // 2 - lx0, y - oy + h // 2 - ly0
    idx = int(labels[cy, cx]) if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1] else 0
    if idx == 0:                       # centre not on blue: take the component overlapping the box the most
        bx0, by0 = max(0, x - ox - lx0), max(0, y - oy - ly0)
        inside = labels[by0:by0 + h, bx0:bx0 + w]
        counts = np.bincount(inside[inside > 0].ravel(), minlength=n) if inside.size else np.zeros(n, int)
        idx = int(counts.argmax()) if counts.size and counts.max() > 0 else 0
        if idx == 0:
            return None
    px, py, pw, ph, _ = (int(v) for v in stats[idx])
    if not (100 * s <= pw <= 330 * s and 40 * s <= ph <= 150 * s):      # not a button-sized pill
        return None

    ix, iy = int(round(pw * inset_x)), int(round(ph * inset_y))
    core = (px + ix, py + iy, pw - 2 * ix, ph - 2 * iy)
    if core[2] < 10 or core[3] < 10:
        return None

    # veto: Khoe / 'Bán tức thì' must not be anywhere near the area we are about to click
    reach = max(3, int(round(0.3 * ph)))
    yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    near = cv2.dilate(yellow, np.ones((2 * reach + 1, 2 * reach + 1), np.uint8))
    if near[core[1]:core[1] + core[3], core[0]:core[0] + core[2]].any():
        return None
    return (core[0] + lx0 + ox, core[1] + ly0 + oy, core[2], core[3])
