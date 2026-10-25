"""
Cloud-shape veto: a thought bubble is not a '!'.

User report: when the character "wants something" the game puts a white fluffy bubble with an icon above its head
(measured on the screenshot: cloud 106x102 px, bowling-pin icon ~33x70 px, plate text 148 px wide). The bot reeled
although no fish had bitten. This kind of false hit is NEW (it appears suddenly), so neither a higher threshold nor
StaticVeto can remove it, and a temporal-novelty detector would fire on it too.

What separates them is SHAPE and CONTEXT. A '!' is a capsule + dot: aspect (h/w) of 2-3.5. The bubble is a filled,
roughly square white blob (aspect ~1.0). So: look at the bright, low-saturation blob that contains the hit; if that
blob is cloud-like (aspect <= 1.5, filled, at least as wide as ~60 % of the hit box) the hit is a bubble, not a '!'.

Deliberately narrow: it only looks at WHITE/grey blobs (coloured '!' - green, pink, yellow, lavender - never
enter the mask), and a white '!' is a tall capsule, so it passes. Known risk: a white '!' drawn over a big white
background (snow, fog) merges into a blob and could be vetoed - the corpus recorder will show if that happens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from core.geometry import Rect


@dataclass(frozen=True)
class CloudResult:
    cloud_like: bool
    blob: Optional[Rect]        # bounding box of the white blob, in the coordinates of the image passed in
    aspect: Optional[float]     # blob h / w
    fill: Optional[float]       # blob area / bbox area


def is_cloud_like(roi_bgr: np.ndarray, box: Rect, *, v_min: int = 200, s_max: int = 60, close: int = 7,
                  min_w_ratio: float = 0.6, max_aspect: float = 1.5, min_fill: float = 0.5) -> CloudResult:
    """`box`: the template hit, in the coordinates of `roi_bgr` (local, not screen)."""
    h, w = roi_bgr.shape[:2]
    pad = int(1.2 * max(box.w, box.h))
    win = Rect(box.x - pad, box.y - pad, box.w + 2 * pad, box.h + 2 * pad).intersect(Rect(0, 0, w, h))
    if win is None:
        return CloudResult(False, None, None, None)
    crop = roi_bgr[win.y:win.bottom, win.x:win.right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return CloudResult(False, None, None, None)

    lb = box.translate(-win.x, -win.y).intersect(Rect(0, 0, crop.shape[1], crop.shape[0]))
    if lb is None:
        return CloudResult(False, None, None, None)
    inside = labels[lb.y:lb.bottom, lb.x:lb.right]
    counts = np.bincount(inside[inside > 0].ravel(), minlength=n)
    if counts.size == 0 or counts.max() < 0.25 * box.w * box.h:
        return CloudResult(False, None, None, None)          # no white blob under the hit
    i = int(counts.argmax())
    bx, by, bw, bh, area = (int(v) for v in stats[i])
    aspect, fill = bh / float(bw), area / float(bw * bh)
    blob = Rect(bx + win.x, by + win.y, bw, bh)
    like = aspect <= max_aspect and fill >= min_fill and bw >= min_w_ratio * box.w
    return CloudResult(like, blob, aspect, fill)
