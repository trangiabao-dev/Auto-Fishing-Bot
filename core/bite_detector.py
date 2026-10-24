"""
Structural bite ('!') detector - replaces grayscale template matching as the trigger for the reel click.

Why template matching was the wrong tool (measured on the recorded evidence):
  * The '!' is a semi-transparent pastel capsule only ~19x48 px, but the templates are 60-90 px boxes that
    are >85% BACKGROUND. Normalised correlation on a mostly-background template scores 0.75-0.82 on
    unrelated scenery (terrain edges, other players, effects) and its colour changes every bite
    (lavender / pink / white ...), so no threshold separates "bite" from "no bite" across maps.

What a real bite has that nothing else in the ROI has - all three are required:
  1. NOVELTY   : it APPEARS suddenly. We compare the current frame with the median of frames 0.35-1.6 s
                 older, so anything static (scenery, the badge, the name plate) can never trigger.
  2. SHAPE     : a compact upright capsule (~19x48 px at 1.0 zoom): aspect >= 1.6, filled >= 45%.
  3. POSITION  : centred horizontally on the name plate (the ROI is built symmetrically around it) and
                 pastel (bright, low chroma). Players/effects to the side or dark objects are rejected.
It must also be seen on `persist_frames` consecutive scans (the '!' lives 400-600 ms, a scan is ~30-60 ms).
"""
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class BiteHit:
    x: int          # bounding box inside the ROI frame passed to update()
    y: int
    w: int
    h: int
    fill: float     # area / bbox area, 0..1 (capsule ~0.8)
    mean_bgr: Tuple[int, int, int]


class BiteDetector:
    _REF_W = 240.0  # ROI width at zoom 1.0 that the size limits below are expressed for

    def __init__(self, history_sec: float = 1.6, min_age_sec: float = 0.35,
                 diff_thresh: int = 24, persist_frames: int = 2):
        self._history = history_sec
        self._min_age = min_age_sec
        self._diff = diff_thresh
        self._persist = persist_frames
        self._frames: Deque[Tuple[float, np.ndarray]] = deque()
        self._streak = 0

    def reset(self) -> None:
        """Call when the ROI moves or a new cast starts: stored frames no longer line up."""
        self._frames.clear()
        self._streak = 0

    def update(self, roi_bgr: np.ndarray, now: float) -> Optional[BiteHit]:
        roi_h, roi_w = roi_bgr.shape[:2]
        k = roi_w / self._REF_W
        cx = roi_w / 2.0
        x0, x1 = int(cx - 0.22 * roi_w), int(cx + 0.22 * roi_w)
        strip = roi_bgr[:, x0:x1]

        refs = [f for t, f in self._frames if self._min_age <= now - t <= self._history]
        self._frames.append((now, strip.copy()))
        while self._frames and now - self._frames[0][0] > self._history:
            self._frames.popleft()

        if len(refs) < 3 or refs[0].shape != strip.shape:
            self._streak = 0
            return None

        ref = np.median(np.stack(refs[-8:]), axis=0).astype(np.int16)
        diff = np.abs(strip.astype(np.int16) - ref).max(axis=2)
        mask = (diff > self._diff).astype(np.uint8)

        if mask.mean() > 0.25:  # camera shake / scene change, not a bite: forget history
            self.reset()
            return None

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)

        best: Optional[BiteHit] = None
        for i in range(1, n):
            X, Y, W, H, area = (int(v) for v in stats[i])
            if not (10 * k <= W <= 34 * k and 28 * k <= H <= 70 * k):
                continue
            if H / W < 1.6 or area / (W * H) < 0.45:
                continue
            if abs((x0 + X + W / 2.0) - cx) > 0.08 * roi_w:
                continue
            if Y < 0.25 * roi_h or Y + H > 0.95 * roi_h:
                continue
            px = strip[Y:Y + H, X:X + W].reshape(-1, 3).astype(np.int16)
            if px.min(axis=1).mean() < 130 or (px.max(axis=1) - px.min(axis=1)).mean() > 100:
                continue
            hit = BiteHit(x0 + X, Y, W, H, area / (W * H), tuple(int(v) for v in px.mean(axis=0)))
            if best is None or area > best.fill * best.w * best.h:
                best = hit

        self._streak = self._streak + 1 if best else 0
        return best if self._streak >= self._persist else None
