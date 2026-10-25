"""
Static-hit veto: a '!' APPEARS, scenery does not.

Evidence (real telemetry, 25 triggers): 17 of 25 triggers were false, with scores 0.727-0.87, while true bites
scored 0.752-0.956, so NO threshold separates them (a 0.85 cut would drop 2 of the 8 real catches and keep a
miss). Several false triggers were the SAME object: five hits on the swan boat at x = 622-623 (+-1 px), always
2.3-2.4 s after the cast, score 0.831-0.833; in the morning log two hits at (1127, 363) and (1126, 361) 30 s apart.
The 2 s "grace period" only postpones them.

Rule: before the bot reels, look at the SAME screen position in ROI frames that are 0.8-3 s OLD. If the template
already scored (almost) as high there, the thing was already on screen: static scenery or shimmer that crossed
the threshold, not a new '!'. A real '!' lives 0.4-0.6 s, so frames older than 0.8 s cannot contain it.

Properties
  * colour-agnostic, no thresholds on saturation/aspect: works for any weather / water colour;
  * rolling reference (not one baseline frame): follows slow zoom/pan and moving foam;
  * conservative: with no old frames, or when the camera moved so the position is outside the old frame, it
    does NOT veto (never blocks a real bite for lack of evidence);
  * one tiny matchTemplate per old frame (~0.1 ms), only when a hit already passed the threshold.
Not covered: FALSE hits that move (foam drifting) or that first appear less than 0.8 s earlier.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple

import cv2
import numpy as np

from core.geometry import Rect


@dataclass(frozen=True)
class VetoResult:
    static: bool
    ref_score: Optional[float]   # best score of the same template at the same place in old frames (None: no evidence)
    ref_age: Optional[float]     # age (s) of the frame that gave it


class StaticVeto:
    def __init__(self, history_s: float = 3.0, min_age_s: float = 0.8, margin: float = 0.10, pad: int = 4,
                 max_refs: int = 4, clock: Callable[[], float] = time.perf_counter):
        self._history = history_s
        self._min_age = min_age_s
        self._margin = margin
        self._pad = pad
        self._max_refs = max_refs
        self._clock = clock
        self._frames: Deque[Tuple[float, np.ndarray, Tuple[int, int]]] = deque()

    def reset(self) -> None:
        self._frames.clear()

    def observe(self, frame_bgr: np.ndarray, origin: Tuple[int, int] = (0, 0), t: Optional[float] = None) -> None:
        """Call once per scan tick with the ROI that is being matched (cheap: gray copy of a small ROI)."""
        now = self._clock() if t is None else t
        gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._frames.append((now, gray.copy(), (int(origin[0]), int(origin[1]))))
        while self._frames and now - self._frames[0][0] > self._history:
            self._frames.popleft()

    def is_static(self, box: Rect, score: float, template_gray: np.ndarray, t: Optional[float] = None) -> VetoResult:
        """`box`: where the template matched, in SCREEN coordinates. `template_gray`: the template as matched."""
        now = self._clock() if t is None else t
        refs = [f for f in self._frames if self._min_age <= now - f[0] <= self._history]
        if not refs:
            return VetoResult(False, None, None)
        if len(refs) > self._max_refs:   # spread over the available ages instead of taking the newest few
            idx = np.linspace(0, len(refs) - 1, self._max_refs).round().astype(int)
            refs = [refs[i] for i in sorted(set(idx.tolist()))]

        best_score, best_age = None, None
        th, tw = template_gray.shape[:2]
        for ft, gray, (ox, oy) in refs:
            lx, ly = box.x - ox, box.y - oy
            x0, y0 = max(0, lx - self._pad), max(0, ly - self._pad)
            x1, y1 = min(gray.shape[1], lx + box.w + self._pad), min(gray.shape[0], ly + box.h + self._pad)
            if lx < 0 or ly < 0 or lx + box.w > gray.shape[1] or ly + box.h > gray.shape[0]:
                continue                  # the position is outside that old frame (camera moved): no evidence
            patch = gray[y0:y1, x0:x1]
            if patch.shape[0] < th or patch.shape[1] < tw:
                continue
            _, sc, _, _ = cv2.minMaxLoc(cv2.matchTemplate(patch, template_gray, cv2.TM_CCOEFF_NORMED))
            if np.isfinite(sc) and (best_score is None or sc > best_score):
                best_score, best_age = float(sc), now - ft
        if best_score is None:
            return VetoResult(False, None, None)
        return VetoResult(best_score >= score - self._margin, best_score, best_age)
