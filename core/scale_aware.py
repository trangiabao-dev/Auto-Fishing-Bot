"""
Scale-aware '!' search: the head ROI and the '!' templates follow the camera zoom.

Symptom (user's log, morning session): works with the camera close to the character, fails when zoomed out.
Cause: PlateTracker already reports the zoom (`PlateHit.scale`, 0.85 -> 0.90 -> 1.11 within 10 s in the log),
but the '!' stage still uses (a) a head ROI of fixed pixel size (above=180, pad_x=50) and (b) templates
matched at native size (multi_scale=False). A '!' that is 0.5x as big can never reach 0.72 against a 1.0x template.

Fix: derive both from the plate scale. This is "estimate, don't search": one resize per template, cached per
quantised scale (5 % steps), instead of the old 16-scale sweep that cost 400-1600 ms.

Assumption to verify: the '!' templates were cut at the same zoom as the name template (ratio 1.0). If they
were not, pass `reference_scale` (plate scale at which the '!' templates were captured).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.geometry import Rect

_MIN_SCALE, _MAX_SCALE = 0.35, 3.0


def clamp_scale(scale: float) -> float:
    return max(_MIN_SCALE, min(_MAX_SCALE, float(scale)))


def scaled_head_roi(plate: Rect, scale: float, above: int = 180, below: int = 10, pad_x: int = 50) -> Rect:
    """
    Search band for the '!' above the plate, with the pixel offsets multiplied by the zoom.
    `plate` is already in zoomed pixels (PlateHit.plate), so only the three constants are scaled.
    """
    s = clamp_scale(scale)
    a, b, p = int(round(above * s)), int(round(below * s)), int(round(pad_x * s))
    return Rect(plate.x - p, plate.y - a, plate.w + 2 * p, a + b)


@dataclass(frozen=True)
class BankMatch:
    score: float
    x: int            # top-left inside the image that was searched
    y: int
    w: int
    h: int
    name: str
    template: np.ndarray   # the resized template that produced the match (re-usable by StaticVeto)


class BiteTemplateBank:
    def __init__(self, directory: str, reference_scale: float = 1.0, quantum: float = 0.05, min_side: int = 8):
        self._ref = float(reference_scale)
        self._q = float(quantum)
        self._min_side = min_side
        self._base: List[Tuple[str, np.ndarray]] = []
        for f in sorted(os.listdir(directory)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            img = cv2.imread(os.path.join(directory, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self._base.append((f, img))
        if not self._base:
            raise ValueError(f"No '!' templates in {directory}")
        self._cache: Dict[Tuple[str, int], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._base)

    def _scaled(self, name: str, img: np.ndarray, scale: float) -> Optional[np.ndarray]:
        key = (name, int(round(scale / self._q)))
        t = self._cache.get(key)
        if t is None:
            s = key[1] * self._q
            w, h = int(round(img.shape[1] * s)), int(round(img.shape[0] * s))
            if w < self._min_side or h < self._min_side:
                return None
            t = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
            self._cache[key] = t
        return t

    def best_match(self, roi_gray: np.ndarray, plate_scale: float, min_contrast: int = 14) -> Optional[BankMatch]:
        """Best template at the zoom implied by the plate. `roi_gray` is single-channel."""
        s = clamp_scale(plate_scale) / self._ref
        best: Optional[BankMatch] = None
        for name, img in self._base:
            t = self._scaled(name, img, s)
            if t is None or roi_gray.shape[0] < t.shape[0] or roi_gray.shape[1] < t.shape[1]:
                continue
            th, tw = t.shape[:2]
            res = cv2.matchTemplate(roi_gray, t, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if np.isfinite(score) and (best is None or score > best.score):
                lx, ly = int(loc[0]), int(loc[1])
                patch = roi_gray[ly:ly + th, lx:lx + tw]
                # Reject flat regions where small variance inflates TM_CCOEFF_NORMED
                if patch.size > 0 and int(patch.max()) - int(patch.min()) < min_contrast:
                    continue
                best = BankMatch(float(score), lx, ly, tw, th, name, t)
        return best
