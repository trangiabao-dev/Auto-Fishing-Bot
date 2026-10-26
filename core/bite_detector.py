"""
Structural bite ('!') detector v2 - COLOUR-AGNOSTIC.

Why the template matcher cannot be the trigger (measured on the recorded evidence): the '!' is a semi-transparent
capsule ~19x48 px whose colour changes with the map (lavender / pink / deep pink / white), while the templates are
60-90 px boxes that are mostly background. Normalised correlation scores 0.75-0.82 on unrelated scenery, so no
threshold separates "bite" from "no bite" across maps.

What a real bite has that nothing else in the head ROI has - ALL FOUR are required, none of them is a colour:
  1. NOVELTY   It APPEARS suddenly. The current frame is compared with the median of frames 0.35-1.6 s older, so
               anything that was already there (sky, water, the title badge, the name plate) can never trigger,
               whatever its colour.
  2. SHAPE     A compact upright capsule (~19x48 px at zoom 1.0): aspect H/W >= 1.6, filled >= 45 %. Sparkles and
               bobbers are round, mist and thought-bubbles are big and shapeless.
  3. POSITION  Centred on the head axis (the ROI is built symmetrically around the plate). A neighbour's props
               to the side are rejected.
  4. CONTRAST  RELATIVE, not absolute: the blob must differ from the ring of pixels right around it by at least
               `min_contrast` (distance in OpenCV 8-bit Lab). This replaces the old absolute test
               "min(B,G,R) mean >= 130 and chroma <= 100", which silently assumed a near-white pastel '!' and would
               reject a deep-pink '!' (e.g. BGR 156,112,168) on a pink or purple map.
It must also be seen on `persist_frames` consecutive scans at (almost) the same place: the '!' lives 400-600 ms,
a scan takes ~30-60 ms.

Zoom: pass `scale` (PlateHit.scale). The old code derived the zoom from the ROI width (roi_w / 240), but roi_w also
depends on the LENGTH OF THE PLAYER'S NAME, so every size limit was wrong whenever the name was not 'columbinaa'.

Known limit (physics, not a bug): a '!' whose colour is within ~`diff_thresh` of the background it appears on has
no signal to detect. min_contrast must be calibrated on real clips (tools/replay_bite_detector.py prints the
contrast of every hit so the value can be set from data).
"""
from __future__ import annotations

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
    contrast: float = 0.0   # Lab distance between the blob and the ring around it (diagnostics + calibration)
    aspect: float = 0.0     # h / w


def _odd(n: float, lo: int = 3) -> int:
    v = max(lo, int(round(n)))
    return v if v % 2 else v + 1


class BiteDetector:
    _REF_W = 240.0                 # ROI width at zoom 1.0; only used when `scale` is not given (legacy)
    _W_RANGE = (10.0, 34.0)        # capsule width limits at zoom 1.0 (px)
    _H_RANGE = (28.0, 70.0)        # capsule height limits at zoom 1.0 (px)
    _Y_RANGE = (0.25, 0.95)        # allowed vertical band inside the ROI (fractions of ROI height)

    def __init__(self, history_sec: float = 1.6, min_age_sec: float = 0.35, diff_thresh: int = 24,
                 persist_frames: int = 2, min_contrast: float = 20.0, center_tol: float = 0.12,
                 min_aspect: float = 1.6, min_fill: float = 0.45, link_ratio: float = 0.6,
                 max_changed_frac: float = 0.25):
        self._history = history_sec
        self._min_age = min_age_sec
        self._diff = diff_thresh
        self._persist = persist_frames
        self._min_contrast = min_contrast
        self._center_tol = center_tol
        self._min_aspect = min_aspect
        self._min_fill = min_fill
        self._link_ratio = link_ratio
        self._max_changed = max_changed_frac
        self._frames: Deque[Tuple[float, np.ndarray]] = deque()
        self._streak = 0
        self._last: Optional[BiteHit] = None

    def reset(self) -> None:
        """Call when the ROI moves / the camera pans or zooms / a new cast starts: stored frames no longer line up."""
        self._frames.clear()
        self._miss()

    def update(self, roi_bgr: np.ndarray, now: float, scale: Optional[float] = None) -> Optional[BiteHit]:
        """Feed ONE head-ROI frame. Returns a BiteHit once the bite has been seen on `persist_frames` scans."""
        roi_h, roi_w = roi_bgr.shape[:2]
        k = float(scale) if scale else roi_w / self._REF_W
        k = min(3.0, max(0.3, k))
        cx = roi_w / 2.0
        x0, x1 = int(cx - 0.22 * roi_w), int(cx + 0.22 * roi_w)
        strip = roi_bgr[:, x0:x1]

        refs = [f for t, f in self._frames if self._min_age <= now - t <= self._history and f.shape == strip.shape]
        self._frames.append((now, strip.copy()))
        while self._frames and now - self._frames[0][0] > self._history:
            self._frames.popleft()
        if len(refs) < 3:
            return self._miss()

        ref = np.median(np.stack(refs[-8:]), axis=0).astype(np.int16)
        diff = np.abs(strip.astype(np.int16) - ref).max(axis=2)
        mask = (diff > self._diff).astype(np.uint8)
        if float(mask.mean()) > self._max_changed:   # camera shake / scene change, not a bite: forget history
            self.reset()
            return None

        ko, kc = _odd(3 * k), _odd(5 * k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((ko, ko), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kc, kc), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        best: Optional[BiteHit] = None
        best_area = -1
        wlo, whi = self._W_RANGE[0] * k, self._W_RANGE[1] * k
        hlo, hhi = self._H_RANGE[0] * k, self._H_RANGE[1] * k
        for i in range(1, n):
            X, Y, W, H, area = (int(v) for v in stats[i])
            if not (wlo <= W <= whi and hlo <= H <= hhi):
                continue
            if H / W < self._min_aspect or area / (W * H) < self._min_fill:
                continue
            if abs((x0 + X + W / 2.0) - cx) > self._center_tol * roi_w:
                continue
            if Y < self._Y_RANGE[0] * roi_h or Y + H > self._Y_RANGE[1] * roi_h:
                continue
            comp = labels[Y:Y + H, X:X + W] == i
            contrast = self._local_contrast(strip, comp, X, Y, W, H, k)
            if contrast is None or contrast < self._min_contrast:
                continue
            mean = strip[Y:Y + H, X:X + W][comp].mean(axis=0)
            if area > best_area:
                best_area = area
                best = BiteHit(x0 + X, Y, W, H, area / float(W * H), tuple(int(v) for v in mean),
                               contrast=float(contrast), aspect=H / float(W))

        if best is None:
            return self._miss()
        linked = self._last is not None and self._linked(best, self._last)
        self._streak = self._streak + 1 if linked else 1
        self._last = best
        return best if self._streak >= self._persist else None

    # ------------------------------------------------------------------ internals
    def _miss(self) -> None:
        self._streak = 0
        self._last = None
        return None

    def _linked(self, a: BiteHit, b: BiteHit) -> bool:
        ax, ay = a.x + a.w / 2.0, a.y + a.h / 2.0
        bx, by = b.x + b.w / 2.0, b.y + b.h / 2.0
        return float(np.hypot(ax - bx, ay - by)) <= self._link_ratio * max(a.h, b.h)

    @staticmethod
    def _local_contrast(strip: np.ndarray, comp: np.ndarray, X: int, Y: int, W: int, H: int, k: float) -> Optional[float]:
        """
        Colour distance (OpenCV 8-bit Lab, median of each region) between the blob and a ring of pixels around it.
        Relative by construction: it never asks "is this pink?", only "does this stand out from what surrounds it?".
        Returns None when there is no usable ring (blob glued to the strip border).
        """
        ring_w = max(3, int(round(0.35 * min(W, H))))
        gap = max(1, int(round(1.5 * k)))
        reach = ring_w + gap
        ph, pw = strip.shape[:2]
        ax0, ay0 = max(0, X - reach), max(0, Y - reach)
        ax1, ay1 = min(pw, X + W + reach), min(ph, Y + H + reach)
        region = strip[ay0:ay1, ax0:ax1]
        m = np.zeros(region.shape[:2], np.uint8)
        m[Y - ay0:Y - ay0 + H, X - ax0:X - ax0 + W] = comp.astype(np.uint8)
        outer = cv2.dilate(m, np.ones((2 * reach + 1, 2 * reach + 1), np.uint8))
        inner_guard = cv2.dilate(m, np.ones((2 * gap + 1, 2 * gap + 1), np.uint8))
        ring = (outer > 0) & (inner_guard == 0)
        if int(m.sum()) < 8 or int(ring.sum()) < 8:
            return None
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        inside = np.median(lab[m > 0], axis=0)
        around = np.median(lab[ring], axis=0)
        return float(np.linalg.norm(inside - around))
