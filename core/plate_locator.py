"""
Name-plate locator v2: blob-based, scale ESTIMATED from geometry, hue-based colour mask.

Why it replaces core/head_locator.NameLocator (measured on the real telemetry, 35 STAGE_2 waits):
  * locate() cost depends on the BOUNDING BOX of every green pixel on screen, not on the plate: on the
    recorded sessions it was 30-60 ms with one player in view and 430-760 ms (up to 1.6 s) with other
    players' names on screen, because the box then spans several players and the old code runs
    16 scales x N templates of matchTemplate over all of it.
  * Its scale list stops at 1.25x; a zoomed-in camera makes the plate ~2.0x and locate() returns None.
  * Its colour mask (G>170, R<120, B<120) keeps only 6-52 % of the text pixels in storm / rain / fog /
    aurora frames (the neon text gets darker), so the correlation collapses.

What this module does instead
  1. HUE mask (H 50-78, S>=140, V>=110): 3-10x more text pixels in weather, foliage (teal, H~84) stays out.
  2. Connected components after a horizontal dilation: ONE blob per name. Work is proportional to the
     blobs, never to the distance between two players.
  3. The blob WIDTH gives the scale directly (scale = blob_width / template_text_width). We verify at
     three scales around that estimate instead of searching 16: covers 0.3x-3x at the same cost.
  4. PlateTracker searches only a window around the last position while tracking (camera rotation moves the
     plate, but not by more than a few hundred pixels between two scans) and falls back to a rate-limited
     global search when the plate is lost. It also refuses to jump to a distant candidate whose score
     is not clearly better (anti-hijack by construction, not by threshold).

Pure OpenCV/NumPy, no Win32, no engine dependency: unit-testable on any machine.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.geometry import Rect

HSV_LO = (50, 140, 110)
HSV_HI = (78, 255, 255)
# Fractions of the frame: (x0, y0, x1, y1). The chat log prints the player's own nickname in the same
# neon green as the plate; it must never be matched as a plate.
CHAT_ZONE = (0.0, 0.50, 0.30, 1.0)


@dataclass(frozen=True)
class PlateHit:
    plate: Rect    # estimated whole-plate box (screen coordinates), same meaning as HeadLocation.plate
    text: Rect     # tight box of the green text (screen coordinates)
    score: float   # 0..1 shape correlation against the template
    name: str      # template file name
    scale: float   # plate size relative to the template image

    def head_roi(self, above: int = 180, below: int = 10, pad_x: int = 50) -> Tuple[int, int, int, int]:
        """Drop-in for HeadLocation.head_roi(); the geometry of the '!' search band is a SEPARATE question."""
        x, y, w, _ = self.plate.to_tuple()
        return (x - pad_x, y - above, w + 2 * pad_x, above + below)


@dataclass(frozen=True)
class _Template:
    name: str
    mask: np.ndarray            # tight crop of the text mask, uint8 0/255
    text_xy: Tuple[int, int]    # where that crop sits inside the template image
    image_wh: Tuple[int, int]


def hue_mask(bgr: np.ndarray) -> np.ndarray:
    """uint8 0/255 mask of 'neon green' pixels (name-plate text)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, HSV_LO, HSV_HI)


def _load_templates(path_or_dir: str) -> List[_Template]:
    if os.path.isdir(path_or_dir):
        paths = [os.path.join(path_or_dir, f) for f in sorted(os.listdir(path_or_dir))
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    elif os.path.isfile(path_or_dir):
        paths = [path_or_dir]
    else:
        raise FileNotFoundError(f"Name template path not found: {path_or_dir}")
    out: List[_Template] = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        m = hue_mask(img)
        ys, xs = np.where(m > 0)
        if len(xs) < 25:      # decorative badge without green text
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        out.append(_Template(os.path.basename(p), m[y0:y1, x0:x1].copy(), (x0, y0),
                             (img.shape[1], img.shape[0])))
    if not out:
        raise ValueError(f"No template with neon-green text in: {path_or_dir}")
    return out


class PlateLocator:
    def __init__(self, template_path_or_dir: str, min_score: float = 0.55,
                 scale_probe: Sequence[float] = (0.92, 1.0, 1.08), max_blobs: int = 16):
        self._templates = _load_templates(template_path_or_dir)
        self._min_score = min_score
        self._probe = tuple(scale_probe)
        self._max_blobs = max_blobs

    # ------------------------------------------------------------------ public
    def locate(self, frame_bgr: np.ndarray, origin: Tuple[int, int] = (0, 0),
               window: Optional[Rect] = None) -> Optional[PlateHit]:
        hits = self.candidates(frame_bgr, origin, window)
        return hits[0] if hits else None

    def candidates(self, frame_bgr: np.ndarray, origin: Tuple[int, int] = (0, 0),
                   window: Optional[Rect] = None) -> List[PlateHit]:
        """All plausible plates (best first), one per text blob. `window` is in SCREEN coordinates."""
        fh, fw = frame_bgr.shape[:2]
        ox, oy = origin
        wx0, wy0, wx1, wy1 = 0, 0, fw, fh
        if window is not None:
            inter = window.intersect(Rect(ox, oy, fw, fh))
            if inter is None:
                return []
            wx0, wy0 = inter.x - ox, inter.y - oy
            wx1, wy1 = wx0 + inter.w, wy0 + inter.h
        crop = frame_bgr[wy0:wy1, wx0:wx1]
        mask = hue_mask(crop)

        # blank the chat log (zone defined on the FULL frame)
        cx0, cy0, cx1, cy1 = CHAT_ZONE
        mask[max(0, int(fh * cy0) - wy0):max(0, int(fh * cy1) - wy0),
             max(0, int(fw * cx0) - wx0):max(0, int(fw * cx1) - wx0)] = 0
        if int(np.count_nonzero(mask)) < 20:
            return []

        hits: List[PlateHit] = []
        for bx, by, bw, bh in self._blobs(mask, fh):
            hit = self._best_for_blob(mask, bx, by, bw, bh, (ox + wx0, oy + wy0))
            if hit is not None:
                hits.append(hit)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    # ------------------------------------------------------------------ internals
    def _blobs(self, mask: np.ndarray, frame_h: int) -> List[Tuple[int, int, int, int]]:
        # Merge the letters of one name into one component. Gap between letters grows with zoom, so the
        # kernel scales with the frame, not with a template.
        kw = max(5, int(round(frame_h * 0.012))) | 1
        merged = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3)))
        n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if h < 5 or w < 12 or h > 0.12 * frame_h or w < 1.6 * h:
                continue                      # a name is a wide, short text line
            boxes.append((area, x, y, w, h))
        boxes.sort(reverse=True)              # biggest first; cap the work in busy scenes
        return [(x, y, w, h) for _, x, y, w, h in boxes[: self._max_blobs]]

    def _best_for_blob(self, mask: np.ndarray, bx: int, by: int, bw: int, bh: int,
                       crop_origin: Tuple[int, int]) -> Optional[PlateHit]:
        sub = mask[by:by + bh, bx:bx + bw]
        ys, xs = np.where(sub > 0)
        if len(xs) < 15:
            return None
        tx0, ty0 = bx + int(xs.min()), by + int(ys.min())
        tw, th = int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1
        pad = 3
        px0, py0 = max(0, tx0 - pad), max(0, ty0 - pad)
        padded = mask[py0:ty0 + th + pad, px0:tx0 + tw + pad]

        best: Optional[PlateHit] = None
        for tpl in self._templates:
            base = tw / float(tpl.mask.shape[1])
            if base < 0.35 or base > 3.5:
                continue
            for k in self._probe:
                s = base * k
                rw, rh = max(4, int(round(tpl.mask.shape[1] * s))), max(3, int(round(tpl.mask.shape[0] * s)))
                interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                t = cv2.resize(tpl.mask, (rw, rh), interpolation=interp)
                hay = padded
                if hay.shape[0] < rh or hay.shape[1] < rw:   # template slightly larger than the blob: pad
                    hay = cv2.copyMakeBorder(hay, 0, max(0, rh - hay.shape[0]), 0, max(0, rw - hay.shape[1]),
                                             cv2.BORDER_CONSTANT, value=0)
                res = cv2.matchTemplate(hay, t, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)
                if not np.isfinite(score) or score < self._min_score:
                    continue
                if best is not None and score <= best.score:
                    continue
                s_eff = rw / float(tpl.mask.shape[1])
                text_left = crop_origin[0] + px0 + loc[0]
                text_top = crop_origin[1] + py0 + loc[1]
                plate_left = int(round(text_left - tpl.text_xy[0] * s_eff))
                plate_top = int(round(text_top - tpl.text_xy[1] * s_eff))
                best = PlateHit(
                    plate=Rect(plate_left, plate_top, int(round(tpl.image_wh[0] * s_eff)),
                               int(round(tpl.image_wh[1] * s_eff))),
                    text=Rect(crop_origin[0] + tx0, crop_origin[1] + ty0, tw, th),
                    score=float(score), name=tpl.name, scale=float(s_eff))
        return best


# ---------------------------------------------------------------------------------------------- tracking
@dataclass(frozen=True)
class TrackResult:
    hit: Optional[PlateHit]     # best knowledge of the plate (may be the last known one when stale)
    state: str                  # "TRACKING" | "STALE" | "LOST"
    ambiguous: bool = False     # a second, distant plate scored within `ambiguity_margin` (global search only)


class PlateTracker:
    """
    Keeps the plate position alive while the camera moves.

    TRACKING : search only a window around the last position (cheap; runs every tick if you want).
    STALE    : nothing found in the window for < `lost_after_s`; the last hit is returned unchanged.
    LOST     : nothing for `lost_after_s`; a global search is tried every `global_interval_s`.

    Anti-hijack rule: a candidate far from the last position replaces it only if it beats the nearest
    candidate by `jump_margin`. A stranger with a slightly higher score can no longer steal the lock.
    """

    def __init__(self, locator: PlateLocator, *, margin: Tuple[int, int] = (360, 260),
                 lost_after_s: float = 1.0, global_interval_s: float = 0.5,
                 jump_margin: float = 0.12, ambiguity_margin: float = 0.05,
                 clock: Callable[[], float] = time.perf_counter):
        self._loc = locator
        self._margin = margin
        self._lost_after = lost_after_s
        self._global_every = global_interval_s
        self._jump_margin = jump_margin
        self._amb_margin = ambiguity_margin
        self._clock = clock
        self._last: Optional[PlateHit] = None
        self._last_seen = -1e9
        self._last_global = -1e9

    def reset(self) -> None:
        self._last = None
        self._last_seen = -1e9

    def update(self, frame_bgr: np.ndarray, origin: Tuple[int, int] = (0, 0)) -> TrackResult:
        now = self._clock()
        if self._last is not None and now - self._last_seen < self._lost_after:
            window = self._last.text.inflate(*self._margin)
            hits = self._loc.candidates(frame_bgr, origin, window)
            chosen = self._choose(hits)
            if chosen is not None:
                self._last, self._last_seen = chosen, now
                return TrackResult(chosen, "TRACKING")
            return TrackResult(self._last, "STALE")

        if now - self._last_global < self._global_every:
            return TrackResult(self._last, "LOST")
        self._last_global = now
        hits = self._loc.candidates(frame_bgr, origin)
        if not hits:
            return TrackResult(self._last, "LOST")
        ambiguous = False
        if len(hits) > 1:
            far = [h for h in hits[1:] if _dist(h, hits[0]) > 0.5 * hits[0].plate.w]
            ambiguous = bool(far) and (hits[0].score - far[0].score) < self._amb_margin
        chosen = self._choose(hits) if self._last is not None else hits[0]
        self._last, self._last_seen = chosen, now
        return TrackResult(chosen, "TRACKING", ambiguous)

    def _choose(self, hits: List[PlateHit]) -> Optional[PlateHit]:
        if not hits:
            return None
        if self._last is None:
            return hits[0]
        nearest = min(hits, key=lambda h: _dist(h, self._last))
        best = hits[0]
        if best is nearest:
            return best
        # best candidate is elsewhere: only follow it when it is CLEARLY better than the nearest one
        return best if best.score - nearest.score >= self._jump_margin else nearest


def _dist(a: PlateHit, b: PlateHit) -> float:
    ac, bc = a.text.center, b.text.center
    return float(np.hypot(ac.x - bc.x, ac.y - bc.y))
