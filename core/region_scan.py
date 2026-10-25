"""
Fast, region-limited, single-scale detection of the game's fixed UI (cast / reel / store / repair / OK).

Why (measured): the log shows "[NGỮ CẢNH] Khởi động tuần tự" every ~10 s while the bot is stuck. That is the 3.5 s
STAGE_1 timeout plus `_auto_detect_state()`, which runs SIX `match_any()` calls on the FULL 1080p frame with
multi_scale=True. An ABSENT template costs ~1.2 s that way (15 resizes + 16 matchTemplate of a 1920x1080 image;
reproduced here), so ~7 s per call. STAGE_REPAIR repeats 5 of those per loop iteration - that is why repairing the
rod and pressing OK "takes forever".

Fix: the modals and buttons are drawn by the game's UI at fixed positions and fixed size (1080p). So search each
one only in the region where it can appear, at native scale (~16 ms for a 200x60 template on a 960x810 region),
convert the frame to gray ONCE, and load the templates ONCE. One full context check drops from ~7 s to well under
0.2 s. Resolution changes are NOT handled here (that is what the old multi-scale was for): run the game at 1080p,
or give the scanner a frame resized to 1080p.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ScanTarget:
    name: str
    templates: str                              # a file or a folder of images
    region: Tuple[float, float, float, float]   # fractions of the frame: (x0, y0, x1, y1)
    threshold: float


@dataclass(frozen=True)
class ScanHit:
    found: bool
    score: float
    x: int = 0          # top-left of the best match, SCREEN coordinates
    y: int = 0
    w: int = 0
    h: int = 0
    template: str = ""


def _load_gray(path_or_dir: str) -> List[Tuple[str, np.ndarray]]:
    if os.path.isdir(path_or_dir):
        paths = [os.path.join(path_or_dir, f) for f in sorted(os.listdir(path_or_dir))
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    else:
        base, _ = os.path.splitext(path_or_dir)
        paths = [p for p in [path_or_dir] + [base + e for e in (".png", ".jpg", ".jpeg", ".bmp")] if os.path.isfile(p)][:1]
    out = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            out.append((os.path.basename(p), img))
    return out


class RegionScanner:
    def __init__(self, targets: Sequence[ScanTarget]):
        self._targets: Dict[str, ScanTarget] = {t.name: t for t in targets}
        self._tpl: Dict[str, List[Tuple[str, np.ndarray]]] = {t.name: _load_gray(t.templates) for t in targets}
        empty = [n for n, v in self._tpl.items() if not v]
        if empty:
            raise ValueError(f"No template images found for: {', '.join(empty)}")

    def scan(self, frame_bgr: np.ndarray, origin: Tuple[int, int] = (0, 0),
             names: Optional[Iterable[str]] = None) -> Dict[str, ScanHit]:
        gray = _to_gray(frame_bgr)
        return {n: self._scan_one(gray, origin, n) for n in (names or self._targets)}

    def first_found(self, frame_bgr: np.ndarray, order: Sequence[str],
                    origin: Tuple[int, int] = (0, 0)) -> Optional[Tuple[str, ScanHit]]:
        """Checks targets in priority order and STOPS at the first hit (later ones cost nothing)."""
        gray = _to_gray(frame_bgr)
        for n in order:
            hit = self._scan_one(gray, origin, n)
            if hit.found:
                return n, hit
        return None

    def _scan_one(self, gray: np.ndarray, origin: Tuple[int, int], name: str) -> ScanHit:
        t = self._targets[name]
        fh, fw = gray.shape[:2]
        x0, y0, x1, y1 = int(fw * t.region[0]), int(fh * t.region[1]), int(fw * t.region[2]), int(fh * t.region[3])
        roi = gray[y0:y1, x0:x1]
        best = ScanHit(False, 0.0)
        for tname, tpl in self._tpl[name]:
            if roi.shape[0] < tpl.shape[0] or roi.shape[1] < tpl.shape[1]:
                continue
            _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED))
            if np.isfinite(score) and score > best.score:
                best = ScanHit(score >= t.threshold, float(score), origin[0] + x0 + loc[0], origin[1] + y0 + loc[1],
                               tpl.shape[1], tpl.shape[0], tname)
        return best


def _to_gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
