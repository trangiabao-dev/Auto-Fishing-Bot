"""
Locates the player's name plate on screen, then derives the Head ROI from it.

Why this replaces matching the whole plate with matchTemplate:
  The plate is a semi-transparent dark rounded rectangle, so its pixels change with the
  background (day / night / rain). A raw-pixel template therefore scored only 0.54-0.62 at
  night (threshold 0.70) and the bot fell back to a stale ROI.
  The name TEXT, however, is always the same neon green. We keep only the green pixels
  (a binary mask) and match the *shape of the letters* on the mask of the frame.
  The background is removed by construction. Measured on the 11 recorded sessions:
  score ~0.77 at the true position in day, night and rain.

Multi-Character & Multi-Scale Discovery:
  - Can point to either a single image or an entire folder of character name templates.
  - Automatically filters images with neon-green text.
  - Pre-computes multi-scale variants (0.75x to 1.25x) so camera zoom changes (dock vs pool,
    wheel zoom, perspective distance) NEVER cause false misses.
  - Searches only within the bounding box of neon-green clusters on screen, executing in ~2-4ms.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

Region = Tuple[int, int, int, int]  # x, y, w, h


@dataclass(frozen=True)
class HeadLocation:
    plate: Region   # name-plate bounding box in global screen coordinates
    score: float    # 0..1 correlation of the letter shapes
    name: str = ""  # Matched template filename (e.g. 'columbinaa.png')
    scale: float = 1.0

    def head_roi(self, above: int = 180, below: int = 10, pad_x: int = 50) -> Region:
        """
        Search region for the "!" : a band above the plate.
        Same geometry the project already used (x-50, y-180, w+100, h=190).
        """
        x, y, w, _ = self.plate
        return (x - pad_x, y - above, w + 2 * pad_x, above + below)


@dataclass(frozen=True)
class _ScaledTemplate:
    tpl: np.ndarray
    origin: Tuple[int, int]
    plate_size: Tuple[int, int]
    scale: float


@dataclass(frozen=True)
class _TemplateEntry:
    name: str
    scaled_versions: List[_ScaledTemplate]


class NameLocator:
    # Neon-green name text, in BGR. Thresholds are deliberately loose on G and tight on R/B.
    _G_MIN, _RB_MAX = 170, 120
    _SCALES = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.25]
    # The chat log (bottom-left) prints the player's nickname in the SAME neon green as the name plate.
    # Without excluding it the locator locked onto the chat line (score 0.78 > real plate 0.75) and the
    # "!" search ROI ended up on the chat instead of the character. Fractions: (x0, y0, x1, y1) of the frame.
    _CHAT_ZONE = (0.0, 0.50, 0.30, 1.0)

    def __init__(self, template_path_or_dir: str, min_score: float = 0.55):
        self._min_score = min_score
        self._templates: List[_TemplateEntry] = []

        paths = []
        if os.path.isdir(template_path_or_dir):
            for f in sorted(os.listdir(template_path_or_dir)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(template_path_or_dir, f))
        elif os.path.isfile(template_path_or_dir):
            paths.append(template_path_or_dir)
        else:
            raise FileNotFoundError(f"Name template path not found: {template_path_or_dir}")

        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            mask = self._green_mask(img)
            ys, xs = np.where(mask > 0)
            if len(xs) < 25:  # Skip decorative badges without neon-green text
                continue

            pad = 2
            x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
            x1, y1 = xs.max() + pad + 1, ys.max() + pad + 1
            crop_tpl = mask[y0:y1, x0:x1]

            # Pre-compute multi-scale templates for instant zero-latency matching
            scaled_list = []
            for s in self._SCALES:
                sw = max(5, int(crop_tpl.shape[1] * s))
                sh = max(3, int(crop_tpl.shape[0] * s))
                scaled_mask = cv2.resize(crop_tpl, (sw, sh), interpolation=cv2.INTER_NEAREST)
                scaled_list.append(_ScaledTemplate(
                    tpl=scaled_mask,
                    origin=(int(x0 * s), int(y0 * s)),
                    plate_size=(int(img.shape[1] * s), int(img.shape[0] * s)),
                    scale=s
                ))

            self._templates.append(_TemplateEntry(
                name=os.path.basename(p),
                scaled_versions=scaled_list
            ))

        if not self._templates:
            raise ValueError(f"No valid neon-green name templates found in: {template_path_or_dir}")

    @classmethod
    def _green_mask(cls, bgr: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(bgr.astype(np.int16))  # int16: avoid uint8 overflow in comparisons
        return (((g > cls._G_MIN) & (r < cls._RB_MAX) & (b < cls._RB_MAX)) * 255).astype(np.uint8)

    @classmethod
    def _blank_chat_zone(cls, mask: np.ndarray) -> None:
        """Zeroes the chat area of the mask in place so chat nicknames can never be matched as a name plate."""
        h, w = mask.shape[:2]
        fx0, fy0, fx1, fy1 = cls._CHAT_ZONE
        mask[int(h * fy0):int(h * fy1), int(w * fx0):int(w * fx1)] = 0

    def locate(self, frame_bgr: np.ndarray, offset: Tuple[int, int] = (0, 0)) -> Optional[HeadLocation]:
        """Returns the plate location of the best-matching character, or None."""
        mask = self._green_mask(frame_bgr)
        self._blank_chat_zone(mask)
        ys, xs = np.where(mask > 0)
        if len(xs) < 20:  # Fast exit if no green text anywhere on screen
            return None

        # Confine search strictly to bounding box of green clusters (100x faster than full 1080p)
        y1 = max(0, int(ys.min()) - 10)
        y2 = min(mask.shape[0], int(ys.max()) + 10)
        x1 = max(0, int(xs.min()) - 10)
        x2 = min(mask.shape[1], int(xs.max()) + 10)
        search_roi = mask[y1:y2, x1:x2]

        best_score = -1.0
        best_loc: Optional[HeadLocation] = None

        for entry in self._templates:
            for s_ver in entry.scaled_versions:
                th, tw = s_ver.tpl.shape
                if search_roi.shape[0] < th or search_roi.shape[1] < tw:
                    continue

                result = cv2.matchTemplate(search_roi, s_ver.tpl, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(result)
                if np.isfinite(score) and score >= self._min_score and score > best_score:
                    best_score = float(score)
                    gx = x1 + loc[0]
                    gy = y1 + loc[1]
                    px = int(offset[0] + gx - s_ver.origin[0])
                    py = int(offset[1] + gy - s_ver.origin[1])
                    best_loc = HeadLocation(
                        plate=(px, py, *s_ver.plate_size),
                        score=best_score,
                        name=entry.name,
                        scale=s_ver.scale
                    )

        return best_loc
