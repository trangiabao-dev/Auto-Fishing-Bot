"""
Digit & Card Reader for Farm Harvest Mode.
Recognizes weights (e.g. 5.08kg, 1.03kg, 125.50kg) and locates harvest buttons across grid cards.
Lightweight (15KB), ultra-fast (< 5ms per screen), fail-closed security for high-value crops.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class HarvestCard:
    card_index: int
    weight: float
    kg_x: int
    kg_y: int
    btn_x: int
    btn_y: int
    btn_w: int
    btn_h: int
    is_clipped: bool = False

    def get_click_point(self, margin_x: int = 25, margin_y: int = 15) -> Tuple[int, int]:
        """Returns a randomized safe click point strictly inside the upper-middle cyan core of 'Thu hoạch' button."""
        x0 = self.btn_x + margin_x
        x1 = self.btn_x + self.btn_w - margin_x
        # Restrict Y to upper-middle (y: 15..50 within 85px button) to guarantee hitting cyan pill
        y0 = self.btn_y + margin_y
        y1 = min(self.btn_y + self.btn_h - margin_y, self.btn_y + 50)
        if x1 <= x0:
            x1 = x0 + 1
        if y1 <= y0:
            y1 = y0 + 1
        return random.randint(x0, x1), random.randint(y0, y1)


class DigitReader:
    def __init__(self, digits_dir: Optional[str] = None):
        if digits_dir is None:
            # Default to assets/farm/digits
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            digits_dir = os.path.join(base_dir, "assets", "farm", "digits")

        self.digits_dir = digits_dir
        self.digits: Dict[str, np.ndarray] = {}
        for d in range(10):
            p = os.path.join(self.digits_dir, f"{d}.png")
            if os.path.exists(p):
                self.digits[str(d)] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

        kg_p = os.path.join(self.digits_dir, "kg.png")
        self.kg_tpl = cv2.imread(kg_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(kg_p) else None

        btn_p = os.path.join(os.path.dirname(digits_dir), "thu_hoach", "nut thu hoach.jpg")
        self.btn_tpl = cv2.imread(btn_p) if os.path.exists(btn_p) else None

    def _classify_digit_patch(self, patch: np.ndarray) -> Tuple[str, float]:
        """Matches a single digit image patch against template digits 0-9."""
        best_d = ""
        best_score = -1.0
        pw, ph = patch.shape[1], patch.shape[0]
        for d, tpl in self.digits.items():
            resized = cv2.resize(tpl, (pw, ph))
            score = cv2.matchTemplate(patch, resized, cv2.TM_CCOEFF_NORMED)[0, 0]
            if score > best_score:
                best_score = score
                best_d = d
        return best_d, best_score

    def read_weight_from_strip(self, strip_mask: np.ndarray, gap_max: int = 8) -> float:
        """
        Parses weight from a binarized mask of the strip to the left of 'kg'.
        Scans right-to-left starting closest to 'kg', clustering contiguous digits and dot.
        Rejects noise blobs, truncations, and handles weights up to 9999.99kg.
        Fails-closed (returns 9999.0) if unreadable or truncated to guarantee giant crops are never harvested.
        """
        n, _, stats, _ = cv2.connectedComponentsWithStats(strip_mask)
        blobs = []
        for bi in range(1, n):
            bx, by, bw, bh, barea = stats[bi]
            # Decimal dot in 1080p can be 3x2px (bh=2, barea>=3)
            is_dot = (2 <= bw <= 6 and 2 <= bh <= 6 and by >= 8)
            # Dot is 2..6px high, digits are >= 10px high. Anything with bh < 2 or barea < 3 is noise!
            if barea < 3 or bh < 2 or (bw < 3 and not is_dot):
                continue
            blobs.append((int(bx), int(by), int(bw), int(bh)))

        if not blobs:
            return 9999.0

        # Sort right-to-left (starting from the one nearest to 'kg')
        blobs.sort(key=lambda b: -b[0])

        group = [blobs[0]]
        for b in blobs[1:]:
            prev_left = group[-1][0]
            curr_right = b[0] + b[2]
            # If distance between left edge of previous blob and right edge of current blob > gap_max,
            # we have reached the crop name boundary!
            if prev_left - curr_right > gap_max:
                break
            group.append(b)

        # Truncation check: if the leftmost digit touches or exceeds the left strip border (bx <= 0),
        # an integer digit might have been cut off (e.g. 1000.50 -> 0.50). Reject as protected giant crop!
        if group[-1][0] <= 0:
            return 9999.0

        # Re-sort left-to-right for sequential parsing
        group.reverse()
        num_str = ""
        has_dot = False
        digits_before_dot = 0
        digits_after_dot = 0

        for bx, by, bw, bh in group:
            # A decimal dot is small and sits near the baseline (by >= 8, 2..6px wide, 2..6px high)
            if 2 <= bw <= 6 and 2 <= bh <= 6 and by >= 8:
                if not has_dot:
                    num_str += "."
                    has_dot = True
                continue

            patch = strip_mask[by:by + bh, bx:bx + bw]
            d, score = self._classify_digit_patch(patch)
            if score < 0.35:
                # Unrecognized or blurry glyph -> fail-closed to protect crops
                return 9999.0

            num_str += d
            if not has_dot:
                digits_before_dot += 1
            else:
                digits_after_dot += 1

        # STRICT PLAY TOGETHER WEIGHT VALIDATION:
        # 1. At least 1 integer digit before dot (e.g. '0.86', '3.17', '1000.50')
        # 2. Must contain exactly 1 decimal dot
        # 3. Game ALWAYS formats weights with exactly 2 decimal places (e.g. .50, .00, .15)
        if digits_before_dot == 0 or not has_dot or digits_after_dot != 2:
            return 9999.0

        try:
            return float(num_str)
        except Exception:
            return 9999.0

    def scan_cards(self, frame_bgr: np.ndarray, origin: Tuple[int, int] = (0, 0)) -> List[HarvestCard]:
        """
        Scans all visible harvest item cards in the 'Thu hoạch trái' window.
        Returns cards sorted top-to-bottom, left-to-right with exact parsed weights and button coords.
        """
        if self.kg_tpl is None:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 115, 205)

        res = cv2.matchTemplate(mask, self.kg_tpl, cv2.TM_CCOEFF_NORMED)
        locs = np.where(res >= 0.75)
        hits = []
        for pt in zip(*locs[::-1]):
            if not any(abs(pt[0] - p[0]) < 10 and abs(pt[1] - p[1]) < 10 for p in hits):
                hits.append(pt)

        # Robust row clustering by vertical proximity (dy > 25px creates a new row)
        hits.sort(key=lambda p: p[1])
        rows: List[List[Tuple[int, int]]] = []
        row_y: Optional[int] = None
        for p in hits:
            if row_y is None or p[1] - row_y > 25:
                rows.append([])
                row_y = p[1]
            rows[-1].append(p)
        hits = [p for r in rows for p in sorted(r, key=lambda q: q[0])]

        fh, fw = frame_bgr.shape[:2]
        hsv_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask_cyan_full = cv2.inRange(hsv_full, np.array([90, 100, 100]), np.array([115, 255, 255]))

        cards: List[HarvestCard] = []
        for idx, (kg_x, kg_y) in enumerate(hits):
            # Dynamic cyan button calibration:
            # Searches for the real 'Thu hoạch' button contour in the card's row slice to the right of 'kg'.
            # Immune to fruit name length differences (e.g. 'Sầu riêng' 9 chars vs 'Hạt dẻ' 6 chars).
            y0_s = max(0, kg_y - 25)
            y1_s = min(fh, kg_y + 90)
            x0_s = kg_x + 30
            x1_s = min(fw, kg_x + 400)
            slice_cyan = mask_cyan_full[y0_s:y1_s, x0_s:x1_s]
            cnts, _ = cv2.findContours(slice_cyan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            exact_btn = None
            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw >= 70 and bh >= 35:
                    exact_btn = (x0_s + bx, y0_s + by, bw, bh)
                    break

            if exact_btn:
                btn_x = origin[0] + exact_btn[0]
                btn_y = origin[1] + exact_btn[1]
                btn_w = exact_btn[2]
                btn_h = exact_btn[3]
            else:
                btn_w = self.btn_tpl.shape[1] if self.btn_tpl is not None else 140
                btn_h = self.btn_tpl.shape[0] if self.btn_tpl is not None else 85
                btn_x = origin[0] + kg_x + 220
                btn_y = origin[1] + kg_y - 9

            # VIEWPORT CLIPPING GUARD:
            # Cards cut off at the bottom of the scroll container or with clipped cyan buttons.
            # In 1080p, row 4 buttons end around Y=820px, bottom bar starts at Y=845px.
            is_clipped = False
            if fh >= 900 and (btn_y + btn_h > int(fh * 0.81) or kg_y > int(fh * 0.79)):
                is_clipped = True
            else:
                x0_b, x1_b = max(0, btn_x), min(fw, btn_x + btn_w)
                y0_b, y1_b = max(0, btn_y), min(fh, btn_y + btn_h)
                if y1_b - y0_b < 40 or x1_b - x0_b < 60:
                    is_clipped = True
                else:
                    crop = frame_bgr[y0_b:y1_b, x0_b:x1_b]
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    mask_cyan = cv2.inRange(hsv, np.array([90, 100, 100]), np.array([115, 255, 255]))
                    if np.mean(mask_cyan > 0) < 0.40:
                        is_clipped = True

            # Crop wide 120px strip to the left of kg, and extend vertical range by (-6..+19)
            # Accommodates giant numbers up to 9999.99kg without truncating high-order digits.
            x0 = max(0, kg_x - 120)
            y0 = max(0, kg_y - 6)
            y1 = min(fh, kg_y + 19)
            strip = mask[y0:y1, x0:kg_x]
            weight = self.read_weight_from_strip(strip)

            cards.append(HarvestCard(
                card_index=idx + 1,
                weight=weight,
                kg_x=origin[0] + kg_x,
                kg_y=origin[1] + kg_y,
                btn_x=btn_x,
                btn_y=btn_y,
                btn_w=btn_w,
                btn_h=btn_h,
                is_clipped=is_clipped,
            ))

        return cards
