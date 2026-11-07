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
from core.path_utils import get_digits_dir


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
            # Default to assets/farm/digits via path_utils
            digits_dir = get_digits_dir()

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

    def read_weight_from_strip(self, strip_mask: np.ndarray, gap_max: int = 8, scale: float = 1.0) -> float:
        """
        Parses weight from a binarized mask of the strip to the left of 'kg'.
        Scans right-to-left starting closest to 'kg', clustering contiguous digits and dot.
        Rejects noise blobs, truncations, and handles weights up to 9999.99kg.
        Fails-closed (returns 9999.0) if unreadable or truncated to guarantee giant crops are never harvested.
        Multi-scale aware across resolutions.
        """
        min_dot_w = max(1, int(1 * scale))
        max_dot_w = max(3, int(6 * scale))
        min_dot_h = max(1, int(2 * scale))
        max_dot_h = max(3, int(6 * scale))
        min_by = max(3, int(6 * scale))
        min_barea = max(1, int(2 * scale * scale))
        min_bh = max(1, int(2 * scale))
        min_bw = max(1, int(2 * scale))

        n, _, stats, _ = cv2.connectedComponentsWithStats(strip_mask)
        blobs = []
        for bi in range(1, n):
            bx, by, bw, bh, barea = stats[bi]
            is_dot = (min_dot_w <= bw <= max_dot_w and min_dot_h <= bh <= max_dot_h and by >= min_by)
            if barea < min_barea or bh < min_bh or (bw < min_bw and not is_dot):
                continue
            blobs.append((int(bx), int(by), int(bw), int(bh)))

        if not blobs:
            return 9999.0

        # Sort right-to-left (starting from the one nearest to 'kg')
        blobs.sort(key=lambda b: -b[0])

        effective_gap = max(gap_max, int(12 * scale))
        group = [blobs[0]]
        for b in blobs[1:]:
            prev_left = group[-1][0]
            curr_right = b[0] + b[2]
            if prev_left - curr_right > effective_gap:
                break
            group.append(b)

        if group[-1][0] <= 0:
            return 9999.0

        # Re-sort left-to-right for sequential parsing
        group.reverse()
        num_str = ""
        has_dot = False
        digits_before_dot = 0
        digits_after_dot = 0

        for bx, by, bw, bh in group:
            if min_dot_w <= bw <= max_dot_w and min_dot_h <= bh <= max_dot_h and by >= min_by:
                if not has_dot:
                    num_str += "."
                    has_dot = True
                continue

            patch = strip_mask[by:by + bh, bx:bx + bw]
            d, score = self._classify_digit_patch(patch)
            min_score = 0.30 if scale < 0.85 else 0.35
            if score < min_score:
                return 9999.0

            num_str += d
            if not has_dot:
                digits_before_dot += 1
            else:
                digits_after_dot += 1

        if digits_before_dot == 0 or not has_dot or digits_after_dot != 2:
            return 9999.0

        try:
            return float(num_str)
        except Exception:
            return 9999.0

    def read_weight_at_pos(self, frame_bgr: np.ndarray, kg_x: int, kg_y: int, scale: Optional[float] = None) -> float:
        """Rapid single-card OCR (< 1ms): reads weight directly from local strip of (kg_x, kg_y)."""
        if self.kg_tpl is None or frame_bgr is None or frame_bgr.size == 0:
            return 9999.0
        s = scale or getattr(self, "detected_scale", None) or 1.0
        fh, fw = frame_bgr.shape[:2]
        x0 = max(0, kg_x - int(120 * s))
        y0 = max(0, kg_y - int(6 * s))
        y1 = min(fh, kg_y + int(19 * s))
        crop = frame_bgr[y0:y1, x0:kg_x]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 10:
            return 9999.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 115, 205)
        return self.read_weight_from_strip(mask, gap_max=max(6, int(8 * s)), scale=s)

    def scan_cards(self, frame_bgr: np.ndarray, origin: Tuple[int, int] = (0, 0)) -> List[HarvestCard]:
        """
        Scans all visible harvest item cards in the 'Thu hoạch trái' window.
        Returns cards sorted top-to-bottom, left-to-right with exact parsed weights and button coords.
        Multi-scale aware across resolutions.
        """
        if self.kg_tpl is None:
            return []

        fh, fw = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 115, 205)

        # Multi-scale kg matching
        scales = (1.0, 0.75, 0.70, 0.67, 0.80, 0.85, 0.53, 0.50, 1.25, 1.33)
        h_ratio = round(fh / 1080.0, 2)
        if h_ratio not in scales and 0.3 <= h_ratio <= 2.5:
            scales = (h_ratio,) + scales

        best_v = 0.0
        best_s = 1.0
        # Fast-track: test cached scale first (< 10ms)
        if getattr(self, "detected_scale", None) is not None:
            s_cached = self.detected_scale
            tw = int(self.kg_tpl.shape[1] * s_cached)
            th = int(self.kg_tpl.shape[0] * s_cached)
            if tw >= 6 and th >= 6 and mask.shape[0] >= th and mask.shape[1] >= tw:
                scaled = cv2.resize(self.kg_tpl, (tw, th))
                r = cv2.matchTemplate(mask, scaled, cv2.TM_CCOEFF_NORMED)
                v = float(r.max())
                locs_c = np.where(r >= 0.74)
                if v >= 0.76 and len(locs_c[0]) >= 2:
                    best_v = v
                    best_s = s_cached

        if best_v < 0.75:
            scale_stats = []
            for s in scales:
                tw = int(self.kg_tpl.shape[1] * s)
                th = int(self.kg_tpl.shape[0] * s)
                if tw >= 6 and th >= 6 and mask.shape[0] >= th and mask.shape[1] >= tw:
                    scaled = cv2.resize(self.kg_tpl, (tw, th))
                    r = cv2.matchTemplate(mask, scaled, cv2.TM_CCOEFF_NORMED)
                    v = float(r.max())
                    if v >= 0.72:
                        locs = np.where(r >= 0.72)
                        pts = list(zip(locs[1], locs[0]))
                        dedup = []
                        for p in pts:
                            if not any(abs(p[0] - d[0]) < 15 and abs(p[1] - d[1]) < 15 for d in dedup):
                                dedup.append(p)
                        scale_stats.append((s, len(dedup), v))
                    else:
                        scale_stats.append((s, 0, v))

            # Prioritize scales that produce multiple card hits (>= 2 hits) over single glitches
            multi_hits = [st for st in scale_stats if st[1] >= 2]
            if multi_hits:
                best_st = max(multi_hits, key=lambda st: (st[1] >= 4, st[2], st[1]))
                best_s = best_st[0]
            elif scale_stats:
                best_st = max(scale_stats, key=lambda st: st[2])
                best_s = best_st[0]

            self.detected_scale = best_s

        scaled_kg = cv2.resize(self.kg_tpl, (int(self.kg_tpl.shape[1] * best_s), int(self.kg_tpl.shape[0] * best_s)))
        res = cv2.matchTemplate(mask, scaled_kg, cv2.TM_CCOEFF_NORMED)
        threshold = 0.60 if best_s < 0.85 else 0.75
        hsv_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask_cyan_full = cv2.inRange(hsv_full, np.array([90, 100, 100]), np.array([115, 255, 255]))

        locs = np.where(res >= threshold)
        hits = []
        tw_kg = int(self.kg_tpl.shape[1] * best_s)
        th_kg = int(self.kg_tpl.shape[0] * best_s)
        for pt in zip(*locs[::-1]):
            # Reject false kg matches occurring inside cyan 'Thu hoach' buttons
            if mask_cyan_full[pt[1]:pt[1] + th_kg, pt[0]:pt[0] + tw_kg].mean() > 30:
                continue
            if not any(abs(pt[0] - p[0]) < 12 and abs(pt[1] - p[1]) < 12 for p in hits):
                hits.append(pt)

        # Robust row clustering by vertical proximity (dy > 20*s creates a new row)
        row_gap = max(12, int(25 * best_s))
        hits.sort(key=lambda p: p[1])
        rows: List[List[Tuple[int, int]]] = []
        row_y: Optional[int] = None
        for p in hits:
            if row_y is None or p[1] - row_y > row_gap:
                rows.append([])
                row_y = p[1]
            rows[-1].append(p)
        hits = [p for r in rows for p in sorted(r, key=lambda q: q[0])]

        cards: List[HarvestCard] = []
        for idx, (kg_x, kg_y) in enumerate(hits):
            # Dynamic cyan button calibration:
            # Searches for the real 'Thu hoạch' button contour in the card's row slice to the right of 'kg'.
            y0_s = max(0, kg_y - int(25 * best_s))
            y1_s = min(fh, kg_y + int(90 * best_s))
            x0_s = kg_x + int(20 * best_s)
            x1_s = min(fw, kg_x + int(420 * best_s))
            slice_cyan = mask_cyan_full[y0_s:y1_s, x0_s:x1_s]
            cnts, _ = cv2.findContours(slice_cyan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            exact_btn = None
            min_bw = max(30, int(60 * best_s))
            min_bh = max(15, int(30 * best_s))
            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw >= min_bw and bh >= min_bh:
                    exact_btn = (x0_s + bx, y0_s + by, bw, bh)
                    break

            if exact_btn:
                btn_x = origin[0] + exact_btn[0]
                btn_y = origin[1] + exact_btn[1]
                btn_w = exact_btn[2]
                btn_h = exact_btn[3]
            else:
                btn_w = int(self.btn_tpl.shape[1] * best_s) if self.btn_tpl is not None else int(140 * best_s)
                btn_h = int(self.btn_tpl.shape[0] * best_s) if self.btn_tpl is not None else int(85 * best_s)
                btn_x = origin[0] + kg_x + int(220 * best_s)
                btn_y = origin[1] + kg_y - int(9 * best_s)

            # VIEWPORT CLIPPING GUARD:
            is_clipped = False
            if btn_y + btn_h > int(fh * 0.82) or kg_y > int(fh * 0.80):
                is_clipped = True
            else:
                x0_b, x1_b = max(0, btn_x), min(fw, btn_x + btn_w)
                y0_b, y1_b = max(0, btn_y), min(fh, btn_y + btn_h)
                if y1_b - y0_b < min_bh or x1_b - x0_b < min_bw:
                    is_clipped = True
                else:
                    crop = frame_bgr[y0_b:y1_b, x0_b:x1_b]
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    mask_cyan = cv2.inRange(hsv, np.array([90, 100, 100]), np.array([115, 255, 255]))
                    if np.mean(mask_cyan > 0) < 0.35:
                        is_clipped = True

            # Crop strip to the left of kg
            x0 = max(0, kg_x - int(120 * best_s))
            y0 = max(0, kg_y - int(6 * best_s))
            y1 = min(fh, kg_y + int(19 * best_s))
            strip = mask[y0:y1, x0:kg_x]
            weight = self.read_weight_from_strip(strip, gap_max=max(6, int(8 * best_s)), scale=best_s)

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
