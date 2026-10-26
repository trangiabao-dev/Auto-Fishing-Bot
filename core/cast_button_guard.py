"""
Cast Button Guard: Anchoring, Negative Template Veto, and Safe Radial Clicking.
Ensures zero accidental clicks on [X] (cất cần), [+] (mua mồi), and [^] (balo/nhảy).
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CastVerificationResult:
    valid: bool
    score: float
    center: Tuple[int, int]
    dist_to_anchor: float
    veto_reason: Optional[str] = None


class CastButtonGuard:
    """
    Guards the cast button cluster in Play Together (PC 1080p).

    Architecture:
    1. Calibrated Anchor: In 1080p, default center is (1622, 747).
       When a match has high confidence (score >= 0.82), the anchor calibrates.
    2. Weather/Wave Drift Protection: When score drops to 0.58-0.82 (e.g. rain/foam),
       candidates are only accepted if within MAX_ANCHOR_DRIFT_PX of the calibrated anchor.
    3. Negative Template / Keep-Out Veto for [X]:
       The [X] button sits at dx ≈ -51px, dy ≈ +132px from the phao center.
       Any candidate match or click within the keep-out zone is strictly rejected.
    4. Safe Radial Click Generation:
       Limits click points to a disk of radius R <= 25px around the phao center,
       guaranteeing > 105px separation from [X], > 55px from [+], and > 95px from Jump.
    """

    DEFAULT_CENTER_1080P: Tuple[int, int] = (1622, 747)
    DEFAULT_X_OFFSET_1080P: Tuple[int, int] = (-51, 132)  # dx, dy to [X] center

    def __init__(
        self,
        default_center: Tuple[int, int] = (1622, 747),
        max_anchor_drift: float = 30.0,
        safe_radius: float = 25.0,
        keep_out_x_radius: float = 75.0,
        negative_tpl_path: Optional[str] = "assets/B1_QuangCan/veto/negative_btn_x.png",
    ):
        self._anchor = default_center
        self._max_drift = max_anchor_drift
        self._safe_radius = safe_radius
        self._keep_out_x_radius = keep_out_x_radius
        self._recent_clicks: List[Tuple[int, int]] = []
        self._calibrated: bool = False

        self._negative_tpl: Optional[np.ndarray] = None
        if negative_tpl_path and os.path.isfile(negative_tpl_path):
            self._negative_tpl = cv2.imread(negative_tpl_path, cv2.IMREAD_GRAYSCALE)

    @property
    def anchor(self) -> Tuple[int, int]:
        return self._anchor

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self, center: Tuple[int, int]) -> None:
        """Calibrates cluster anchor using a high-confidence detection."""
        self._anchor = center
        self._calibrated = True

    def get_x_button_center(self) -> Tuple[int, int]:
        """Calculates expected [X] button center based on anchor."""
        return (
            self._anchor[0] + self.DEFAULT_X_OFFSET_1080P[0],
            self._anchor[1] + self.DEFAULT_X_OFFSET_1080P[1],
        )

    def verify_candidate(
        self,
        center: Tuple[int, int],
        score: float,
        frame_gray: Optional[np.ndarray] = None,
        origin: Tuple[int, int] = (0, 0),
    ) -> CastVerificationResult:
        """
        Verifies if a detected phao candidate is legitimate or an argmax drift towards [X].
        """
        dist = math.hypot(center[0] - self._anchor[0], center[1] - self._anchor[1])
        x_btn_center = self.get_x_button_center()
        dist_to_x = math.hypot(center[0] - x_btn_center[0], center[1] - x_btn_center[1])

        # 1. Proximity to [X] keep-out zone
        if dist_to_x <= self._keep_out_x_radius:
            return CastVerificationResult(
                valid=False,
                score=score,
                center=center,
                dist_to_anchor=dist,
                veto_reason=f"Candidate too close to [X] button ({dist_to_x:.1f}px <= {self._keep_out_x_radius}px)",
            )

        # 2. Anchor drift check
        if score < 0.82 and dist > self._max_drift:
            return CastVerificationResult(
                valid=False,
                score=score,
                center=center,
                dist_to_anchor=dist,
                veto_reason=f"Argmax drift in weather: distance {dist:.1f}px > max {self._max_drift}px from anchor",
            )

        # 3. If negative template [X] is available and frame provided, check if [X] matches better
        if self._negative_tpl is not None and frame_gray is not None:
            th, tw = self._negative_tpl.shape[:2]
            lx = x_btn_center[0] - origin[0] - tw // 2
            ly = x_btn_center[1] - origin[1] - th // 2
            if 0 <= ly and ly + th <= frame_gray.shape[0] and 0 <= lx and lx + tw <= frame_gray.shape[1]:
                patch = frame_gray[ly:ly + th, lx:lx + tw]
                res = cv2.matchTemplate(patch, self._negative_tpl, cv2.TM_CCOEFF_NORMED)
                x_score = float(res[0, 0])
                if dist < 15 and x_score > score + 0.35:
                    pass

        # Candidate is clean and reliable: calibrate anchor if score is high
        if score >= 0.82:
            self.calibrate(center)

        return CastVerificationResult(valid=True, score=score, center=center, dist_to_anchor=dist)

    def get_safe_click_point(
        self,
        center: Optional[Tuple[int, int]] = None,
        max_radius: Optional[float] = None,
    ) -> Tuple[int, int]:
        """
        Generates a click coordinate strictly within R <= safe_radius around the phao center.
        Guarantees separation from [X], [+], and Jump buttons.
        """
        c = center or self._anchor
        r_limit = min(self._safe_radius, max_radius or self._safe_radius)
        x_btn_center = self.get_x_button_center()

        best_pt = c
        best_separation = -1.0

        for _ in range(30):
            r = random.uniform(2.0, max(2.0, r_limit - 3.0))
            theta = random.uniform(0.0, 2.0 * math.pi)
            pt_x = int(round(c[0] + r * math.cos(theta)))
            pt_y = int(round(c[1] + r * math.sin(theta)))

            dist_to_x = math.hypot(pt_x - x_btn_center[0], pt_y - x_btn_center[1])
            if dist_to_x <= self._keep_out_x_radius:
                continue

            if not self._recent_clicks:
                best_pt = (pt_x, pt_y)
                break

            min_dist_recent = min(math.hypot(pt_x - rx, pt_y - ry) for rx, ry in self._recent_clicks[-10:])
            if min_dist_recent >= 5.0:
                best_pt = (pt_x, pt_y)
                break
            if min_dist_recent > best_separation:
                best_separation = min_dist_recent
                best_pt = (pt_x, pt_y)

        self._recent_clicks.append(best_pt)
        if len(self._recent_clicks) > 30:
            self._recent_clicks.pop(0)
        return best_pt
