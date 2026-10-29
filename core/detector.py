import os
import random
import cv2
import collections
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Union


@dataclass
class MatchResult:
    found: bool
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    confidence: float = 0.0
    template_name: str = ""

    @property
    def center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


class TemplateMatcher:
    """
    Template matcher powered by OpenCV with in-memory caching, multi-scale fallback,
    auto-extension resolution, and random click calculation.
    """

    _recent_clicks = collections.deque(maxlen=40)
    _click_seq: int = random.randint(100, 1000)

    def __init__(self):
        self._template_cache: Dict[str, np.ndarray] = {}
        self._missing_templates_warned = set()

    @staticmethod
    def resolve_template_path(template_path: str) -> Optional[str]:
        """
        Flexibly resolves template path by testing alternative file extensions (.png, .jpg, .jpeg, .bmp).
        Handles user screenshots saved with different extensions.
        """
        if not template_path:
            return None
        if os.path.isfile(template_path):
            return template_path

        base, _ = os.path.splitext(template_path)
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            candidate = base + ext
            if os.path.isfile(candidate):
                return candidate
        return None

    def load_template(self, template_path: str, grayscale: bool = True) -> Optional[Tuple[np.ndarray, str]]:
        """
        Loads and caches template image from disk.
        Returns: (image_array, resolved_path) or None.
        """
        resolved_path = self.resolve_template_path(template_path)

        if resolved_path is None:
            if template_path not in self._missing_templates_warned:
                print(f"[CẢNH BÁO] Không tìm thấy file ảnh mẫu: '{template_path}'.")
                self._missing_templates_warned.add(template_path)
            return None

        cache_key = f"{resolved_path}_{grayscale}"
        if cache_key in self._template_cache:
            return self._template_cache[cache_key], resolved_path

        img = cv2.imread(resolved_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if img is not None:
            self._template_cache[cache_key] = img
            return img, resolved_path
        return None

    def match(
        self,
        screen_bgr: np.ndarray,
        template_path: str,
        threshold: float = 0.8,
        screen_offset: Tuple[int, int] = (0, 0),
        grayscale: bool = True,
        multi_scale: bool = True
    ) -> MatchResult:
        """
        Searches for template within screen image.
        Uses native 1.0x scale first; only if below threshold and multi_scale is True does it evaluate multi-scale.
        Always finds the GLOBAL MAXIMUM across all scales to avoid false downscale matches.
        """
        load_res = self.load_template(template_path, grayscale=grayscale)
        if load_res is None:
            return MatchResult(found=False, confidence=0.0)

        template, resolved_path = load_res
        tpl_name = os.path.basename(resolved_path)

        # Convert screen if grayscale requested
        if grayscale:
            if len(screen_bgr.shape) == 3:
                screen = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
            else:
                screen = screen_bgr
        else:
            if len(screen_bgr.shape) == 2 and len(template.shape) == 3:
                screen = cv2.cvtColor(screen_bgr, cv2.COLOR_GRAY2BGR)
            elif len(screen_bgr.shape) == 3 and len(template.shape) == 2:
                template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
            else:
                screen = screen_bgr

        # Ensure screen is larger than template
        th, tw = template.shape[:2]
        sh, sw = screen.shape[:2]
        if sh < th or sw < tw:
            return MatchResult(found=False, confidence=0.0, template_name=tpl_name)

        # 1. Native Scale (1.0)
        res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold or not multi_scale:
            global_x = screen_offset[0] + max_loc[0]
            global_y = screen_offset[1] + max_loc[1]
            return MatchResult(
                found=(max_val >= threshold),
                x=global_x,
                y=global_y,
                w=tw,
                h=th,
                confidence=float(max_val),
                template_name=tpl_name
            )

        # 2. Multi-Scale Fallback: Find the GLOBAL PEAK across scales close to 1.0
        best_val = max_val
        best_loc = max_loc
        best_w, best_h = tw, th

        # Ordered by proximity to 1.0, including standard window scales (720p=0.67, 540p=0.53)
        scales = [0.95, 0.90, 1.05, 1.10, 0.85, 1.15, 0.80, 1.25, 0.75, 0.67, 0.53, 0.50, 1.50, 1.88, 2.00]
        for scale in scales:
            scaled_w = int(tw * scale)
            scaled_h = int(th * scale)
            # Prevent scaling down too small (noise) or larger than screen
            if scaled_w < 25 or scaled_h < 18 or scaled_w > sw or scaled_h > sh:
                continue
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            scaled_tpl = cv2.resize(template, (scaled_w, scaled_h), interpolation=interp)
            s_res = cv2.matchTemplate(screen, scaled_tpl, cv2.TM_CCOEFF_NORMED)
            _, s_max_val, _, s_max_loc = cv2.minMaxLoc(s_res)

            if s_max_val > best_val:
                best_val = s_max_val
                best_loc = s_max_loc
                best_w, best_h = scaled_w, scaled_h

        global_x = screen_offset[0] + best_loc[0]
        global_y = screen_offset[1] + best_loc[1]

        if best_val >= threshold:
            return MatchResult(
                found=True,
                x=global_x,
                y=global_y,
                w=best_w,
                h=best_h,
                confidence=float(best_val),
                template_name=tpl_name
            )

        return MatchResult(
            found=False,
            confidence=float(best_val),
            x=global_x,
            y=global_y,
            w=best_w,
            h=best_h,
            template_name=tpl_name
        )

    def match_any(
        self,
        screen_bgr: np.ndarray,
        template_target: Any,
        threshold: float = 0.75,
        screen_offset: Tuple[int, int] = (0, 0),
        grayscale: bool = True,
        multi_scale: bool = True
    ) -> MatchResult:
        """
        Matches against multiple template variants.
        'template_target' can be:
          - A list of file paths
          - A folder path (e.g. 'assets/B1_QuangCan')
          - A single file path (auto-resolves alternative extensions)
        Returns the match with the highest confidence.
        """
        paths: List[str] = []
        if isinstance(template_target, list):
            paths = template_target
        elif isinstance(template_target, str):
            if os.path.isdir(template_target):
                for fname in sorted(os.listdir(template_target)):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        paths.append(os.path.join(template_target, fname))
            else:
                resolved = self.resolve_template_path(template_target)
                if resolved:
                    paths.append(resolved)
                else:
                    folder_candidate = os.path.splitext(template_target)[0]
                    if os.path.isdir(folder_candidate):
                        for fname in sorted(os.listdir(folder_candidate)):
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                                paths.append(os.path.join(folder_candidate, fname))

        if not paths:
            return MatchResult(found=False, confidence=0.0)

        best_result = MatchResult(found=False, confidence=0.0)

        for path in paths:
            res = self.match(screen_bgr, path, threshold=threshold, screen_offset=screen_offset, grayscale=grayscale, multi_scale=multi_scale)
            # Keep the result with highest confidence
            if res.confidence > best_result.confidence:
                best_result = res
                if best_result.confidence >= 0.90:
                    # Early exit on high match
                    return best_result

        return best_result

    @classmethod
    def get_random_click_point(cls, match: MatchResult, margin: int = 6,
                               avoid: Optional[Tuple[int, int, int, int]] = None) -> Tuple[int, int]:
        """
        Calculates an anti-clustering randomized click point inside the bounding box.
        Prevents repeating recent coordinates or clustering within the target.
        'avoid' is an optional keep-out rectangle (x, y, w, h): candidate points inside it are rejected.
        """
        if not match.found:
            raise ValueError("Cannot calculate click point for an unfound match.")

        safe_margin_x = min(margin, max(2, match.w // 6))
        safe_margin_y = min(margin, max(2, match.h // 6))

        min_x = match.x + safe_margin_x
        max_x = match.x + match.w - safe_margin_x
        min_y = match.y + safe_margin_y
        max_y = match.y + match.h - safe_margin_y

        if min_x >= max_x or min_y >= max_y:
            return match.center

        # Geometry-aware validation: prevent clicking outer transparent corners of round/capsule buttons
        aspect = match.w / float(match.h) if match.h > 0 else 1.0
        is_circle = 0.75 <= aspect <= 1.33
        is_capsule = aspect > 1.33

        def is_geom_valid(lx: int, ly: int) -> bool:
            if is_circle:
                cx_local, cy_local = match.w / 2.0, match.h / 2.0
                r = min(cx_local, cy_local) - safe_margin_y
                return (lx - cx_local) ** 2 + (ly - cy_local) ** 2 <= (r ** 2)
            elif is_capsule:
                r = match.h / 2.0
                safe_r = max(2.0, r - safe_margin_y)
                if lx < r:
                    return (lx - r) ** 2 + (ly - r) ** 2 <= (safe_r ** 2)
                elif lx > match.w - r:
                    return (lx - (match.w - r)) ** 2 + (ly - r) ** 2 <= (safe_r ** 2)
                else:
                    return safe_margin_y <= ly <= (match.h - safe_margin_y)
            return True

        # Filter recent clicks within this target region
        region_recent = [
            (rx, ry) for rx, ry in cls._recent_clicks
            if match.x <= rx <= match.x + match.w and match.y <= ry <= match.y + match.h
        ]

        # Minimum Euclidean separation threshold to prevent spatial clustering
        min_dist = max(10.0, min(match.w, match.h) * 0.25)
        best_pt = match.center
        best_d = -1.0

        # R2 Quasi-Monte Carlo generator (Plastic constant ratio) for maximum dispersion across the entire button surface
        phi2 = 1.324717957244746
        a1 = 1.0 / phi2
        a2 = 1.0 / (phi2 ** 2)

        for _ in range(60):
            cls._click_seq += 1
            # Low-discrepancy sampling with Gaussian micro-jitter to prevent mathematical fingerprinting
            u = (0.5 + a1 * cls._click_seq + random.gauss(0, 0.035)) % 1.0
            v = (0.5 + a2 * cls._click_seq + random.gauss(0, 0.035)) % 1.0
            cx = int(min_x + u * (max_x - min_x))
            cy = int(min_y + v * (max_y - min_y))

            lx, ly = cx - match.x, cy - match.y
            if not is_geom_valid(lx, ly):
                continue
            if avoid and avoid[0] <= cx <= avoid[0] + avoid[2] and avoid[1] <= cy <= avoid[1] + avoid[3]:
                continue
            if not region_recent:
                best_pt = (cx, cy)
                break
            min_d_to_recent = min(float(np.hypot(cx - rx, cy - ry)) for rx, ry in region_recent)
            if min_d_to_recent >= min_dist:
                best_pt = (cx, cy)
                break
            if min_d_to_recent > best_d:
                best_d = min_d_to_recent
                best_pt = (cx, cy)

        cls._recent_clicks.append(best_pt)
        return best_pt

    @classmethod
    def get_safe_core_click_point(
        cls,
        match: MatchResult,
        x_ratio_range: Tuple[float, float] = (0.15, 0.55),
        y_ratio_range: Tuple[float, float] = (0.30, 0.70),
    ) -> Tuple[int, int]:
        """
        Calculates a click point strictly within the safe core zone of a button.
        For the 'Bảo Quản' button, this restricts clicks to [0.15*W, 0.55*W] and [0.30*H, 0.70*H],
        leaving a generous safety margin (> 80px) away from the 'Khoe' button on the right
        and > 50px away from the 'Bán tức thì' button on the left.
        """
        if not match.found:
            raise ValueError("Cannot calculate safe click point for an unfound match.")

        min_x = match.x + int(match.w * x_ratio_range[0])
        max_x = match.x + int(match.w * x_ratio_range[1])
        min_y = match.y + int(match.h * y_ratio_range[0])
        max_y = match.y + int(match.h * y_ratio_range[1])

        if min_x >= max_x or min_y >= max_y:
            return match.center

        cx = random.randint(min_x, max_x)
        cy = random.randint(min_y, max_y)
        cls._recent_clicks.append((cx, cy))
        return (cx, cy)
