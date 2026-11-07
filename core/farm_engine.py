"""
Farm Engine: Automated Crop Harvesting and Selling State Machine for Play Together.
Features:
  - Dual-Phase Pipeline:
      * Phase 1 (Category A): Sequential single-fruit filtering with OCR weight threshold (< max_kg).
        Giant fruits (>= max_kg) are strictly skipped and protected.
      * Phase 2 (Category B): Multi-fruit batch filtering without OCR (rapid click harvest-all).
        Terminates on empty-filter banner: 'Không có trái nào phù hợp với bộ lọc đã chọn.'
  - Shift-delay compensation (0.6s - 0.8s) with pre-click re-verification.
  - Automated full-bag detection ('Túi đã đầy nên không thể thu hoạch.').
  - Auto-sell loop at NPC Oliver (Select All -> Sell -> Confirm -> OK -> Re-enter Harvest).
  - Clean threading with stop/pause events and mockable dependencies.
"""
from __future__ import annotations

import os
import random
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.digit_reader import DigitReader, HarvestCard
from core.farm_config_manager import FarmConfigManager, FruitRule
from core.input_sim import InputSimulator
from core.screen_capture import ScreenCapturer
from core.path_utils import get_farm_assets_dir


class FarmState(Enum):
    STOPPED = "STOPPED"
    PAUSED = "PAUSED"
    TELEPORTING = "TELEPORTING"
    OPENING_HARVEST = "OPENING_HARVEST"
    APPLYING_FILTER = "APPLYING_FILTER"
    HARVESTING = "HARVESTING"
    BAG_FULL_DETECTED = "BAG_FULL_DETECTED"
    SELLING = "SELLING"
    SELLING_OPEN_SHOP = "SELLING_OPEN_SHOP"
    SELLING_SELECT_ALL = "SELLING_SELECT_ALL"
    SELLING_CONFIRM = "SELLING_CONFIRM"
    SELLING_DONE = "SELLING_DONE"
    SELLING_EXIT = "SELLING_EXIT"
    COMPLETED = "COMPLETED"


class FarmEngine:
    def __init__(
        self,
        config_manager: Optional[FarmConfigManager] = None,
        digit_reader: Optional[DigitReader] = None,
        capturer: Optional[ScreenCapturer] = None,
        input_sim: Optional[InputSimulator] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        state_callback: Optional[Callable[[FarmState], None]] = None,
        assets_dir: Optional[str] = None,
    ):
        self.assets_dir = assets_dir or get_farm_assets_dir()

        self.cfg_mgr = config_manager or FarmConfigManager()
        self.digit_reader = digit_reader or DigitReader()
        self.capturer = capturer or ScreenCapturer()
        self.input_sim = input_sim or InputSimulator()
        self.log_callback = log_callback
        self.state_callback = state_callback

        self.state = FarmState.STOPPED
        self._stop_event = False
        self._pause_event = False
        self._thread = None

        # Statistics
        self.stats = {
            "harvested_count": 0,
            "sell_cycles": 0,
            "skipped_heavy_count": 0,
            "current_fruit": "",
            "max_kg": 0.0,
        }

        # Cache templates
        self._templates: Dict[str, np.ndarray] = {}
        self._load_templates()

        # Session-scoped harvest click coordinate (fixed per harvest entry to eliminate jitter)
        self._fixed_harvest_point: Optional[Tuple[int, int]] = None
        self._detected_ui_scale: Optional[float] = None

    def _log(self, msg: str) -> None:
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(f"[FarmEngine] {msg}")

    def _set_state(self, new_state: FarmState) -> None:
        self.state = new_state
        if self.state_callback:
            self.state_callback(new_state)

    def _load_templates(self) -> None:
        tpl_files = {
            # Harvest screen templates (B1 to B7)
            "harvest_b1": os.path.join(self.assets_dir, "thu_hoach", "B1.png"),
            "harvest_b2": os.path.join(self.assets_dir, "thu_hoach", "B2.png"),
            "core_b1_ban": os.path.join(self.assets_dir, "thu_hoach", "core_b1_ban.png"),
            "core_b2_harvest": os.path.join(self.assets_dir, "thu_hoach", "core_b2_harvest.png"),
            "btn_loc": os.path.join(self.assets_dir, "thu_hoach", "B3.jpg"),
            "chk_dot_bien": os.path.join(self.assets_dir, "thu_hoach", "B4.jpg"),
            "tab_nong_san": os.path.join(self.assets_dir, "thu_hoach", "B5.jpg"),
            "item_hat_de": os.path.join(self.assets_dir, "thu_hoach", "item_hat_de.png"),
            "btn_cai_dat_loc": os.path.join(self.assets_dir, "thu_hoach", "B7.jpg"),
            "toast_day_tui": os.path.join(self.assets_dir, "thu_hoach", "anh thong bao day tui.jpg"),
            "btn_thu_hoach": os.path.join(self.assets_dir, "thu_hoach", "nut thu hoach.jpg"),
            "tpl_het_trai_1080p": os.path.join(self.assets_dir, "thu_hoach", "tpl_het_trai_1080p.png"),
            "btn_close_x": os.path.join(self.assets_dir, "thu_hoach", "btn_close_x.png"),
            # Sell screen templates
            "oliver_basket": os.path.join(self.assets_dir, "ban", "B1.jpg"),
            "oliver_sell_crops": os.path.join(self.assets_dir, "ban", "B2.jpg"),
            "oliver_sell_crops_old": os.path.join(self.assets_dir, "ban", "B2_old_cay_trong.jpg"),
            "oliver_dialogue_smile": os.path.join(self.assets_dir, "ban", "icon_dialogue_smile.png"),
            "sell_chon_tu_dong": os.path.join(self.assets_dir, "ban", "B3.jpg"),
            "sell_coin_btn": os.path.join(self.assets_dir, "ban", "B4.jpg"),
            "sell_confirm_btn": os.path.join(self.assets_dir, "ban", "B5.jpg"),
            "sell_confirm_high_tier": os.path.join(self.assets_dir, "ban", "B6.jpg"),
            "sell_ok_btn": os.path.join(self.assets_dir, "ban", "B7.jpg") if os.path.exists(os.path.join(self.assets_dir, "ban", "B7.jpg")) else os.path.join(self.assets_dir, "ban", "B6.jpg"),
            "prompt_space_cyan": os.path.join(self.assets_dir, "prompt_space_cyan.png"),
            "prompt_space_brown": os.path.join(self.assets_dir, "prompt_space_brown_1080p.png"),
            "btn_khong_thoat": os.path.join(self.assets_dir, "btn_khong_thoat.png"),
            "btn_space_ok": os.path.join(self.assets_dir, "btn_space_ok.png"),
        }
        for k, p in tpl_files.items():
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    self._templates[k] = img

        # Dynamically load all fruit item templates in thu_hoach
        th_dir = os.path.join(self.assets_dir, "thu_hoach")
        if os.path.exists(th_dir):
            for fname in os.listdir(th_dir):
                if fname.startswith("item_") and fname.endswith(".png"):
                    key = fname[:-4]
                    if key not in self._templates:
                        img = cv2.imread(os.path.join(th_dir, fname))
                        if img is not None:
                            self._templates[key] = img

        has_b1 = ("core_b1_ban" in self._templates) or ("harvest_b1" in self._templates)
        has_b2 = ("core_b2_harvest" in self._templates) or ("harvest_b2" in self._templates)
        has_oliver = ("oliver_basket" in self._templates)

        if not (has_b1 and has_b2 and has_oliver):
            self._templates_valid = False
            self._log(f"⚠️ [LỖI TẢI ẢNH MẪU] Thư mục '{self.assets_dir}' thiếu ảnh mẫu cốt lõi (Bán/Thu hoạch/Oliver)!")
        else:
            self._templates_valid = True

    def _get_fruit_template_key(self, fruit_name: str) -> Optional[str]:
        """Resolves fruit template key from fruit name."""
        fn = fruit_name.lower().strip()
        if "sầu" in fn or "riêng" in fn:
            return "item_sau_rieng"
        if "dừa" in fn:
            return "item_dua"
        if "xoài" in fn:
            return "item_xoai"
        if "đậu" in fn:
            return "item_cay_dau"
        if "đu" in fn or "đủ" in fn:
            return "item_du_du"
        if "dẻ" in fn:
            return "item_hat_de"

        slug = f"item_{fn.replace(' ', '_')}"
        if slug in self._templates:
            return slug
        return None

    def _get_scale_candidates(self, fh: int, multiscale: bool = True) -> List[float]:
        """Generates a complete yet fast list of scale candidates matching Play Together PC window/HUD modes."""
        if not multiscale:
            return [1.0]

        # Fast path 1: If scale is already detected in FarmEngine
        if getattr(self, "_detected_ui_scale", None) is not None:
            det_s = round(self._detected_ui_scale, 2)
            cands = [det_s]
            for delta in (-0.03, 0.03):
                adj = round(det_s + delta, 2)
                if 0.35 <= adj <= 2.5 and adj not in cands:
                    cands.append(adj)
            return cands

        candidates: List[float] = []

        # Fast path 2: If DigitReader already locked a scale (e.g. 0.75 or 0.70)
        dr_scale = getattr(getattr(self, "digit_reader", None), "detected_scale", None)
        if dr_scale is not None:
            dr_s = round(float(dr_scale), 2)
            if 0.35 <= dr_s <= 2.5:
                candidates.append(dr_s)

        # Resolution prediction based on window height
        base_s = round(fh / 1080.0, 2)
        if base_s not in candidates:
            candidates.append(base_s)

        # Common Play Together UI scales across windowed / fullscreen modes
        for s in (1.0, 0.75, 0.70, 0.67, 0.53, 0.50):
            if s not in candidates:
                candidates.append(s)

        return candidates

    def find_template(
        self,
        frame: np.ndarray,
        tpl_key: str,
        threshold: float = 0.80,
        roi: Optional[Tuple[int, int, int, int]] = None,
        multiscale: bool = True,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Finds a template in the frame across adaptive scales, returning (x, y, w, h) if found."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        tpl = self._templates.get(tpl_key)
        if tpl is None:
            return None

        search_area = frame
        ox, oy = 0, 0
        if roi is not None:
            rx, ry, rw, rh = roi
            rx0 = max(0, rx)
            ry0 = max(0, ry)
            rx1 = min(frame.shape[1], rx + rw)
            ry1 = min(frame.shape[0], ry + rh)
            if rx1 <= rx0 or ry1 <= ry0:
                return None
            search_area = frame[ry0:ry1, rx0:rx1]
            ox, oy = rx0, ry0

        gray_f = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY) if len(search_area.shape) == 3 else search_area
        gray_t = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if len(tpl.shape) == 3 else tpl

        fh = frame.shape[0]
        candidates = self._get_scale_candidates(fh, multiscale=multiscale)

        best_val = 0.0
        best_loc = None
        best_size = None
        best_s = 1.0

        for s in candidates:
            tw = int(gray_t.shape[1] * s)
            th = int(gray_t.shape[0] * s)
            if gray_f.shape[0] >= th and gray_f.shape[1] >= tw and tw >= 8 and th >= 8:
                scaled_t = cv2.resize(gray_t, (tw, th), interpolation=cv2.INTER_LINEAR)
                res = cv2.matchTemplate(gray_f, scaled_t, cv2.TM_CCOEFF_NORMED)
                _, v, _, l = cv2.minMaxLoc(res)
                if v > best_val:
                    best_val = v
                    best_loc = l
                    best_size = (tw, th)
                    best_s = s
                if v >= 0.80:
                    break

        if best_val >= threshold and best_loc is not None and best_size is not None:
            if best_val >= 0.75 and getattr(self, "_detected_ui_scale", None) is None:
                self._detected_ui_scale = best_s
            return (ox + best_loc[0], oy + best_loc[1], best_size[0], best_size[1])
        return None

    def click_rect(self, rect: Tuple[int, int, int, int], margin_ratio: float = 0.2) -> None:
        """Clicks randomly inside a rectangular target."""
        rx, ry, rw, rh = rect
        # Safeguard: if rect matches HUD button aspect ratio (~90x115), limit vertical click to upper circular half
        if abs(rw / max(1, rh) - (90 / 115)) < 0.15 and rh >= 80:
            mx = int(rw * 0.25)
            cx = random.randint(rx + mx, rx + rw - mx)
            cy = random.randint(ry + int(rh * 0.18), ry + int(rh * 0.52))
            self.input_sim.click(cx, cy)
            return

        mx = int(rw * margin_ratio)
        my = int(rh * margin_ratio)
        cx = random.randint(rx + mx, rx + rw - mx)
        cy = random.randint(ry + my, ry + rh - my)
        self.input_sim.click(cx, cy)

    def click_hud_harvest_button(self, hit: Tuple[int, int, int, int]) -> None:
        """Clicks inside the circular green 'Thu hoạch' HUD button."""
        rx, ry, rw, rh = hit
        if rw < 70:  # Core template hit
            cx = rx + rw // 2 + random.randint(-6, 6)
            cy = ry + rh // 2 + random.randint(-6, 6)
        else:  # Legacy full box hit
            cx = rx + int(46 * rw / 90.0) + random.randint(-8, 8)
            cy = ry + int(38 * rh / 115.0) + random.randint(-8, 8)
        self.input_sim.click(cx, cy)

    def click_hud_sell_button(self, hit: Tuple[int, int, int, int]) -> None:
        """Clicks inside the circular pink/red 'Bán' HUD button."""
        rx, ry, rw, rh = hit
        if rw < 70:  # Core template hit
            cx = rx + rw // 2 + random.randint(-6, 6)
            cy = ry + rh // 2 + random.randint(-6, 6)
        else:  # Legacy full box hit
            cx = rx + int(48 * rw / 90.0) + random.randint(-8, 8)
            cy = ry + int(42 * rh / 115.0) + random.randint(-8, 8)
        self.input_sim.click(cx, cy)

    # =========================================================================
    # High-level Automation Routines
    # =========================================================================

    def step_open_harvest_window(self, frame: np.ndarray) -> bool:
        """
        Ensures harvest interface is active following strict sequential logic:
        Step 1: Check if character is ALREADY in front of Oliver (Hình 2: find_oliver_basket visible):
          -> If YES: Do NOT click 'Bán' (Hình 1). Skip directly to Step 2!
          -> If NO: Must click 'Bán' (Hình 1 on HUD) to teleport to Oliver.
                    Wait for teleport, close shop popup if opened.
        Step 2: Once at Oliver, click 'Thu hoạch' (B2 / Hình 6 on HUD) to open harvest window.
        """
        # Pre-check: dismiss any quit confirmation or notice modal blocking HUD
        if self.dismiss_quit_game_dialog_if_present(frame):
            time.sleep(0.4)
            f_clean, _ = self.capturer.capture(None)
            frame = f_clean if f_clean is not None else frame

        if self.handle_space_prompt_if_present(frame):
            time.sleep(0.4)
            f_clean, _ = self.capturer.capture(None)
            frame = f_clean if f_clean is not None else frame
        # Case A: Filter modal already open
        if self.is_filter_modal_open(frame):
            self._log("🔍 [TỰ ĐỘNG NHẬN DIỆN] Bảng Cài Đặt Bộ Lọc đã mở sẵn trên màn hình!")
            return True

        # Case B: Harvest cards window already open
        if self.is_harvest_window_open(frame):
            self._log("🌾 [TỰ ĐỘNG NHẬN DIỆN] Bảng Thu Hoạch đã mở sẵn trên màn hình!")
            return True

        # Case C: Shop modal open -> Close it cleanly to reveal HUD!
        if self.is_shop_open(frame):
            self._log("🚪 [ƯU TIÊN MÀN HÌNH] Cửa hàng Oliver đang mở! Đóng để quay lại màn hình chính...")
            close_btn = self.find_close_button(frame)
            if close_btn:
                self.click_rect(close_btn)
            else:
                self.input_sim.press_key("esc")
            time.sleep(0.6)
            return False

        # Case D: Oliver selection menu open -> Close with ESC
        if self.find_oliver_sell_option(frame) is not None:
            self._log("🚪 [ƯU TIÊN MÀN HÌNH] Đang mở menu đối thoại của Oliver! Bấm ESC để đóng...")
            self.input_sim.press_key("esc")
            time.sleep(0.6)
            return False

        # STEP 1: Check Oliver interaction bubble (Hình 2)
        at_oliver = (self.find_oliver_basket(frame, threshold=0.70) is not None)

        if at_oliver:
            # QUY TẮC CỐT LÕI: Thấy Hình 2 (giỏ hàng Oliver) -> KHÔNG ĐƯỢC bấm nút Bán (Hình 1)!
            self._log("ℹ️ [ĐÃ Ở TRƯỚC CỬA HÀNG] Đã đứng trước ông Oliver (thấy icon giỏ hàng Hình 2)! Bỏ qua nút 'Bán', chuyển thẳng tới nút 'Thu hoạch'...")
            self._teleported_to_oliver = True
            hit_b2 = self.find_hud_harvest_button(frame)
            if hit_b2:
                self._set_state(FarmState.OPENING_HARVEST)
                self._log("🌾 [BƯỚC 2] Thấy nút 'Thu hoạch' trên HUD! Bấm mở bảng nông sản...")
                self.click_hud_harvest_button(hit_b2)
                time.sleep(1.0)
                f_check, _ = self.capturer.capture(None)
                if (self.find_template(f_check, "btn_loc", threshold=0.68) or
                    self.find_template(f_check, "btn_thu_hoach", threshold=0.68) or
                    self.is_filter_empty(f_check)):
                    self._log("🌾 [B8] Đã mở thành công Bảng Thu Hoạch!")
                    return True
                return True
            return False

        # Chưa ở trước Oliver -> Bắt buộc bấm 'Bán' (Hình 1) trên HUD để dịch chuyển
        hit_b1 = self.find_hud_sell_button(frame)
        if hit_b1:
            self._set_state(FarmState.TELEPORTING)
            self._log("🚀 [BƯỚC 1] Chưa ở trước Oliver: Thấy nút 'Bán' (Hình 1) trên HUD! Bấm để dịch chuyển nhân vật tới trước mặt ông bán quả...")
            self.click_hud_sell_button(hit_b1)
            time.sleep(2.2)

            # Đóng popup cửa hàng của Oliver nếu bị tự động bật lên khi dịch chuyển
            f_after, _ = self.capturer.capture(None)
            close_btn = self.find_close_button(f_after)
            if close_btn and self.find_template(f_after, "sell_chon_tu_dong", threshold=0.75):
                self.click_rect(close_btn)
                time.sleep(0.8)

            # Chụp lại màn hình sau dịch chuyển: nhân vật đã đứng trước Oliver
            f_check, _ = self.capturer.capture(None)
            hit_b2 = self.find_hud_harvest_button(f_check)
            if hit_b2:
                self._set_state(FarmState.OPENING_HARVEST)
                self._log("🌾 [BƯỚC 2] Đã dịch chuyển tới ông bán quả! Bấm nút 'Thu hoạch' mở bảng nông sản...")
                self.click_hud_harvest_button(hit_b2)
                self._teleported_to_oliver = True
                time.sleep(1.0)
                f_final, _ = self.capturer.capture(None)
                if (self.find_template(f_final, "btn_loc", threshold=0.68) or
                    self.find_template(f_final, "btn_thu_hoach", threshold=0.68) or
                    self.is_filter_empty(f_final)):
                    self._log("🌾 [B8] Đã mở thành công Bảng Thu Hoạch!")
                    return True
                return True
            self._teleported_to_oliver = True
            return False

        # Chưa thấy nút Bán và chưa ở Oliver
        self._log("⏳ [BƯỚC 1] Đang tìm nút 'Bán' (Hình 1) trên màn hình để dịch chuyển tới ông Oliver...")
        time.sleep(0.4)
        return False

    def step_setup_filter(self, frame: np.ndarray, fruit_name: str) -> bool:
        """Applies filter for a single fruit (Category A) with deselection of other configured fruits across all screen resolutions."""
        in_modal = self.is_filter_modal_open(frame)
        if not in_modal:
            loc_btn = self.find_filter_button(frame)
            if not loc_btn:
                return False
            self._log(f"🔍 [BỘ LỌC] Bấm 'Lọc' để cấu hình riêng quả: '{fruit_name}'...")
            self.click_rect(loc_btn)
            time.sleep(0.25)
            f_cur, _ = self.capturer.capture(None)
        else:
            f_cur = frame

        # BƯỚC B4: Xử lý checkbox "Ngoại trừ trái đột biến" TRƯỚC TIÊN (khi modal vừa mở)
        chk = self.find_dot_bien(f_cur)
        if chk:
            fx, fy, fw, fh = chk
            box_w = min(fw, max(20, int(fh * 0.75)))
            roi_chk = f_cur[fy:fy+fh, fx:fx+box_w]
            if self.has_blue_checkbox(roi_chk):
                self._log("🛡️ [BỘ LỌC B4] 'Ngoại trừ trái đột biến' đã được tích sẵn [✓]! Giữ nguyên, không bấm hủy.")
            else:
                self._log("🛡️ [BỘ LỌC B4] Đã chọn 'Ngoại trừ trái đột biến' để bảo vệ quả đặc biệt!")
                self.click_rect((fx, fy, box_w, fh))
                time.sleep(0.3)
            f_cur, _ = self.capturer.capture(None)

        # BƯỚC B5: Mở mục "Nông sản" để hiển thị danh sách cây trái
        tab_ns = self.find_tab_nong_san(f_cur)
        if tab_ns:
            self._log("🌾 [BỘ LỌC B5] Bấm chọn mục 'Nông sản' để hiển thị danh sách cây trái...")
            self.click_rect(tab_ns)
            time.sleep(0.20)
            f_cur, _ = self.capturer.capture(None)

        # Defensive retry: Nếu chk_dot_bien chưa thấy ở frame 1 mà xuất hiện sau khi mở tab
        if not chk:
            chk_retry = self.find_dot_bien(f_cur)
            if chk_retry:
                fx, fy, fw, fh = chk_retry
                box_w = min(fw, max(20, int(fh * 0.75)))
                roi_chk = f_cur[fy:fy+fh, fx:fx+box_w]
                if self.has_blue_checkbox(roi_chk):
                    self._log("🛡️ [BỘ LỌC B4] 'Ngoại trừ trái đột biến' đã được tích sẵn [✓]! Giữ nguyên, không bấm hủy.")
                else:
                    self._log("🛡️ [BỘ LỌC B4] Đã chọn 'Ngoại trừ trái đột biến' để bảo vệ quả đặc biệt!")
                    self.click_rect((fx, fy, box_w, fh))
                    time.sleep(0.3)
                f_cur, _ = self.capturer.capture(None)

        # DESELECTION: Uncheck any other fruits that might be checked from previous rounds
        if isinstance(self.cfg_mgr.rules, dict):
            configured_fruits = list(self.cfg_mgr.rules.keys())
        elif isinstance(self.cfg_mgr.rules, list):
            configured_fruits = [r.name for r in self.cfg_mgr.rules]
        else:
            configured_fruits = []

        for other_name in configured_fruits:
            if other_name == fruit_name:
                continue
            other_key = self._get_fruit_template_key(other_name)
            if other_key:
                other_item = self.find_template(f_cur, other_key, threshold=0.55)
                if other_item:
                    ofx, ofy, ofw, ofh = other_item
                    card_w = max(int(ofh * 3.5), int(ofw * 1.4))
                    x_chk_start = int(card_w * 0.70)
                    roi_other_chk = f_cur[ofy:ofy+ofh, ofx + x_chk_start : ofx + card_w]
                    if self.has_blue_checkbox(roi_other_chk):
                        self._log(f"🌾 [BỘ LỌC] Hủy tích quả cũ '{other_name}' để chỉ giữ '{fruit_name}'...")
                        ocx = random.randint(ofx + int(card_w * 0.20), ofx + int(card_w * 0.75))
                        ocy = random.randint(ofy + int(ofh * 0.20), ofy + int(ofh * 0.80))
                        self.input_sim.click(ocx, ocy)
                        time.sleep(0.3)
                        f_cur, _ = self.capturer.capture(None)

        fruit_key = self._get_fruit_template_key(fruit_name)
        if fruit_key:
            fruit_item = self.find_template(f_cur, fruit_key, threshold=0.55)
            if fruit_item:
                fx, fy, fw, fh = fruit_item
                card_w = max(int(fh * 3.5), int(fw * 1.4))
                x_chk_start = int(card_w * 0.70)
                roi_card_chk = f_cur[fy:fy+fh, fx + x_chk_start : fx + card_w]
                if self.has_blue_checkbox(roi_card_chk):
                    self._log(f"🌾 [BỘ LỌC] Quả '{fruit_name}' đã được tích sẵn [✓]! Giữ nguyên, không bấm hủy.")
                else:
                    self._log(f"🌾 [BỘ LỌC B6] Đã chọn loại quả: '{fruit_name}'!")
                    cx = random.randint(fx + int(card_w * 0.20), fx + int(card_w * 0.75))
                    cy = random.randint(fy + int(fh * 0.20), fy + int(fh * 0.80))
                    self.input_sim.click(cx, cy)
                    time.sleep(0.3)
            else:
                self._log(f"⚠️ [BỘ LỌC B6] Không tìm thấy icon/tên '{fruit_name}' trên trang lọc!")
        else:
            self._log(f"⚠️ [BỘ LỌC B6] Chưa có template icon cho quả '{fruit_name}'!")

        # Final check: make sure chk_dot_bien is selected if it was only revealed after tab click
        f_final, _ = self.capturer.capture(None)
        chk_after = self.find_dot_bien(f_final)
        if chk_after:
            fx, fy, fw, fh = chk_after
            box_w = min(fw, max(20, int(fh * 1.1)))
            roi_chk = f_final[fy:fy+fh, fx:fx+box_w]
            if not self.has_blue_checkbox(roi_chk):
                self._log("🛡️ [BỘ LỌC] Đã chọn 'Ngoại trừ trái đột biến' để bảo vệ quả đặc biệt!")
                self.click_rect((fx, fy, box_w, fh))
                time.sleep(0.3)
                f_final, _ = self.capturer.capture(None)

        fh_fin, fw_fin = f_final.shape[:2]
        roi_apply = (int(fw_fin * 0.25), int(fh_fin * 0.60), int(fw_fin * 0.50), int(fh_fin * 0.38))
        btn_apply = self.find_template(f_final, "btn_cai_dat_loc", threshold=0.52, roi=roi_apply)
        if not btn_apply:
            btn_apply = self.find_template(f_final, "btn_cai_dat_loc", threshold=0.52)
        if not btn_apply:
            s = getattr(self, "_detected_ui_scale", None) or (fh_fin / 1080.0)
            btn_apply = (int(fw_fin * 0.50 - 130 * s), int(fh_fin * 0.88 - 35 * s), int(260 * s), int(70 * s))

        self._log("✅ [BỘ LỌC] Áp dụng bộ lọc thành công!")
        self.click_rect(btn_apply)
        time.sleep(0.30)
        return True

    def step_setup_multi_filter(self, frame: np.ndarray, fruit_names: List[str]) -> bool:
        """Applies filter for multiple Category B fruits at once with idempotent check across all screen resolutions."""
        in_modal = self.is_filter_modal_open(frame)
        if not in_modal:
            loc_btn = self.find_filter_button(frame)
            if not loc_btn:
                return False
            self._log(f"🔍 [BỘ LỌC ĐA QUẢ] Bấm 'Lọc' cấu hình {len(fruit_names)} loại quả thu tự do...")
            self.click_rect(loc_btn)
            time.sleep(0.25)
            f_cur, _ = self.capturer.capture(None)
        else:
            f_cur = frame

        # BƯỚC B4: Xử lý checkbox "Ngoại trừ trái đột biến" TRƯỚC TIÊN (khi modal vừa mở)
        chk = self.find_dot_bien(f_cur)
        if chk:
            fx, fy, fw, fh = chk
            box_w = min(fw, max(20, int(fh * 0.75)))
            roi_chk = f_cur[fy:fy+fh, fx:fx+box_w]
            if self.has_blue_checkbox(roi_chk):
                self._log("🛡️ [BỘ LỌC B4] 'Ngoại trừ trái đột biến' đã được tích sẵn [✓]! Giữ nguyên, không bấm hủy.")
            else:
                self._log("🛡️ [BỘ LỌC B4] Đã chọn 'Ngoại trừ trái đột biến' để bảo vệ quả đặc biệt!")
                self.click_rect((fx, fy, box_w, fh))
                time.sleep(0.3)
            f_cur, _ = self.capturer.capture(None)

        # BƯỚC B5: Mở mục "Nông sản" để hiển thị danh sách cây trái
        tab_ns = self.find_tab_nong_san(f_cur)
        if tab_ns:
            self._log("🌾 [BỘ LỌC B5] Bấm chọn mục 'Nông sản' để hiển thị danh sách cây trái...")
            self.click_rect(tab_ns)
            time.sleep(0.20)
            f_cur, _ = self.capturer.capture(None)

        # Defensive retry: Nếu chk_dot_bien chưa thấy ở frame 1 mà xuất hiện sau khi mở tab
        if not chk:
            chk_retry = self.find_dot_bien(f_cur)
            if chk_retry:
                fx, fy, fw, fh = chk_retry
                box_w = min(fw, max(20, int(fh * 0.75)))
                roi_chk = f_cur[fy:fy+fh, fx:fx+box_w]
                if self.has_blue_checkbox(roi_chk):
                    self._log("🛡️ [BỘ LỌC B4] 'Ngoại trừ trái đột biến' đã được tích sẵn [✓]! Giữ nguyên, không bấm hủy.")
                else:
                    self._log("🛡️ [BỘ LỌC B4] Đã chọn 'Ngoại trừ trái đột biến' để bảo vệ quả đặc biệt!")
                    self.click_rect((fx, fy, box_w, fh))
                    time.sleep(0.3)
                f_cur, _ = self.capturer.capture(None)

        for name in fruit_names:
            fruit_key = self._get_fruit_template_key(name)
            if fruit_key:
                fruit_item = self.find_template(f_cur, fruit_key, threshold=0.55)
                if fruit_item:
                    fx, fy, fw, fh = fruit_item
                    card_w = max(int(fh * 3.5), int(fw * 1.4))
                    x_chk_start = int(card_w * 0.70)
                    roi_card_chk = f_cur[fy:fy+fh, fx + x_chk_start : fx + card_w]
                    if self.has_blue_checkbox(roi_card_chk):
                        self._log(f"🌾 [BỘ LỌC] Quả '{name}' đã được tích sẵn [✓]! Giữ nguyên.")
                    else:
                        self._log(f" -> Tích chọn quả: '{name}'")
                        cx = random.randint(fx + int(card_w * 0.20), fx + int(card_w * 0.75))
                        cy = random.randint(fy + int(fh * 0.20), fy + int(fh * 0.80))
                        self.input_sim.click(cx, cy)
                        time.sleep(0.3)
                        f_cur, _ = self.capturer.capture(None)
                else:
                    self._log(f"⚠️ [BỘ LỌC B6] Chưa thấy icon/tên '{name}' trên trang lọc!")
            else:
                self._log(f"⚠️ [BỘ LỌC B6] Chưa có template icon cho quả '{name}'!")

        f_apply, _ = self.capturer.capture(None)
        fh_fin, fw_fin = f_apply.shape[:2]
        roi_apply = (int(fw_fin * 0.30), int(fh_fin * 0.65), int(fw_fin * 0.40), int(fh_fin * 0.32))
        btn_apply = self.find_template(f_apply, "btn_cai_dat_loc", threshold=0.52, roi=roi_apply)
        if not btn_apply:
            s = getattr(self, "_detected_ui_scale", None) or (fh_fin / 1080.0)
            btn_apply = (int(fw_fin * 0.50 - 130 * s), int(fh_fin * 0.88 - 35 * s), int(260 * s), int(70 * s))

        self._log("✅ [BỘ LỌC] Áp dụng bộ lọc đa quả thành công!")
        self.click_rect(btn_apply)
        time.sleep(0.30)
        return True

    @staticmethod
    def has_blue_checkbox(roi_bgr: Optional[np.ndarray]) -> bool:
        """
        Detects if a checkbox is checked [✓] with a vibrant cyan-blue square.
        HSV color range: H in [95..125], S >= 140, V >= 140.
        Uses adaptive pixel threshold proportional to ROI area to support all resolutions.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return False
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([95, 140, 140], dtype=np.uint8), np.array([125, 255, 255], dtype=np.uint8))
        nz = int(cv2.countNonZero(mask))
        area = roi_bgr.shape[0] * roi_bgr.shape[1]
        return nz >= max(25, int(area * 0.08))

    def is_harvest_window_showing_fruit(self, frame: np.ndarray, fruit_name: str) -> bool:
        """
        Determines if the Harvest Window is already open and showing cards for `fruit_name`.
        Allows resuming harvest mid-flight without re-opening the filter modal.
        """
        in_window = (
            self.find_template(frame, "btn_loc", threshold=0.68) is not None or
            self.find_template(frame, "btn_thu_hoach", threshold=0.68) is not None or
            self.is_filter_empty(frame)
        )
        if not in_window:
            return False

        cards = self.digit_reader.scan_cards(frame)
        if not cards:
            return self.is_filter_empty(frame)

        fruit_key = self._get_fruit_template_key(fruit_name)
        if not fruit_key or fruit_key not in self._templates:
            return True

        tpl = self._templates[fruit_key]
        res = cv2.matchTemplate(frame, tpl, cv2.TM_CCOEFF_NORMED)
        return bool(res.max() >= 0.44)

    def is_bag_full(self, frame: np.ndarray) -> bool:
        """Checks if 'Túi đã đầy nên không thể thu hoạch.' toast is displayed."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False
        fh, fw = frame.shape[:2]
        roi_toast = (int(fw * 0.15), int(fh * 0.05), int(fw * 0.70), int(fh * 0.50))
        return self.find_template(frame, "toast_day_tui", threshold=0.85, roi=roi_toast) is not None

    def find_quit_game_cancel_button(self, frame: np.ndarray, threshold: float = 0.70) -> Optional[Tuple[int, int, int, int]]:
        """
        Locates the 'Không, tôi muốn chơi thêm chút nữa.' cyan button on the quit confirmation modal
        ('Bạn có chắc muốn thoát trò chơi không?') using multi-scale template matching in the center ROI.
        Zero hardcoded coordinates; dynamically returns (x, y, w, h) of the button.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        tpl = self._templates.get("btn_khong_thoat")
        if tpl is None:
            return None

        fh, fw = frame.shape[:2]
        # Confirmation dialog always renders in center ROI: 20%..80% width, 25%..75% height
        rx0, rx1 = int(fw * 0.20), int(fw * 0.80)
        ry0, ry1 = int(fh * 0.25), int(fh * 0.75)
        roi = frame[ry0:ry1, rx0:rx1]
        if roi.size == 0:
            return None

        base_s = fw / 1024.0
        scales = [base_s * factor for factor in (0.80, 0.90, 1.00, 1.10, 1.20)]

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        gray_tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if len(tpl.shape) == 3 else tpl

        best_val = -1.0
        best_loc = None
        best_size = None

        for s in scales:
            tw = int(gray_tpl.shape[1] * s)
            th = int(gray_tpl.shape[0] * s)
            if tw >= gray_roi.shape[1] or th >= gray_roi.shape[0] or tw < 20 or th < 10:
                continue
            tpl_s = cv2.resize(gray_tpl, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
            res = cv2.matchTemplate(gray_roi, tpl_s, cv2.TM_CCOEFF_NORMED)
            min_v, max_v, min_l, max_l = cv2.minMaxLoc(res)
            if max_v > best_val:
                best_val = max_v
                best_loc = (rx0 + max_l[0], ry0 + max_l[1])
                best_size = (tw, th)
            if max_v >= 0.85:
                break

        if best_val >= threshold and best_loc is not None and best_size is not None:
            return (best_loc[0], best_loc[1], best_size[0], best_size[1])
        return None

    def dismiss_quit_game_dialog_if_present(self, frame: np.ndarray) -> bool:
        """
        Detects if 'Bạn có chắc muốn thoát trò chơi không?' is open on screen, and cancels exit
        by clicking 'Không, tôi muốn chơi thêm chút nữa.' at dynamically matched coordinates.
        Returns True if dialog was detected and safely dismissed.
        """
        btn = self.find_quit_game_cancel_button(frame)
        if btn is not None:
            self._log("🛡️ [CHẶN THOÁT GAME] Phát hiện bảng 'Bạn có chắc muốn thoát trò chơi không?'! Nhận diện và bấm 'Không, tôi muốn chơi thêm chút nữa.'...")
            self.click_rect(btn)
            time.sleep(0.5)
            return True
        return False

    def handle_space_prompt_if_present(self, frame: np.ndarray) -> bool:
        """
        Detects any sudden on-screen Spacebar prompt symbol [␣] or OK notice modal (Image 1: #102509).
        100% HARDWARE SPACE CONTROL: Sends DirectInput Space key directly, strictly avoiding mouse clicks
        to eliminate any risk of misclicking background UI or disturbing the harvest coordinate.
        Returns True if a prompt was detected and handled.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False

        # 1. Cyan Space Prompt / Notice Modal OK button (Image 1: e.g. #102509, B7 OK button)
        tpl_c = self._templates.get("prompt_space_cyan")
        tpl_ok = self._templates.get("sell_ok_btn")
        tpl_s_ok = self._templates.get("btn_space_ok")

        is_cyan_prompt = False

        # Fast search in center modal ROI
        fh, fw = frame.shape[:2]
        c_roi = frame[int(fh * 0.35):, int(fw * 0.15):int(fw * 0.85)]
        if c_roi.size > 0:
            if tpl_c is not None and cv2.matchTemplate(c_roi, tpl_c, cv2.TM_CCOEFF_NORMED).max() >= 0.78:
                is_cyan_prompt = True
            elif tpl_ok is not None and cv2.matchTemplate(c_roi, tpl_ok, cv2.TM_CCOEFF_NORMED).max() >= 0.75:
                is_cyan_prompt = True
            elif tpl_s_ok is not None and cv2.matchTemplate(c_roi, tpl_s_ok, cv2.TM_CCOEFF_NORMED).max() >= 0.78:
                is_cyan_prompt = True

        # Fallback to full frame match if not found in ROI
        if not is_cyan_prompt:
            if tpl_c is not None and cv2.matchTemplate(frame, tpl_c, cv2.TM_CCOEFF_NORMED).max() >= 0.78:
                is_cyan_prompt = True
            elif tpl_ok is not None and cv2.matchTemplate(frame, tpl_ok, cv2.TM_CCOEFF_NORMED).max() >= 0.75:
                is_cyan_prompt = True

        if is_cyan_prompt:
            self._log("⌨️ [SPACE PROMPT] Phát hiện bảng thông báo/nút Space [␣]! Bấm phím Space (né chuột 100%)...")
            self._fixed_harvest_point = None  # Reset coordinate anchor if a modal/error interrupted harvest
            self._fixed_card_kg_pos = None
            self.input_sim.press_key("space")
            time.sleep(0.4)
            return True

        # 2. Brown Space Prompt (e.g. NPC Cedric/Oliver dialogue bubble)
        # Only check if harvest window is NOT open to avoid chestnut card ambiguity
        is_harvest_window = self.find_template(frame, "btn_loc", threshold=0.70) is not None
        if not is_harvest_window:
            tpl_b = self._templates.get("prompt_space_brown")
            if tpl_b is not None:
                roi_b = frame[int(fh * 0.60):, :]
                res_b = cv2.matchTemplate(roi_b, tpl_b, cv2.TM_CCOEFF_NORMED)
                if res_b.max() >= 0.76:
                    self._log("⌨️ [SPACE PROMPT] Phát hiện bong bóng thoại có ký hiệu phím Space [␣] (nâu)! Bấm phím Space (né chuột 100%)...")
                    self.input_sim.press_key("space")
                    time.sleep(0.4)
                    return True

        return False

    def scroll_harvest_list_up(self, frame: np.ndarray, dy: Optional[int] = None) -> bool:
        """
        Scrolls the harvest list back UP to verify remaining crops near the top.
        - If dy is provided: Uses drag.
        - If dy is None: Uses mouse wheel rolling (2-3 notches x 2 rolls up) with randomized
          cursor positioning in the neutral empty/card body area (NO mouse button down).
        """
        fh, fw = frame.shape[:2]
        if dy is not None and dy > 0:
            drag_x = int(fw * (400 / 1920))
            y0 = int(fh * (350 / 1080))
            y1 = min(int(fh * 0.85), y0 + dy)
            self.input_sim.drag(drag_x, y0, drag_x, y1, duration_s=0.35, steps=20)
        else:
            rx = int(fw * random.uniform(0.18, 0.28))
            ry = int(fh * random.uniform(0.40, 0.65))
            self.input_sim.move_to(rx, ry, human_curve=False)
            time.sleep(0.04)
            for _ in range(2):
                notches = random.randint(2, 3)
                delta = 120 * notches
                self.input_sim.scroll(delta=delta)
                time.sleep(0.08)
        time.sleep(0.25)
        return True

    def is_filter_empty(self, frame: np.ndarray) -> bool:
        """
        Checks if 'Không có trái nào phù hợp với bộ lọc đã chọn.' banner is displayed.
        Restricts template matching strictly to the center ROI of the harvest modal.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False
        fh, fw = frame.shape[:2]
        roi = (int(fw * 0.15), int(fh * 0.25), int(fw * 0.70), int(fh * 0.50))
        return self.find_template(frame, "tpl_het_trai_1080p", threshold=0.68, roi=roi) is not None

    def find_oliver_basket(self, frame: np.ndarray, threshold: float = 0.70) -> Optional[Tuple[int, int, int, int]]:
        """
        Locates the Oliver basket interaction bubble above Oliver's head (Image 6 / Hinh 2).
        Uses adaptive multi-scale matching (threshold 0.70) in central upper 3D view.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        tpl = self._templates.get("oliver_basket")
        if tpl is None:
            return None

        # Guard: Filter modal and harvest cards modal take precedence
        if (self.find_template(frame, "btn_cai_dat_loc", threshold=0.65) is not None or
            self.find_template(frame, "chk_dot_bien", threshold=0.65) is not None or
            self.is_harvest_window_open(frame)):
            return None

        fh, fw = frame.shape[:2]
        center_roi = (int(fw * 0.20), int(fh * 0.05), int(fw * 0.65), int(fh * 0.70))
        hit = self.find_template(frame, "oliver_basket", threshold=threshold, roi=center_roi)
        if not hit:
            hit = self.find_template(frame, "oliver_basket", threshold=threshold)
        if hit:
            bx, by, bw, bh = hit
            roi = frame[by:by+bh, bx:bx+bw]
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                if gray_roi.mean() >= 120:
                    return hit
        return None

    def find_hud_sell_button(self, frame: np.ndarray, threshold: float = 0.58) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects the circular pink/red 'Bán' button (Image 4 / B1) in HUD across resolutions and window sizes.
        Uses core icon template matching ('core_b1_ban') across adaptive scales.
        Falls back to legacy full template ('harvest_b1') if available.
        Uses fast right-side ROI first, with full-frame fallback for windowed / non-standard screens.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        fh, fw = frame.shape[:2]
        hud_roi = (int(fw * 0.40), int(fh * 0.01), int(fw * 0.60), int(fh * 0.95))

        # 1. Fast right-side ROI
        hit = self.find_template(frame, "core_b1_ban", threshold=threshold, roi=hud_roi)
        if hit:
            return hit
        hit = self.find_template(frame, "harvest_b1", threshold=threshold, roi=hud_roi)
        if hit:
            return hit

        # 2. Full-frame fallback
        hit = self.find_template(frame, "core_b1_ban", threshold=threshold)
        if hit:
            return hit
        return self.find_template(frame, "harvest_b1", threshold=threshold)

    def find_hud_harvest_button(self, frame: np.ndarray, threshold: float = 0.58) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects the circular green 'Thu hoạch' button (Image 5 / B2) in HUD across resolutions and window sizes.
        Uses core icon template matching ('core_b2_harvest') across adaptive scales.
        Falls back to legacy full template ('harvest_b2') if available.
        Uses fast right-side ROI first, with full-frame fallback for windowed / non-standard screens.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        fh, fw = frame.shape[:2]
        hud_roi = (int(fw * 0.40), int(fh * 0.01), int(fw * 0.60), int(fh * 0.95))

        # 1. Fast right-side ROI
        hit = self.find_template(frame, "core_b2_harvest", threshold=threshold, roi=hud_roi)
        if hit:
            return hit
        hit = self.find_template(frame, "harvest_b2", threshold=threshold, roi=hud_roi)
        if hit:
            return hit

        # 2. Full-frame fallback
        hit = self.find_template(frame, "core_b2_harvest", threshold=threshold)
        if hit:
            return hit
        return self.find_template(frame, "harvest_b2", threshold=threshold)

    def find_filter_button(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Locates the 'Lọc' button in the harvest window top-right corner (< 30ms).
        Directly matches 'btn_loc' in top-right ROI first. Only checks modal guard if missing.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        fh, fw = frame.shape[:2]
        top_right_roi = (int(fw * 0.55), 0, int(fw * 0.45), int(fh * 0.40))
        hit = self.find_template(frame, "btn_loc", threshold=0.52, roi=top_right_roi)
        if hit:
            return hit
        hit = self.find_template(frame, "btn_loc", threshold=0.52)
        if hit:
            return hit

        # Guard: Only verify if modal is open when btn_loc was not found (saves 400ms!)
        if self.is_filter_modal_open(frame):
            return None

        # Fallback relative to close [X] button
        close_hit = self.find_close_button(frame)
        if close_hit:
            cx, cy, cw, ch = close_hit
            s = getattr(self, "_detected_ui_scale", None) or (fh / 1080.0)
            lx = cx - int(190 * s)
            ly = cy + int(10 * s)
            lw = int(180 * s)
            lh = int(50 * s)
            if 0 <= lx and lx + lw <= fw and 0 <= ly and ly + lh <= fh:
                return (lx, ly, lw, lh)

        return None

    def find_tab_nong_san(self, frame: np.ndarray, threshold: float = 0.58) -> Optional[Tuple[int, int, int, int]]:
        """
        Locates the 'Nông sản' tab/category on the harvest filter modal.
        Uses adaptive matching in left-side modal ROI with full-frame fallback.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        fh, fw = frame.shape[:2]
        roi_tab = (int(fw * 0.08), int(fh * 0.08), int(fw * 0.40), int(fh * 0.45))
        hit = self.find_template(frame, "tab_nong_san", threshold=threshold, roi=roi_tab)
        if hit is not None:
            return hit
        return self.find_template(frame, "tab_nong_san", threshold=threshold)

    def find_dot_bien(self, frame: np.ndarray, threshold: float = 0.55) -> Optional[Tuple[int, int, int, int]]:
        """
        Locates the 'Ngoại trừ trái đột biến' (Biến thể) checkbox row on the harvest filter modal.
        Uses adaptive matching in bottom-left modal ROI with full-frame fallback.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        fh, fw = frame.shape[:2]
        roi_chk = (int(fw * 0.08), int(fh * 0.50), int(fw * 0.55), int(fh * 0.45))
        hit = self.find_template(frame, "chk_dot_bien", threshold=threshold, roi=roi_chk)
        if hit is not None:
            return hit
        return self.find_template(frame, "chk_dot_bien", threshold=threshold)

    def can_harvest_list_scroll(self, cards: List[HarvestCard], frame: np.ndarray) -> bool:
        """
        Determines if the harvest card table can physically scroll down further.
        Prevents unnecessary, redundant mouse dragging when:
          1. The visible card count is small (< 6 cards out of 10-slot full screen).
          2. The last visible row is incomplete (odd number of cards: 1, 3, 5, 7, 9).
          3. The scrollbar thumb has reached the bottom (bottom_ratio >= 0.95 and thumb_ratio < 0.85).
        """
        if not cards:
            return False

        # Check scrollbar position first
        thumb_ratio, bottom_ratio, is_at_bottom = self.get_scrollbar_info(frame)
        if is_at_bottom and bottom_ratio >= 0.95 and thumb_ratio < 0.85:
            return False

        # Less than 6 cards -> Definitely cannot fill a 10-card page (e.g. 5 cards like Mango screenshot)
        if len(cards) < 6:
            return False

        # Incomplete last row (odd number of cards) -> End of items
        if len(cards) % 2 != 0:
            return False

        return True

    def close_harvest_window(self) -> bool:
        """
        Closes harvest modal and verifies return to the main screen in front of Oliver
        (where HUD harvest button / Oliver basket is visible).
        """
        self._fixed_harvest_point = None
        self._fixed_card_kg_pos = None
        for attempt in range(4):
            f_cur, _ = self.capturer.capture(None)
            # Check if HUD is visible and harvest modal is closed
            is_harvest_open = self.is_harvest_window_open(f_cur)
            is_shop_open = self.is_shop_open(f_cur)
            if not is_harvest_open and not is_shop_open:
                if self.find_hud_harvest_button(f_cur) is not None or self.find_oliver_basket(f_cur) is not None:
                    return True

            close_btn = self.find_close_button(f_cur)
            if close_btn:
                self.click_rect(close_btn)
                time.sleep(0.5)
            elif is_harvest_open:
                # ONLY press ESC if harvest window was confirmed open to avoid HUD quit prompt
                self.input_sim.press_key("esc")
                time.sleep(0.6)
            else:
                return True

        return True

    def is_filter_modal_open(self, frame: np.ndarray) -> bool:
        """Determines if Crop Filter modal dialog is open on screen (< 40ms, ROI targeted)."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False
        fh, fw = frame.shape[:2]
        roi_apply = (int(fw * 0.20), int(fh * 0.55), int(fw * 0.60), int(fh * 0.43))
        if self.find_template(frame, "btn_cai_dat_loc", threshold=0.55, roi=roi_apply) is not None:
            return True
        roi_chk = (int(fw * 0.05), int(fh * 0.45), int(fw * 0.60), int(fh * 0.50))
        if self.find_template(frame, "chk_dot_bien", threshold=0.55, roi=roi_chk) is not None:
            return True
        return False

    def find_close_button(self, frame: np.ndarray, allow_fallback: bool = False) -> Optional[Tuple[int, int, int, int]]:
        """
        Finds the close [X] button on top-right of modal dialogs using template matching.
        Only falls back to calibrated modal corner coordinates if allow_fallback=True.
        Strictly returns None on the main HUD screen or in 3D world to prevent false modal detection.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        fh, fw = frame.shape[:2]
        top_right_roi = (int(fw * 0.55), 0, int(fw * 0.45), int(fh * 0.40))
        hit = self.find_template(frame, "btn_close_x", threshold=0.62, roi=top_right_roi)
        if hit:
            return hit
        hit = self.find_template(frame, "btn_close_x", threshold=0.62)
        if hit:
            return hit
        if allow_fallback:
            return (int(fw * 0.831), int(fh * 0.136), int(67 * (fh / 1080.0)), int(60 * (fh / 1080.0)))
        return None

    def is_harvest_window_open(self, frame: np.ndarray) -> bool:
        """
        Determines if the Harvest Cards window is actually open and visible on screen:
          - Filter empty banner showing ('Không có trái nào phù hợp...').
          - Cyan 'btn_thu_hoach' buttons on crop cards.
          - 'Lọc' button (template match >= 0.52).
        Strictly returns False if the Filter Modal or Shop modal is open, or on the main HUD screen.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False

        fh, fw = frame.shape[:2]
        top_right_roi = (int(fw * 0.55), int(fh * 0.02), int(fw * 0.40), int(fh * 0.35))
        has_loc = self.find_template(frame, "btn_loc", threshold=0.52, roi=top_right_roi) is not None
        has_empty = self.is_filter_empty(frame)
        has_cards_btn = self.find_template(frame, "btn_thu_hoach", threshold=0.68) is not None

        if not (has_loc or has_empty or has_cards_btn):
            return False

        # Guard 1: Filter modal
        if self.is_filter_modal_open(frame):
            return False

        # Guard 2: Shop modal (only if sell_chon_tu_dong template matches)
        roi_shop = (int(fw * 0.40), int(fh * 0.70), int(fw * 0.40), int(fh * 0.25))
        if self.find_template(frame, "sell_chon_tu_dong", threshold=0.65, roi=roi_shop) is not None:
            return False

        return True

    def find_shop_coin_button(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects the active cyan coin sell button (B4) in Oliver's shop window.
        Restricted to bottom bar below crop cards to prevent false matches with crop icons.
        Verifies that crops are actually selected (red notification badge present or active B4 template match).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        fh, fw = frame.shape[:2]
        rx0, rx1 = int(fw * 0.60), fw
        ry0, ry1 = int(fh * 0.72), fh
        roi = (rx0, ry0, rx1 - rx0, ry1 - ry0)

        # Layer 1: Template matching with sell_coin_btn within bottom-right ROI
        hit = self.find_template(frame, "sell_coin_btn", threshold=0.55, roi=roi)
        if hit:
            if self._has_shop_coin_badge(frame, hit):
                return hit
            # If high threshold match (active B4 template)
            tpl = self._templates.get("sell_coin_btn")
            if tpl is not None:
                hx, hy, hw, hh = hit
                roi_c = frame[hy:hy+hh, hx:hx+hw]
                if roi_c.shape[:2] == tpl.shape[:2]:
                    r = cv2.matchTemplate(roi_c, tpl, cv2.TM_CCOEFF_NORMED)
                    if float(r.max()) >= 0.70:
                        return hit

        # Layer 2: Color segmentation fallback in bottom bar
        roi_img = frame[ry0:ry1, rx0:rx1]
        if roi_img.size > 0:
            hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([90, 120, 120]), np.array([115, 255, 255]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                # Coin button is a wide horizontal pill button (w / h >= 1.8) at bottom-right
                if w >= int(fw * 0.10) and h >= int(fh * 0.06) and (w / float(h)) >= 1.8:
                    btn_rect = (rx0 + x, ry0 + y, w, h)
                    if self._has_shop_coin_badge(frame, btn_rect):
                        return btn_rect

        return None

    def _has_shop_coin_badge(self, frame: np.ndarray, coin_rect: Tuple[int, int, int, int]) -> bool:
        """Checks if red notification badge is present at top-right of coin button."""
        x, y, w, h = coin_rect
        fh, fw = frame.shape[:2]
        bx0 = x + int(w * 0.65)
        bx1 = min(fw, x + int(w * 1.15))
        by0 = max(0, y - int(h * 0.35))
        by1 = min(fh, y + int(h * 0.50))
        if by1 <= by0 or bx1 <= bx0:
            return False
        roi = frame[by0:by1, bx0:bx1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([165, 100, 100]), np.array([180, 255, 255]))
        return float(np.sum((m1 | m2) > 0)) > 15

    def find_shop_auto_select_button(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects the yellow '(0/100) Chọn tự động' or '(.. / 100) Bỏ chọn tất cả' button (B3) in Oliver's shop.
        Uses template matching with 'sell_chon_tu_dong' (threshold 0.50) with full-frame fallback.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        fh, fw = frame.shape[:2]
        roi_auto = (int(fw * 0.35), int(fh * 0.65), int(fw * 0.50), int(fh * 0.33))
        hit = self.find_template(frame, "sell_chon_tu_dong", threshold=0.50, roi=roi_auto)
        if hit is not None:
            return hit
        return self.find_template(frame, "sell_chon_tu_dong", threshold=0.50)

    def find_oliver_sell_option(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Locates Oliver's 'Bán nông sản' / 'Bán cây trồng' dialogue option (A1).
        Multi-tier detection:
        1. New template 'oliver_sell_crops' ('Bán nông sản' - B2.jpg)
        2. Old template 'oliver_sell_crops_old' ('Bán cây trồng')
        3. Cyan Capsule Color Contour fallback (Hue 85..105, Value > 180, Aspect 4.0..8.5, W > 200)
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        # Tier 1: New template 'Bán nông sản' (B2.jpg)
        hit = self.find_template(frame, "oliver_sell_crops", threshold=0.62)
        if hit is not None:
            return hit

        # Tier 2: Old template 'Bán cây trồng'
        hit = self.find_template(frame, "oliver_sell_crops_old", threshold=0.62)
        if hit is not None:
            return hit

        # Tier 3: Cyan Capsule Color Fallback in dialogue area
        fh, fw = frame.shape[:2]
        roi_x0 = int(fw * 0.40)
        roi_y0 = int(fh * 0.30)
        roi_x1 = int(fw * 0.95)
        roi_y1 = int(fh * 0.85)
        roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]
        if roi.size > 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([85, 35, 180]), np.array([105, 255, 255]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 5000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = w / max(1, h)
                    if 4.0 <= aspect <= 8.5 and w > 200 and h > 30:
                        return (roi_x0 + x, roi_y0 + y, w, h)

        return None

    def is_shop_open(self, frame: np.ndarray) -> bool:
        """
        Determines if Oliver's crop shop modal is open and visible on screen.
        Guarantees zero false positives on harvest cards window or filter modal.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False

        # Fast gate: If no close button, no shop modal can be open!
        close_btn = self.find_close_button(frame)
        if not close_btn:
            return False

        # Guard 1: Filter modal has close button, but is NOT shop modal
        if self.is_filter_modal_open(frame):
            return False

        # Guard 2: Harvest window has close button and crop cards, but is NOT shop modal
        if self.is_harvest_window_open(frame):
            return False

        # Check positive shop indicators (strictly buttons unique to shop modal)
        if self.find_shop_auto_select_button(frame) is not None:
            return True
        if self.find_shop_coin_button(frame) is not None:
            return True

        return False

    def find_sell_confirm_button(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects the cyan pill button with coin amount (B5) on popup 'Xác nhận bán hàng'.
        Uses color segmentation for cyan pill button in center bottom of modal:
          X in [0.35..0.65], Y in [0.60..0.85].
        Falls back to template matching with threshold 0.55.
        Immune to changing dynamic coin numbers!
        """
        fh, fw = frame.shape[:2]
        ry0, ry1 = int(fh * 0.60), int(fh * 0.85)
        rx0, rx1 = int(fw * 0.35), int(fw * 0.65)
        if ry1 <= fh and rx1 <= fw:
            roi = frame[ry0:ry1, rx0:rx1]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([90, 120, 120]), np.array([115, 255, 255]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w > 100 and h > 35:
                    return (rx0 + x, ry0 + y, w, h)
        return self.find_template(frame, "sell_confirm_btn", threshold=0.55)

    def find_high_tier_confirm_button(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects the cyan 'Xác nhận' button (B6) on popup 'Xác nhận bán' (High-Tier Warning).
        Strict dual-button verification:
          1. Verifies the orange 'Không' button is present at bottom-left (X: 0.28..0.50, Y: 0.60..0.85).
          2. Locates the cyan 'Xác nhận' button at bottom-right (X: 0.48..0.72, Y: 0.60..0.85).
        Guarantees zero false positives against the shop's coin button or crop grid.
        """
        fh, fw = frame.shape[:2]
        ry0, ry1 = int(fh * 0.60), int(fh * 0.85)
        if ry1 > fh:
            return None

        # Step 1: Guard with orange [Không] button
        rx_orange0, rx_orange1 = int(fw * 0.28), int(fw * 0.50)
        roi_orange = frame[ry0:ry1, rx_orange0:rx_orange1]
        hsv_orange = cv2.cvtColor(roi_orange, cv2.COLOR_BGR2HSV)
        mask_orange = cv2.inRange(hsv_orange, np.array([10, 140, 140]), np.array([25, 255, 255]))
        contours_o, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_orange = any(cv2.boundingRect(c)[2] > 70 and cv2.boundingRect(c)[3] > 30 for c in contours_o)
        if not has_orange:
            return None  # High-tier modal is NOT open!

        # Step 2: Detect cyan [Xác nhận] button
        rx_cyan0, rx_cyan1 = int(fw * 0.48), int(fw * 0.72)
        roi_cyan = frame[ry0:ry1, rx_cyan0:rx_cyan1]
        hsv_cyan = cv2.cvtColor(roi_cyan, cv2.COLOR_BGR2HSV)
        mask_cyan = cv2.inRange(hsv_cyan, np.array([90, 120, 120]), np.array([115, 255, 255]))
        contours_c, _ = cv2.findContours(mask_cyan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_c:
            x, y, w, h = cv2.boundingRect(c)
            if w > 80 and h > 30:
                return (rx_cyan0 + x, ry0 + y, w, h)

        return self.find_template(frame, "sell_confirm_high_tier", threshold=0.68)

    def _has_harvest_button_near(self, frame: np.ndarray, pt: Tuple[int, int]) -> bool:
        """Verifies if cyan harvest button color is present within a 30px radius of click point."""
        cx, cy = pt
        fh, fw = frame.shape[:2]
        r = 30
        x0, x1 = max(0, cx - r), min(fw, cx + r)
        y0, y1 = max(0, cy - r), min(fh, cy + r)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 100, 100]), np.array([115, 255, 255]))
        return float(np.mean(mask > 0)) >= 0.15

    def _wait_for_card_shift(self, pt: Tuple[int, int], pre_frame: np.ndarray) -> np.ndarray:
        """
        Active Settlement Guard:
        Waits for the harvested card at pt to physically change or disappear from the screen.
        Eliminates duplicate 'inertia' clicks and stale reads of already harvested fruits.
        Returns the settled new frame.
        """
        if not hasattr(self, "capturer") or self.capturer is None or pre_frame is None or not isinstance(pre_frame, np.ndarray) or pre_frame.size == 0:
            time.sleep(max(getattr(self.cfg_mgr, "shift_delay_s", 0.20), 0.15))
            return pre_frame

        cx, cy = pt
        fh, fw = pre_frame.shape[:2]
        x0, x1 = max(0, cx - 35), min(fw, cx + 35)
        y0, y1 = max(0, cy - 20), min(fh, cy + 20)
        crop = pre_frame[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            time.sleep(0.05)
            return pre_frame

        pre_roi = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        deadline = time.time() + 0.55  # Max 550ms timeout
        time.sleep(0.12)  # Minimum physical click-to-animation response in Unity
        
        last_f = pre_frame
        while time.time() < deadline:
            cap_res = self.capturer.capture(None)
            f_cur = cap_res[0] if (isinstance(cap_res, tuple) and len(cap_res) > 0) else cap_res
            if f_cur is None or not isinstance(f_cur, np.ndarray) or f_cur.size == 0:
                time.sleep(0.04)
                continue
            cur_crop = f_cur[y0:y1, x0:x1]
            if cur_crop.size == 0 or cur_crop.shape != crop.shape:
                time.sleep(0.04)
                continue
            cur_roi = cv2.cvtColor(cur_crop, cv2.COLOR_BGR2GRAY)
            diff = float(np.mean(cv2.absdiff(cur_roi, pre_roi)))
            last_f = f_cur
            # If visual difference > 12.0: card has transitioned/disappeared!
            if diff > 12.0:
                time.sleep(0.08)  # Let new card complete settling
                cap_fin = self.capturer.capture(None)
                f_settled = cap_fin[0] if (isinstance(cap_fin, tuple) and len(cap_fin) > 0) else cap_fin
                if f_settled is not None and isinstance(f_settled, np.ndarray) and f_settled.size > 0:
                    return f_settled
                return last_f
            time.sleep(0.04)

        return last_f

    def _group_cards_by_row(self, cards: List[HarvestCard]) -> List[List[HarvestCard]]:
        """Groups scanned cards into horizontal rows based on Y coordinates."""
        rows: List[List[HarvestCard]] = []
        for c in cards:
            placed = False
            for r in rows:
                if abs(r[0].btn_y - c.btn_y) < 25:
                    r.append(c)
                    placed = True
                    break
            if not placed:
                rows.append([c])
        return sorted(rows, key=lambda r: r[0].btn_y)

    def align_boundary_row_to_top(self, frame: np.ndarray, max_kg: float) -> Tuple[np.ndarray, bool]:
        """
        Bottom-up Boundary Alignment with Anti-Overshoot:
        If heavy crops (>= max_kg) occupy Rows 1-3 and eligible crops start at Row 4 or 5 (like Image 1),
        scrolls the list so that the boundary row (e.g. 4.23kg | 1.09kg) lands precisely at Row 1 (Image 4).
        Uses calibrated dy (0.72x) to prevent overshooting past valid crops.
        """
        cards = self.digit_reader.scan_cards(frame)
        rows = self._group_cards_by_row(cards)
        if not rows or len(rows) < 2:
            return frame, False

        row1_y = rows[0][0].btn_y
        boundary_idx = None
        # Scan from bottom up (highest Y to lowest Y)
        for idx in range(len(rows) - 1, -1, -1):
            if any(0.0 < c.weight < max_kg for c in rows[idx]):
                boundary_idx = idx

        # If eligible crops start below Row 1 (e.g. Row 4 or Row 5) and there are heavy crops above
        if boundary_idx is not None and boundary_idx > 0:
            has_heavy_above = any(any(c.weight >= max_kg for c in rows[i]) for i in range(boundary_idx))
            if has_heavy_above:
                boundary_y = rows[boundary_idx][0].btn_y
                raw_dy = boundary_y - row1_y
                calibrated_dy = int(raw_dy * 0.72)
                if calibrated_dy > 20:
                    self._log(f"📜 [CĂN CHỈNH RANH GIỚI] Thấy quả lớn ở phía trên và quả nhỏ bắt đầu ở Hàng {boundary_idx+1} (Y={boundary_y}). Đang lướt chuẩn xác đưa dòng ranh giới lên Hàng 1 (dy={calibrated_dy}px)...")
                    self.scroll_harvest_list_down(frame, dy=calibrated_dy)
                    self._did_scroll_boundary = True
                    time.sleep(0.30)
                    cap_res = self.capturer.capture(None) if hasattr(self, "capturer") and self.capturer else None
                    f_aligned = cap_res[0] if (isinstance(cap_res, tuple) and len(cap_res) > 0 and cap_res[0] is not None) else frame
                    return f_aligned, True

        return frame, False

    def execute_harvest_round(self, frame: np.ndarray, max_kg: float) -> Tuple[bool, bool]:
        """
        Scans visible harvest cards with OCR.
        Locks onto a single fixed harvest button and coordinate (self._fixed_harvest_point) selected in Rows 2 & 3.
        Clicks the exact same point repeatedly as crops shift forward.
        - Active Settlement: waits for card to physically vanish/shift (> 12 diff).
        - If point is empty/white space: Resets fixed point, stops immediately (zero inertia clicks).
        - If a crop shifting into the point is >= max_kg: VETOES click to protect giant crop.
        - Returns: (harvested_any, should_scroll)
        """
        if self.is_filter_empty(frame):
            self._fixed_harvest_point = None
            return False, False

        # --- Sub-loop: Locked Click Point Handling ---
        if self._fixed_harvest_point is not None:
            cx, cy = self._fixed_harvest_point

            # Check if harvest button has disappeared completely (empty slot / white space)
            if not self._has_harvest_button_near(frame, (cx, cy)):
                time.sleep(0.12)
                cap_conf = self.capturer.capture(None) if hasattr(self, "capturer") and self.capturer else None
                f_confirm = cap_conf[0] if (isinstance(cap_conf, tuple) and len(cap_conf) > 0 and cap_conf[0] is not None) else frame
                if not self._has_harvest_button_near(f_confirm, (cx, cy)):
                    self._log(f"ℹ️ [HẾT QUẢ TẠI Ô] Ô thu hoạch cố định ({cx}, {cy}) đã hết quả (trở thành khoảng trắng)! Chọn vị trí khác...")
                    self._fixed_harvest_point = None
                    self._fixed_card_kg_pos = None
                    self._fixed_point_unsettled_count = 0
                    # Fall through to scan remaining cards on screen

            if self._fixed_harvest_point is not None:
                card_weight = None
                card_index = getattr(self, "_fixed_card_index", 1)

                if getattr(self, "_fixed_card_kg_pos", None) is not None:
                    kg_x, kg_y = self._fixed_card_kg_pos
                    card_weight = self.digit_reader.read_weight_at_pos(frame, kg_x, kg_y)

                if card_weight is None or card_weight >= 9000.0:
                    cards = self.digit_reader.scan_cards(frame)
                    under = [c for c in cards if c.btn_x <= cx <= c.btn_x + c.btn_w and c.btn_y <= cy <= c.btn_y + c.btn_h]
                    if under:
                        card = under[0]
                        card_weight = card.weight
                        card_index = getattr(card, "card_index", 1)
                        if getattr(card, "kg_x", None) is not None and getattr(card, "kg_y", None) is not None:
                            self._fixed_card_kg_pos = (card.kg_x, card.kg_y)
                        else:
                            self._fixed_card_kg_pos = None
                        self._fixed_card_index = card_index
                    else:
                        card_weight = 9999.0

                if 0.0 < card_weight < max_kg:
                    self._fixed_point_unsettled_count = 0
                    self._log(f"🧺 [THU HOẠCH CỐ ĐỊNH] Đang thu ô #{card_index}: {card_weight:.2f}kg (< {max_kg:.2f}kg) tại ({cx}, {cy})...")
                    self.input_sim.click(cx, cy, hold_min_ms=20, hold_max_ms=40)
                    self.stats["harvested_count"] += 1

                    # Active settlement: wait for animation
                    f_settled = self._wait_for_card_shift((cx, cy), frame)
                    self._last_settled_frame = f_settled

                    # Check for giant crop shifting in: VETO immediately!
                    if getattr(self, "_fixed_card_kg_pos", None) is not None:
                        kx, ky = self._fixed_card_kg_pos
                        shifted_w = self.digit_reader.read_weight_at_pos(f_settled, kx, ky)
                        if shifted_w >= max_kg and shifted_w < 9000.0:
                            self._log(f"🛡️ [VETO BẢO VỆ TUYỆT ĐỐI] Quả vừa dồn vào ô ({cx},{cy}) là {shifted_w:.2f}kg >= {max_kg:.2f}kg! VETO, hủy ô cố định bảo vệ quả lớn!")
                            self._fixed_harvest_point = None
                            self._fixed_card_kg_pos = None

                    # KEEP _fixed_harvest_point LOCKED! Do not destroy here so next click hits exact same spot!
                    return True, False
                elif card_weight >= max_kg and card_weight < 9000.0:
                    self._log(f"🛡️ [VETO BẢO VỆ TUYỆT ĐỐI] Quả dồn vào ô ({cx},{cy}) là {card_weight:.2f}kg >= {max_kg:.2f}kg! Hủy click bảo vệ quả lớn, chuyển sang nút khác...")
                    self._fixed_harvest_point = None
                    self._fixed_card_kg_pos = None
                    self._fixed_point_unsettled_count = 0
                    # Fall through to scan remaining cards!
                else:
                    self._fixed_point_unsettled_count = getattr(self, "_fixed_point_unsettled_count", 0) + 1
                    if self._fixed_point_unsettled_count >= 3:
                        self._log(f"🛡️ [BẢO VỆ AN TOÀN] Ô cố định ({cx},{cy}) không thể xác định số kg an toàn! Đổi sang vị trí khác để tránh thu nhầm...")
                        self._fixed_harvest_point = None
                        self._fixed_card_kg_pos = None
                        self._fixed_point_unsettled_count = 0
                    else:
                        time.sleep(0.08)
                        return False, False

        # --- Phase 1: Boundary Alignment Check ---
        f_aligned, aligned = self.align_boundary_row_to_top(frame, max_kg)
        if aligned:
            frame = f_aligned
            self._last_settled_frame = f_aligned

        cards = self.digit_reader.scan_cards(frame)
        if not cards:
            self._fixed_harvest_point = None
            return False, True

        rows = self._group_cards_by_row(cards)
        eligible = [c for c in cards if 0.0 < c.weight < max_kg and not c.is_clipped]

        if eligible:
            # --- Phase 2: Safe Row Selection (Row 2 & 3 Preferred) ---
            # Center Row 2 (index 1) is highest priority, then Row 3 (index 2)
            row2_cards = [c for c in rows[1] if 0.0 < c.weight < max_kg and not c.is_clipped] if len(rows) >= 2 else []
            row3_cards = [c for c in rows[2] if 0.0 < c.weight < max_kg and not c.is_clipped] if len(rows) >= 3 else []

            if row2_cards:
                target = random.choice(row2_cards)
            elif row3_cards:
                target = random.choice(row3_cards)
            else:
                row1_has_heavy = any(c.weight >= max_kg for c in rows[0]) if rows else False
                safe_other = [c for c in eligible if not (row1_has_heavy and abs(c.btn_y - rows[0][0].btn_y) < 25)]
                target = random.choice(safe_other) if safe_other else random.choice(eligible)

            # Random click point inside the button
            cx, cy = target.get_click_point()
            self._fixed_harvest_point = (cx, cy)
            if getattr(target, "kg_x", None) is not None and getattr(target, "kg_y", None) is not None:
                self._fixed_card_kg_pos = (target.kg_x, target.kg_y)
            else:
                self._fixed_card_kg_pos = None
            t_idx = getattr(target, "card_index", 1)
            self._fixed_card_index = t_idx
            self._log(f"🎯 [CỐ ĐỊNH NÚT AN TOÀN] Đã chọn ô #{t_idx} ({target.weight:.2f}kg, Y={cy}) tại ({cx}, {cy}) ở Hàng an toàn để thu hoạch liên tục...")
            self.input_sim.click(cx, cy, hold_min_ms=20, hold_max_ms=40)
            self.stats["harvested_count"] += 1

            # Active Settlement: wait for card to vanish or shift
            f_settled = self._wait_for_card_shift((cx, cy), frame)
            self._last_settled_frame = f_settled

            # Note: Do not immediately unset fixed_point here on initial selection; let next iteration confirm state
            return True, False
        else:
            self._fixed_harvest_point = None
            self.stats["skipped_heavy_count"] += len(cards)
            can_scroll = self.can_harvest_list_scroll(cards, frame)
            # If scrollbar is not at bottom, heavy cards may sit at top while smaller crops are below
            _, bottom_ratio, is_at_bottom = self.get_scrollbar_info(frame)
            if not is_at_bottom and bottom_ratio < 0.88:
                can_scroll = True
            return False, can_scroll

    def execute_fast_harvest_round(self, frame: np.ndarray) -> Tuple[bool, bool]:
        """
        Rapid harvest round without OCR for Category B.
        Locks onto a single fixed harvest button and coordinate (self._fixed_harvest_point) selected randomly once.
        Clicks the exact same point repeatedly as crops shift forward.
        If point is empty/white space: Resets fixed point, re-randomizes another button.
        Returns: (harvested_any, should_scroll)
        """
        if self.is_filter_empty(frame):
            self._fixed_harvest_point = None
            return False, False  # Empty banner: stop immediately, DO NOT SCROLL!

        cards = self.digit_reader.scan_cards(frame)
        actionable_cards = [c for c in cards if not c.is_clipped]

        # Check if we currently have a locked click point
        if self._fixed_harvest_point is not None:
            cx, cy = self._fixed_harvest_point
            under = [c for c in actionable_cards if c.btn_x <= cx <= c.btn_x + c.btn_w and c.btn_y <= cy <= c.btn_y + c.btn_h]
            if not under:
                self._log("ℹ️ [ĐỔI VỊ TRÍ TỰ DO] Vị trí ô thu hoạch cũ đã trống (khoảng trắng). Đang chọn random nút khác...")
                self._fixed_harvest_point = None
            else:
                target = under[0]
                self._log(f"⚡ [THU HOẠCH TỰ DO] Thu ô #{target.card_index} tại ({cx}, {cy})...")
                self.input_sim.click(cx, cy, hold_min_ms=25, hold_max_ms=45)
                self.stats["harvested_count"] += 1
                f_settled = self._wait_for_card_shift((cx, cy), frame)
                self._last_settled_frame = f_settled
                if not self._has_harvest_button_near(f_settled, (cx, cy)):
                    self._fixed_harvest_point = None
                return True, False

        # If _fixed_harvest_point is None: pick random actionable card
        if actionable_cards:
            target = random.choice(actionable_cards)
            cx, cy = target.get_click_point()
            self._fixed_harvest_point = (cx, cy)
            self._log(f"🎯 [CỐ ĐỊNH NÚT TỰ DO] Đã random chọn ô #{target.card_index} tại ({cx}, {cy}) để thu hoạch liên tục...")
            self.input_sim.click(cx, cy, hold_min_ms=25, hold_max_ms=45)
            self.stats["harvested_count"] += 1
            f_settled = self._wait_for_card_shift((cx, cy), frame)
            self._last_settled_frame = f_settled
            if not self._has_harvest_button_near(f_settled, (cx, cy)):
                self._fixed_harvest_point = None
            return True, False

        tpl = self._templates.get("btn_thu_hoach")
        if tpl is not None:
            fh = frame.shape[0]
            s = getattr(self, "_detected_ui_scale", None) or (fh / 1080.0)
            tw = max(10, int(tpl.shape[1] * s))
            th = max(10, int(tpl.shape[0] * s))
            scaled_btn = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_LINEAR) if abs(s - 1.0) > 0.05 else tpl
            res = cv2.matchTemplate(frame, scaled_btn, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= 0.75)
            hits = []
            for pt in zip(*locs[::-1]):
                if not any(abs(pt[0] - p[0]) < 15 and abs(pt[1] - p[1]) < 15 for p in hits):
                    hits.append(pt)
            if hits:
                target_pt = random.choice(hits)
                bx, by = target_pt[0], target_pt[1]
                rx = random.randint(bx + int(tw * 0.15), bx + int(tw * 0.85))
                ry = random.randint(by + int(th * 0.15), by + int(th * 0.85))
                self._fixed_harvest_point = (rx, ry)
                self.input_sim.click(rx, ry, hold_min_ms=25, hold_max_ms=45)
                self.stats["harvested_count"] += 1
                self._log(f"⚡ [THU HOẠCH TỰ DO] Cố định nút tại ({rx}, {ry})...")
                time.sleep(max(self.cfg_mgr.shift_delay_s, 0.20))
                return True, False

        self._fixed_harvest_point = None
        return False, True

    def get_scrollbar_info(self, frame: np.ndarray) -> Tuple[float, float, bool]:
        """
        Analyzes the vertical scrollbar on the right edge of the harvest window.
        Returns:
            thumb_height_ratio: Length of thumb relative to track (0.0 to 1.0).
            thumb_bottom_ratio: Position of the bottom edge of thumb (0.0 to 1.0).
            is_at_bottom: True if thumb is touching the bottom of track (bottom_ratio >= 0.95).
        """
        fh, fw = frame.shape[:2]
        rx0, rx1 = int(fw * (1625 / 1920)), int(fw * (1655 / 1920))
        ry0, ry1 = int(fh * (235 / 1080)), int(fh * (850 / 1080))
        if ry1 > fh or rx1 > fw:
            return 1.0, 1.0, True

        crop = frame[ry0:ry1, rx0:rx1]
        if crop.size == 0:
            return 1.0, 1.0, True

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        w_crop = crop.shape[1]
        cx0, cx1 = int(w_crop * 0.35), int(w_crop * 0.65)
        sb_center = gray[:, cx0:cx1].mean(axis=1)

        is_thumb = sb_center > 225
        thumb_idx = np.where(is_thumb)[0]
        if len(thumb_idx) == 0:
            return 1.0, 1.0, True

        t_bottom = thumb_idx.max() / float(len(sb_center))
        thumb_height_ratio = (thumb_idx.max() - thumb_idx.min()) / float(len(sb_center))
        is_at_bottom = bool(t_bottom >= 0.95)
        return thumb_height_ratio, t_bottom, is_at_bottom

    def scroll_harvest_list_down(self, frame: np.ndarray, dy: Optional[int] = None) -> bool:
        """
        Performs a verified scroll-down gesture on the harvest table.
        - If dy is provided (boundary alignment): Uses calibrated drag with anti-overshoot.
        - If dy is None (probing/near-end): Uses mouse wheel rolling (2-3 notches) with
          randomized cursor positioning in the neutral empty/card body area (NO mouse button down).
        Captures frame before and after to verify whether list content actually shifted.

        Returns:
            True if the list moved (weights or card positions shifted).
            False if the list is stationary / reached the absolute bottom.
        """
        fh, fw = frame.shape[:2]
        cards_before = self.digit_reader.scan_cards(frame)
        sig_before = [(round(c.weight, 2), getattr(c, "kg_y", 0)) for c in cards_before]

        if dy is not None and dy > 0:
            # Boundary alignment drag with anti-overshoot hold
            drag_x = int(fw * (400 / 1920))
            y0 = int(fh * (680 / 1080))
            y1 = max(int(fh * 0.15), y0 - dy)
            self.input_sim.drag(drag_x, y0, drag_x, y1, duration_s=0.35, steps=20)
        else:
            # Mouse Wheel Scroll (2-3 notches ~ 1.5 rows) with random neutral position
            # Strictly NO mouse button click/hold!
            rx = int(fw * random.uniform(0.18, 0.28))
            ry = int(fh * random.uniform(0.40, 0.65))
            self.input_sim.move_to(rx, ry, human_curve=False)
            time.sleep(0.04)
            notches = random.randint(2, 3)
            delta = -120 * notches
            self.input_sim.scroll(delta=delta)

        # Allow Unity physics / scroll view to settle
        time.sleep(0.25)

        # Capture new frame and compare card signatures
        cap_res = self.capturer.capture(None) if hasattr(self, "capturer") and self.capturer else None
        new_frame = cap_res[0] if (isinstance(cap_res, tuple) and len(cap_res) > 0 and cap_res[0] is not None) else frame
        cards_after = self.digit_reader.scan_cards(new_frame)
        sig_after = [(round(c.weight, 2), getattr(c, "kg_y", 0)) for c in cards_after]

        if not sig_before and not sig_after:
            return False

        return sig_before != sig_after

    def count_sellable_crops_in_shop(self, frame: np.ndarray) -> int:
        """
        Scans the 4x2 grid of crop cards in Oliver's shop window.
        Counts how many cards are visible AND unlocked (not locked with a padlock).
        Returns 0 if the grid is empty or contains only locked crops.
        """
        fh, fw = frame.shape[:2]
        rx = int(fw * (780 / 1920))
        ry = int(fh * (240 / 1080))
        rw = int(fw * (860 / 1920))
        rh = int(fh * (520 / 1080))
        if ry + rh > fh or rx + rw > fw or rw <= 0 or rh <= 0:
            return 0

        grid = frame[ry:ry+rh, rx:rx+rw]
        gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
        mask = (gray > 240).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_w = int(100 * (fw / 1920))
        min_h = int(120 * (fh / 1080))

        sellable = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > min_w and h > min_h:
                # Check top-left corner of the card for padlock icon
                lock_w = int(50 * (fw / 1920))
                lock_h = int(50 * (fh / 1080))
                lock_roi = gray[y:min(y + lock_h, grid.shape[0]), x:min(x + lock_w, grid.shape[1])]
                dark_pixels = (lock_roi < 175).sum()
                area = max(1, lock_roi.shape[0] * lock_roi.shape[1])
                is_locked = (dark_pixels / float(area)) > 0.008
                if not is_locked:
                    sellable += 1

        return sellable

    def step_sell_cycle(self) -> bool:
        """
        Executes verified closed-loop multi-batch auto-sell at NPC Oliver:
          1. Close harvest window (verifies disappearance using btn_close_x / ESC).
          2. Talk to Oliver / Open shop (via dialogue, basket icon, or HUD Sell button).
          3. Multi-batch sell loop: Repeatedly selects (0/100) and sells all batches
             until no more crops remain in bag.
          4. Robust verification: Polls for OK confirmation (threshold 0.62, up to 3s).
          5. Closes shop cleanly using close button. Zero random spacebar keypresses.
        """
        self._fixed_harvest_point = None  # Reset harvest coordinate on selling
        self._log("💰 [BÁN NÔNG SẢN] Bắt đầu quy trình tự động bán nông sản tại NPC Oliver...")

        # Phase 1: Verify and ensure harvest window is closed
        harvest_closed = False
        for attempt in range(4):
            f_cur, _ = self.capturer.capture(None)
            is_harvest_open = (
                self.find_template(f_cur, "btn_loc", threshold=0.68) is not None or
                self.find_template(f_cur, "btn_thu_hoach", threshold=0.68) is not None or
                self.is_filter_empty(f_cur)
            )
            if not is_harvest_open:
                harvest_closed = True
                break
            close_btn = self.find_close_button(f_cur)
            if close_btn:
                self._log(f" -> Đang đóng bảng thu hoạch (bấm dấu [X], lần {attempt+1}/4)...")
                self.click_rect(close_btn)
                time.sleep(0.4)
                if attempt > 0:
                    self.input_sim.press_key("esc")
                time.sleep(0.6)
            else:
                self.input_sim.press_key("esc")
                time.sleep(0.8)

        if not harvest_closed:
            f_check, _ = self.capturer.capture(None)
            if (self.find_template(f_check, "btn_loc", threshold=0.68) is not None or
                self.find_template(f_check, "btn_thu_hoach", threshold=0.68) is not None):
                self._log("⚠️ [CHƯA ĐÓNG ĐƯỢC BẢNG] Bảng thu hoạch vẫn đang mở! Bấm ESC giải phóng...")
                self.input_sim.press_key("esc")
                time.sleep(0.8)
                f_check2, _ = self.capturer.capture(None)
                if not (self.find_template(f_check2, "btn_loc", threshold=0.68) is not None or
                        self.find_template(f_check2, "btn_thu_hoach", threshold=0.68) is not None):
                    harvest_closed = True
                else:
                    return False

        # Phase 2: Open Oliver shop if not yet open (A1 & A2 with polling & retry)
        f_shop, _ = self.capturer.capture(None)
        in_shop = self.is_shop_open(f_shop)
        if not in_shop:
            btn_crops = self.find_oliver_sell_option(f_shop)
            if btn_crops:
                self._log(" -> Thấy nút 'Bán nông sản'! Bấm mở cửa hàng Oliver...")
                self.click_rect(btn_crops)
                # Poll for shop modal to open (up to 3.0s)
                for _ in range(10):
                    time.sleep(0.3)
                    f_chk, _ = self.capturer.capture(None)
                    if self.is_shop_open(f_chk):
                        in_shop = True
                        break
            else:
                oliver_basket = self.find_template(f_shop, "oliver_basket", threshold=0.65)
                if oliver_basket:
                    self._log(" -> Bấm icon giỏ hàng trên đầu ông Oliver để vào bán...")
                    self.click_rect(oliver_basket)
                    # A2: Poll for dialogue options with timeout (up to 2.5s)
                    btn_crops = None
                    for _ in range(8):
                        time.sleep(0.3)
                        f_dlg, _ = self.capturer.capture(None)
                        if self.is_shop_open(f_dlg):
                            in_shop = True
                            break
                        btn_crops = self.find_oliver_sell_option(f_dlg)
                        if btn_crops:
                            break

                    if btn_crops:
                        self._log(" -> Thấy nút 'Bán nông sản'! Bấm mở cửa hàng Oliver...")
                        self.click_rect(btn_crops)
                        # Poll for shop to open (up to 2.5s)
                        for _ in range(8):
                            time.sleep(0.3)
                            f_chk, _ = self.capturer.capture(None)
                            if self.is_shop_open(f_chk):
                                in_shop = True
                                break
                    elif not in_shop:
                        # Retry clicking basket once if lag
                        self._log(" -> Chưa thấy menu thoại, thử bấm lại icon giỏ hàng lần 2...")
                        f_retry, _ = self.capturer.capture(None)
                        basket_retry = self.find_template(f_retry, "oliver_basket", threshold=0.65)
                        if basket_retry:
                            self.click_rect(basket_retry)
                            for _ in range(8):
                                time.sleep(0.3)
                                f_dlg2, _ = self.capturer.capture(None)
                                if self.is_shop_open(f_dlg2):
                                    in_shop = True
                                    break
                                btn_crops2 = self.find_oliver_sell_option(f_dlg2)
                                if btn_crops2:
                                    self._log(" -> Thấy nút 'Bán nông sản' (lần 2)! Bấm mở cửa hàng Oliver...")
                                    self.click_rect(btn_crops2)
                                    time.sleep(1.0)
                                    break
                else:
                    hit_b1 = self.find_hud_sell_button(f_shop)
                    if hit_b1:
                        self._log(" -> Bấm nút 'Bán' trên HUD để mở lại cửa hàng Oliver...")
                        self.click_hud_sell_button(hit_b1)
                        time.sleep(1.5)

        # Phase 3-6: Multi-batch Sell Loop (sells all full batches of 100)
        total_batches_sold = 0
        for batch_idx in range(1, 6): # Up to 5 consecutive batches (500 items max)
            f_cur, _ = self.capturer.capture(None)
            if not self.is_shop_open(f_cur):
                break

            # Check if coin button is ALREADY active (crops already selected)
            coin_btn = self.find_shop_coin_button(f_cur)
            if not coin_btn:
                # Crops not selected yet: Click '(0/100) Chọn tự động'
                auto_btn = self.find_shop_auto_select_button(f_cur)
                if not auto_btn:
                    break

                sellable_count = self.count_sellable_crops_in_shop(f_cur)
                if sellable_count > 0:
                    self._log(f" -> Đợt {batch_idx}: Thấy {sellable_count} ô nông sản có thể bán! Bấm '(0/100) Chọn tự động'...")
                else:
                    self._log(f" -> Đợt {batch_idx}: Bấm '(0/100) Chọn tự động' để chọn nông sản...")
                self.click_rect(auto_btn)
                time.sleep(0.6)

                f_coin, _ = self.capturer.capture(None)
                coin_btn = self.find_shop_coin_button(f_coin)

            if not coin_btn:
                if total_batches_sold > 0:
                    self._log(f"✨ [BÁN XONG] Lưới cửa hàng đã sạch! Túi đồ đã bán hết.")
                else:
                    self._log(f"ℹ️ [CỬA HÀNG] Không có nông sản nào được chọn để bán (hoặc nông sản đã bị khóa)! Đóng cửa hàng...")
                break

            # Click coin button to proceed to sell
            self._log(f" -> Đợt {batch_idx}: Bấm nút 'Bán'...")
            self.click_rect(coin_btn)
            time.sleep(0.8)

            # Confirm popup 1 (B5) with polling (up to 2.0s)
            conf_btn = None
            for _ in range(5):
                f_conf, _ = self.capturer.capture(None)
                conf_btn = self.find_sell_confirm_button(f_conf)
                if conf_btn:
                    self._log(f" -> Đợt {batch_idx}: [B5] Bấm xác nhận bán hàng...")
                    self.click_rect(conf_btn)
                    time.sleep(0.8)
                    break
                time.sleep(0.3)

            # Check for High-Tier Warning popup 2 (B6) with polling (up to 2.0s)
            # "Có bao gồm vật phẩm cấp cao. Bạn vẫn muốn bán?" -> Click [Xác nhận]
            high_tier_btn = None
            for _ in range(6):
                f_ht, _ = self.capturer.capture(None)
                high_tier_btn = self.find_high_tier_confirm_button(f_ht)
                if high_tier_btn:
                    self._log(f" -> Đợt {batch_idx}: [B6] Phát hiện cảnh báo 'Vật phẩm cấp cao'! Bấm 'Xác nhận'...")
                    self.click_rect(high_tier_btn)
                    time.sleep(0.8)
                    break
                if (self.find_template(f_ht, "sell_ok_btn", threshold=0.62) or
                    self.find_template(f_ht, "prompt_space_cyan", threshold=0.75) or
                    self.is_shop_open(f_ht)):
                    break
                time.sleep(0.3)

            # Poll for OK popup 3 (B7) or immediate return to shop grid
            ok_clicked = False
            for poll in range(8):
                time.sleep(0.3)
                f_ok, _ = self.capturer.capture(None)
                # Check for Space prompt (B7 button [(␣) OK] or space prompt icon)
                if (self.find_template(f_ok, "sell_ok_btn", threshold=0.62) is not None or
                    self.find_template(f_ok, "prompt_space_cyan", threshold=0.78) is not None):
                    self._log(f" -> Đợt {batch_idx}: [B7] Thấy ký hiệu [␣] trên nút OK, bấm phím Space hoàn tất bán hàng...")
                    self.input_sim.press_key("space")
                    ok_clicked = True
                    time.sleep(0.8)
                    break
                # If already back in shop grid, transaction finished directly
                if self.is_shop_open(f_ok):
                    ok_clicked = True
                    break

            total_batches_sold += 1

        # Phase 7: Close shop modal cleanly
        for close_attempt in range(4):
            f_after, _ = self.capturer.capture(None)
            if not self.is_shop_open(f_after):
                break
            close_shop = self.find_close_button(f_after)
            if close_shop:
                self._log(f" -> Đóng cửa hàng Oliver (bấm [X], lần {close_attempt+1}/4)...")
                self.click_rect(close_shop)
                time.sleep(0.6)
                if close_attempt > 0:
                    self.input_sim.press_key("esc")
                    time.sleep(0.4)
            else:
                self.input_sim.press_key("esc")
                time.sleep(0.6)

        # Phase 8: Dismiss Oliver farewell dialogue if active (speech bubble blocking HUD)
        for _ in range(4):
            f_dlg, _ = self.capturer.capture(None)
            if self.find_hud_harvest_button(f_dlg) is not None or self.find_oliver_basket(f_dlg) is not None:
                break  # HUD already visible and clean
            if self.handle_space_prompt_if_present(f_dlg):
                time.sleep(0.5)
                continue
            if self.find_oliver_sell_option(f_dlg) is not None:
                self._log(" -> Đóng menu lựa chọn của Oliver (bấm ESC)...")
                self.input_sim.press_key("esc")
            else:
                self._log(" -> Đóng lời thoại ông Oliver (bấm ESC)...")
                self.input_sim.press_key("esc")
            time.sleep(0.6)

        # Zero Space keypresses: Character will NEVER jump in 3D world!

        if total_batches_sold > 0:
            self.stats["sell_cycles"] += total_batches_sold
            self._bag_is_full = False
            self._log(f"✨ [BÁN THÀNH CÔNG] Đã bán {total_batches_sold} đợt nông sản! Túi đồ đã trống, quay lại vườn thu hoạch tiếp...")
            return True
        else:
            self._log("⚠️ [CHƯA BÁN ĐƯỢC] Không chọn được món nào để bán! Vẫn giữ trạng thái túi đầy...")
            return False

    # =========================================================================
    # Main Engine Execution Loop
    # =========================================================================

    def start(self) -> None:
        """Starts the Farm harvesting loop in a background thread."""
        if self.state not in (FarmState.STOPPED, FarmState.COMPLETED):
            return

        if not getattr(self, "_templates_valid", True) or len(self._templates) == 0:
            self._log(f"❌ [KHỞI ĐỘNG THẤT BẠI] Thư viện ảnh mẫu nông trại rỗng hoặc thiếu file cốt lõi tại: {self.assets_dir}")
            self._log("💡 [KHẮC PHỤC] Hãy đảm bảo thư mục 'assets/farm' tồn tại và đầy đủ ảnh mẫu cạnh file chạy.")
            self._set_state(FarmState.STOPPED)
            return

        enabled_rules = self.cfg_mgr.get_enabled_rules()
        if not enabled_rules:
            self._log("⚠️ [LỖI] Chưa có loại quả nào được tích chọn trong cấu hình Nông trại!")
            return

        self.stats["harvested_count"] = 0
        self.stats["sell_cycles"] = 0
        self.stats["skipped_heavy_count"] = 0
        self._teleported_to_oliver = False
        self._bag_is_full = False

        self._stop_event = False
        self._pause_event = False
        self._set_state(FarmState.TELEPORTING)

        import threading
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="FarmEngineThread")
        self._thread.start()
        self._log(f"🚀 [NÔNG TRẠI] Bắt đầu chu trình tự động thu hoạch cho {len(enabled_rules)} loại quả đã chọn...")

    @property
    def _stop_requested(self) -> bool:
        """Compatibility alias for self._stop_event."""
        return self._stop_event

    def stop(self) -> None:
        """Stops the Farm engine cleanly."""
        self._stop_event = True
        self._teleported_to_oliver = False
        self._set_state(FarmState.STOPPED)
        self._log("⏹ [NÔNG TRẠI] Đã dừng bot Nông trại.")

    def pause(self) -> None:
        self._pause_event = True
        self._set_state(FarmState.PAUSED)
        self._log("⏸ [NÔNG TRẠI] Đã tạm dừng.")

    def resume(self) -> None:
        self._pause_event = False
        self._set_state(FarmState.HARVESTING)
        self._log("▶ [NÔNG TRẠI] Tiếp tục chạy.")

    def _harvest_single_fruit(self, fruit_name: str, max_kg: float) -> None:
        """
        Vision-driven harvest loop for Category A (< max_kg):
        Fully state-aware and idempotent:
          1. Bag Full Toast / Persistent State -> Close harvest modal & execute Sell Cycle.
          2. Filter Modal open:
             - 'Ngoại trừ trái đột biến': Checks if ALREADY checked [✓]; only clicks if unchecked!
             - Tab 'Nông sản': Checks if target fruit is ALREADY checked [✓]; only clicks if unchecked!
             - Clicks 'Cài đặt bộ lọc' to apply.
          3. Harvest Window open:
             - Resumes mid-harvest if cards already match fruit_name without re-filtering!
             - If not filtered yet: Click 'Lọc' (B3).
             - If filtered: Check empty / Harvest eligible cards (< max_kg) / Scroll down.
          4. Main Screen HUD:
             - First teleport: Click 'Bán' (B1) to move to Oliver.
             - Ready to open: Click 'Thu hoạch' (B2) to open harvest window.
        """
        filter_done = False
        stuck_scrolls = 0
        total_scrolls = 0
        last_wait_log = 0.0
        step1_start_time = time.time()
        step2_start_time = None
        self._fixed_harvest_point = None

        while not self._stop_event:
            if self._pause_event:
                time.sleep(0.2)
                continue

            try:
                frame, _ = self.capturer.capture(None)

                # 0. Global safety handlers (Quit dialogs, Space prompts)
                if self.dismiss_quit_game_dialog_if_present(frame):
                    time.sleep(0.4)
                    continue

                if self.handle_space_prompt_if_present(frame):
                    time.sleep(0.4)
                    continue

                # 1. PRIORITY 1: Oliver shop modal still open (Ưu tiên màn hình hiện gì xử lý đó)
                if self.is_shop_open(frame):
                    coin_btn = self.find_shop_coin_button(frame)
                    if coin_btn and getattr(self, "_bag_is_full", False):
                        self._log("💰 [ƯU TIÊN MÀN HÌNH] Cửa hàng đang mở và còn nông sản cần bán! Bán tiếp...")
                        self.step_sell_cycle()
                    else:
                        self._log("🚪 [ƯU TIÊN MÀN HÌNH] Cửa hàng vẫn đang mở nhưng đã bán xong/cần thu hoạch! Đóng cửa hàng...")
                        close_btn = self.find_close_button(frame)
                        if close_btn:
                            self.click_rect(close_btn)
                        else:
                            self.input_sim.press_key("esc")
                        time.sleep(0.6)
                    continue

                # 2. PRIORITY 2: Oliver dialogue / selection menu blocking HUD
                btn_crops_opt = self.find_oliver_sell_option(frame)
                if btn_crops_opt is not None:
                    if getattr(self, "_bag_is_full", False):
                        self._log("💰 [ƯU TIÊN MÀN HÌNH] Thấy bảng lựa chọn của Oliver và túi đầy! Bấm 'Bán nông sản'...")
                        self.click_rect(btn_crops_opt)
                        time.sleep(1.0)
                    else:
                        self._log("🚪 [ƯU TIÊN MÀN HÌNH] Thấy menu đối thoại của Oliver nhưng đang cần thu hoạch! Bấm ESC đóng thoại...")
                        self.input_sim.press_key("esc")
                        time.sleep(0.6)
                    continue

                # 3. PRIORITY 3: Detect Bag Full Toast or persistent Bag Full state (A3)
                if self.is_bag_full(frame) or getattr(self, "_bag_is_full", False):
                    self._bag_is_full = True
                    self._set_state(FarmState.SELLING)
                    self._log("🛑 [ĐẦY TÚI] Túi đồ đang đầy! Chuyển sang quy trình bán tại NPC Oliver...")
                    
                    garden_has_more = True
                    if self.is_harvest_window_open(frame):
                        if self.is_filter_empty(frame):
                            garden_has_more = False
                        else:
                            cards = self.digit_reader.scan_cards(frame)
                            visible_eligible = [c for c in cards if 0.0 < getattr(c, "weight", 0.0) < max_kg and not getattr(c, "is_clipped", False)]
                            _, bottom_ratio, is_at_bottom = self.get_scrollbar_info(frame)
                            if not visible_eligible and (is_at_bottom or bottom_ratio >= 0.95):
                                garden_has_more = False
                        self.close_harvest_window()

                    sold = self.step_sell_cycle()
                    if sold:
                        self._bag_is_full = False
                        if not garden_has_more:
                            self._log(f"🏁 [ĐÃ HẾT QUẢ THU] Trước khi xả túi, vườn đã hết sạch quả < {max_kg:.2f}kg! Hoàn tất.")
                            break
                        self._log("🔄 [QUAY LẠI VƯỜN] Đã xả trống túi đồ và vườn vẫn còn quả! Mở lại bảng thu hoạch để tiếp tục thu...")
                        filter_done = False  # Reset filter to re-apply for remaining fruits
                        stuck_scrolls = 0
                        total_scrolls = 0
                        self._fixed_harvest_point = None
                        step1_start_time = None
                        step2_start_time = None
                    time.sleep(1.0)
                    continue

                # 2. Detect Filter Modal
                in_filter_modal = self.is_filter_modal_open(frame)
                if in_filter_modal:
                    self._set_state(FarmState.APPLYING_FILTER)
                    if self.step_setup_filter(frame, fruit_name):
                        filter_done = True
                    time.sleep(0.5)
                    continue

                # 3. Detect Harvest Window (B3, Cards, or Empty Banner)
                in_harvest_window = self.is_harvest_window_open(frame)
                if in_harvest_window:
                    if not filter_done:
                        self._set_state(FarmState.APPLYING_FILTER)
                        btn_loc = self.find_filter_button(frame)
                        if btn_loc:
                            self._log(f"🔍 [BẢNG VƯỜN] Đang mở bảng thu hoạch! Bấm nút 'Lọc' để cấu hình quả '{fruit_name}'...")
                            self.click_rect(btn_loc)
                            time.sleep(0.8)
                        else:
                            now = time.time()
                            if now - last_wait_log > 3.0:
                                last_wait_log = now
                                self._log(f"⏳ [BẢNG VƯỜN] Đang tìm nút 'Lọc' trên bảng nông trại...")
                        continue

                    self._set_state(FarmState.HARVESTING)
                    if self.is_filter_empty(frame):
                        self._log(f"🏁 [HẾT QUẢ] Đã thu hết toàn bộ trái phù hợp cho quả '{fruit_name}'!")
                        break

                    # RAPID HARVEST FAST-PATH (< 0.25s per fruit):
                    # Stays on the harvest window clicking the locked slot as crops shift forward.
                    # Bypasses 2.4s of redundant full-screen modal checks per fruit!
                    rapid_active = True
                    while rapid_active and not self._stop_event:
                        # Safety Guard A: Check quit confirmation dialog
                        if self.dismiss_quit_game_dialog_if_present(frame):
                            self._fixed_harvest_point = None
                            self._fixed_card_kg_pos = None
                            break

                        # Safety Guard B: Check notice popup (#102509) -> Press Space, ZERO mouse clicks!
                        if self.handle_space_prompt_if_present(frame):
                            self._fixed_harvest_point = None
                            self._fixed_card_kg_pos = None
                            time.sleep(0.3)
                            frame, _ = self.capturer.capture(None)
                            continue

                        # Safety Guard C: If harvest window was closed, exit rapid loop immediately to recover
                        if not self.is_harvest_window_open(frame):
                            self._fixed_harvest_point = None
                            self._fixed_card_kg_pos = None
                            break

                        # Quick bag full check
                        if self.is_bag_full(frame):
                            self._bag_is_full = True
                            self._log("🛑 [ĐẦY TÚI] Phát hiện thông báo đầy túi trong lúc thu hoạch!")
                            break

                        if self.is_filter_empty(frame):
                            self._log(f"🏁 [HẾT QUẢ] Đã thu hết toàn bộ trái phù hợp cho quả '{fruit_name}'!")
                            break

                        harvested, should_scroll = self.execute_harvest_round(frame, max_kg)
                        if harvested:
                            stuck_scrolls = 0
                            total_scrolls = 0
                            if getattr(self, "_last_settled_frame", None) is not None:
                                frame = self._last_settled_frame
                                self._last_settled_frame = None
                            else:
                                frame, _ = self.capturer.capture(None)
                            continue
                        elif self._fixed_harvest_point is not None:
                            time.sleep(0.08)
                            frame, _ = self.capturer.capture(None)
                            continue
                        else:
                            # Not harvested and no locked point: exit rapid loop to handle scrolling or recovery
                            rapid_active = False

                    if self._bag_is_full or self.is_filter_empty(frame):
                        continue

                    # AUTOMATIC RECOVERY: If harvest window was accidentally closed mid-harvest!
                    if not self.is_harvest_window_open(frame):
                        self._log(f"🔄 [TỰ ĐỘNG KHÔI PHỤC] Đang thu dở quả '{fruit_name}' nhưng bảng thu hoạch bị đóng ngoài ý muốn! Mở lại bảng để tiếp tục...")
                        filter_done = False
                        self._fixed_harvest_point = None
                        self._fixed_card_kg_pos = None
                        continue

                    # If not harvested and no more cards at current viewport
                    thumb_ratio, bottom_ratio, is_at_bottom = self.get_scrollbar_info(frame)
                    if not should_scroll:
                        # Sweep-back check: if we scrolled boundary earlier, scroll to top to sweep any crops near top
                        if getattr(self, "_did_scroll_boundary", False):
                            self._did_scroll_boundary = False
                            self._log("🔍 [VÉT QUẢ ĐỈNH DANH SÁCH] Kiểm tra lại phần trên danh sách để thu nốt quả nhỏ còn sót...")
                            self.scroll_harvest_list_up(frame)
                            time.sleep(0.35)
                            f_top, _ = self.capturer.capture(None)
                            cards_top = self.digit_reader.scan_cards(f_top)
                            remaining_top = [c for c in cards_top if 0.0 < c.weight < max_kg and not c.is_clipped]
                            if remaining_top:
                                self._log(f"🌾 [CÒN SÓT {len(remaining_top)} QUẢ] Phát hiện {len(remaining_top)} quả hợp lệ ở phần trên! Tiếp tục thu hoạch...")
                                frame = f_top
                                continue

                        self._log(f"🏁 [HẾT QUẢ THU] Tất cả quả hiển thị đều >= {max_kg:.2f}kg và danh sách không thể cuộn thêm. Dừng ngay, 0 lần kéo chuột thừa!")
                        break
                    else:
                        cards = self.digit_reader.scan_cards(frame)
                        if not self.can_harvest_list_scroll(cards, frame):
                            self._log(f"🏁 [HẾT QUẢ THU] Bảng không đủ quả để cuộn (chỉ có {len(cards)} thẻ). Kết thúc quả '{fruit_name}'!")
                            break

                        total_scrolls += 1
                        if total_scrolls > 40:
                            self._log(f"🏁 [XONG QUẢ] Đã đạt giới hạn quét tối đa (40 lần cuộn)! Kết thúc quả '{fruit_name}'.")
                            break

                        self._log(f"📜 [CUỘN TRANG] Đang cuộn nhẹ xuống tìm quả nhỏ hơn... (Thanh cuộn: {bottom_ratio*100:.0f}%, Lần {total_scrolls})")
                        scrolled = self.scroll_harvest_list_down(frame)
                        self._fixed_harvest_point = None
                        self._fixed_card_kg_pos = None
                        if not scrolled or is_at_bottom:
                            stuck_scrolls += 1
                            if stuck_scrolls >= 2:
                                # FAIL-SAFE BOTTOM EXIT VERIFICATION:
                                time.sleep(0.4)
                                f_last, _ = self.capturer.capture(None)
                                last_cards = self.digit_reader.scan_cards(f_last)
                                fh_last = f_last.shape[0]
                                remaining = [
                                    c for c in last_cards
                                    if 0.0 < c.weight < max_kg and (c.btn_y + 40 <= int(fh_last * 0.785))
                                ]
                                if remaining:
                                    self._log(f"🌾 [ĐÁY BẢNG NHƯNG CÒN QUẢ] Còn {len(remaining)} quả hợp lệ (< {max_kg:.2f}kg) ở đáy danh sách! Thu hoạch nốt trước khi kết thúc...")
                                    stuck_scrolls = 0
                                    continue
                                self._log(f"🏁 [ĐÁY DANH SÁCH] Đã cuộn đến đáy bảng nông sản (thanh cuộn chạm đáy)! Tất cả các quả còn lại đều >= {max_kg:.2f}kg.")
                                break
                        else:
                            stuck_scrolls = 0
                    continue

                # 4. Detect Main Screen HUD (Nút Bán / Nút Thu hoạch)
                now = time.time()
                if step1_start_time is None:
                    step1_start_time = now
                elif now - step1_start_time > getattr(self, "step1_timeout_s", 25.0):
                    self._log("❌ [QUÁ THỜI GIAN] Sau 20s không tìm thấy nút 'Bán' hoặc ông Oliver trên màn hình!")
                    self._log("💡 [HƯỚNG DẪN KHẮC PHỤC] Hãy kiểm tra: 1) Game có đang mở ở màn hình chính hay không. 2) Đảm bảo độ phân giải 1080p và không bị che khuất. 3) Đảm bảo ngôn ngữ game là Tiếng Việt.")
                    self.stop()
                    break

                opened = self.step_open_harvest_window(frame)
                if opened:
                    step1_start_time = None
                else:
                    time.sleep(0.3)
                continue

            except Exception as e:
                self._log(f"⚠️ [LỖI NGOẠI LỆ] {e}")
                time.sleep(1.0)

    def _harvest_multi_fruits_fast(self, fruit_names: List[str]) -> None:
        """
        Vision-driven harvest loop for Category B (multi fruits, no OCR):
        Fully state-aware and idempotent across all configured fruits.
        """
        filter_done = False
        stuck_scrolls = 0
        total_scrolls = 0
        last_wait_log = 0.0
        self._fixed_harvest_point = None
        step1_start_time = time.time()
        step2_start_time = None

        while not self._stop_event:
            if self._pause_event:
                time.sleep(0.2)
                continue

            try:
                frame, _ = self.capturer.capture(None)

                # 0. Global sudden space prompt handler
                if self.handle_space_prompt_if_present(frame):
                    time.sleep(0.5)
                    continue

                # 1. PRIORITY 1: Oliver shop modal still open (Ưu tiên màn hình hiện gì xử lý đó)
                if self.is_shop_open(frame):
                    coin_btn = self.find_shop_coin_button(frame)
                    if coin_btn and getattr(self, "_bag_is_full", False):
                        self._log("💰 [ƯU TIÊN MÀN HÌNH] Cửa hàng đang mở và còn nông sản cần bán! Bán tiếp...")
                        self.step_sell_cycle()
                    else:
                        self._log("🚪 [ƯU TIÊN MÀN HÌNH] Cửa hàng vẫn đang mở nhưng đã bán xong/cần thu hoạch! Đóng cửa hàng...")
                        close_btn = self.find_close_button(frame)
                        if close_btn:
                            self.click_rect(close_btn)
                        else:
                            self.input_sim.press_key("esc")
                        time.sleep(0.6)
                    continue

                # 2. PRIORITY 2: Oliver dialogue / selection menu blocking HUD
                btn_crops_opt = self.find_oliver_sell_option(frame)
                if btn_crops_opt is not None:
                    if getattr(self, "_bag_is_full", False):
                        self._log("💰 [ƯU TIÊN MÀN HÌNH] Thấy bảng lựa chọn của Oliver và túi đầy! Bấm 'Bán nông sản'...")
                        self.click_rect(btn_crops_opt)
                        time.sleep(1.0)
                    else:
                        self._log("🚪 [ƯU TIÊN MÀN HÌNH] Thấy menu đối thoại của Oliver nhưng đang cần thu hoạch! Bấm ESC đóng thoại...")
                        self.input_sim.press_key("esc")
                        time.sleep(0.6)
                    continue

                # 3. PRIORITY 3: Bag Full Toast or persistent Bag Full state (A3)
                if self.is_bag_full(frame) or getattr(self, "_bag_is_full", False):
                    self._bag_is_full = True
                    self._set_state(FarmState.SELLING)
                    self._log("🛑 [ĐẦY TÚI] Túi đồ đang đầy! Chuyển sang quy trình bán tại NPC Oliver...")
                    
                    garden_has_more = True
                    if self.is_harvest_window_open(frame):
                        if self.is_filter_empty(frame):
                            garden_has_more = False
                        else:
                            cards = self.digit_reader.scan_cards(frame)
                            visible_eligible = [c for c in cards if 0.0 < getattr(c, "weight", 0.0) < max_kg and not getattr(c, "is_clipped", False)]
                            _, bottom_ratio, is_at_bottom = self.get_scrollbar_info(frame)
                            if not visible_eligible and (is_at_bottom or bottom_ratio >= 0.95):
                                garden_has_more = False
                        self.close_harvest_window()

                    sold = self.step_sell_cycle()
                    if sold:
                        self._bag_is_full = False
                        if not garden_has_more:
                            self._log(f"🏁 [ĐÃ HẾT QUẢ THU] Trước khi xả túi, vườn đã hết sạch quả < {max_kg:.2f}kg! Hoàn tất loại quả '{fruit_name}'.")
                            break
                        self._log(f"🔄 [QUAY LẠI VƯỜN] Đã xả trống túi đồ và vườn vẫn còn quả! Mở lại bảng thu hoạch để tiếp tục thu...")
                        filter_done = False  # Reset filter to re-apply
                        stuck_scrolls = 0
                        total_scrolls = 0
                        self._fixed_harvest_point = None
                        step1_start_time = None
                        step2_start_time = None
                    time.sleep(1.0)
                    continue

                # 2. Filter Modal
                in_filter_modal = self.is_filter_modal_open(frame)
                if in_filter_modal:
                    self._set_state(FarmState.APPLYING_FILTER)
                    if self.step_setup_multi_filter(frame, fruit_names):
                        filter_done = True
                    time.sleep(0.5)
                    continue

                # 3. Harvest Window
                in_harvest_window = self.is_harvest_window_open(frame)
                if in_harvest_window:
                    if not filter_done:
                        self._set_state(FarmState.APPLYING_FILTER)
                        btn_loc = self.find_filter_button(frame)
                        if btn_loc:
                            self._log(f"🔍 [BẢNG VƯỜN] Đang mở bảng thu hoạch! Bấm nút 'Lọc' để cấu hình {len(fruit_names)} loại quả...")
                            self.click_rect(btn_loc)
                            time.sleep(0.8)
                        else:
                            now = time.time()
                            if now - last_wait_log > 3.0:
                                last_wait_log = now
                                self._log("⏳ [BẢNG VƯỜN] Đang tìm nút 'Lọc'...")
                        continue

                    self._set_state(FarmState.HARVESTING)
                    if self.is_filter_empty(frame):
                        self._log("🏁 [XONG NHÓM 2] Phát hiện thông báo: 'Không có trái nào phù hợp với bộ lọc đã chọn.'!")
                        break

                    # RAPID HARVEST FAST-PATH FOR MULTI-FRUITS (< 0.25s per fruit):
                    rapid_multi = True
                    while rapid_multi and not self._stop_event:
                        # Safety Guard A: Check quit confirmation dialog
                        if self.dismiss_quit_game_dialog_if_present(frame):
                            self._fixed_harvest_point = None
                            break

                        # Safety Guard B: Check notice popup (#102509) -> Press Space, ZERO mouse clicks!
                        if self.handle_space_prompt_if_present(frame):
                            self._fixed_harvest_point = None
                            time.sleep(0.3)
                            frame, _ = self.capturer.capture(None)
                            continue

                        # Safety Guard C: If harvest window was closed, exit rapid loop immediately to recover
                        if not self.is_harvest_window_open(frame):
                            self._fixed_harvest_point = None
                            break

                        if self.is_bag_full(frame):
                            self._bag_is_full = True
                            break
                        if self.is_filter_empty(frame):
                            break
                        harvested, should_scroll = self.execute_fast_harvest_round(frame)
                        if harvested:
                            stuck_scrolls = 0
                            total_scrolls = 0
                            if getattr(self, "_last_settled_frame", None) is not None:
                                frame = self._last_settled_frame
                                self._last_settled_frame = None
                            else:
                                frame, _ = self.capturer.capture(None)
                            continue
                        elif self._fixed_harvest_point is not None:
                            time.sleep(0.08)
                            frame, _ = self.capturer.capture(None)
                            continue
                        else:
                            rapid_multi = False

                    if self._bag_is_full or self.is_filter_empty(frame):
                        continue

                    # AUTOMATIC RECOVERY: If harvest window was accidentally closed mid-harvest!
                    if not self.is_harvest_window_open(frame):
                        self._log("🔄 [TỰ ĐỘNG KHÔI PHỤC] Đang thu dở nhóm quả tự do nhưng bảng thu hoạch bị đóng ngoài ý muốn! Mở lại bảng để tiếp tục...")
                        filter_done = False
                        self._fixed_harvest_point = None
                        continue

                    thumb_ratio, bottom_ratio, is_at_bottom = self.get_scrollbar_info(frame)
                    if not should_scroll:
                        self._log("🏁 [XONG NHÓM 2] Danh sách không thể cuộn thêm! Hoàn tất nhóm thu tự do.")
                        break
                    else:
                        cards = self.digit_reader.scan_cards(frame)
                        if not self.can_harvest_list_scroll(cards, frame):
                            self._log(f"🏁 [XONG NHÓM 2] Bảng không đủ quả để cuộn (chỉ có {len(cards)} thẻ). Dừng lại!")
                            break
                        total_scrolls += 1
                        if total_scrolls > 40:
                            self._log("🏁 [XONG NHÓM 2] Đã đạt giới hạn quét tối đa (40 lần cuộn)!")
                            break

                        self._log(f"📜 [CUỘN TRANG] Đang cuộn xuống tiếp tục thu hoạch... (Thanh cuộn: {bottom_ratio*100:.0f}%, Lần {total_scrolls})")
                        scrolled = self.scroll_harvest_list_down(frame)
                        self._fixed_harvest_point = None
                        if not scrolled or is_at_bottom:
                            stuck_scrolls += 1
                            if stuck_scrolls >= 2:
                                time.sleep(0.4)
                                f_last, _ = self.capturer.capture(None)
                                last_cards = self.digit_reader.scan_cards(f_last)
                                fh_last = f_last.shape[0]
                                remaining = [c for c in last_cards if c.btn_y + 40 <= int(fh_last * 0.785)]
                                if remaining:
                                    self._log(f"🌾 [ĐÁY BẢNG NHƯNG CÒN QUẢ] Còn {len(remaining)} quả ở đáy danh sách! Thu hoạch nốt trước khi kết thúc...")
                                    stuck_scrolls = 0
                                    continue
                                self._log("🏁 [ĐÁY DANH SÁCH] Đã cuộn hết toàn bộ danh sách và thu hoạch xong!")
                                break
                        else:
                            stuck_scrolls = 0
                    continue

                # 4. Main HUD (Nút Bán / Nút Thu hoạch)
                now = time.time()
                if step1_start_time is None:
                    step1_start_time = now
                elif now - step1_start_time > getattr(self, "step1_timeout_s", 25.0):
                    self._log("❌ [QUÁ THỜI GIAN] Sau 20s không tìm thấy nút 'Bán' hoặc ông Oliver trên màn hình!")
                    self._log("💡 [HƯỚNG DẪN KHẮC PHỤC] Hãy kiểm tra: 1) Game có đang mở ở màn hình chính hay không. 2) Đảm bảo độ phân giải 1080p và không bị che khuất. 3) Đảm bảo ngôn ngữ game là Tiếng Việt.")
                    self.stop()
                    break

                opened = self.step_open_harvest_window(frame)
                if opened:
                    step1_start_time = None
                else:
                    time.sleep(0.3)
                continue

            except Exception as e:
                self._log(f"⚠️ [LỖI NGOẠI LỆ] {e}")
                time.sleep(1.0)

    def _run_loop(self) -> None:
        enabled_rules = self.cfg_mgr.get_enabled_rules()
        cat_a = [r for r in enabled_rules if r.max_kg > 0.0]
        cat_b = [r for r in enabled_rules if r.max_kg <= 0.0]

        self._log(f"📋 [KẾ HOẠCH] {len(cat_a)} loại quả có cài kg (Nhóm 1), {len(cat_b)} loại quả thu tự do (Nhóm 2).")

        # 1. Phase 1: Category A (Tuần tự từng loại quả, tự động chuyển tiếp)
        for idx, rule in enumerate(cat_a):
            if self._stop_event:
                break
            self.stats["current_fruit"] = rule.name
            self.stats["max_kg"] = rule.max_kg
            self._log(f"👉 [NHÓM 1 - {idx+1}/{len(cat_a)}] Bắt đầu thu hoạch '{rule.name}' (< {rule.max_kg:.2f} kg)...")
            
            harvested_before = self.stats["harvested_count"]
            self._harvest_single_fruit(rule.name, rule.max_kg)
            harvested_in_round = self.stats["harvested_count"] - harvested_before

            if self._stop_event:
                break

            # Tự động bán hoặc đóng bảng về màn hình chính trước Oliver để chuẩn bị cho quả tiếp theo
            if self.cfg_mgr.auto_sell and harvested_in_round > 0:
                self._log(f"💰 [BÁN TỰ ĐỘNG] Đã thu {harvested_in_round} quả '{rule.name}'. Tự động bán tại Oliver trước khi đổi quả...")
                self.step_sell_cycle()
            else:
                self._log("🚪 [HOÀN TẤT QUẢ] Đóng bảng thu hoạch để trở về màn hình chính trước Oliver...")
                self.close_harvest_window()

            time.sleep(1.0)

        # 2. Phase 2: Category B (Thu hoạch tự do không OCR)
        if cat_b and not self._stop_event:
            cat_b_names = [r.name for r in cat_b]
            self.stats["current_fruit"] = "Nhóm Thu Tự Do"
            self.stats["max_kg"] = 0.0
            self._log(f"👉 [NHÓM 2] Bắt đầu thu hoạch tự do: {', '.join(cat_b_names)}...")
            
            harvested_before = self.stats["harvested_count"]
            self._harvest_multi_fruits_fast(cat_b_names)
            harvested_in_round = self.stats["harvested_count"] - harvested_before

            if not self._stop_event:
                if self.cfg_mgr.auto_sell and harvested_in_round > 0:
                    self.step_sell_cycle()
                else:
                    self.close_harvest_window()

        # 3. Phase 3: Final Sell (Nếu còn sót nông sản trong túi)
        if self.cfg_mgr.auto_sell and not self._stop_event and self.stats["harvested_count"] > 0:
            f_check, _ = self.capturer.capture(None)
            sellable_now = self.count_sellable_crops_in_shop(f_check)
            if sellable_now > 0:
                self._log("💰 [BÁN CUỐI CÙNG] Thực hiện đợt bán cuối cùng để làm sạch túi đồ...")
                self.step_sell_cycle()

        if not self._stop_event:
            self._set_state(FarmState.COMPLETED)
            self._log(f"🎉 [HOÀN TẤT TOÀN BỘ] Thu hoạch: {self.stats['harvested_count']} quả | Bán: {self.stats['sell_cycles']} đợt.")

        self._set_state(FarmState.STOPPED)
