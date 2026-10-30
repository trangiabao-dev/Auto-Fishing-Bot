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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assets_dir = assets_dir or os.path.join(base_dir, "assets", "farm")

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
            "sell_chon_tu_dong": os.path.join(self.assets_dir, "ban", "B3.jpg"),
            "sell_coin_btn": os.path.join(self.assets_dir, "ban", "B4.jpg"),
            "sell_confirm_btn": os.path.join(self.assets_dir, "ban", "B5.jpg"),
            "sell_confirm_high_tier": os.path.join(self.assets_dir, "ban", "B6.jpg"),
            "sell_ok_btn": os.path.join(self.assets_dir, "ban", "B7.jpg") if os.path.exists(os.path.join(self.assets_dir, "ban", "B7.jpg")) else os.path.join(self.assets_dir, "ban", "B6.jpg"),
            "prompt_space_cyan": os.path.join(self.assets_dir, "prompt_space_cyan.png"),
            "prompt_space_brown": os.path.join(self.assets_dir, "prompt_space_brown_1080p.png"),
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

    def find_template(
        self,
        frame: np.ndarray,
        tpl_key: str,
        threshold: float = 0.80,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Finds a template in the frame, returning (x, y, w, h) if found."""
        tpl = self._templates.get(tpl_key)
        if tpl is None:
            return None

        search_area = frame
        ox, oy = 0, 0
        if roi is not None:
            rx, ry, rw, rh = roi
            search_area = frame[ry:ry + rh, rx:rx + rw]
            ox, oy = rx, ry

        if search_area.shape[0] < tpl.shape[0] or search_area.shape[1] < tpl.shape[1]:
            return None

        res = cv2.matchTemplate(search_area, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            tw, th = tpl.shape[1], tpl.shape[0]
            return (ox + max_loc[0], oy + max_loc[1], tw, th)
        return None

    def click_rect(self, rect: Tuple[int, int, int, int], margin_ratio: float = 0.2) -> None:
        """Clicks randomly inside a rectangular target."""
        rx, ry, rw, rh = rect
        mx = int(rw * margin_ratio)
        my = int(rh * margin_ratio)
        cx = random.randint(rx + mx, rx + rw - mx)
        cy = random.randint(ry + my, ry + rh - my)
        self.input_sim.click(cx, cy)

    # =========================================================================
    # High-level Automation Routines
    # =========================================================================

    def step_open_harvest_window(self, frame: np.ndarray) -> bool:
        """
        Ensures harvest interface is active:
        Step 1: On bot start, click Nút Bán (B1.png) to teleport in front of Oliver.
        Step 2: Once at Oliver, click Nút Thu hoạch (B2.png) to open harvest window.
        """
        # Case A: Filter modal already open
        if (self.find_template(frame, "btn_cai_dat_loc", threshold=0.75) or
            self.find_template(frame, "tab_nong_san", threshold=0.75) or
            self.find_template(frame, "chk_dot_bien", threshold=0.75)):
            self._log("🔍 [TỰ ĐỘNG NHẬN DIỆN] Bảng Cài Đặt Bộ Lọc đã mở sẵn trên màn hình!")
            return True

        # Case B: Harvest cards window already open
        if (self.find_template(frame, "btn_loc", threshold=0.70) or
            self.find_template(frame, "btn_thu_hoach", threshold=0.70) or
            self.is_filter_empty(frame)):
            self._log("🌾 [TỰ ĐỘNG NHẬN DIỆN] Bảng Thu Hoạch đã mở sẵn trên màn hình!")
            return True

        # Step 1: Teleport to Oliver if not yet done this run
        if not getattr(self, "_teleported_to_oliver", False):
            hit_b1 = self.find_hud_sell_button(frame)
            if hit_b1:
                self._set_state(FarmState.TELEPORTING)
                self._log("🚀 [BƯỚC 1] Bấm nút 'Bán' để dịch chuyển nhân vật tới trước mặt ông bán quả (Oliver)...")
                self.click_rect(hit_b1)
                self._teleported_to_oliver = True
                time.sleep(2.0)
                # If clicking Bán opened a shop modal, close it
                f_after, _ = self.capturer.capture(None)
                close_btn = self.find_close_button(f_after)
                if close_btn and self.find_template(f_after, "sell_chon_tu_dong", threshold=0.75):
                    self.click_rect(close_btn)
                    time.sleep(0.8)
                f_check, _ = self.capturer.capture(None)
                frame = f_check

        # Step 2: Click 'Thu hoạch' (B2.png) to open harvest cards
        hit_b2 = self.find_hud_harvest_button(frame)
        if hit_b2:
            self._set_state(FarmState.OPENING_HARVEST)
            self._log("🌾 [BƯỚC 2] Đã đứng trước ông bán quả! Bấm nút 'Thu hoạch' mở bảng nông sản...")
            self.click_rect(hit_b2)
            time.sleep(1.0)
            f_check, _ = self.capturer.capture(None)
            if (self.find_template(f_check, "btn_loc", threshold=0.68) or
                self.find_template(f_check, "btn_thu_hoach", threshold=0.68) or
                self.is_filter_empty(f_check)):
                self._log("🌾 [B8] Đã mở thành công Bảng Thu Hoạch!")
            return True

        return False

    def step_setup_filter(self, frame: np.ndarray, fruit_name: str) -> bool:
        """Applies filter for a single fruit (Category A) with idempotent check."""
        in_modal = (self.find_template(frame, "btn_cai_dat_loc", threshold=0.75) is not None or
                    self.find_template(frame, "tab_nong_san", threshold=0.75) is not None)
        if not in_modal:
            loc_btn = self.find_template(frame, "btn_loc", threshold=0.75)
            if not loc_btn:
                return False
            self._log(f"🔍 [BỘ LỌC] Bấm 'Lọc' để cấu hình quả: '{fruit_name}'...")
            self.click_rect(loc_btn)
            time.sleep(0.6)
            f2, _ = self.capturer.capture(None)
        else:
            f2 = frame

        chk = self.find_template(f2, "chk_dot_bien", threshold=0.75)
        if chk:
            fx, fy, fw, fh = chk
            roi_chk = f2[fy:fy+fh, fx:fx+65]
            if self.has_blue_checkbox(roi_chk):
                self._log("🛡️ [BỘ LỌC] 'Ngoại trừ trái đột biến' đã được tích sẵn [✓]! Giữ nguyên, không bấm hủy.")
            else:
                self._log("🛡️ [BỘ LỌC] Đã chọn 'Ngoại trừ trái đột biến' để bảo vệ quả đặc biệt!")
                self.click_rect((fx, fy, 50, fh))
                time.sleep(0.4)

        f3, _ = self.capturer.capture(None)
        tab_ns = self.find_template(f3, "tab_nong_san", threshold=0.75)
        if tab_ns:
            self.click_rect(tab_ns)
            time.sleep(0.5)

        f4, _ = self.capturer.capture(None)
        fruit_key = self._get_fruit_template_key(fruit_name)
        if fruit_key:
            fruit_item = self.find_template(f4, fruit_key, threshold=0.68)
            if fruit_item:
                fx, fy, fw, fh = fruit_item
                card_w = max(fw + 90, 230)
                roi_card_chk = f4[fy:fy+fh, fx+190:fx+card_w]
                if self.has_blue_checkbox(roi_card_chk):
                    self._log(f"🌾 [BỘ LỌC] Quả '{fruit_name}' đã được tích sẵn [✓]! Giữ nguyên, không bấm hủy.")
                else:
                    self._log(f"🌾 [BỘ LỌC B6] Đã chọn loại quả: '{fruit_name}'!")
                    cx = random.randint(fx + 10, fx + card_w - 15)
                    cy = random.randint(fy + 8, fy + fh - 8)
                    self.input_sim.click(cx, cy)
                    time.sleep(0.4)
            else:
                self._log(f"⚠️ [BỘ LỌC B6] Không tìm thấy icon/tên '{fruit_name}' trên trang lọc!")
        else:
            self._log(f"⚠️ [BỘ LỌC B6] Chưa có template icon cho quả '{fruit_name}'!")

        f5, _ = self.capturer.capture(None)
        btn_apply = self.find_template(f5, "btn_cai_dat_loc", threshold=0.75)
        if btn_apply:
            self._log("✅ [BỘ LỌC] Áp dụng bộ lọc thành công!")
            self.click_rect(btn_apply)
            time.sleep(0.8)
            return True
        return False

    def step_setup_multi_filter(self, frame: np.ndarray, fruit_names: List[str]) -> bool:
        """Applies filter for multiple Category B fruits at once with idempotent check."""
        in_modal = (self.find_template(frame, "btn_cai_dat_loc", threshold=0.75) is not None or
                    self.find_template(frame, "tab_nong_san", threshold=0.75) is not None)
        if not in_modal:
            loc_btn = self.find_template(frame, "btn_loc", threshold=0.75)
            if not loc_btn:
                return False
            self._log(f"🔍 [BỘ LỌC ĐA QUẢ] Bấm 'Lọc' cấu hình {len(fruit_names)} loại quả thu tự do...")
            self.click_rect(loc_btn)
            time.sleep(0.6)
            f2, _ = self.capturer.capture(None)
        else:
            f2 = frame

        chk = self.find_template(f2, "chk_dot_bien", threshold=0.75)
        if chk:
            fx, fy, fw, fh = chk
            roi_chk = f2[fy:fy+fh, fx:fx+65]
            if self.has_blue_checkbox(roi_chk):
                self._log("🛡️ [BỘ LỌC] 'Ngoại trừ trái đột biến' đã được tích sẵn [✓]! Giữ nguyên, không bấm hủy.")
            else:
                self._log("🛡️ [BỘ LỌC] Chọn 'Ngoại trừ trái đột biến'...")
                self.click_rect((fx, fy, 50, fh))
                time.sleep(0.4)

        f3, _ = self.capturer.capture(None)
        tab_ns = self.find_template(f3, "tab_nong_san", threshold=0.75)
        if tab_ns:
            self.click_rect(tab_ns)
            time.sleep(0.5)

        f4, _ = self.capturer.capture(None)
        for name in fruit_names:
            fruit_key = self._get_fruit_template_key(name)
            if fruit_key:
                fruit_item = self.find_template(f4, fruit_key, threshold=0.68)
                if fruit_item:
                    fx, fy, fw, fh = fruit_item
                    card_w = max(fw + 90, 230)
                    roi_card_chk = f4[fy:fy+fh, fx+190:fx+card_w]
                    if self.has_blue_checkbox(roi_card_chk):
                        self._log(f"🌾 [BỘ LỌC] Quả '{name}' đã được tích sẵn [✓]! Giữ nguyên.")
                    else:
                        self._log(f" -> Tích chọn quả: '{name}'")
                        cx = random.randint(fx + 10, fx + card_w - 15)
                        cy = random.randint(fy + 8, fy + fh - 8)
                        self.input_sim.click(cx, cy)
                        time.sleep(0.3)
                else:
                    self._log(f"⚠️ [BỘ LỌC B6] Chưa thấy icon/tên '{name}' trên trang lọc!")
            else:
                self._log(f"⚠️ [BỘ LỌC B6] Chưa có template icon cho quả '{name}'!")

        f5, _ = self.capturer.capture(None)
        btn_apply = self.find_template(f5, "btn_cai_dat_loc", threshold=0.75)
        if btn_apply:
            self._log("✅ [BỘ LỌC] Áp dụng bộ lọc đa quả thành công!")
            self.click_rect(btn_apply)
            time.sleep(0.8)
            return True
        return False

    @staticmethod
    def has_blue_checkbox(roi_bgr: Optional[np.ndarray]) -> bool:
        """
        Detects if a checkbox is checked [✓] with a vibrant cyan-blue square.
        HSV color range: H in [95..125], S >= 140, V >= 140.
        Returns True if active blue pixels >= 150.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return False
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([95, 140, 140], dtype=np.uint8), np.array([125, 255, 255], dtype=np.uint8))
        return int(cv2.countNonZero(mask)) >= 150

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
        return self.find_template(frame, "toast_day_tui", threshold=0.88) is not None

    def handle_space_prompt_if_present(self, frame: np.ndarray) -> bool:
        """
        Detects any sudden on-screen Spacebar prompt symbol [␣] (Cyan modal button or Brown NPC dialogue).
        100% HARDWARE SPACE CONTROL: Sends DirectInput Space key directly, strictly avoiding mouse clicks
        to eliminate any risk of misclicking background UI or disturbing the harvest coordinate.
        Returns True if a prompt was detected and handled.
        """
        # 1. Cyan Space Prompt (e.g. B7 OK button, harvest error modal #102509, reward popups)
        tpl_c = self._templates.get("prompt_space_cyan")
        tpl_ok = self._templates.get("sell_ok_btn")

        is_cyan_prompt = False
        if tpl_c is not None and cv2.matchTemplate(frame, tpl_c, cv2.TM_CCOEFF_NORMED).max() >= 0.78:
            is_cyan_prompt = True
        elif tpl_ok is not None and cv2.matchTemplate(frame, tpl_ok, cv2.TM_CCOEFF_NORMED).max() >= 0.75:
            is_cyan_prompt = True

        if is_cyan_prompt:
            self._log("⌨️ [SPACE PROMPT] Phát hiện nút có ký hiệu phím Space [␣] (xanh)! Bấm phím Space (né chuột 100%)...")
            self._fixed_harvest_point = None  # Reset coordinate anchor if a modal/error interrupted harvest
            self.input_sim.press_key("space")
            time.sleep(0.5)
            return True

        # 2. Brown Space Prompt (e.g. NPC Cedric/Oliver dialogue bubble)
        # Only check if harvest window is NOT open to avoid chestnut card ambiguity
        is_harvest_window = self.find_template(frame, "btn_loc", threshold=0.70) is not None
        if not is_harvest_window:
            tpl_b = self._templates.get("prompt_space_brown")
            if tpl_b is not None:
                fh = frame.shape[0]
                roi_b = frame[int(fh * 0.60):, :]
                res_b = cv2.matchTemplate(roi_b, tpl_b, cv2.TM_CCOEFF_NORMED)
                if res_b.max() >= 0.76:
                    self._log("⌨️ [SPACE PROMPT] Phát hiện bong bóng thoại có ký hiệu phím Space [␣] (nâu)! Bấm phím Space (né chuột 100%)...")
                    self.input_sim.press_key("space")
                    time.sleep(0.5)
                    return True

        return False

    def is_filter_empty(self, frame: np.ndarray) -> bool:
        """
        Checks if 'Không có trái nào phù hợp với bộ lọc đã chọn.' banner is displayed.
        Restricts template matching strictly to the center ROI of the harvest modal
        to eliminate false positives from the chat bar, hotbar, or crop cards.
        """
        tpl = self._templates.get("tpl_het_trai_1080p")
        if tpl is None:
            return False
        fh, fw = frame.shape[:2]
        # In 1080p: Y in [350..680], X in [500..1420]
        y0, y1 = int(fh * 0.32), int(fh * 0.63)
        x0, x1 = int(fw * 0.26), int(fw * 0.74)
        if y1 > fh or x1 > fw:
            return False
        roi = frame[y0:y1, x0:x1]
        if roi.shape[0] < tpl.shape[0] or roi.shape[1] < tpl.shape[1]:
            return False
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        return bool(res.max() >= 0.70)

    def find_hud_sell_button(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects the circular red/pink 'Bán' button in the top-right HUD.
        Centered tightly on the circular apple button at (1680, 218) with 70x70 bounding box.
        Immune to day/night lighting cycles, camera rotation, and resolution scaling.
        """
        fh, fw = frame.shape[:2]
        rx = int(fw * (1645 / 1920))
        ry = int(fh * (183 / 1080))
        rw = int(fw * (70 / 1920))
        rh = int(fh * (70 / 1080))
        if ry + rh <= fh and rx + rw <= fw and rw * rh > 0:
            roi = frame[ry:ry+rh, rx:rx+rw]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([15, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 60, 60]), np.array([180, 255, 255]))
            ratio = (int(cv2.countNonZero(mask1)) + int(cv2.countNonZero(mask2))) / float(rw * rh)
            if ratio >= 0.35:
                return (rx, ry, rw, rh)
        return self.find_template(frame, "harvest_b1", threshold=0.60)

    def find_hud_harvest_button(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects the circular green 'Thu hoạch' button in the top-right HUD.
        Centered tightly on the green sprout circle at (1587, 218) with 70x70 bounding box.
        Immune to day/night lighting cycles, camera rotation, and resolution scaling.
        """
        fh, fw = frame.shape[:2]
        rx = int(fw * (1552 / 1920))
        ry = int(fh * (183 / 1080))
        rw = int(fw * (70 / 1920))
        rh = int(fh * (70 / 1080))
        if ry + rh <= fh and rx + rw <= fw and rw * rh > 0:
            roi = frame[ry:ry+rh, rx:rx+rw]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_g = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255]))
            ratio = int(cv2.countNonZero(mask_g)) / float(rw * rh)
            if ratio >= 0.30:
                return (rx, ry, rw, rh)
        return self.find_template(frame, "harvest_b2", threshold=0.60)

    def find_close_button(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Finds the close [X] button on top-right of modal dialogs using template matching.
        Falls back to calibrated modal corner coordinates if template not found.
        """
        hit = self.find_template(frame, "btn_close_x", threshold=0.70)
        if hit:
            return hit
        # Calibrated 1080p fallback: modal top-right X is at ~ (1596, 147) -> (0.831, 0.136)
        fh, fw = frame.shape[:2]
        return (int(fw * 0.831), int(fh * 0.136), 67, 60)

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

    def execute_harvest_round(self, frame: np.ndarray, max_kg: float) -> Tuple[bool, bool]:
        """
        Scans visible harvest cards with OCR.
        Locks onto a single fixed harvest button and coordinate (self._fixed_harvest_point) selected randomly once.
        Clicks the exact same point repeatedly as crops shift forward.
        - If a crop shifting into the point is >= max_kg: Vetoes click to protect giant crop, resets fixed point, picks another button.
        - If the point is empty/white space: Resets fixed point, re-randomizes another eligible button.
        - Returns: (harvested_any, should_scroll)
        """
        if self.is_filter_empty(frame):
            self._fixed_harvest_point = None
            return False, False  # Empty banner showing: stop immediately, DO NOT SCROLL!

        cards = self.digit_reader.scan_cards(frame)
        if not cards:
            self._fixed_harvest_point = None
            return False, True

        # Check if we currently have a locked click point
        if self._fixed_harvest_point is not None:
            cx, cy = self._fixed_harvest_point
            # Find the card under the fixed point
            under = [c for c in cards if c.btn_x <= cx <= c.btn_x + c.btn_w and c.btn_y <= cy <= c.btn_y + c.btn_h]
            if not under:
                # Slot became empty / white space
                self._log("ℹ️ [ĐỔI VỊ TRÍ] Vị trí ô thu hoạch cũ đã trống (khoảng trắng)! Đang chọn random nút thu hoạch khác...")
                self._fixed_harvest_point = None
            else:
                card = under[0]
                if card.weight >= max_kg or card.weight <= 0.0:
                    self._log(f"🛡️ [VETO BẢO VỆ] Quả dồn vào ô cố định ({cx},{cy}) là {card.weight:.2f}kg >= {max_kg:.2f}kg! Hủy click bảo vệ quả lớn, chuyển sang nút khác...")
                    self._fixed_harvest_point = None
                elif card.is_clipped:
                    self._fixed_harvest_point = None
                else:
                    # Valid eligible card under fixed point: Keep clicking this exact coordinate!
                    self._log(f"🧺 [THU HOẠCH] Đang thu ô #{card.card_index}: {card.weight:.2f}kg (< {max_kg:.2f}kg) tại ({cx}, {cy})...")
                    self.input_sim.click(cx, cy, hold_min_ms=25, hold_max_ms=45)
                    self.stats["harvested_count"] += 1
                    time.sleep(max(self.cfg_mgr.shift_delay_s, 0.65))
                    return True, False

        # If _fixed_harvest_point is None (first time after filter, or after white space / veto)
        eligible = [c for c in cards if 0.0 < c.weight < max_kg and not c.is_clipped]
        if eligible:
            target = random.choice(eligible)
            cx, cy = target.get_click_point()
            self._fixed_harvest_point = (cx, cy)
            self._log(f"🎯 [CỐ ĐỊNH NÚT] Đã random chọn ô #{target.card_index} ({target.weight:.2f}kg) tại ({cx}, {cy}) để thu hoạch liên tục...")
            self.input_sim.click(cx, cy, hold_min_ms=25, hold_max_ms=45)
            self.stats["harvested_count"] += 1
            time.sleep(max(self.cfg_mgr.shift_delay_s, 0.65))
            return True, False
        else:
            self._fixed_harvest_point = None
            self.stats["skipped_heavy_count"] += len(cards)
            return False, True

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
                time.sleep(max(self.cfg_mgr.shift_delay_s, 0.50))
                return True, False

        # If _fixed_harvest_point is None: pick random actionable card
        if actionable_cards:
            target = random.choice(actionable_cards)
            cx, cy = target.get_click_point()
            self._fixed_harvest_point = (cx, cy)
            self._log(f"🎯 [CỐ ĐỊNH NÚT TỰ DO] Đã random chọn ô #{target.card_index} tại ({cx}, {cy}) để thu hoạch liên tục...")
            self.input_sim.click(cx, cy, hold_min_ms=25, hold_max_ms=45)
            self.stats["harvested_count"] += 1
            time.sleep(max(self.cfg_mgr.shift_delay_s, 0.50))
            return True, False

        tpl = self._templates.get("btn_thu_hoach")
        if tpl is not None:
            res = cv2.matchTemplate(frame, tpl, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= 0.75)
            hits = []
            for pt in zip(*locs[::-1]):
                if not any(abs(pt[0] - p[0]) < 15 and abs(pt[1] - p[1]) < 15 for p in hits):
                    hits.append(pt)
            if hits:
                target_pt = random.choice(hits)
                tw, th = tpl.shape[1], tpl.shape[0]
                bx, by = target_pt[0], target_pt[1]
                rx = random.randint(bx + 15, bx + tw - 15)
                ry = random.randint(by + 10, by + th - 10)
                self._fixed_harvest_point = (rx, ry)
                self.input_sim.click(rx, ry, hold_min_ms=25, hold_max_ms=45)
                self.stats["harvested_count"] += 1
                self._log(f"⚡ [THU HOẠCH TỰ DO] Cố định nút tại ({rx}, {ry})...")
                time.sleep(max(self.cfg_mgr.shift_delay_s, 0.50))
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

    def scroll_harvest_list_down(self, frame: np.ndarray) -> bool:
        """
        Performs a verified scroll-down gesture on the harvest table.
        Uses native swipe-up drag on the neutral body of the crop card (X: ~550, Y: 680 -> 280)
        combined with mouse wheel as fallback.
        Captures frame before and after to verify whether list content actually shifted.

        Returns:
            True if the list moved (weights or card positions shifted).
            False if the list is stationary / reached the absolute bottom.
        """
        fh, fw = frame.shape[:2]
        cards_before = self.digit_reader.scan_cards(frame)
        sig_before = [(round(c.weight, 2), c.kg_y) for c in cards_before]

        # 1. Native swipe-up drag (touch gesture emulation for Unity ScrollRect)
        x0 = int(fw * (550 / 1920))
        y0 = int(fh * (680 / 1080))
        x1 = int(fw * (550 / 1920))
        y1 = int(fh * (280 / 1080))
        self.input_sim.drag(x0, y0, x1, y1, duration_s=0.22, steps=12)

        # 2. Wheel fallback directly over the scroll container
        self.input_sim.move_to(x0, int(fh * (480 / 1080)), human_curve=False)
        for _ in range(4):
            self.input_sim.scroll(-120)
            time.sleep(0.03)

        time.sleep(0.4)  # Wait for Unity physics / inertia to settle

        # 3. Capture new frame and compare card signatures
        new_frame, _ = self.capturer.capture(None)
        cards_after = self.digit_reader.scan_cards(new_frame)
        sig_after = [(round(c.weight, 2), c.kg_y) for c in cards_after]

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

        # Phase 2: Open Oliver shop if not yet open
        f_shop, _ = self.capturer.capture(None)
        in_shop = (
            self.find_template(f_shop, "sell_chon_tu_dong", threshold=0.68) is not None or
            self.find_template(f_shop, "sell_coin_btn", threshold=0.68) is not None
        )
        if not in_shop:
            btn_crops = self.find_template(f_shop, "oliver_sell_crops", threshold=0.70)
            if btn_crops:
                self.click_rect(btn_crops)
                time.sleep(1.0)
            else:
                oliver_basket = self.find_template(f_shop, "oliver_basket", threshold=0.65)
                if oliver_basket:
                    self._log(" -> Bấm icon giỏ hàng trên đầu ông Oliver để vào bán...")
                    self.click_rect(oliver_basket)
                    time.sleep(1.0)
                    f_dlg, _ = self.capturer.capture(None)
                    btn_crops = self.find_template(f_dlg, "oliver_sell_crops", threshold=0.70)
                    if btn_crops:
                        self.click_rect(btn_crops)
                        time.sleep(1.0)
                else:
                    hit_b1 = self.find_hud_sell_button(f_shop)
                    if hit_b1:
                        self._log(" -> Bấm nút 'Bán' trên HUD để mở lại cửa hàng Oliver...")
                        self.click_rect(hit_b1)
                        time.sleep(1.5)

        # Phase 3-6: Multi-batch Sell Loop (sells all full batches of 100)
        total_batches_sold = 0
        for batch_idx in range(1, 6): # Up to 5 consecutive batches (500 items max)
            f_cur, _ = self.capturer.capture(None)
            auto_btn = self.find_template(f_cur, "sell_chon_tu_dong", threshold=0.68)
            if not auto_btn:
                # Shop modal not open or already finished
                break

            # Pre-scan: Check if there are any sellable (unlocked) crops on the shop grid
            sellable_count = self.count_sellable_crops_in_shop(f_cur)
            if sellable_count == 0:
                if total_batches_sold > 0:
                    self._log("✨ [BÁN XONG] Lưới cửa hàng đã sạch (không còn nông sản mở khóa để bán)! Kết thúc đợt bán.")
                else:
                    self._log("⚠️ [CỬA HÀNG] Không có nông sản nào có thể bán trên lưới! Đóng cửa hàng...")
                break

            self._log(f" -> Đợt {batch_idx}: Thấy {sellable_count} ô nông sản có thể bán! Bấm '(0/100) Chọn tự động'...")
            self.click_rect(auto_btn)
            time.sleep(0.6)

            # Check if coin button is active with items to sell
            f_coin, _ = self.capturer.capture(None)
            coin_btn = self.find_template(f_coin, "sell_coin_btn", threshold=0.68)
            if not coin_btn:
                self._log(f" -> Đợt {batch_idx}: Không còn nông sản nào có thể bán! Túi đồ đã sạch.")
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
                if self.find_template(f_ht, "sell_ok_btn", threshold=0.62) or self.find_template(f_ht, "sell_chon_tu_dong", threshold=0.68):
                    break
                time.sleep(0.3)

            # Poll for OK popup 3 (B7) or immediate return to shop grid
            ok_clicked = False
            for poll in range(8):
                time.sleep(0.3)
                f_ok, _ = self.capturer.capture(None)
                # Check for Space prompt (B7 button [(␣) OK] or space prompt icon)
                if (self.find_template(f_ok, "sell_ok_btn", threshold=0.62) is not None or
                    self.find_template(f_ok, "prompt_space_cyan", threshold=0.80) is not None):
                    self._log(f" -> Đợt {batch_idx}: [B7] Thấy ký hiệu [␣] trên nút OK, bấm phím Space hoàn tất bán hàng...")
                    self.input_sim.press_key("space")
                    ok_clicked = True
                    time.sleep(0.8)
                    break
                # If already back in shop grid, transaction finished directly
                if self.find_template(f_ok, "sell_chon_tu_dong", threshold=0.68):
                    ok_clicked = True
                    break

            total_batches_sold += 1

        # Phase 7: Close shop modal cleanly
        for close_attempt in range(3):
            f_after, _ = self.capturer.capture(None)
            close_shop = self.find_close_button(f_after)
            if close_shop and (
                self.find_template(f_after, "sell_chon_tu_dong", threshold=0.65) or
                self.find_template(f_after, "sell_coin_btn", threshold=0.65)
            ):
                self.click_rect(close_shop)
                time.sleep(0.6)
            else:
                break

        # Phase 8: Dismiss Oliver farewell dialogue if active (speech bubble blocking HUD)
        for _ in range(3):
            f_dlg, _ = self.capturer.capture(None)
            if self.find_hud_harvest_button(f_dlg) is not None or self.find_hud_sell_button(f_dlg) is not None:
                break  # HUD already visible and clean
            self._log(" -> Đóng lời thoại ông Oliver (bấm Space)...")
            self.input_sim.press_key("space")
            time.sleep(0.8)

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
        self._set_state(FarmState.OPENING_HARVEST)

        import threading
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="FarmEngineThread")
        self._thread.start()
        self._log(f"🚀 [NÔNG TRẠI] Bắt đầu chu trình tự động thu hoạch cho {len(enabled_rules)} loại quả đã chọn...")

    def stop(self) -> None:
        """Stops the Farm engine cleanly."""
        self._stop_event = True
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
        self._fixed_harvest_point = None

        while not self._stop_event:
            if self._pause_event:
                time.sleep(0.2)
                continue

            try:
                frame, _ = self.capturer.capture(None)

                # 0. Global sudden space prompt handler (unexpected popups, rewards, dialogues)
                if self.handle_space_prompt_if_present(frame):
                    time.sleep(0.5)
                    continue

                # 1. Detect Bag Full Toast or persistent Bag Full state
                if self.is_bag_full(frame) or getattr(self, "_bag_is_full", False):
                    self._bag_is_full = True
                    self._set_state(FarmState.SELLING)
                    self._log("🛑 [ĐẦY TÚI] Túi đồ đang đầy! Chuyển sang quy trình bán tại NPC Oliver...")
                    sold = self.step_sell_cycle()
                    if sold:
                        self._bag_is_full = False
                        filter_done = False  # Reset filter to re-apply for remaining fruits
                        stuck_scrolls = 0
                        total_scrolls = 0
                    time.sleep(1.0)
                    continue

                # 2. Detect Filter Modal (B4, B5, B7)
                in_filter_modal = (
                    self.find_template(frame, "btn_cai_dat_loc", threshold=0.72) is not None or
                    self.find_template(frame, "tab_nong_san", threshold=0.72) is not None or
                    self.find_template(frame, "chk_dot_bien", threshold=0.72) is not None
                )
                if in_filter_modal:
                    self._set_state(FarmState.APPLYING_FILTER)
                    chk = self.find_template(frame, "chk_dot_bien", threshold=0.72)
                    if chk:
                        fx, fy, fw, fh = chk
                        roi_chk = frame[fy:fy+fh, fx:fx+65]
                        if self.has_blue_checkbox(roi_chk):
                            self._log("🛡️ [BỘ LỌC] 'Ngoại trừ trái đột biến' đã được tích sẵn [✓]! Giữ nguyên, không bấm hủy.")
                        else:
                            self._log("🛡️ [BỘ LỌC] Đã chọn 'Ngoại trừ trái đột biến' để bảo vệ quả đặc biệt!")
                            self.click_rect((fx, fy, 50, fh))
                            time.sleep(0.4)

                    f_tab, _ = self.capturer.capture(None)
                    tab_ns = self.find_template(f_tab, "tab_nong_san", threshold=0.72)
                    if tab_ns:
                        self.click_rect(tab_ns)
                        time.sleep(0.5)

                    f_fruit, _ = self.capturer.capture(None)
                    fruit_key = self._get_fruit_template_key(fruit_name)
                    if fruit_key:
                        fruit_item = self.find_template(f_fruit, fruit_key, threshold=0.68)
                        if fruit_item:
                            fx, fy, fw, fh = fruit_item
                            card_w = max(fw + 90, 230)
                            roi_card_chk = f_fruit[fy:fy+fh, fx+190:fx+card_w]
                            if self.has_blue_checkbox(roi_card_chk):
                                self._log(f"🌾 [BỘ LỌC] Quả '{fruit_name}' đã được tích sẵn [✓]! Giữ nguyên, không bấm hủy.")
                            else:
                                self._log(f"🌾 [BỘ LỌC] Đã tìm thấy '{fruit_name}'! Bấm chọn...")
                                cx = random.randint(fx + 10, fx + card_w - 15)
                                cy = random.randint(fy + 8, fy + fh - 8)
                                self.input_sim.click(cx, cy)
                                time.sleep(0.4)
                        else:
                            self._log(f"⚠️ [BỘ LỌC] Chưa thấy dòng quả '{fruit_name}' trên bảng lọc!")

                    f_apply, _ = self.capturer.capture(None)
                    btn_apply = self.find_template(f_apply, "btn_cai_dat_loc", threshold=0.72)
                    if btn_apply:
                        self._log("✅ [BỘ LỌC] Bấm 'Cài đặt bộ lọc' hoàn tất cấu hình!")
                        self.click_rect(btn_apply)
                        filter_done = True
                        time.sleep(0.8)
                    continue

                # 3. Detect Harvest Window (B3, Cards, or Empty Banner)
                in_harvest_window = (
                    self.find_template(frame, "btn_loc", threshold=0.68) is not None or
                    self.find_template(frame, "btn_thu_hoach", threshold=0.68) is not None or
                    self.is_filter_empty(frame)
                )
                if in_harvest_window:
                    if not filter_done:
                        self._set_state(FarmState.APPLYING_FILTER)
                        btn_loc = self.find_template(frame, "btn_loc", threshold=0.68)
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

                    # Monitor scrollbar status to detect bottom position and shrinking content
                    thumb_ratio, bottom_ratio, is_at_bottom = self.get_scrollbar_info(frame)

                    harvested, should_scroll = self.execute_harvest_round(frame, max_kg)
                    if harvested:
                        stuck_scrolls = 0
                        total_scrolls = 0
                    elif should_scroll:
                        total_scrolls += 1
                        if total_scrolls > 40:
                            self._log(f"🏁 [XONG QUẢ] Đã đạt giới hạn quét tối đa (40 lần cuộn)! Kết thúc quả '{fruit_name}'.")
                            break

                        self._log(f"📜 [CUỘN TRANG] Các quả hiện tại đều >= {max_kg:.2f}kg (Thanh cuộn: {bottom_ratio*100:.0f}%). Đang cuộn xuống tìm quả nhỏ hơn... (Lần {total_scrolls})")
                        scrolled = self.scroll_harvest_list_down(frame)
                        self._fixed_harvest_point = None
                        if not scrolled or is_at_bottom:
                            stuck_scrolls += 1
                            if stuck_scrolls >= 2:
                                # FAIL-SAFE BOTTOM EXIT VERIFICATION:
                                # Ensure zero remaining harvestable crops are left unharvested before exiting!
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
                hit_b1 = self.find_hud_sell_button(frame)
                hit_b2 = self.find_hud_harvest_button(frame)

                if not getattr(self, "_teleported_to_oliver", False) and hit_b1:
                    self._set_state(FarmState.TELEPORTING)
                    self._log("🚀 [BƯỚC 1] Thấy nút 'Bán' trên HUD! Bấm để dịch chuyển nhân vật tới trước mặt ông bán quả (Oliver)...")
                    self.click_rect(hit_b1)
                    self._teleported_to_oliver = True
                    time.sleep(2.0)
                    f_chk, _ = self.capturer.capture(None)
                    close_btn = self.find_close_button(f_chk)
                    if close_btn and self.find_template(f_chk, "sell_chon_tu_dong", threshold=0.75):
                        self.click_rect(close_btn)
                        time.sleep(0.8)
                    continue

                if hit_b2:
                    self._set_state(FarmState.OPENING_HARVEST)
                    self._log("🌾 [BƯỚC 2] Thấy nút 'Thu hoạch' trên HUD! Bấm mở bảng nông sản...")
                    self.click_rect(hit_b2)
                    time.sleep(1.2)
                    continue

                # 5. Scanning Fallback
                now = time.time()
                if now - last_wait_log > 3.0:
                    last_wait_log = now
                    if not getattr(self, "_teleported_to_oliver", False):
                        self._log("⏳ [ĐANG QUÉT MÀN HÌNH] Bước 1: Đang tìm nút 'Bán' (B1) trên HUD để dịch chuyển tới Oliver...")
                    else:
                        self._log("⏳ [ĐANG QUÉT MÀN HÌNH] Bước 2: Đang tìm nút 'Thu hoạch' (B2), bảng Nông trại hoặc bảng Lọc...")
                time.sleep(0.4)

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

                # 1. Bag Full Toast or persistent Bag Full state
                if self.is_bag_full(frame) or getattr(self, "_bag_is_full", False):
                    self._bag_is_full = True
                    self._set_state(FarmState.SELLING)
                    self._log("🛑 [ĐẦY TÚI] Túi đồ đang đầy! Chuyển sang quy trình bán tại NPC Oliver...")
                    sold = self.step_sell_cycle()
                    if sold:
                        self._bag_is_full = False
                        filter_done = False  # Reset filter to re-apply
                        stuck_scrolls = 0
                        total_scrolls = 0
                    time.sleep(1.0)
                    continue

                # 2. Filter Modal
                in_filter_modal = (
                    self.find_template(frame, "btn_cai_dat_loc", threshold=0.72) is not None or
                    self.find_template(frame, "tab_nong_san", threshold=0.72) is not None or
                    self.find_template(frame, "chk_dot_bien", threshold=0.72) is not None
                )
                if in_filter_modal:
                    self._set_state(FarmState.APPLYING_FILTER)
                    chk = self.find_template(frame, "chk_dot_bien", threshold=0.72)
                    if chk:
                        fx, fy, fw, fh = chk
                        roi_chk = frame[fy:fy+fh, fx:fx+65]
                        if self.has_blue_checkbox(roi_chk):
                            self._log("🛡️ [BỘ LỌC] 'Ngoại trừ trái đột biến' đã được tích sẵn [✓]! Giữ nguyên, không bấm hủy.")
                        else:
                            self._log("🛡️ [BỘ LỌC] Chọn 'Ngoại trừ trái đột biến'...")
                            self.click_rect((fx, fy, 50, fh))
                            time.sleep(0.4)

                    f_tab, _ = self.capturer.capture(None)
                    tab_ns = self.find_template(f_tab, "tab_nong_san", threshold=0.72)
                    if tab_ns:
                        self.click_rect(tab_ns)
                        time.sleep(0.5)

                    f_fruits, _ = self.capturer.capture(None)
                    for name in fruit_names:
                        f_key = self._get_fruit_template_key(name)
                        if f_key:
                            fruit_item = self.find_template(f_fruits, f_key, threshold=0.68)
                            if fruit_item:
                                fx, fy, fw, fh = fruit_item
                                card_w = max(fw + 90, 230)
                                roi_card_chk = f_fruits[fy:fy+fh, fx+190:fx+card_w]
                                if self.has_blue_checkbox(roi_card_chk):
                                    self._log(f"🌾 [BỘ LỌC] Quả '{name}' đã được tích sẵn [✓]! Giữ nguyên.")
                                else:
                                    self._log(f" -> Tích chọn quả: '{name}'")
                                    cx = random.randint(fx + 10, fx + card_w - 15)
                                    cy = random.randint(fy + 8, fy + fh - 8)
                                    self.input_sim.click(cx, cy)
                                    time.sleep(0.3)

                    f_apply, _ = self.capturer.capture(None)
                    btn_apply = self.find_template(f_apply, "btn_cai_dat_loc", threshold=0.72)
                    if btn_apply:
                        self._log("✅ [BỘ LỌC] Bấm 'Cài đặt bộ lọc' hoàn tất cấu hình đa quả!")
                        self.click_rect(btn_apply)
                        filter_done = True
                        time.sleep(0.8)
                    continue

                # 3. Harvest Window
                in_harvest_window = (
                    self.find_template(frame, "btn_loc", threshold=0.68) is not None or
                    self.find_template(frame, "btn_thu_hoach", threshold=0.68) is not None or
                    self.is_filter_empty(frame)
                )
                if in_harvest_window:
                    if not filter_done:
                        self._set_state(FarmState.APPLYING_FILTER)
                        btn_loc = self.find_template(frame, "btn_loc", threshold=0.68)
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

                    thumb_ratio, bottom_ratio, is_at_bottom = self.get_scrollbar_info(frame)

                    harvested, should_scroll = self.execute_fast_harvest_round(frame)
                    if harvested:
                        stuck_scrolls = 0
                        total_scrolls = 0
                    elif should_scroll:
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

                # 4. Main HUD
                hit_b1 = self.find_hud_sell_button(frame)
                hit_b2 = self.find_hud_harvest_button(frame)

                if not getattr(self, "_teleported_to_oliver", False) and hit_b1:
                    self._set_state(FarmState.TELEPORTING)
                    self._log("🚀 [BƯỚC 1] Thấy nút 'Bán' trên HUD! Bấm để dịch chuyển nhân vật tới trước mặt ông bán quả (Oliver)...")
                    self.click_rect(hit_b1)
                    self._teleported_to_oliver = True
                    time.sleep(2.0)
                    f_chk, _ = self.capturer.capture(None)
                    close_btn = self.find_close_button(f_chk)
                    if close_btn and self.find_template(f_chk, "sell_chon_tu_dong", threshold=0.75):
                        self.click_rect(close_btn)
                        time.sleep(0.8)
                    continue

                if hit_b2:
                    self._set_state(FarmState.OPENING_HARVEST)
                    self._log("🌾 [BƯỚC 2] Thấy nút 'Thu hoạch' trên HUD! Bấm mở bảng nông sản...")
                    self.click_rect(hit_b2)
                    time.sleep(1.2)
                    continue

                # 5. Scanning Fallback
                now = time.time()
                if now - last_wait_log > 3.0:
                    last_wait_log = now
                    if not getattr(self, "_teleported_to_oliver", False):
                        self._log("⏳ [ĐANG QUÉT MÀN HÌNH] Bước 1: Đang tìm nút 'Bán' (B1) trên HUD để dịch chuyển tới Oliver...")
                    else:
                        self._log("⏳ [ĐANG QUÉT MÀN HÌNH] Bước 2: Đang tìm nút 'Thu hoạch' (B2), bảng Nông trại hoặc bảng Lọc...")
                time.sleep(0.4)

            except Exception as e:
                self._log(f"⚠️ [LỖI NGOẠI LỆ] {e}")
                time.sleep(1.0)

    def _run_loop(self) -> None:
        enabled_rules = self.cfg_mgr.get_enabled_rules()
        cat_a = [r for r in enabled_rules if r.max_kg > 0.0]
        cat_b = [r for r in enabled_rules if r.max_kg <= 0.0]

        self._log(f"📋 [KẾ HOẠCH] {len(cat_a)} loại quả có cài kg (Nhóm 1), {len(cat_b)} loại quả thu tự do (Nhóm 2).")

        # 1. Phase 1: Category A
        for rule in cat_a:
            if self._stop_event:
                break
            self.stats["current_fruit"] = rule.name
            self.stats["max_kg"] = rule.max_kg
            self._log(f"👉 [NHÓM 1] Bắt đầu thu hoạch '{rule.name}' (< {rule.max_kg:.2f} kg)...")
            self._harvest_single_fruit(rule.name, rule.max_kg)

        # 2. Phase 2: Category B
        if cat_b and not self._stop_event:
            cat_b_names = [r.name for r in cat_b]
            self.stats["current_fruit"] = "Nhóm Thu Tự Do"
            self.stats["max_kg"] = 0.0
            self._log(f"👉 [NHÓM 2] Bắt đầu thu hoạch tự do: {', '.join(cat_b_names)}...")
            self._harvest_multi_fruits_fast(cat_b_names)

        # 3. Phase 3: Final Sell
        if self.cfg_mgr.auto_sell and not self._stop_event and self.stats["harvested_count"] > 0:
            self._log("💰 [BÁN CUỐI CÙNG] Thực hiện đợt bán cuối cùng để làm sạch túi đồ...")
            self.step_sell_cycle()

        if not self._stop_event:
            self._set_state(FarmState.COMPLETED)
            self._log(f"🎉 [HOÀN TẤT TOÀN BỘ] Thu hoạch: {self.stats['harvested_count']} quả | Bán: {self.stats['sell_cycles']} đợt.")

        self._set_state(FarmState.STOPPED)
