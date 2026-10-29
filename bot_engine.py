import math
import time
import random
import threading
from enum import Enum
from typing import Dict, Any, Optional, Tuple, Callable
from datetime import datetime

from core.screen_capture import ScreenCapturer
from core.detector import TemplateMatcher, MatchResult
from core.input_sim import InputSimulator
from core.diagnostics import StageTwoRecorder
from core.head_locator import NameLocator
from core.plate_locator import PlateLocator, PlateTracker
from core.scale_aware import BiteTemplateBank, scaled_head_roi
from core.static_veto import StaticVeto
from core.outcome_recorder import attach_outcome_recorder
from core.geometry import Rect
from core.shape_veto import is_cloud_like
from core.region_scan import RegionScanner, ScanTarget
from core.progress_watchdog import ProgressWatchdog
from core.bite_detector import BiteDetector
from core.bite_policy import TriggerPolicy
from core.cast_button_guard import CastButtonGuard
from core.missed_catch_logger import MissedCatchLogger
import os
import cv2


class BotState(Enum):
    IDLE = "IDLE (Chờ F1)"
    STAGE_1 = "BƯỚC 1: Quét & Click Nút Quăng Cần"
    STAGE_2 = "BƯỚC 2: Chờ Dấu Chấm Than -> Giật Cần"
    STAGE_3 = "BƯỚC 3: Lưu Cá Vào Balo"
    STAGE_REPAIR = "BƯỚC PHỤ: Sửa Cần Câu Bị Hỏng"
    STOPPED = "ĐÃ DỪNG (F2)"


class BotEngine:
    """
    Finite State Machine orchestrator managing the vision-based automation loop.
    Features:
      - Dynamic Initial State Discovery (Auto-detects context: currently fishing, popup open, or ready to cast)
      - Self-Healing State Recovery (Handles desyncs, lag, missed bites)
      - Anchor-based Head Tracking (Immune to fish exclamation marks in water)
      - Folder & Extension Agnostic Image Matching
    """

    def __init__(
        self,
        config: Dict[str, Any],
        on_log: Optional[Callable[[str], None]] = None,
        on_state_change: Optional[Callable[[BotState], None]] = None,
        on_fish_caught: Optional[Callable[[], None]] = None
    ):
        self.config = config
        self.state = BotState.IDLE
        self.on_log = on_log
        self.on_state_change = on_state_change
        self.on_fish_caught = on_fish_caught

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        # Dependencies
        self.capturer = ScreenCapturer()
        self.matcher = TemplateMatcher()
        self.input_sim = InputSimulator()
        self.stats = {"bite_seen": 0, "caught": 0, "missed_after_click": 0, "lost_before_bite": 0}
        self.recorder = StageTwoRecorder()  # Step 0: evidence for bite-detection debugging

        # Character locator: PlateLocator + PlateTracker (P1a: blob-based, hue-mask, adaptive zoom)
        name_tpl = config.get("templates", {}).get("b2_ten_nhan_vat", "assets/B2_DangCau/vat_neo_ten")
        name_thresh = config.get("thresholds", {}).get("b2_ten_nhan_vat", 0.60)
        self.plate_locator = PlateLocator(name_tpl, min_score=name_thresh)
        self.plate_tracker = PlateTracker(self.plate_locator)
        self.name_locator = self.plate_locator  # backward compatibility / telemetry fallback

        # P1b: Scale-aware '!' template bank & StaticVeto
        b2_cham_than_dir = config.get("templates", {}).get("b2_cham_than", "assets/B2_DangCau/dau_cham_than")
        self.bite_bank = BiteTemplateBank(b2_cham_than_dir, reference_scale=1.0)
        self.static_veto = StaticVeto()
        self.stats["static_vetoed"] = 0
        self.stats["shape_vetoed"] = 0

        # P2: BiteDetector (không phụ thuộc màu) + chính sách triển khai: off | shadow | confirm | primary
        bd_cfg = config.get("bite_detector", {})
        self.bite_detector = BiteDetector(min_contrast=bd_cfg.get("min_contrast", 20.0))
        self.bite_policy = TriggerPolicy(bd_cfg.get("mode", "shadow"), bd_cfg.get("confirm_window_s", 0.6))
        self._reel_det_agreed = False
        for key in ("caught_det_agreed", "caught_det_missed", "miss_det_agreed", "miss_det_rejected"):
            self.stats[key] = 0

        # P1c: Fast RegionScanner for fixed game UI modals/buttons & ProgressWatchdog
        tpls = config.get("templates", {})
        ths = config.get("thresholds", {})
        self.scanner = RegionScanner([
            ScanTarget("bag_toast",    tpls.get("b5_tui_day_toast", "assets/B5_TuiDay/toast_tui_day.png"),  (0.25, 0.15, 0.75, 0.45), ths.get("b5_tui_day", 0.75)),
            ScanTarget("bag_speech",   tpls.get("b5_tui_day_speech", "assets/B5_TuiDay/speech_tui_day.png"), (0.15, 0.15, 0.85, 0.85), ths.get("b5_tui_day", 0.72)),
            ScanTarget("repair_title", tpls.get("b4_sua_can_title", "assets/B4_SuaCan/title"), (0.15, 0.05, 0.85, 0.85), ths.get("b4_sua_can", 0.70)),
            ScanTarget("repair_btn",   tpls.get("b4_sua_can_btn", "assets/B4_SuaCan/button"),  (0.20, 0.40, 0.80, 0.90), ths.get("b4_sua_can", 0.70)),
            ScanTarget("done_title",   tpls.get("b4_hoan_thanh_title", "assets/B4_SuaCan/title_hoan_thanh"), (0.15, 0.05, 0.85, 0.85), ths.get("b4_sua_can", 0.70)),
            ScanTarget("ok",           tpls.get("b4_ok_btn", "assets/B4_SuaCan/btn_ok"),       (0.20, 0.40, 0.80, 0.90), ths.get("b4_btn_ok", 0.70)),
            ScanTarget("keep",         tpls.get("b3_bao_quan", "assets/B3_BaoQuan"),          (0.65, 0.65, 0.95, 0.95), ths.get("b3_bao_quan", 0.70)),
            ScanTarget("reel",         tpls.get("b2_giat_can", "assets/B2_DangCau/nut_giat_can.png"), (0.65, 0.55, 1.00, 1.00), ths.get("b2_giat_can", 0.70)),
            ScanTarget("cast",         tpls.get("b1_quang_can", "assets/B1_QuangCan"),         (0.65, 0.55, 1.00, 1.00), ths.get("b1_quang_can", 0.70)),
            ScanTarget("card_open",    tpls.get("b6_btn_mo", "assets/B6_GoiThe/btn_mo.png"),   (0.70, 0.65, 0.98, 0.95), ths.get("b6_goi_the", 0.70)),
            ScanTarget("card_prompt",  tpls.get("b6_prompt_space", "assets/B6_GoiThe/prompt_space.png"), (0.35, 0.85, 0.65, 1.00), ths.get("b6_goi_the", 0.70)),
            ScanTarget("card_title",   tpls.get("b6_title_ketqua", "assets/B6_GoiThe/title_ketqua.png"), (0.35, 0.08, 0.65, 0.28), ths.get("b6_goi_the", 0.70)),
            ScanTarget("card_ok",      tpls.get("b6_btn_ok", "assets/B6_GoiThe/btn_ok_card.png"), (0.38, 0.68, 0.62, 0.92), ths.get("b6_goi_the", 0.70)),
        ])
        self.watchdog = ProgressWatchdog(timeout_s=90.0)
        self.cast_guard = CastButtonGuard()

        # Cached last known match for img1 and head ROI
        self._last_img1_match: Optional[MatchResult] = None
        self._last_good_head_roi: Optional[Tuple[int, int, int, int]] = None  # only ever set from a real name match

        # P0: telemetry tuy chon, chi quan sat, khong doi hanh vi
        tel_cfg = config.get("telemetry", {})
        if tel_cfg.get("enabled", False):
            from core.instrumentation import attach_telemetry
            attach_telemetry(self, jsonl_path=tel_cfg.get("jsonl_path", "telemetry/telemetry.jsonl"))
        # P1b: Outcome recorder (corpus/<caught|missed|rod_retracted>)
        attach_outcome_recorder(self, "corpus")
        self.missed_logger = MissedCatchLogger(base_dir="debug_hut_ca")
        self._last_head_frame_at_bite: Optional[np.ndarray] = None
        self._last_bite_score: float = 0.0
        self._last_bite_tpl: str = "N/A"

    @staticmethod
    def _clamp_region(region, frame, offset):
        """Keeps a capture region inside the captured monitor so mss never gets an off-screen box."""
        x, y, w, h = region
        ox, oy = offset
        fh, fw = frame.shape[:2]
        x0, y0 = max(ox, x), max(oy, y)
        x1, y1 = min(ox + fw, x + w), min(oy + fh, y + h)
        return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    def _find_repair_button(
        self,
        frame,
        title_match: Optional[MatchResult] = None,
        cash_match: Optional[MatchResult] = None
    ) -> Optional[MatchResult]:
        """
        Dynamically locates the entire blue capsule repair button using color segmentation.
        Provides the full button box (~260x90 in 1080p) so anti-clustering random clicks
        can land anywhere across the entire blue surface, regardless of the price text.
        """
        import cv2
        th, tw = frame.shape[:2]
        if cash_match and cash_match.found:
            search_y1 = max(0, cash_match.y - int(cash_match.h * 0.8))
            search_y2 = min(th, cash_match.y + int(cash_match.h * 1.8))
            search_x1 = max(0, cash_match.x - int(cash_match.w * 1.5))
            search_x2 = min(tw, cash_match.x + int(cash_match.w * 5.0))
        elif title_match and title_match.found:
            search_y1 = title_match.y + int(title_match.h * 2.0)
            search_y2 = min(th, title_match.y + int(title_match.h * 10.0))
            search_x1 = max(0, title_match.x - int(title_match.w * 0.4))
            search_x2 = min(tw, title_match.x + int(title_match.w * 1.4))
        else:
            # Fallback trung tâm: Modal sửa cần luôn nằm ở trung tâm màn hình (y: 45%-85%, x: 25%-75%)
            search_y1 = int(th * 0.45)
            search_y2 = int(th * 0.85)
            search_x1 = int(tw * 0.25)
            search_x2 = int(tw * 0.75)

        roi = frame[search_y1:search_y2, search_x1:search_x2]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (90, 110, 170), (108, 255, 255))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        max_area = 0
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_cnt = cnt

        if best_cnt is not None and max_area > 200:
            bx, by, bw, bh = cv2.boundingRect(best_cnt)
            return MatchResult(
                found=True,
                x=search_x1 + bx,
                y=search_y1 + by,
                w=bw,
                h=bh,
                confidence=1.0,
                template_name="nut_sua_can_xanh"
            )

        if cash_match and cash_match.found:
            btn_x = max(0, cash_match.x - int(cash_match.w * 0.8))
            btn_w = int(cash_match.w * 4.0)
            return MatchResult(found=True, x=btn_x, y=cash_match.y - 5, w=btn_w, h=cash_match.h + 10, confidence=1.0, template_name="nut_sua_can_expanded")

        if title_match and title_match.found:
            bw = int(title_match.w * 0.72)
            bx = title_match.x + (title_match.w - bw) // 2
            by = title_match.y + int(title_match.h * 6.7)
            bh = int(title_match.h * 1.4)
            return MatchResult(found=True, x=bx, y=by, w=bw, h=bh, confidence=1.0, template_name="nut_sua_can_inferred")

        return None

    def _find_ok_button(
        self,
        frame,
        title_match: Optional[MatchResult] = None,
        screen_offset: Tuple[int, int] = (0, 0)
    ) -> Optional[MatchResult]:
        """
        Định vị nút xanh OK ("(-) OK") trên modal 'Sửa chữa hoàn thành'.
        Kết hợp:
        1. Khớp mẫu đa tỉ lệ với thư mục assets/B4_SuaCan/btn_ok
        2. Phân đoạn màu HSV (cyan/xanh dương) trong vùng dialog trung tâm
        3. Suy luận không gian bên dưới tiêu đề 'Sửa chữa hoàn thành' nếu có
        """
        import cv2

        ok_tpl = self.config.get("templates", {}).get("b4_ok_btn", "assets/B4_SuaCan/btn_ok")
        thresh_ok = self.config.get("thresholds", {}).get("b4_btn_ok", 0.70)

        # 1. Khớp mẫu template matching trong vùng dialog trung tâm (đơn tỉ lệ)
        th, tw = frame.shape[:2]
        c_y1, c_y2 = int(th * 0.40), int(th * 0.90)
        c_x1, c_x2 = int(tw * 0.20), int(tw * 0.80)
        c_roi = frame[c_y1:c_y2, c_x1:c_x2]
        c_off = (screen_offset[0] + c_x1, screen_offset[1] + c_y1)
        m_tpl = self.matcher.match_any(c_roi, ok_tpl, threshold=thresh_ok, screen_offset=c_off, multi_scale=False)
        if m_tpl.found:
            return m_tpl

        # Nếu Modal 1 ("Thông tin sửa chữa") đang hiện, nút xanh trên màn hình là nút tiền, KHÔNG PHẢI nút OK!
        sua_can_title = self.config.get("templates", {}).get("b4_sua_can_title", "assets/B4_SuaCan/title")
        thresh_repair = self.config.get("thresholds", {}).get("b4_sua_can", 0.70)
        m_modal1 = self.matcher.match_any(frame, sua_can_title, threshold=thresh_repair, screen_offset=screen_offset)
        if m_modal1.found:
            return None

        # 2. Phân đoạn màu HSV cho viên thuốc màu xanh ở giữa màn hình
        th, tw = frame.shape[:2]
        roi_y1, roi_y2 = int(th * 0.45), int(th * 0.85)
        roi_x1, roi_x2 = int(tw * 0.25), int(tw * 0.75)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size > 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (90, 110, 170), (108, 255, 255))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_cnt = None
            max_area = 0
            for c in cnts:
                area = cv2.contourArea(c)
                if area > max_area:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    aspect = bw / float(bh) if bh > 0 else 0
                    if 2.0 <= aspect <= 4.5 and 20 <= bh <= 150:
                        max_area = area
                        best_cnt = (bx, by, bw, bh)

            if best_cnt is not None and max_area > 400:
                bx, by, bw, bh = best_cnt
                return MatchResult(
                    found=True,
                    x=screen_offset[0] + roi_x1 + bx,
                    y=screen_offset[1] + roi_y1 + by,
                    w=bw,
                    h=bh,
                    confidence=1.0,
                    template_name="btn_ok_hsv"
                )

        # 3. Suy luận theo vị trí tiêu đề nếu tiêu đề khớp
        if title_match and title_match.found:
            bw = int(title_match.w * 0.72)
            bx = title_match.x + (title_match.w - bw) // 2
            by = title_match.y + int(title_match.h * 6.7)
            bh = int(title_match.h * 1.4)
            return MatchResult(
                found=True,
                x=bx,
                y=by,
                w=bw,
                h=bh,
                confidence=1.0,
                template_name="btn_ok_inferred"
            )

        return None

    def _log_stats(self):
        st = self.stats
        total = st["caught"] + st["missed_after_click"] + st["lost_before_bite"]
        rate = (100.0 * st["caught"] / total) if total else 0.0
        vetoed = st.get('static_vetoed', 0)
        shape_v = st.get('shape_vetoed', 0)
        self._log(f"📊 [THỐNG KÊ] Thấy '!': {st['bite_seen']} | Dính cá: {st['caught']} | "
                  f"Giật nhưng hụt: {st['missed_after_click']} | Mất cá chưa '!': {st['lost_before_bite']} | "
                  f"Chặn vật tĩnh: {vetoed} | Chặn bong bóng: {shape_v} => tỉ lệ dính {rate:.0f}% ({st['caught']}/{total})")
        if hasattr(self, "bite_policy"):
            self._log(f"🔬 [SHADOW:{self.bite_policy.mode}] Cá dính: detector đồng ý {st.get('caught_det_agreed', 0)}, bỏ sót {st.get('caught_det_missed', 0)} | "
                      f"Giật hụt: detector cũng bắn {st.get('miss_det_agreed', 0)}, đã chặn được {st.get('miss_det_rejected', 0)}")

    def _log(self, message: str):
        now = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{now}] {message}"
        print(formatted)
        if self.on_log:
            try:
                self.on_log(formatted)
            except Exception:
                pass

    def _set_state(self, new_state: BotState):
        self.state = new_state
        if self.on_state_change:
            try:
                self.on_state_change(new_state)
            except Exception:
                pass

    def _auto_detect_state(self) -> BotState:
        """
        Tự động quét màn hình theo vùng (RegionScanner) để xác định ngữ cảnh hiện tại (< 300ms):
        1. Bảng 'Thông tin sửa chữa' / 'Sửa chữa hoàn thành' / Nút OK -> Vào thẳng BƯỚC SỬA CẦN
        2. Popup 'Bảo quản' -> Vào thẳng BƯỚC 3
        3. Nút giật cần -> Vào thẳng BƯỚC 2
        4. Nút quăng cần -> Vào BƯỚC 1
        """
        full_frame, offset = self.capturer.capture(None)

        first = self.scanner.first_found(
            full_frame,
            ["bag_toast", "bag_speech", "repair_title", "done_title", "ok", "keep", "card_open", "card_prompt", "card_title", "card_ok", "reel", "cast"],
            origin=offset
        )

        if first is not None:
            name, hit = first
            if name in ("bag_toast", "bag_speech"):
                self._log(f"🛑 [TÚI ĐỒ ĐÃ ĐẦY] Phát hiện thông báo túi đầy ({hit.score:.2f} - {hit.template})! Dừng bot an toàn để bạn đi bán cá.")
                try:
                    import winsound
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                except Exception:
                    pass
                return BotState.STOPPED
            elif name in ("repair_title", "done_title", "ok"):
                self._log(f"🔍 [NGỮ CẢNH] Phát hiện bảng sửa cần ({hit.score:.2f} - {hit.template})! Vào BƯỚC SỬA CẦN.")
                return BotState.STAGE_REPAIR
            elif name in ("keep", "card_open", "card_prompt", "card_title", "card_ok"):
                self._log(f"🔍 [NGỮ CẢNH] Phát hiện popup nhận cá / mở gói thẻ ({hit.score:.2f} - {hit.template})! Vào thẳng BƯỚC 3.")
                return BotState.STAGE_3
            elif name == "reel":
                self._last_img1_match = MatchResult(found=True, confidence=hit.score, x=hit.x, y=hit.y, w=hit.w, h=hit.h, template_name=hit.template)
                self._log(f"🔍 [NGỮ CẢNH] Phát hiện đang câu cá dở dang (nút giật cần: {hit.score:.2f})! Vào thẳng BƯỚC 2 (Chờ cá cắn).")
                return BotState.STAGE_2
            elif name == "cast":
                self._last_img1_match = MatchResult(found=True, confidence=hit.score, x=hit.x, y=hit.y, w=hit.w, h=hit.h, template_name=hit.template)
                self._log(f"🔍 [NGỮ CẢNH] Phát hiện nút quăng cần sẵn sàng ({hit.score:.2f} - {hit.template})! Vào BƯỚC 1 (Quăng cần).")
                return BotState.STAGE_1

        hits = self.scanner.scan(full_frame, origin=offset)
        self._log(f"🔍 [NGỮ CẢNH] Khởi động tuần tự: QuăngCần={hits['cast'].score:.2f}, GiậtCần={hits['reel'].score:.2f}, BảoQuản={hits['keep'].score:.2f}, SửaCần={hits['repair_title'].score:.2f}")
        return BotState.STAGE_1

    def start(self):
        """Starts or resumes the bot loop."""
        if self._thread is not None and self._thread.is_alive():
            if self._pause_event.is_set():
                self._pause_event.clear()
                self._log("▶ ĐÃ TIẾP TỤC CHẠY (Resumed).")
            return

        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._log("🚀 BOT ĐÃ KHỞI ĐỘNG (Nhấn F2 để dừng bất kỳ lúc nào).")

    def stop(self):
        """Stops the bot loop cleanly."""
        self._stop_event.set()
        self._set_state(BotState.STOPPED)
        self._log("⏹ BOT ĐÃ DỪNG (Nhấn F1 để bắt đầu lại).")

    def _sleep_with_check(self, seconds: float) -> bool:
        """
        Sleeps in small chunks while checking for stop or pause flags.
        Returns False if stop was triggered during sleep.
        """
        step = 0.05
        elapsed = 0.0
        while elapsed < seconds:
            if self._stop_event.is_set():
                return False
            time.sleep(step)
            elapsed += step
        return True

    def _run_loop(self):
        """Main State Machine Loop."""
        try:
            cfg = self.config
            templates = cfg.get("templates", {})
            thresholds = cfg.get("thresholds", {})
            regions = cfg.get("regions", {})
            delays = cfg.get("delays", {})

            scan_interval = delays.get("scan_interval_ms", 100) / 1000.0
            hold_min = delays.get("click_hold_duration_min_ms", 40)
            hold_max = delays.get("click_hold_duration_max_ms", 90)
            margin = delays.get("click_margin_px", 5)

            self._set_state(self._auto_detect_state())

            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    time.sleep(0.1)
                    continue

                # ========================================================
                # STAGE 1: Tìm nút Quăng Cần và Click (Fast Path)
                # ========================================================
                if self.state == BotState.STAGE_1:
                    # Duy trì điểm neo liên tục (Sticky Anchor), KHÔNG reset plate_tracker giữa các vòng câu
                    self._log(f"[{self.state.value}] Đang quét tìm nút Quăng Cần...")
                    img1_path = templates.get("b1_quang_can") or templates.get("img1")
                    thresh1 = thresholds.get("b1_quang_can") or thresholds.get("img1", 0.70)

                    sua_can_title = templates.get("b4_sua_can_title", "assets/B4_SuaCan/title")
                    hoan_thanh_title = templates.get("b4_hoan_thanh_title", "assets/B4_SuaCan/title_hoan_thanh")
                    ok_btn = templates.get("b4_ok_btn", "assets/B4_SuaCan/btn_ok")
                    thresh_repair = thresholds.get("b4_sua_can", 0.70)
                    thresh_ok = thresholds.get("b4_btn_ok", 0.70)

                    stage1_start = time.time()
                    last_modal_check = 0.0
                    cast_attempt = 0

                    while not self._stop_event.is_set() and not self._pause_event.is_set():
                        frame_full, off_full = self.capturer.capture(None)
                        now = time.time()
                        elapsed_st1 = now - stage1_start

                        # Sensory Closed-Loop: Quét toàn diện thực tế màn hình qua RegionScanner (<25ms)
                        sc1 = self.scanner.scan(frame_full, origin=off_full)

                        # 0. ƯU TIÊN TUYỆT ĐỐI: Túi đồ đầy -> Dừng bot ngay lập tức
                        if sc1["bag_toast"].found or sc1["bag_speech"].found:
                            self._log("🛑 [TÚI ĐỒ ĐÃ ĐẦY] Game báo túi đã đầy! Dừng bot an toàn để bạn đi bán cá.")
                            try:
                                import winsound
                                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                            except Exception:
                                pass
                            self.stop()
                            break

                        # 1. ƯU TIÊN SỐ 1 (CHỐNG KẸT MODAL): Bảng Sửa Cần đang mở -> Vào thẳng BƯỚC SỬA CẦN
                        if sc1["repair_title"].found or sc1["done_title"].found or sc1["ok"].found:
                            self._log("⚠️ [TỰ SỬA TRẠNG THÁI] Bảng sửa cần đang mở! -> Vào thẳng BƯỚC SỬA CẦN...")
                            self._set_state(BotState.STAGE_REPAIR)
                            break

                        # 2. ƯU TIÊN SỐ 2 (CHỐNG KẸT MODAL): Popup Bảo Quản hoặc Gói Thẻ đang mở!
                        if sc1["keep"].found or sc1["card_open"].found or sc1["card_prompt"].found or sc1["card_ok"].found or sc1["card_title"].found:
                            self._log(f"🔄 [TỰ SỬA TRẠNG THÁI] Phát hiện Popup nhận cá / Gói Thẻ đang mở! -> Vào thẳng BƯỚC 3...")
                            self._set_state(BotState.STAGE_3)
                            break

                        # 3. ƯU TIÊN SỐ 3: Nút Giật Cần đang hiện -> Thực tế đang câu dở dang!
                        if sc1["reel"].found:
                            self._last_img1_match = MatchResult(found=True, confidence=sc1["reel"].score, x=sc1["reel"].x, y=sc1["reel"].y, w=sc1["reel"].w, h=sc1["reel"].h, template_name=sc1["reel"].template)
                            self._log(f"🔄 [TỰ SỬA TRẠNG THÁI] Nút Giật Cần đang hiện ({sc1['reel'].score:.2f})! Màn hình báo đang câu cá -> Vào thẳng BƯỚC 2...")
                            self._set_state(BotState.STAGE_2)
                            break

                        # 4. ƯU TIÊN SỐ 4: Nút Quăng Cần sẵn sàng -> Quăng cần (chỉ khi không có modal chặn)
                        if sc1["cast"].found:
                            cast_hit = sc1["cast"]
                            cand_center = (cast_hit.x + cast_hit.w // 2, cast_hit.y + cast_hit.h // 2)
                            verify_res = self.cast_guard.verify_candidate(cand_center, cast_hit.score)
                            if not verify_res.valid:
                                self._log(f"⚠️ [CHẶN CLICK NHẦM] Từ chối vị trí quăng cần tại {cand_center} ({verify_res.veto_reason})")
                                time.sleep(0.04)
                                continue

                            self.watchdog.note_progress("cast")
                            match1 = MatchResult(found=True, confidence=cast_hit.score, x=cast_hit.x, y=cast_hit.y, w=cast_hit.w, h=cast_hit.h, template_name=cast_hit.template)
                            self._last_img1_match = match1

                            cast_attempt += 1
                            self.input_sim.focus_game_window()

                            if cast_attempt == 1:
                                self._log(f" -> Đã thấy nút Quăng Cần ({match1.template_name}, độ khớp {match1.confidence:.2f}). Bấm phím F để quăng cần (DirectInput, né chuột 100%)...")
                                self.input_sim.press_key("f", hold_min_ms=60, hold_max_ms=100)
                            else:
                                click_pt = self.cast_guard.get_safe_click_point(cand_center, max_radius=20)
                                self._log(f"⚠️ [HỤT LỆNH PHÍM F - LẦN {cast_attempt}] Game hiển thị icon chuột (🖱️). Kích hoạt Fallback click chuột an toàn tại {click_pt} (né nút X > 115px)...")
                                self.input_sim.click(click_pt[0], click_pt[1])

                            # Phản xạ nhận thức vòng kín (Closed-Loop Post-Action Verification):
                            # Không nhảy mù quáng sang BƯỚC 2. Quan sát phản hồi thực tế của game ngay sau khi bấm F!
                            post_cast_start = time.time()
                            cast_confirmed = False
                            cast_disappear_count = 0
                            while (time.time() - post_cast_start) < 1.4:
                                if self._stop_event.is_set() or self._pause_event.is_set():
                                    break
                                time.sleep(0.06)
                                f_post, o_post = self.capturer.capture(None)
                                sc_post = self.scanner.scan(f_post, origin=o_post)

                                # 0. Túi đồ đã đầy (game hiện thông báo ngay sau khi bấm F)
                                if sc_post["bag_toast"].found or sc_post["bag_speech"].found:
                                    self._log("🛑 [TÚI ĐỒ ĐÃ ĐẦY SAU KHI THẢ CẦN] Game báo 'Không thể bắt cá vì túi đã đầy'! Dừng bot an toàn để bạn đi bán cá.")
                                    try:
                                        import winsound
                                        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                                    except Exception:
                                        pass
                                    self.stop()
                                    cast_confirmed = True
                                    break

                                # 1. Bảng sửa cần hoặc modal xuất hiện ngay sau khi bấm F (cần hỏng)
                                if sc_post["repair_title"].found or sc_post["done_title"].found or sc_post["ok"].found:
                                    self._log("⚠️ [PHẢN HỒI SAU KHI BẤM F] Game hiện bảng Sửa Cần! Vào thẳng bước sửa cần, KHÔNG quăng tiếp...")
                                    self._set_state(BotState.STAGE_REPAIR)
                                    cast_confirmed = True
                                    break

                                # 2. Popup bảo quản xuất hiện
                                if sc_post["keep"].found:
                                    self._log("🔄 [PHẢN HỒI SAU KHI BẤM F] Popup 'Bảo quản' xuất hiện! Vào thẳng BƯỚC 3...")
                                    self._set_state(BotState.STAGE_3)
                                    cast_confirmed = True
                                    break

                                # 3. Nút Giật Cần xuất hiện -> Cần đã ở dưới nước thực sự
                                if sc_post["reel"].found:
                                    self._log(f"🎣 [XÁC NHẬN VÀO NƯỚC] Đã thấy nút Giật Cần ({sc_post['reel'].score:.2f}) -> Cần đã ở dưới nước, vào BƯỚC 2!")
                                    self._set_state(BotState.STAGE_2)
                                    cast_confirmed = True
                                    break

                                # 4. Nút Quăng Cần biến mất liên tục >= 3 nhịp -> Phao đang bay ra mặt nước thực sự
                                if not sc_post["cast"].found:
                                    cast_disappear_count += 1
                                    if cast_disappear_count >= 3:
                                        self._log("🎣 [XÁC NHẬN VÀO NƯỚC] Nút Quăng Cần đã biến mất -> Chuyển sang BƯỚC 2!")
                                        self._set_state(BotState.STAGE_2)
                                        cast_confirmed = True
                                        break
                                else:
                                    cast_disappear_count = 0

                            if cast_confirmed:
                                time.sleep(random.uniform(0.15, 0.25))
                                break

                            # Nếu sau 1.2s nút quăng cần vẫn còn và không có gì thay đổi: bấm F bị miss / lag
                            self._log("⚠️ [HỤT LỆNH / LAG] Bấm F nhưng nút Quăng Cần vẫn còn nguyên và không có phản hồi! Giữ BƯỚC 1 thử lại...")
                            time.sleep(random.uniform(0.3, 0.5))

                        # Watchdog kiểm tra kẹt
                        if self.watchdog.warn_due(30):
                            stuck_dir = os.path.join("corpus", "stuck")
                            os.makedirs(stuck_dir, exist_ok=True)
                            ts = datetime.now().strftime("%H%M%S")
                            stuck_path = os.path.join(stuck_dir, f"{ts}.jpg")
                            cv2.imwrite(stuck_path, frame_full)
                            self._log(f"⚠️ [KẸT >30s] Bot chưa có tiến triển trong 30s (Quăng={sc1['cast'].score:.2f})! Đã lưu ảnh hiện tại vào {stuck_path}")

                        if self.watchdog.stalled():
                            self._log("🚨 [KẸT >90s] Không có tiến triển trong 90s! Tự động dừng bot để bảo vệ an toàn.")
                            try:
                                import winsound
                                winsound.Beep(1000, 600)
                            except Exception:
                                pass
                            self.stop()
                            break

                        # Tự phục hồi: Nếu tìm >3.5s chưa có kết quả, reset lại stage1_start
                        if elapsed_st1 > 3.5:
                            stage1_start = time.time()

                        time.sleep(0.04)


                # ========================================================
                # STAGE 2: Chờ Dấu Chấm Than xuất hiện trên đầu nhân vật -> Giật cần
                # ========================================================
                elif self.state == BotState.STAGE_2:
                    img2_path = templates.get("b2_cham_than") or templates.get("img2")
                    thresh2 = thresholds.get("b2_cham_than") or thresholds.get("img2", 0.72)
                    timeout_sec = delays.get("stage2_wait_timeout_sec", 120)
                    relocate_interval = 1.5  # 1.5s keeps loop fast (<15ms) to catch 400ms '!' without lag spikes

                    start_wait_time = time.time()
                    found_img2 = False

                    # The ROI is NEVER taken from a template-matching fallback. It is either located by the
                    # character's name (this cast) or the last ROI that was located the same way (seed).
                    effective_region = self._last_good_head_roi
                    is_first_locate = True
                    last_locate_time = 0.0
                    top_bite_score = 0.0
                    top_bite_template = "None"
                    top_matches_dict: Dict[str, float] = {}
                    last_head_frame: Optional[np.ndarray] = None
                    last_warn_time = 0.0
                    last_repair_check = 0.0
                    last_roi_move_time = start_wait_time
                    roi_jump_streak = 0
                    pending_jump_roi = None

                    # Reel button: a fixed UI element that is already on screen for the whole wait.
                    # LOCATE it now (argmax in the lower-right area, low floor) instead of re-detecting it
                    # after the bite, where a miss cost 0.8-0.9s (multi-scale search) - longer than the bite itself.
                    btn_path = templates.get("b2_giat_can") or templates.get("img2_btn") or templates.get("img1")
                    reel_floor = thresholds.get("b2_giat_can_locate", 0.40)
                    reel_match: Optional[MatchResult] = None

                    self.static_veto.reset()
                    self.bite_detector.reset()
                    self.bite_policy.reset()
                    last_det_hit = None
                    self.recorder.begin_cast(effective_region, {"thresh2": thresh2})
                    self._log(f"[{self.state.value}] Đang canh chừng Dấu Chấm Than (tần số cao, tối đa {timeout_sec}s)...")

                    fast_scan_interval = 0.03  # 30ms (~33 FPS) để phản xạ tức thì khi cá cắn
                    scan_count = 0
                    last_scan_t = None
                    max_scan_gap = 0.0  # watchdog: khoảng nghỉ dài nhất giữa 2 lần quét "!" (phải << 400ms)

                    # Peak Candidate Memory: Ghi nhớ frame đạt điểm nghi ngờ cao nhất lúc '!' lóe lên
                    best_candidate_head: Optional[np.ndarray] = None
                    best_candidate_score: float = 0.0
                    best_candidate_template: str = ""
                    best_candidate_time: float = 0.0
                    best_candidate_reason: str = ""
                    while not self._stop_event.is_set() and not self._pause_event.is_set():
                        # Kiểm tra timeout
                        if (time.time() - start_wait_time) > timeout_sec:
                            self._log(f"[HẾT GIỜ] Đã đợi {timeout_sec}s mà không thấy Dấu Chấm Than! Tự động kiểm tra lại màn hình...")
                            self.recorder.dump("timeout")
                            try:
                                timeout_full = self.capturer.capture(None)[0]
                            except Exception:
                                timeout_full = None
                            p_info = {
                                "state": track_res.state if 'track_res' in locals() else "UNKNOWN",
                                "name": loc.name if ('loc' in locals() and loc) else "N/A",
                                "score": loc.score if ('loc' in locals() and loc) else 0.0,
                                "plate": loc.plate.to_tuple() if ('loc' in locals() and loc) else None,
                                "scale": loc.scale if ('loc' in locals() and loc) else 1.0,
                            }
                            b_info = {
                                "threshold": thresh2,
                                "top_score": top_bite_score,
                                "top_template": top_bite_template,
                                "matches": top_matches_dict,
                            }
                            b_info["peak_time_s"] = best_candidate_time
                            b_info["peak_reason"] = best_candidate_reason or (
                                "Không có tín hiệu giống dấu '!'" if top_bite_score < 0.50 else f"Điểm cao nhất {top_bite_score:.2f} < Ngưỡng yêu cầu {thresh2:.2f}"
                            )
                            case_id = self.missed_logger.log_incident(
                                full_frame=timeout_full,
                                head_roi_frame=best_candidate_head if best_candidate_head is not None else last_head_frame,
                                head_roi_coords=effective_region,
                                reason=f"Hết thời gian chờ {timeout_sec}s mà không thấy dấu '!'",
                                plate_info=p_info,
                                bite_analysis=b_info,
                                duration_s=time.time() - start_wait_time,
                                scan_count=scan_count,
                            )
                            self._log(f"📸 [GHI NHẬN HỤT CÁ] Đã chụp ảnh và lưu bộ debug vào 'debug_hut_ca' (Mã: {case_id})")
                            self._set_state(self._auto_detect_state())
                            break

                        # Re-locate the character every 0.5s: camera/character may move after the cast.
                        now = time.time()
                        if now - last_locate_time >= relocate_interval:
                            last_locate_time = now
                            full_frame, full_offset = self.capturer.capture(None)

                            # 1. Định vị nhân vật qua PlateTracker (P1a: hue-mask, blob-based, windowed)
                            track_res = self.plate_tracker.update(full_frame, full_offset)
                            if track_res.state == "LOST":
                                self.static_veto.reset()
                                self.bite_detector.reset()
                            if track_res.ambiguous:
                                self._log(" -> [CẢNH BÁO] Phát hiện nhiều người chơi có tên màu xanh tương tự. Giữ nguyên mục tiêu hiện tại.")
                            loc = track_res.hit
                            if (track_res.state == "TRACKING" or effective_region is None or is_first_locate) and loc is not None:
                                # P1b: scaled_head_roi theo zoom của camera (above=145px để ôm sát đầu, pad_x=12 gọn gàng)
                                s_roi = scaled_head_roi(loc.plate, loc.scale, above=145, pad_x=12)
                                # Giới hạn bề ngang tối đa 135px quanh trục giữa để loại trừ triệt để người lạ đứng chen chúc
                                cx = loc.plate.x + loc.plate.w // 2
                                max_w = int(round(135 * loc.scale))
                                if s_roi.w > max_w:
                                    s_roi = Rect(cx - max_w // 2, s_roi.y, max_w, s_roi.h)
                                new_roi = self._clamp_region(s_roi.to_tuple(), full_frame, full_offset)
                                if effective_region is None or is_first_locate:
                                    is_first_locate = False
                                    effective_region = self._last_good_head_roi = new_roi
                                    self._log(f" -> Định vị nhân vật '{loc.name}' (độ khớp {loc.score:.2f}, scale {loc.scale:.2f}): vùng đầu = {new_roi}")
                                    last_roi_move_time = now
                                    self.bite_detector.reset()
                                    roi_jump_streak = 0
                                    pending_jump_roi = None
                                else:
                                    dist = math.hypot(new_roi[0] - effective_region[0], new_roi[1] - effective_region[1])
                                    if dist <= 35:
                                        # Dao động vi mô / camera rung nhẹ: bám theo mượt mà, KHÔNG reset bộ đếm thời gian
                                        effective_region = self._last_good_head_roi = new_roi
                                        roi_jump_streak = 0
                                        pending_jump_roi = None
                                    else:
                                        # Nhảy vị trí lớn (> 35px): Có thể do người lạ đi ngang hoặc xoay camera
                                        if pending_jump_roi is not None and math.hypot(new_roi[0] - pending_jump_roi[0], new_roi[1] - pending_jump_roi[1]) <= 25:
                                            roi_jump_streak += 1
                                        else:
                                            roi_jump_streak = 1
                                            pending_jump_roi = new_roi

                                        # Chỉ chấp nhận góc mới nếu duy trì liên tục >= 2 nhịp quét (1.0s)
                                        if roi_jump_streak >= 2:
                                            self._log(f" -> [XOAY CAMERA] Xác nhận vị trí mới của '{loc.name}': vùng đầu = {new_roi}")
                                            effective_region = self._last_good_head_roi = new_roi
                                            last_roi_move_time = now
                                            self.bite_detector.reset()
                                            roi_jump_streak = 0
                                            pending_jump_roi = None

                            # 2. Định vị nút Giật Cần trong vùng thao tác (chỉ tìm khi chưa định vị)
                            if reel_match is None:
                                fh, fw = full_frame.shape[:2]
                                rx, ry = int(fw * 0.65), int(fh * 0.55)
                                action_roi = full_frame[ry:, rx:]
                                action_offset = (full_offset[0] + rx, full_offset[1] + ry)
                                cand = self.matcher.match(action_roi, btn_path, threshold=reel_floor,
                                                          screen_offset=action_offset,
                                                          multi_scale=False)
                                if cand.found:
                                    self._log(f" -> Đã định vị nút Giật Cần tại ({cand.x}, {cand.y}) {cand.w}x{cand.h} (độ khớp {cand.confidence:.2f})")
                                    reel_match = cand

                            # 3. Kiểm tra sự cố: Cần hỏng, Bảo quản sẵn, hoặc cần giật lên lại (< 25ms qua RegionScanner)
                            if (now - start_wait_time) > 1.5:
                                # ƯU TIÊN SỐ 1: Túi đầy, Bảng sửa cần hoặc Popup bảo quản đang mở
                                sc2 = self.scanner.scan(full_frame, origin=full_offset, names=["bag_toast", "bag_speech", "repair_title", "repair_btn", "done_title", "ok", "keep", "cast"])
                                if sc2["bag_toast"].found or sc2["bag_speech"].found:
                                    self._log("🛑 [TÚI ĐỒ ĐÃ ĐẦY] Game báo túi đã đầy! Dừng bot an toàn để bạn đi bán cá.")
                                    try:
                                        import winsound
                                        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                                    except Exception:
                                        pass
                                    self.stop()
                                    break

                                if sc2["repair_title"].found or sc2["done_title"].found or sc2["ok"].found:
                                    self._log("⚠️ Cần câu bị hỏng (phát hiện bảng sửa cần)! Vào bước sửa cần...")
                                    self._set_state(BotState.STAGE_REPAIR)
                                    break

                                if sc2["keep"].found:
                                    self._log(f"🔄 [TỰ SỬA TRẠNG THÁI] Popup 'Bảo quản' ({sc2['keep'].score:.2f}) đã mở sẵn! Vào thẳng BƯỚC 3...")
                                    self._set_state(BotState.STAGE_3)
                                    break

                                # ƯU TIÊN SỐ 2: Nút quăng cần hiện lại (cần bị giật lên / chưa vào câu) - qua RegionScanner siêu tốc
                                if sc2["cast"].found:
                                    wait_duration = time.time() - start_wait_time
                                    if wait_duration < 6.0:
                                        # Cần chưa ở dưới nước đủ lâu -> Đây là hụt lệnh quăng cần (chưa quăng thành công), không tính là mất cá
                                        self._log(f"⚠️ [HỤT LỆNH QUĂNG CẦN] Cần chưa vào nước sau {wait_duration:.1f}s (Nút Quăng Cần vẫn còn, khớp {sc2['cast'].score:.2f})! Quay về BƯỚC 1 quăng lại...")
                                        self._set_state(BotState.STAGE_1)
                                        break

                                    self.stats["lost_before_bite"] += 1
                                    p_info = {
                                        "state": track_res.state,
                                        "name": loc.name if loc else "N/A",
                                        "score": loc.score if loc else 0.0,
                                        "plate": loc.plate.to_tuple() if loc else None,
                                        "scale": loc.scale if loc else 1.0,
                                    }
                                    b_info = {
                                        "threshold": thresh2,
                                        "top_score": top_bite_score,
                                        "top_template": top_bite_template,
                                        "matches": top_matches_dict,
                                    }
                                    # Bổ sung thông tin phân tích khoảnh khắc đỉnh
                                    b_info["peak_time_s"] = best_candidate_time
                                    b_info["peak_reason"] = best_candidate_reason or (
                                        "Không có tín hiệu giống dấu '!'" if top_bite_score < 0.50 else f"Điểm cao nhất {top_bite_score:.2f} < Ngưỡng yêu cầu {thresh2:.2f}"
                                    )
                                    case_id = self.missed_logger.log_incident(
                                        full_frame=full_frame,
                                        head_roi_frame=best_candidate_head if best_candidate_head is not None else last_head_frame,
                                        head_roi_coords=effective_region,
                                        reason="Cần câu bị giật lên lại sau khi ngâm mồi (Cá thoát / Chưa phát hiện dấu '!')",
                                        plate_info=p_info,
                                        bite_analysis=b_info,
                                        duration_s=wait_duration,
                                        scan_count=scan_count,
                                    )
                                    self._log(f"⚠️ Cần câu bị giật lên lại / cá thoát (Nút Quăng Cần hiện lại sau {wait_duration:.1f}s, khớp {sc2['cast'].score:.2f})!")
                                    self._log(f"📸 [GHI NHẬN HỤT CÁ] Đã chụp ảnh và lưu bộ debug vào 'debug_hut_ca' (Mã: {case_id})")
                                    self._log_stats()
                                    self._set_state(BotState.STAGE_1)
                                    break

                        if effective_region is None:
                            # Never guess: without a located character there is nothing meaningful to scan.
                            if now - last_warn_time > 5.0:
                                self._log(" -> [CẢNH BÁO] Chưa định vị được nhân vật (tên bị che hoặc ngoài màn hình). Đang thử lại...")
                                last_warn_time = now
                            time.sleep(fast_scan_interval)
                            continue

                        frame, offset = self.capturer.capture(effective_region)
                        last_head_frame = frame

                        # P1b Step C: StaticVeto quan sát mỗi nhịp quét
                        self.static_veto.observe(frame, offset)

                        # P2: Cho BiteDetector ăn mỗi khung hình (không phụ thuộc màu)
                        t_det = time.time()
                        det_raw = self.bite_detector.update(frame, t_det, scale=(loc.scale if loc is not None else None))

                        # P1b Step B & C: Quét '!' theo scale của PlateTracker và kiểm tra StaticVeto
                        match2 = MatchResult(found=False)
                        if (now - start_wait_time) >= 1.5 and (now - last_roi_move_time) >= 0.6:
                            scale = loc.scale if loc is not None else 1.0
                            gray_strip = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            bm = self.bite_bank.best_match(gray_strip, scale)
                            if bm is not None:
                                top_matches_dict[bm.name] = max(top_matches_dict.get(bm.name, 0.0), bm.score)
                                if bm.score > top_bite_score:
                                    top_bite_score = bm.score
                                    top_bite_template = bm.name

                                # Ghi nhận vào bộ nhớ khoảnh khắc đỉnh (Peak Candidate)
                                if bm.score > best_candidate_score:
                                    best_candidate_score = bm.score
                                    best_candidate_template = bm.name
                                    best_candidate_head = frame.copy()
                                    best_candidate_time = round(now - start_wait_time, 2)
                                    if bm.score < thresh2:
                                        best_candidate_reason = f"Điểm số ({bm.score:.2f}) thấp hơn ngưỡng kích hoạt ({thresh2:.2f})"

                            if bm is not None and bm.score >= thresh2:
                                # Guardrail B: Dấu '!' trong Play Together luôn là icon phát quang (V >= 135)
                                # Bỏ qua vệt bùn tối / bóng râm / mặt nước ban đêm bị TM_CCOEFF_NORMED phóng đại
                                patch_bgr = frame[bm.y:bm.y + bm.h, bm.x:bm.x + bm.w]
                                vetoed_reason = None
                                if patch_bgr.size > 0:
                                    hsv_p = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
                                    max_v = int(hsv_p[:, :, 2].max())
                                    if max_v < 135:
                                        vetoed_reason = f"Bị loại bởi bộ lọc phát quang (Độ sáng V={max_v} < 135, nghi bóng râm/bùn tối)"
                                        bm = None
                                if bm is not None:
                                    gx, gy = offset[0] + bm.x, offset[1] + bm.y
                                    veto_res = self.static_veto.is_static(Rect(gx, gy, bm.w, bm.h), bm.score, bm.template)
                                    if veto_res.static:
                                        self.stats["static_vetoed"] = self.stats.get("static_vetoed", 0) + 1
                                        self._log(f" -> [CHẶN VẬT TĨNH] Bỏ qua '{bm.name}' (khớp {bm.score:.2f}) tại ({gx}, {gy}) vì đã tồn tại từ {veto_res.ref_age:.1f}s trước (điểm cũ {veto_res.ref_score:.2f})")
                                        vetoed_reason = f"Bị chặn bởi StaticVeto (Vật thể tĩnh tồn tại từ {veto_res.ref_age:.1f}s trước)"
                                    else:
                                        # P1d: Chặn bong bóng mong muốn/nhiệm vụ (cloud-like veto)
                                        cloud_res = is_cloud_like(frame, Rect(bm.x, bm.y, bm.w, bm.h))
                                        if cloud_res.cloud_like:
                                            self.stats["shape_vetoed"] = self.stats.get("shape_vetoed", 0) + 1
                                            self._log(f" -> [CHẶN BONG BÓNG] Bỏ qua bong bóng mong muốn/nhiệm vụ (đám mây {cloud_res.blob.w}x{cloud_res.blob.h}, aspect {cloud_res.aspect:.2f})")
                                            vetoed_reason = f"Bị chặn bởi CloudVeto (Bong bóng dạng mây {cloud_res.blob.w}x{cloud_res.blob.h})"
                                        else:
                                            match2 = MatchResult(found=True, confidence=bm.score, x=gx, y=gy,
                                                                 w=bm.w, h=bm.h, template_name=bm.name)
                                            if hasattr(self, "outcome_recorder") and self.outcome_recorder is not None:
                                                self.outcome_recorder.note_hit(x=gx, y=gy, w=bm.w, h=bm.h,
                                                                               score=round(bm.score, 3), template=bm.name)
                                if vetoed_reason and bm is not None and bm.score >= best_candidate_score:
                                    best_candidate_reason = vetoed_reason

                        # P2: Quyết định kích hoạt giật cần qua TriggerPolicy (off | shadow | confirm | primary)
                        det_hit = det_raw
                        armed = (now - start_wait_time) >= 1.5 and (now - last_roi_move_time) >= 0.6
                        if det_hit is not None and (not armed or is_cloud_like(
                                frame, Rect(det_hit.x, det_hit.y, det_hit.w, det_hit.h)).cloud_like):
                            det_hit = None
                        if det_hit is not None:
                            last_det_hit = det_hit
                        t_now = time.time()
                        self.bite_policy.note_detector(t_now, det_hit is not None)
                        self.bite_policy.note_template(t_now, match2.found)
                        if det_hit is not None and not match2.found:
                            self._log(f" -> [SHADOW:{self.bite_policy.mode}] Detector thấy '!' mà template không thấy: "
                                      f"tương phản {det_hit.contrast:.0f}, {det_hit.w}x{det_hit.h}, fill {det_hit.fill:.2f}")
                        if self.bite_policy.should_reel(t_now):
                            if not match2.found and last_det_hit is not None:
                                d = last_det_hit
                                match2 = MatchResult(found=True, confidence=1.0, x=offset[0] + d.x, y=offset[1] + d.y,
                                                     w=d.w, h=d.h, template_name="bite_detector")
                        else:
                            match2 = MatchResult(found=False)

                        scan_t = time.time()
                        if last_scan_t is not None:
                            max_scan_gap = max(max_scan_gap, scan_t - last_scan_t)
                        last_scan_t = scan_t

                        # --- Step 0 diagnostics: record what the bot sees, every frame ---
                        self.recorder.record(frame, match2.confidence, match2.template_name)
                        scan_count += 1
                        if scan_count % 8 == 0:
                            self.recorder.record_full(self.capturer.capture(None)[0])
                        mark = self.recorder.consume_manual_request()
                        if mark is not None:
                            folder = self.recorder.dump("manual_f3", mark_time=mark)
                            self._log(f" -> [F3] Đã lưu bằng chứng vào: {folder}")

                        if match2.found:
                            self._reel_det_agreed = self.bite_policy.detector_recent(time.time())
                            self._last_head_frame_at_bite = frame.copy() if frame is not None else None
                            self._last_bite_score = match2.confidence
                            self._last_bite_tpl = match2.template_name
                            self.stats["bite_seen"] += 1
                            self._log(f" -> PHÁT HIỆN DẤU CHẤM THAN TRÊN ĐẦU! (Độ khớp: {match2.confidence:.2f} [{match2.template_name}] tại {match2.x}, {match2.y})")
                            found_img2 = True
                            break

                        time.sleep(fast_scan_interval)

                    if max_scan_gap > 0.25:
                        self._log(f" -> [HIỆU NĂNG] Vòng quét '!' từng đứng {max_scan_gap*1000:.0f}ms (phải < 250ms, dấu '!' chỉ hiện 400-600ms)")

                    if found_img2 and not self._stop_event.is_set():
                        # Thực hiện hành động: Click vào nút giật cần (đã định vị sẵn trong lúc chờ -> phản xạ tức thì)
                        target_match = reel_match
                        if target_match is None and btn_path:
                            # Fallback: tìm trong vùng thao tác góc dưới bên phải (chỉ single-scale)
                            full_frame, full_offset = self.capturer.capture(None)
                            fh, fw = full_frame.shape[:2]
                            rx, ry = int(fw * 0.65), int(fh * 0.55)
                            target_match = self.matcher.match(
                                full_frame[ry:, rx:],
                                btn_path,
                                threshold=reel_floor,
                                screen_offset=(full_offset[0] + rx, full_offset[1] + ry),
                                multi_scale=False
                            )

                        # Khi đã thấy '!', phím F giật cần trực tiếp mà không cần chờ nút giật cần
                        self.watchdog.note_progress("reel")
                        self._log(" -> KÍCH HOẠT: Bấm phím F để giật cần câu tức thì (DirectInput, phản xạ 0ms, né chuột 100%)!")
                        self.input_sim.press_key("f", hold_min_ms=30, hold_max_ms=60)

                        # Chuyển sang STAGE 3
                        self._set_state(BotState.STAGE_3)
                        self._log(f"==> Chuyển sang {self.state.value}")

                # ========================================================
                # STAGE 3: Chờ kết quả -> Tìm và click nút Bảo Quản hoặc phục hồi (Fast Path)
                # ========================================================
                elif self.state == BotState.STAGE_3:
                    wait_sec = delays.get("stage3_wait_delay_sec", 0.0)
                    if wait_sec > 0:
                        self._log(f"[{self.state.value}] Nghỉ {wait_sec}s trước khi quét kết quả...")
                        if not self._sleep_with_check(wait_sec):
                            break

                    img3_path = templates.get("b3_bao_quan") or templates.get("img3")
                    thresh3 = thresholds.get("b3_bao_quan") or thresholds.get("img3", 0.70)

                    img1_path = templates.get("b1_quang_can") or templates.get("img1")
                    thresh1 = thresholds.get("b1_quang_can") or thresholds.get("img1", 0.70)

                    sua_can_title = templates.get("b4_sua_can_title", "assets/B4_SuaCan/title")
                    hoan_thanh_title = templates.get("b4_hoan_thanh_title", "assets/B4_SuaCan/title_hoan_thanh")
                    ok_btn = templates.get("b4_ok_btn", "assets/B4_SuaCan/btn_ok")
                    thresh_repair = thresholds.get("b4_sua_can", 0.70)
                    thresh_ok = thresholds.get("b4_btn_ok", 0.70)

                    self._log(f"[{self.state.value}] Đang quét tìm nút Bảo Quản hoặc trạng thái phục hồi...")
                    timeout_stage3 = 10  # Đợi tối đa 10s (thay vì 25s mù quáng)
                    start_stage3 = time.time()

                    while not self._stop_event.is_set() and not self._pause_event.is_set():
                        now = time.time()
                        elapsed_st3 = now - start_stage3
                        if elapsed_st3 > timeout_stage3:
                            self._log("[HẾT GIỜ] Không tìm thấy nút Bảo Quản sau 10s! Tự động kiểm tra lại màn hình...")
                            self._set_state(self._auto_detect_state())
                            break

                        frame3, offset3 = self.capturer.capture(None)

                        # Closed-Loop Perception: Quét đồng thời toàn bộ mục tiêu qua RegionScanner (<25ms)
                        sc3 = self.scanner.scan(frame3, origin=offset3)

                        # 1. ƯU TIÊN SỐ 1: Nút Bảo Quản (bắt được cá bình thường)
                        if sc3["keep"].found:
                            self.watchdog.note_progress("keep")
                            self._log(f" -> Đã thấy popup Bảo Quản ({sc3['keep'].template}, độ khớp {sc3['keep'].score:.2f}).")
                            self._log(" -> [AN TOÀN TUYỆT ĐỐI] Gửi phím SPACE để nhận cá (triệt tiêu 100% rủi ro click nhầm Khoe hoặc Bán)...")
                            self.input_sim.press_key("space")
                            self.stats["caught"] += 1
                            self.stats["caught_det_agreed" if self._reel_det_agreed else "caught_det_missed"] += 1
                            self._log_stats()
                            if self.on_fish_caught:
                                try:
                                    self.on_fish_caught()
                                except Exception:
                                    pass
                            time.sleep(random.uniform(0.20, 0.35))
                            self._log("🔄 HOÀN THÀNH 1 VÒNG CÂU! Tự chuyển vòng mới...")
                            self._set_state(BotState.STAGE_1)
                            break

                        # 1b. GÓI THẺ CÂU ĐƯỢC (Túi thẻ xanh / vàng): Click 'MỞ' -> Space lật thẻ -> Space đóng KẾT QUẢ
                        if sc3["card_open"].found:
                            self.watchdog.note_progress("card_open")
                            self._log(f"🎁 [GÓI THẺ CÂU ĐƯỢC] Phát hiện túi thẻ ({sc3['card_open'].template}, khớp {sc3['card_open'].score:.2f})!")
                            self._log(" -> Click nút 'MỞ' để bắt đầu mở các thẻ...")
                            mo_x = sc3["card_open"].x + sc3["card_open"].w // 2
                            mo_y = sc3["card_open"].y + sc3["card_open"].h // 2
                            self.input_sim.click(mo_x, mo_y)
                            time.sleep(0.5)

                            # Vòng lặp mở từng thẻ qua phím Space cho đến khi hiện màn hình KẾT QUẢ
                            card_open_start = time.time()
                            last_space_press = 0.0
                            while not self._stop_event.is_set() and not self._pause_event.is_set():
                                if (time.time() - card_open_start) > 25.0:
                                    self._log("⚠️ [HẾT GIỜ] Mở gói thẻ quá 25s! Thoát ra kiểm tra lại...")
                                    break

                                f_card, o_card = self.capturer.capture(None)
                                sc_card = self.scanner.scan(f_card, origin=o_card, names=["card_prompt", "card_ok", "card_title", "cast"])

                                # Nếu đã về màn hình bình thường (nút quăng cần hiện):
                                if sc_card["cast"].found:
                                    self._log("🎣 Màn hình đã quay lại chế độ câu cá.")
                                    break

                                # Nếu tới màn hình KẾT QUẢ (có nút OK hoặc tiêu đề KẾT QUẢ):
                                if sc_card["card_ok"].found or sc_card["card_title"].found:
                                    self._log("✨ [HOÀN TẤT MỞ THẺ] Đã mở hết thẻ! Màn hình KẾT QUẢ xuất hiện -> Bấm SPACE để đóng...")
                                    self.input_sim.press_key("space")
                                    time.sleep(0.6)
                                    break

                                # Nếu đang có dòng chữ 'Xin hãy nhấn nút [Space]':
                                if sc_card["card_prompt"].found:
                                    now_card = time.time()
                                    if now_card - last_space_press >= 0.4:
                                        self._log("🃏 Phát hiện prompt 'Xin hãy nhấn nút [Space]' -> Bấm SPACE để mở thẻ tiếp theo...")
                                        self.input_sim.press_key("space")
                                        last_space_press = now_card
                                        time.sleep(0.35)
                                        continue

                                time.sleep(0.06)

                            self.stats["caught"] = self.stats.get("caught", 0) + 1
                            self._log("🔄 MỞ TÚI THẺ THÀNH CÔNG! Quay lại BƯỚC 1 để tiếp tục quăng cần...")
                            self._set_state(BotState.STAGE_1)
                            break

                        # Nếu màn hình đang ở giữa chừng hoạt cảnh mở thẻ (prompt Space):
                        if sc3["card_prompt"].found:
                            self._log("🃏 [GÓI THẺ ĐANG MỞ] Đang ở màn hình lật thẻ -> Bấm SPACE...")
                            self.input_sim.press_key("space")
                            time.sleep(0.4)

                        # Nếu màn hình đang ở KẾT QUẢ mở thẻ:
                        if sc3["card_ok"].found or sc3["card_title"].found:
                            self._log("✨ [KẾT QUẢ GÓI THẺ] Màn hình KẾT QUẢ đang mở -> Bấm SPACE để quay lại câu cá...")
                            self.input_sim.press_key("space")
                            time.sleep(0.6)
                            self._set_state(BotState.STAGE_1)
                            break

                        # 0. Túi đồ đã đầy
                        if sc3["bag_toast"].found or sc3["bag_speech"].found:
                            self._log("🛑 [TÚI ĐỒ ĐÃ ĐẦY] Game báo túi đã đầy! Dừng bot an toàn để bạn đi bán cá.")
                            try:
                                import winsound
                                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                            except Exception:
                                pass
                            self.stop()
                            break

                        # 2. ƯU TIÊN SỐ 2: Cần câu bị hỏng (Modal Sửa Cần xuất hiện khi giật cần)
                        if sc3["repair_title"].found or sc3["done_title"].found or sc3["ok"].found:
                            self._log("⚠️ Cần câu bị hỏng (phát hiện bảng sửa cần)! Vào bước sửa cần...")
                            self._set_state(BotState.STAGE_REPAIR)
                            break

                        # 3. ƯU TIÊN SỐ 3: Đứt dây / hụt cá -> Nút Quăng Cần xuất hiện lại (chỉ khi không có bảng sửa cần)
                        if sc3["cast"].found:
                            self.stats["missed_after_click"] += 1
                            self.stats["miss_det_agreed" if self._reel_det_agreed else "miss_det_rejected"] += 1
                            case_id = self.missed_logger.log_incident(
                                full_frame=frame3,
                                head_roi_frame=self._last_head_frame_at_bite,
                                head_roi_coords=self._last_good_head_roi,
                                reason="Giật cần nhưng hụt cá / đứt dây (Thấy '!' nhưng không hiện popup Bảo Quản)",
                                plate_info={"name": "columbinaa.png", "state": "REELED_MISSED"},
                                bite_analysis={"threshold": 0.80, "top_score": self._last_bite_score, "top_template": self._last_bite_tpl},
                                duration_s=elapsed_st3,
                                scan_count=0,
                            )
                            self._log(f"⚠️ [ĐỨT DÂY / HỤT CÁ] Nút Quăng Cần ({sc3['cast'].template}, {sc3['cast'].score:.2f}) đã mở lại!")
                            self._log(f"📸 [GHI NHẬN HỤT CÁ] Đã chụp ảnh và lưu bộ debug vào 'debug_hut_ca' (Mã: {case_id})")
                            self._log_stats()
                            self._set_state(BotState.STAGE_1)
                            break

                        # 4. ƯU TIÊN SỐ 4 (CLOSED-LOOP RECOVERY CHỐNG KẸT): NÚT GIẬT CẦN VẪN CÒN TRÊN MÀN HÌNH!
                        # Nếu đã vào BƯỚC 3 mà sau 0.8s nút Giật Cần vẫn còn, chứng tỏ thực tế nhân vật VẪN ĐANG CÂU CÁ.
                        # Lập tức tự sửa trạng thái quay lại BƯỚC 2 tiếp tục canh cá, không được đứng chờ mù quáng!
                        if elapsed_st3 > 0.8 and sc3["reel"].found:
                            self._log(f"🔄 [TỰ SỬA TRẠNG THÁI] Nút Giật Cần vẫn còn ({sc3['reel'].score:.2f})! Màn hình chứng minh vẫn đang câu cá -> Quay lại BƯỚC 2 canh cá tiếp...")
                            self._set_state(BotState.STAGE_2)
                            break

                        time.sleep(0.04)

                # ========================================================
                # STAGE REPAIR: Sửa cần câu bị hỏng (2 giai đoạn: Bấm Tiền -> Bấm OK)
                # ========================================================
                elif self.state == BotState.STAGE_REPAIR:
                    self._log("🔧 [BƯỚC PHỤ: Sửa Cần] Đang xử lý sửa cần câu tự động...")
                    btn_sua_can = templates.get("b4_sua_can_btn", "assets/B4_SuaCan/button")
                    sua_can_title = templates.get("b4_sua_can_title", "assets/B4_SuaCan/title")
                    hoan_thanh_title = templates.get("b4_hoan_thanh_title", "assets/B4_SuaCan/title_hoan_thanh")
                    thresh_repair = thresholds.get("b4_sua_can", 0.70)
                    thresh_ok = thresholds.get("b4_btn_ok", 0.70)

                    # Kiểm tra xem có đang ở sẵn Modal 2 ("Sửa chữa hoàn thành" với nút OK) hay không
                    frame_init, off_init = self.capturer.capture(None)
                    sc_init = self.scanner.scan(frame_init, origin=off_init, names=["done_title", "ok"])
                    already_in_modal2 = sc_init["done_title"].found or sc_init["ok"].found

                    repair_cash_clicked = already_in_modal2
                    if already_in_modal2:
                        self._log(" -> Phát hiện bảng 'Sửa chữa hoàn thành' đã mở sẵn! Vào thẳng bước ấn SPACE.")

                    # GIAI ĐOẠN 1: Bấm nút Tiền Sao sửa cần (nếu chưa ở Modal 2)
                    if not repair_cash_clicked:
                        start_repair = time.time()
                        while not self._stop_event.is_set() and not self._pause_event.is_set():
                            if (time.time() - start_repair) > 8.0:
                                self._log("[HẾT GIỜ] Không tìm thấy nút sửa cần trong 8s. Tự động kiểm tra lại màn hình...")
                                self._set_state(self._auto_detect_state())
                                break

                            frame_r, off_r = self.capturer.capture(None)

                            # Dùng RegionScanner để kiểm tra modal hoàn thành & nút sửa cần (~30ms)
                            sc_hits = self.scanner.scan(frame_r, origin=off_r, names=["done_title", "ok", "repair_title", "repair_btn"])
                            if sc_hits["done_title"].found or sc_hits["ok"].found:
                                self._log(" -> Đã phát hiện bảng 'Sửa chữa hoàn thành'. Chuyển ngay sang bước ấn OK...")
                                repair_cash_clicked = True
                                break

                            m_title = MatchResult(found=sc_hits["repair_title"].found, confidence=sc_hits["repair_title"].score,
                                                  x=sc_hits["repair_title"].x - off_r[0], y=sc_hits["repair_title"].y - off_r[1],
                                                  w=sc_hits["repair_title"].w, h=sc_hits["repair_title"].h, template_name=sc_hits["repair_title"].template)
                            m_cash = MatchResult(found=sc_hits["repair_btn"].found, confidence=sc_hits["repair_btn"].score,
                                                 x=sc_hits["repair_btn"].x - off_r[0], y=sc_hits["repair_btn"].y - off_r[1],
                                                 w=sc_hits["repair_btn"].w, h=sc_hits["repair_btn"].h, template_name=sc_hits["repair_btn"].template)

                            # Định vị toàn bộ bề mặt nút bấm màu xanh (bất chấp mệnh giá tiền)
                            full_btn = self._find_repair_button(frame_r, title_match=m_title, cash_match=m_cash)
                            if full_btn is not None and full_btn.found:
                                full_btn.x += off_r[0]
                                full_btn.y += off_r[1]
                                cx, cy = self.matcher.get_random_click_point(full_btn, margin=8)
                                self._log(f" -> Đã định vị nút Sửa Cần ({full_btn.w}x{full_btn.h}). Click ngẫu nhiên khắp nút tại ({cx}, {cy})")
                                self.input_sim.focus_game_window()
                                self.input_sim.click(cx, cy, hold_min_ms=hold_min, hold_max_ms=hold_max)
                                repair_cash_clicked = True
                                time.sleep(random.uniform(0.6, 0.9))  # Chờ hiệu ứng hiện modal hoàn thành
                                break

                            time.sleep(scan_interval)

                    # GIAI ĐOẠN 2: Bấm phím SPACE xác nhận hoàn thành sửa cần (né chuột 100%)
                    if repair_cash_clicked and not self._stop_event.is_set() and not self._pause_event.is_set():
                        self._log(" -> Đang chờ bảng 'Sửa chữa hoàn thành' để bấm phím SPACE...")
                        start_ok = time.time()
                        ok_clicked = False

                        while not self._stop_event.is_set() and not self._pause_event.is_set():
                            if (time.time() - start_ok) > 8.0:
                                self._log("[HẾT GIỜ] Không tìm thấy bảng hoàn thành trong 8s. Tự động kiểm tra lại màn hình...")
                                self._set_state(self._auto_detect_state())
                                break

                            frame_ok, off_ok = self.capturer.capture(None)
                            sc_ok_hits = self.scanner.scan(frame_ok, origin=off_ok, names=["done_title", "ok"])

                            modal2_present = sc_ok_hits["done_title"].found or sc_ok_hits["ok"].found
                            if not modal2_present:
                                m_title_ht = MatchResult(found=sc_ok_hits["done_title"].found, confidence=sc_ok_hits["done_title"].score,
                                                         x=sc_ok_hits["done_title"].x - off_ok[0], y=sc_ok_hits["done_title"].y - off_ok[1],
                                                         w=sc_ok_hits["done_title"].w, h=sc_ok_hits["done_title"].h, template_name=sc_ok_hits["done_title"].template)
                                btn_ok_match = self._find_ok_button(frame_ok, title_match=m_title_ht, screen_offset=off_ok)
                                modal2_present = btn_ok_match is not None and btn_ok_match.found

                            if modal2_present:
                                self._log(" -> Đã phát hiện bảng 'Sửa chữa hoàn thành'. Bấm phím SPACE để xác nhận OK (DirectInput, né chuột 100%)...")
                                self.input_sim.focus_game_window()
                                self.input_sim.press_key("space", hold_min_ms=50, hold_max_ms=90)
                                ok_clicked = True
                                self.watchdog.note_progress("repair-ok")
                                break

                            time.sleep(scan_interval)

                        if ok_clicked:
                            # Chờ modal đóng hoàn toàn
                            time.sleep(random.uniform(0.8, 1.2))
                            self._log("✅ SỬA CẦN HOÀN THÀNH VÀ ĐÃ BẤM SPACE! Quay lại BƯỚC 1 để tiếp tục quăng cần...")
                            self._set_state(BotState.STAGE_1)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._log(f"🚨 [SỰ CỐ ĐỘT NGỘT] Luồng bot gặp lỗi nghiêm trọng: {e}\n{tb}")
            if hasattr(self, "on_error") and callable(self.on_error):
                try:
                    self.on_error(str(e), tb)
                except Exception:
                    pass
        finally:
            self._set_state(BotState.STOPPED)
            self._log("Bot loop đã kết thúc an toàn.")
