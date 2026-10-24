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

        # Character locator: colour-mask of the neon-green name text (robust to day/night/rain)
        name_tpl = config.get("templates", {}).get("b2_ten_nhan_vat", "assets/B2_DangCau/vat_neo_ten")
        name_thresh = config.get("thresholds", {}).get("b2_ten_nhan_vat", 0.60)
        self.name_locator = NameLocator(name_tpl, min_score=name_thresh)

        # Cached last known match for img1 and head ROI
        self._last_img1_match: Optional[MatchResult] = None
        self._last_good_head_roi: Optional[Tuple[int, int, int, int]] = None  # only ever set from a real name match

        # P0: telemetry tuy chon, chi quan sat, khong doi hanh vi
        tel_cfg = config.get("telemetry", {})
        if tel_cfg.get("enabled", False):
            from core.instrumentation import attach_telemetry
            attach_telemetry(self, jsonl_path=tel_cfg.get("jsonl_path", "telemetry/telemetry.jsonl"))

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
            return None

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

        # 1. Khớp mẫu template matching
        m_tpl = self.matcher.match_any(frame, ok_tpl, threshold=thresh_ok, screen_offset=screen_offset)
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
        self._log(f"📊 [THỐNG KÊ] Thấy '!': {st['bite_seen']} | Dính cá: {st['caught']} | "
                  f"Giật nhưng hụt: {st['missed_after_click']} | Mất cá khi chưa thấy '!': {st['lost_before_bite']} "
                  f"=> tỉ lệ dính {rate:.0f}% ({st['caught']}/{total})")

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
        Tự động quét màn hình để xác định ngữ cảnh hiện tại:
        1. Bảng 'Thông tin sửa chữa' đang mở -> Vào thẳng BƯỚC SỬA CẦN
        2. Popup 'Bảo quản' đang mở -> Vào thẳng BƯỚC 3
        3. Nút giật cần đang mở (đang trong lúc câu chờ cá) -> Vào thẳng BƯỚC 2
        4. Nút quăng cần sẵn sàng -> Vào BƯỚC 1
        """
        cfg = self.config
        templates = cfg.get("templates", {})
        thresholds = cfg.get("thresholds", {})

        full_frame, offset = self.capturer.capture(None)

        # 1. Kiểm tra popup Sửa Cần (cả bảng Thông tin sửa chữa và bảng Sửa chữa hoàn thành)
        sua_can_title = templates.get("b4_sua_can_title", "assets/B4_SuaCan/title")
        hoan_thanh_title = templates.get("b4_hoan_thanh_title", "assets/B4_SuaCan/title_hoan_thanh")
        ok_btn = templates.get("b4_ok_btn", "assets/B4_SuaCan/btn_ok")
        thresh_repair = thresholds.get("b4_sua_can", 0.70)
        thresh_ok = thresholds.get("b4_btn_ok", 0.70)

        m_rep = self.matcher.match_any(full_frame, sua_can_title, threshold=thresh_repair, screen_offset=offset)
        m_ht = self.matcher.match_any(full_frame, hoan_thanh_title, threshold=thresh_repair, screen_offset=offset)
        m_ok = self.matcher.match_any(full_frame, ok_btn, threshold=thresh_ok, screen_offset=offset)

        if m_rep.found or m_ht.found or m_ok.found:
            detected_tag = (m_rep.template_name if m_rep.found
                            else (m_ht.template_name if m_ht.found else m_ok.template_name))
            best_conf = max(m_rep.confidence, m_ht.confidence, m_ok.confidence)
            self._log(f"🔍 [NGỮ CẢNH] Phát hiện bảng sửa cần đang mở ({best_conf:.2f} - {detected_tag})! Vào BƯỚC SỬA CẦN.")
            return BotState.STAGE_REPAIR

        # 2. Kiểm tra popup Bảo Quản (BƯỚC 3)
        img3_path = templates.get("b3_bao_quan") or templates.get("img3")
        thresh3 = thresholds.get("b3_bao_quan") or thresholds.get("img3", 0.70)
        m3 = self.matcher.match_any(full_frame, img3_path, threshold=thresh3, screen_offset=offset)
        if m3.found:
            self._log(f"🔍 [NGỮ CẢNH] Phát hiện popup 'Bảo quản' đang mở ({m3.confidence:.2f} - {m3.template_name})! Vào thẳng BƯỚC 3 (Lưu cá).")
            return BotState.STAGE_3

        # 3. Kiểm tra có đang câu dở dang không (BƯỚC 2: nút giật cần đang hiện)
        img2_btn = templates.get("b2_giat_can") or templates.get("img2_btn")
        thresh_btn = thresholds.get("b2_giat_can") or thresholds.get("img2_btn", 0.70)
        m2 = self.matcher.match_any(full_frame, img2_btn, threshold=thresh_btn, screen_offset=offset)
        if m2.found:
            self._last_img1_match = m2
            self._log(f"🔍 [NGỮ CẢNH] Phát hiện đang câu cá dở dang (nút giật cần hiện: {m2.confidence:.2f})! Vào thẳng BƯỚC 2 (Chờ cá cắn).")
            return BotState.STAGE_2

        # 4. Kiểm tra nút quăng cần (BƯỚC 1)
        img1_path = templates.get("b1_quang_can") or templates.get("img1")
        thresh1 = thresholds.get("b1_quang_can") or thresholds.get("img1", 0.70)
        m1 = self.matcher.match_any(full_frame, img1_path, threshold=thresh1, screen_offset=offset)
        if m1.found:
            self._last_img1_match = m1
            self._log(f"🔍 [NGỮ CẢNH] Phát hiện nút quăng cần sẵn sàng ({m1.confidence:.2f} - {m1.template_name})! Vào BƯỚC 1 (Quăng cần).")
            return BotState.STAGE_1

        self._log(f"🔍 [NGỮ CẢNH] Khởi động tuần tự: QuăngCần={m1.confidence:.2f}, GiậtCần={m2.confidence:.2f}, BảoQuản={m3.confidence:.2f}, SửaCần={m_rep.confidence:.2f}")
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

                while not self._stop_event.is_set() and not self._pause_event.is_set():
                    frame_full, off_full = self.capturer.capture(None)
                    fh, fw = frame_full.shape[:2]
                    now = time.time()
                    elapsed_st1 = now - stage1_start

                    # 1. ƯU TIÊN SỐ 1 (Fast Path): Quét nút Quăng Cần trong vùng thao tác (góc dưới bên phải)
                    # Quét đơn tỉ lệ trước (~20ms), chỉ bật đa tỉ lệ nếu >1.5s chưa thấy
                    rx, ry = int(fw * 0.65), int(fh * 0.55)
                    action_frame = frame_full[ry:, rx:]
                    action_offset = (off_full[0] + rx, off_full[1] + ry)

                    use_multi = (elapsed_st1 > 1.5)
                    match1 = self.matcher.match_any(action_frame, img1_path, threshold=thresh1, screen_offset=action_offset, multi_scale=use_multi)

                    if match1.found:
                        self._last_img1_match = match1
                        cx, cy = self.matcher.get_random_click_point(match1, margin=margin)
                        self._log(f" -> Đã thấy nút Quăng Cần ({match1.template_name}, độ khớp {match1.confidence:.2f}). Click tại ({cx}, {cy})")
                        self.input_sim.click(cx, cy, hold_min_ms=hold_min, hold_max_ms=hold_max)

                        # Chuyển ngay sang STAGE 2 (STAGE 2 đã có sẵn watchdog tự động phát hiện cần hỏng / hụt cá)
                        self._set_state(BotState.STAGE_2)
                        self._log(f"==> Chuyển sang {self.state.value}")
                        time.sleep(random.uniform(0.20, 0.35))  # Delay nhỏ để game bắt đầu quăng phao
                        break

                    # 2. ƯU TIÊN SỐ 2 (Fallback): Chỉ khi không thấy nút Quăng Cần sau 0.6s mới kiểm tra bảng Sửa Cần
                    # Quét trong ROI trung tâm màn hình, đơn tỉ lệ để tránh nghẽn luồng (~30ms)
                    if elapsed_st1 > 0.6 and (now - last_modal_check >= 1.0):
                        last_modal_check = now
                        center_y1, center_y2 = int(fh * 0.15), int(fh * 0.60)
                        center_x1, center_x2 = int(fw * 0.25), int(fw * 0.75)
                        center_roi = frame_full[center_y1:center_y2, center_x1:center_x2]
                        center_off = (off_full[0] + center_x1, off_full[1] + center_y1)

                        m_repair = self.matcher.match_any(center_roi, sua_can_title, threshold=thresh_repair, screen_offset=center_off, multi_scale=False)
                        m_ht = self.matcher.match_any(center_roi, hoan_thanh_title, threshold=thresh_repair, screen_offset=center_off, multi_scale=False)
                        m_ok = self.matcher.match_any(center_roi, ok_btn, threshold=thresh_ok, screen_offset=center_off, multi_scale=False)

                        if m_repair.found or m_ht.found or m_ok.found:
                            self._log("⚠️ Phát hiện bảng sửa cần đang mở! Vào bước sửa cần.")
                            self._set_state(BotState.STAGE_REPAIR)
                            break

                    # Tự phục hồi: Nếu tìm 3.5s không thấy, tự kiểm tra lại ngữ cảnh
                    if elapsed_st1 > 3.5:
                        new_state = self._auto_detect_state()
                        if new_state != BotState.STAGE_1:
                            self._set_state(new_state)
                            break
                        stage1_start = time.time()

                    time.sleep(0.04)

            # ========================================================
            # STAGE 2: Chờ Dấu Chấm Than xuất hiện trên đầu nhân vật -> Giật cần
            # ========================================================
            elif self.state == BotState.STAGE_2:
                img2_path = templates.get("b2_cham_than") or templates.get("img2")
                thresh2 = thresholds.get("b2_cham_than") or thresholds.get("img2", 0.72)
                timeout_sec = delays.get("stage2_wait_timeout_sec", 120)
                relocate_interval = 0.5  # seconds between re-locating the character during the wait

                start_wait_time = time.time()
                found_img2 = False

                # The ROI is NEVER taken from a template-matching fallback. It is either located by the
                # character's name (this cast) or the last ROI that was located the same way (seed).
                effective_region = self._last_good_head_roi
                last_locate_time = 0.0
                last_warn_time = 0.0
                last_repair_check = 0.0

                # Reel button: a fixed UI element that is already on screen for the whole wait.
                # LOCATE it now (argmax in the lower-right area, low floor) instead of re-detecting it
                # after the bite, where a miss cost 0.8-0.9s (multi-scale search) - longer than the bite itself.
                btn_path = templates.get("b2_giat_can") or templates.get("img2_btn") or templates.get("img1")
                reel_floor = thresholds.get("b2_giat_can_locate", 0.40)
                reel_match: Optional[MatchResult] = None

                self.recorder.begin_cast(effective_region, {"thresh2": thresh2})
                self._log(f"[{self.state.value}] Đang canh chừng Dấu Chấm Than (tần số cao, tối đa {timeout_sec}s)...")

                fast_scan_interval = 0.03  # 30ms (~33 FPS) để phản xạ tức thì khi cá cắn
                scan_count = 0
                last_scan_t = None
                max_scan_gap = 0.0  # watchdog: khoảng nghỉ dài nhất giữa 2 lần quét "!" (phải << 400ms)
                while not self._stop_event.is_set() and not self._pause_event.is_set():
                    # Kiểm tra timeout
                    if (time.time() - start_wait_time) > timeout_sec:
                        self._log(f"[HẾT GIỜ] Đã đợi {timeout_sec}s mà không thấy Dấu Chấm Than! Tự động kiểm tra lại màn hình...")
                        self.recorder.dump("timeout")
                        self._set_state(self._auto_detect_state())
                        break

                    # Re-locate the character every 0.5s: camera/character may move after the cast.
                    now = time.time()
                    if now - last_locate_time >= relocate_interval:
                        last_locate_time = now
                        full_frame, full_offset = self.capturer.capture(None)

                        # 1. Định vị nhân vật
                        loc = self.name_locator.locate(full_frame, full_offset)
                        if loc is not None:
                            new_roi = self._clamp_region(loc.head_roi(), full_frame, full_offset)
                            moved = effective_region is None or abs(new_roi[0] - effective_region[0]) > 20 or abs(new_roi[1] - effective_region[1]) > 20
                            if moved:
                                self._log(f" -> Định vị nhân vật '{loc.name}' (độ khớp {loc.score:.2f}): vùng đầu = {new_roi}")
                            effective_region = self._last_good_head_roi = new_roi

                        # 2. Định vị nút Giật Cần trong vùng thao tác (góc dưới bên phải)
                        fh, fw = full_frame.shape[:2]
                        rx, ry = int(fw * 0.65), int(fh * 0.55)
                        action_roi = full_frame[ry:, rx:]
                        action_offset = (full_offset[0] + rx, full_offset[1] + ry)

                        cand = self.matcher.match(action_roi, btn_path, threshold=reel_floor,
                                                  screen_offset=action_offset,
                                                  multi_scale=False)
                        if cand.found and (reel_match is None or cand.confidence > reel_match.confidence):
                            if reel_match is None or abs(cand.x - reel_match.x) > 10 or abs(cand.y - reel_match.y) > 10:
                                self._log(f" -> Đã định vị nút Giật Cần tại ({cand.x}, {cand.y}) {cand.w}x{cand.h} (độ khớp {cand.confidence:.2f})")
                            reel_match = cand

                        # 3. Kiểm tra sự cố, mọi phép dò đều đơn tỉ lệ và đứng < 150ms (dấu '!' hiện >= 400ms).
                        # Tín hiệu loại trừ lẫn nhau MẠNH là nút Quăng Cần: hiện (0.95) => cần KHÔNG ở dưới nước;
                        # lúc đang câu nó chỉ đạt ~0.57. (Nút Giật Cần thì yếu: 0.47 khi vắng vs 0.54 khi có.)
                        if (now - start_wait_time) > 2.0:
                            img1_path = templates.get("b1_quang_can") or templates.get("img1")
                            m_quang_st2 = self.matcher.match_any(action_roi, img1_path, threshold=thresholds.get("b1_quang_can", 0.72),
                                                                 screen_offset=action_offset, multi_scale=False)
                            if m_quang_st2.found:
                                self.stats["lost_before_bite"] += 1
                                self._log(f"⚠️ Cần câu bị giật lên lại / chưa vào câu (Nút Quăng Cần hiện lại, khớp {m_quang_st2.confidence:.2f})! Quay về BƯỚC 1 quăng lại...")
                                self._log_stats()
                                self._set_state(BotState.STAGE_1)
                                break

                            # Popup sửa cần: kiểm tra thưa (3s/lần), đơn tỉ lệ ~105ms. (Bản cũ: đa tỉ lệ ~1000ms mỗi 0.5s.)
                            if now - last_repair_check >= 3.0:
                                last_repair_check = now
                                sua_can_title = templates.get("b4_sua_can_title", "assets/B4_SuaCan/title")
                                thresh_repair = thresholds.get("b4_sua_can", 0.70)
                                m_rep2 = self.matcher.match_any(full_frame, sua_can_title, threshold=thresh_repair,
                                                                screen_offset=full_offset, multi_scale=False)
                                if m_rep2.found:
                                    self._log(f"⚠️ Cần câu bị hỏng (phát hiện bảng sửa cần khớp {m_rep2.confidence:.2f})! Vào bước sửa cần...")
                                    self._set_state(BotState.STAGE_REPAIR)
                                    break

                    if effective_region is None:
                        # Never guess: without a located character there is nothing meaningful to scan.
                        if now - last_warn_time > 5.0:
                            self._log(" -> [CẢNH BÁO] Chưa định vị được nhân vật (tên bị che hoặc ngoài màn hình). Đang thử lại...")
                            last_warn_time = now
                        time.sleep(fast_scan_interval)
                        continue

                    frame, offset = self.capturer.capture(effective_region)
                    # Grace period: trong 2.0s đầu sau khi quăng cần, cần chưa xuống nước và cá chưa thể cắn.
                    # Bỏ qua kiểm tra '!' trong 2.0s đầu để triệt tiêu hoàn toàn false-positive (màu áo/nón/cần).
                    if (now - start_wait_time) >= 2.0:
                        # Quét siêu nhanh ở tỷ lệ gốc 1.0x (chỉ mất ~20ms) để không bỏ lỡ khoảnh khắc cá cắn 0.5s
                        match2 = self.matcher.match_any(frame, img2_path, threshold=thresh2, screen_offset=offset, multi_scale=False)
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

                    if target_match is not None and target_match.found:
                        cx, cy = self.matcher.get_random_click_point(target_match, margin=margin)
                        self._log(f" -> KÍCH HOẠT: Giật cần câu tại nút bấm ({cx}, {cy})")
                        self.input_sim.click(cx, cy, hold_min_ms=hold_min, hold_max_ms=hold_max)

                        # Chuyển sang STAGE 3
                        self._set_state(BotState.STAGE_3)
                        self._log(f"==> Chuyển sang {self.state.value}")
                    else:
                        self._log("[CẢNH BÁO] Không tìm thấy nút giật cần! Tự kiểm tra lại màn hình...")
                        self._set_state(self._auto_detect_state())

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
                timeout_stage3 = 25  # Đợi tối đa 25s
                start_stage3 = time.time()
                last_repair_check_st3 = 0.0

                while not self._stop_event.is_set() and not self._pause_event.is_set():
                    now = time.time()
                    elapsed_st3 = now - start_stage3
                    if elapsed_st3 > timeout_stage3:
                        self._log("[HẾT GIỜ] Không tìm thấy nút Bảo Quản! Tự động kiểm tra lại màn hình...")
                        self._set_state(self._auto_detect_state())
                        break

                    frame3, offset3 = self.capturer.capture(None)
                    fh3, fw3 = frame3.shape[:2]

                    # 1. ƯU TIÊN SỐ 1: Nút Bảo Quản trong vùng góc dưới bên phải (X >= 45%, Y >= 45%)
                    # Nút Bảo Quản luôn nằm ở nửa dưới bên phải màn hình
                    bq_rx, bq_ry = int(fw3 * 0.45), int(fh3 * 0.45)
                    bq_roi = frame3[bq_ry:, bq_rx:]
                    bq_offset = (offset3[0] + bq_rx, offset3[1] + bq_ry)

                    # Quét đơn tỉ lệ trước (~15ms), nếu >2.0s chưa thấy mới quét đa tỉ lệ
                    use_bq_multi = (elapsed_st3 > 2.0)
                    match3 = self.matcher.match_any(bq_roi, img3_path, threshold=thresh3, screen_offset=bq_offset, multi_scale=use_bq_multi)

                    if match3.found:
                        cx, cy = self.matcher.get_random_click_point(match3, margin=margin)
                        self._log(f" -> Đã thấy nút Bảo Quản ({match3.template_name}, độ khớp {match3.confidence:.2f}). Click tại ({cx}, {cy})")
                        self.input_sim.click(cx, cy, hold_min_ms=hold_min, hold_max_ms=hold_max)
                        self.stats["caught"] += 1
                        self._log_stats()
                        if self.on_fish_caught:
                            try:
                                self.on_fish_caught()
                            except Exception:
                                pass
                        time.sleep(random.uniform(0.20, 0.35))  # Phản xạ nhanh: chỉ cần 200-350ms cho game ghi nhận click
                        self._log("🔄 HOÀN THÀNH 1 VÒNG CÂU! Tự chuyển vòng mới...")
                        self._set_state(BotState.STAGE_1)
                        break

                    # 2. ƯU TIÊN SỐ 2: Tự phục hồi đứt dây câu / hụt cá -> Nút Quăng Cần xuất hiện lại
                    rx3, ry3 = int(fw3 * 0.65), int(fh3 * 0.55)
                    action_f3 = frame3[ry3:, rx3:]
                    action_off3 = (offset3[0] + rx3, offset3[1] + ry3)
                    match_quang = self.matcher.match_any(action_f3, img1_path, threshold=thresh1, screen_offset=action_off3, multi_scale=False)
                    if match_quang.found:
                        self.stats["missed_after_click"] += 1
                        self._log(f"⚠️ [ĐỨT DÂY / HỤT CÁ] Phát hiện nút Quăng Cần ({match_quang.template_name}, {match_quang.confidence:.2f}) đã mở lại! Tự động quay về BƯỚC 1 quăng cần mới...")
                        self._log_stats()
                        self._set_state(BotState.STAGE_1)
                        break

                    # 3. ƯU TIÊN SỐ 3: Kiểm tra cần hỏng thưa thớt (sau 1.5s, mỗi 1.5s/lần) trong vùng trung tâm
                    if elapsed_st3 > 1.5 and (now - last_repair_check_st3 >= 1.5):
                        last_repair_check_st3 = now
                        center_y1, center_y2 = int(fh3 * 0.15), int(fh3 * 0.60)
                        center_x1, center_x2 = int(fw3 * 0.25), int(fw3 * 0.75)
                        center_roi3 = frame3[center_y1:center_y2, center_x1:center_x2]
                        center_off3 = (offset3[0] + center_x1, offset3[1] + center_y1)

                        m_rep3 = self.matcher.match_any(center_roi3, sua_can_title, threshold=thresh_repair, screen_offset=center_off3, multi_scale=False)
                        m_ht3 = self.matcher.match_any(center_roi3, hoan_thanh_title, threshold=thresh_repair, screen_offset=center_off3, multi_scale=False)
                        m_ok3 = self.matcher.match_any(center_roi3, ok_btn, threshold=thresh_ok, screen_offset=center_off3, multi_scale=False)
                        if m_rep3.found or m_ht3.found or m_ok3.found:
                            self._log("⚠️ Phát hiện cần câu bị hỏng sau khi giật! Chuyển sang sửa cần tự động...")
                            self._set_state(BotState.STAGE_REPAIR)
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
                m_init_ht = self.matcher.match_any(frame_init, hoan_thanh_title, threshold=thresh_repair, screen_offset=off_init)
                btn_init_ok = self._find_ok_button(frame_init, title_match=m_init_ht, screen_offset=off_init)
                already_in_modal2 = (btn_init_ok is not None and btn_init_ok.found)

                repair_cash_clicked = already_in_modal2
                if already_in_modal2:
                    self._log(" -> Phát hiện bảng 'Sửa chữa hoàn thành' đã mở sẵn! Vào thẳng bước ấn OK.")

                # GIAI ĐOẠN 1: Bấm nút Tiền Sao sửa cần (nếu chưa ở Modal 2)
                if not repair_cash_clicked:
                    start_repair = time.time()
                    while not self._stop_event.is_set() and not self._pause_event.is_set():
                        if (time.time() - start_repair) > 8.0:
                            self._log("[HẾT GIỜ] Không tìm thấy nút sửa cần trong 8s. Tự động kiểm tra lại màn hình...")
                            self._set_state(self._auto_detect_state())
                            break

                        frame_r, off_r = self.capturer.capture(None)

                        # Kiểm tra nếu bảng đã chuyển sang 'Sửa chữa hoàn thành'
                        m_ht_check = self.matcher.match_any(frame_r, hoan_thanh_title, threshold=thresh_repair, screen_offset=off_r)
                        m_ok_check = self._find_ok_button(frame_r, title_match=m_ht_check, screen_offset=off_r)
                        if m_ok_check is not None and m_ok_check.found:
                            self._log(" -> Đã phát hiện bảng 'Sửa chữa hoàn thành'. Chuyển ngay sang bước ấn OK...")
                            repair_cash_clicked = True
                            break

                        m_title = self.matcher.match_any(frame_r, sua_can_title, threshold=thresh_repair, screen_offset=off_r)
                        m_cash = self.matcher.match_any(frame_r, btn_sua_can, threshold=thresh_repair, screen_offset=off_r)

                        # Định vị toàn bộ bề mặt nút bấm màu xanh (bất chấp mệnh giá tiền)
                        full_btn = self._find_repair_button(frame_r, title_match=m_title, cash_match=m_cash)
                        if full_btn is not None and full_btn.found:
                            full_btn.x += off_r[0]
                            full_btn.y += off_r[1]
                            cx, cy = self.matcher.get_random_click_point(full_btn, margin=8)
                            self._log(f" -> Đã định vị nút Sửa Cần ({full_btn.w}x{full_btn.h}). Click ngẫu nhiên khắp nút tại ({cx}, {cy})")
                            self.input_sim.click(cx, cy, hold_min_ms=hold_min, hold_max_ms=hold_max)
                            repair_cash_clicked = True
                            time.sleep(random.uniform(0.6, 0.9))  # Chờ hiệu ứng hiện modal hoàn thành
                            break

                        time.sleep(scan_interval)

                # GIAI ĐOẠN 2: Bấm nút xanh OK xác nhận hoàn thành sửa cần
                if repair_cash_clicked and not self._stop_event.is_set() and not self._pause_event.is_set():
                    self._log(" -> Đang chờ nút xanh OK ('Sửa chữa hoàn thành')...")
                    start_ok = time.time()
                    ok_clicked = False

                    while not self._stop_event.is_set() and not self._pause_event.is_set():
                        if (time.time() - start_ok) > 8.0:
                            self._log("[HẾT GIỜ] Không tìm thấy nút OK trong 8s. Tự động kiểm tra lại màn hình...")
                            self._set_state(self._auto_detect_state())
                            break

                        frame_ok, off_ok = self.capturer.capture(None)
                        m_title_ht = self.matcher.match_any(frame_ok, hoan_thanh_title, threshold=thresh_repair, screen_offset=off_ok)
                        btn_ok_match = self._find_ok_button(frame_ok, title_match=m_title_ht, screen_offset=off_ok)

                        if btn_ok_match is not None and btn_ok_match.found:
                            cx, cy = self.matcher.get_random_click_point(btn_ok_match, margin=10)
                            self._log(f" -> Đã định vị nút OK ({btn_ok_match.w}x{btn_ok_match.h}, {btn_ok_match.template_name}). Click ngẫu nhiên khắp nút tại ({cx}, {cy})")
                            self.input_sim.click(cx, cy, hold_min_ms=hold_min, hold_max_ms=hold_max)
                            ok_clicked = True
                            break

                        time.sleep(scan_interval)

                    if ok_clicked:
                        # Chờ modal đóng hoàn toàn
                        time.sleep(random.uniform(0.8, 1.2))
                        self._log("✅ SỬA CẦN HOÀN THÀNH VÀ ĐÃ BẤM OK! Quay lại BƯỚC 1 để tiếp tục quăng cần...")
                        self._set_state(BotState.STAGE_1)

        self._log("Bot loop đã kết thúc an toàn.")
