import os
import sys
import json
import time
import queue
import re
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from bot_engine import BotEngine, BotState
from core.hotkey_manager import HotkeyManager

BASE_DIR = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def load_config(config_path: str = "config.json"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class AutoFishingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Auto Fishing Bot - Play Together [Pro Edition]")
        self.root.geometry("620x720")
        self.root.minsize(540, 560)
        self.root.configure(bg="#181825")

        # Window and option states
        self.is_topmost = tk.BooleanVar(value=True)
        self.auto_clear_simple = tk.BooleanVar(value=False)
        self.auto_clear_debug = tk.BooleanVar(value=True)
        self.root.attributes("-topmost", True)

        # Thread-safe log queue
        self.log_queue = queue.Queue()
        self.fish_count = 0

        # Load engine configuration
        self.config = load_config()

        # Initialize Bot Engine with callbacks
        self.engine = BotEngine(
            self.config,
            on_log=self.on_engine_log,
            on_state_change=self.on_engine_state_change,
            on_fish_caught=self.on_fish_caught
        )

        # Initialize Hotkey Manager
        self.hotkey_mgr = HotkeyManager()
        self._setup_hotkeys()

        # Build Modern UI
        self._build_ui()

        # Schedule auto-clear log timer (every 60s)
        self.root.after(60000, self._schedule_auto_clear)

        # Start periodic queue polling
        self.root.after(80, self._process_log_queue)

        # Cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_hotkeys(self):
        start_key = self.config.get("hotkeys", {}).get("start", "f1").lower()
        stop_key = self.config.get("hotkeys", {}).get("stop", "f2").lower()

        # Global system hooks
        self.hotkey_mgr.register(start_key, self.start_bot)
        self.hotkey_mgr.register(stop_key, self.stop_bot)
        self.hotkey_mgr.register("f3", self.engine.recorder.request_dump)

    def _build_ui(self):
        # 1. Header Frame with Modern Styling
        header = tk.Frame(self.root, bg="#181825", pady=10)
        header.pack(fill="x", padx=15)

        title_box = tk.Frame(header, bg="#181825")
        title_box.pack(fill="x")

        title_lbl = tk.Label(
            title_box,
            text="🎣 AUTO FISHING BOT",
            font=("Segoe UI", 16, "bold"),
            fg="#8aadf4",
            bg="#181825"
        )
        title_lbl.pack(side="left")

        ver_pill = tk.Label(
            title_box,
            text="PRO v2.5",
            font=("Segoe UI", 8, "bold"),
            bg="#313244",
            fg="#a6da95",
            padx=7,
            pady=2
        )
        ver_pill.pack(side="left", padx=10)

        sub_lbl = tk.Label(
            header,
            text="Vòng Lặp Khép Kín • Tự Phục Hồi Lệch Bước • Chống Giật Nhầm • Khóa Tọa Độ",
            font=("Segoe UI", 8),
            fg="#a5adcb",
            bg="#181825"
        )
        sub_lbl.pack(anchor="w", pady=(3, 0))

        # 2. Modern Status & Counter Card
        card = tk.Frame(self.root, bg="#1e1e2e", padx=15, pady=12, highlightbackground="#313244", highlightthickness=1)
        card.pack(fill="x", padx=15, pady=5)

        status_box = tk.Frame(card, bg="#1e1e2e")
        status_box.pack(fill="x")

        tk.Label(status_box, text="Trạng thái:", font=("Segoe UI", 10, "bold"), fg="#cad3f5", bg="#1e1e2e").pack(side="left")
        self.status_lbl = tk.Label(status_box, text="🔴 ĐÃ DỪNG (Sẵn sàng)", font=("Segoe UI", 10, "bold"), fg="#ed8796", bg="#1e1e2e")
        self.status_lbl.pack(side="left", padx=10)

        self.counter_lbl = tk.Label(status_box, text="🐟 Đã câu: 0 con", font=("Segoe UI", 10, "bold"), fg="#a6da95", bg="#1e1e2e")
        self.counter_lbl.pack(side="right")

        # 3. Big Modern Control Buttons
        btn_frame = tk.Frame(self.root, bg="#181825", pady=8)
        btn_frame.pack(fill="x", padx=15)

        self.btn_start = tk.Button(
            btn_frame,
            text="▶ BẮT ĐẦU (F1)",
            font=("Segoe UI", 11, "bold"),
            bg="#a6da95",
            fg="#181825",
            activebackground="#8bd57e",
            activeforeground="#181825",
            relief="flat",
            cursor="hand2",
            padx=15,
            pady=8,
            command=self.start_bot
        )
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.btn_stop = tk.Button(
            btn_frame,
            text="⏹ DỪNG LẠI (F2)",
            font=("Segoe UI", 11, "bold"),
            bg="#ed8796",
            fg="#181825",
            activebackground="#e0697a",
            activeforeground="#181825",
            relief="flat",
            cursor="hand2",
            padx=15,
            pady=8,
            command=self.stop_bot
        )
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=(6, 0))

        # 4. Top Quick Options Bar
        opts_bar = tk.Frame(self.root, bg="#181825")
        opts_bar.pack(fill="x", padx=15, pady=(0, 4))

        top_check = tk.Checkbutton(
            opts_bar,
            text="📌 Luôn nổi (Top)",
            variable=self.is_topmost,
            font=("Segoe UI", 9),
            fg="#cad3f5",
            bg="#181825",
            activebackground="#181825",
            activeforeground="#cad3f5",
            selectcolor="#1e1e2e",
            command=self._toggle_topmost
        )
        top_check.pack(side="left")

        test_btn = tk.Button(
            opts_bar,
            text="📸 Soi Mắt Bot",
            font=("Segoe UI", 8, "bold"),
            bg="#8aadf4",
            fg="#181825",
            activebackground="#7dc4e4",
            relief="flat",
            cursor="hand2",
            padx=10,
            pady=2,
            command=self._test_bot_vision
        )
        test_btn.pack(side="right")

        # 5. Segmented Notebook Tabs (Trạng Thái Rút Gọn vs Log Chi Tiết Debug)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Custom.TNotebook', background='#181825', borderwidth=0)
        style.configure('Custom.TNotebook.Tab', background='#1e1e2e', foreground='#cad3f5', padding=[14, 7], font=('Segoe UI', 9, 'bold'), borderwidth=0)
        style.map('Custom.TNotebook.Tab',
                  background=[('selected', '#313244'), ('active', '#24273a')],
                  foreground=[('selected', '#8aadf4'), ('active', '#b4befe')])

        self.notebook = ttk.Notebook(self.root, style='Custom.TNotebook')
        self.notebook.pack(fill='both', expand=True, padx=15, pady=5)

        # Tab 1: Trạng Thái Rút Gọn
        self.tab_simple = tk.Frame(self.notebook, bg="#181825")
        self.notebook.add(self.tab_simple, text="📋 Trạng Thái Bot (Rút Gọn)")

        t1_toolbar = tk.Frame(self.tab_simple, bg="#181825", pady=4)
        t1_toolbar.pack(fill="x")

        tk.Checkbutton(
            t1_toolbar,
            text="🧹 Tự xóa log (60s)",
            variable=self.auto_clear_simple,
            font=("Segoe UI", 8),
            fg="#a6adc8",
            bg="#181825",
            activebackground="#181825",
            selectcolor="#1e1e2e"
        ).pack(side="left")

        tk.Button(
            t1_toolbar,
            text="🧹 Xóa Màn Hình",
            font=("Segoe UI", 8),
            bg="#313244",
            fg="#cad3f5",
            activebackground="#45475a",
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=1,
            command=self._clear_simple_log
        ).pack(side="right")

        self.simple_text = scrolledtext.ScrolledText(
            self.tab_simple,
            bg="#1e1e2e",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            font=("Segoe UI", 9),
            wrap="word",
            relief="flat",
            bd=0,
            padx=8,
            pady=8
        )
        self.simple_text.pack(fill="both", expand=True)

        # Tags styling for Tab 1
        self.simple_text.tag_config("ROUND_COMPLETE", foreground="#a6da95", font=("Segoe UI", 9, "bold"))
        self.simple_text.tag_config("STEP_CAST", foreground="#89b4fa", font=("Segoe UI", 9, "bold"))
        self.simple_text.tag_config("STEP_WAIT", foreground="#f9e2af", font=("Segoe UI", 9))
        self.simple_text.tag_config("STEP_BITE", foreground="#f38ba8", font=("Segoe UI", 10, "bold"))
        self.simple_text.tag_config("STEP_KEEP", foreground="#a6e3a1", font=("Segoe UI", 9, "bold"))
        self.simple_text.tag_config("STEP_REPAIR", foreground="#cba6f7", font=("Segoe UI", 9, "bold"))
        self.simple_text.tag_config("STEP_ADJUST", foreground="#89dceb", font=("Segoe UI", 9, "italic"))
        self.simple_text.tag_config("STEP_WARN", foreground="#fab387", font=("Segoe UI", 9))
        self.simple_text.tag_config("STEP_INFO", foreground="#cad3f5", font=("Segoe UI", 9))
        self.simple_text.tag_config("SEPARATOR", foreground="#fab387", font=("Segoe UI", 9, "bold"))

        # Tab 2: Log Chi Tiết (Debug)
        self.tab_debug = tk.Frame(self.notebook, bg="#181825")
        self.notebook.add(self.tab_debug, text="🛠 Log Chi Tiết (Debug)")

        t2_toolbar = tk.Frame(self.tab_debug, bg="#181825", pady=4)
        t2_toolbar.pack(fill="x")

        tk.Checkbutton(
            t2_toolbar,
            text="🧹 Tự xóa log (60s)",
            variable=self.auto_clear_debug,
            font=("Segoe UI", 8),
            fg="#a6adc8",
            bg="#181825",
            activebackground="#181825",
            selectcolor="#1e1e2e"
        ).pack(side="left")

        self.copy_status_lbl = tk.Label(t2_toolbar, text="", font=("Segoe UI", 8, "bold"), fg="#a6da95", bg="#181825")
        self.copy_status_lbl.pack(side="left", padx=10)

        tk.Button(
            t2_toolbar,
            text="📋 Copy Toàn Bộ Log",
            font=("Segoe UI", 8, "bold"),
            bg="#8aadf4",
            fg="#181825",
            activebackground="#7dc4e4",
            relief="flat",
            cursor="hand2",
            padx=10,
            pady=1,
            command=self._copy_debug_log
        ).pack(side="right", padx=(5, 0))

        tk.Button(
            t2_toolbar,
            text="🧹 Xóa Log",
            font=("Segoe UI", 8),
            bg="#313244",
            fg="#cad3f5",
            activebackground="#45475a",
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=1,
            command=self._clear_debug_log
        ).pack(side="right")

        self.debug_text = scrolledtext.ScrolledText(
            self.tab_debug,
            bg="#1e1e2e",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            font=("Consolas", 9),
            wrap="word",
            relief="flat",
            bd=0,
            padx=6,
            pady=6
        )
        self.debug_text.pack(fill="both", expand=True)

        # Tags styling for Tab 2
        self.debug_text.tag_config("SUCCESS", foreground="#a6da95")
        self.debug_text.tag_config("WARN", foreground="#f9e2af")
        self.debug_text.tag_config("ALERT", foreground="#ed8796")
        self.debug_text.tag_config("INFO", foreground="#89b4fa")
        self.debug_text.tag_config("STATS", foreground="#cba6f7")

        # Initial Welcome Logs
        self.on_engine_log("✨ Ứng dụng đã sẵn sàng! Bấm [F1] để bắt đầu, [F2] để dừng.")
        self.on_engine_log("🎯 [BƯỚC 0 CHẨN ĐOÁN] Thấy dấu '!' trên đầu hãy bấm [F3] để lưu clip.")

    def _toggle_topmost(self):
        self.root.attributes("-topmost", self.is_topmost.get())

    def _copy_debug_log(self):
        content = self.debug_text.get("1.0", tk.END).strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.root.update()
            self.copy_status_lbl.config(text="✅ Đã copy vào Clipboard!")
            self.root.after(2500, lambda: self.copy_status_lbl.config(text=""))
        else:
            self.copy_status_lbl.config(text="⚠️ Log trống!")
            self.root.after(2000, lambda: self.copy_status_lbl.config(text=""))

    def _clear_simple_log(self):
        self.simple_text.delete("1.0", tk.END)

    def _clear_debug_log(self):
        self.debug_text.delete("1.0", tk.END)

    def _schedule_auto_clear(self):
        if self.engine.state != BotState.STAGE_2:
            if self.auto_clear_simple.get():
                self.simple_text.delete("1.0", tk.END)
            if self.auto_clear_debug.get():
                self.debug_text.delete("1.0", tk.END)
                now = datetime.now().strftime("%H:%M:%S")
                self.debug_text.insert(tk.END, f"[{now}] 🧹 [TỰ ĐỘNG DỌN DẸP] Đã xóa log định kỳ 60s.\n", "INFO")
        self.root.after(60000, self._schedule_auto_clear)

    def _test_bot_vision(self):
        """Captures what the bot currently sees and reports exact match scores."""
        try:
            import cv2
            full_frame, offset = self.engine.capturer.capture(None)
            h, w = full_frame.shape[:2]
            save_path = r"assets\anh_man_hinh_bot_thay.png"
            cv2.imwrite(save_path, full_frame)

            templates = self.config.get("templates", {})
            thresholds = self.config.get("thresholds", {})
            m1 = self.engine.matcher.match_any(full_frame, templates.get("b1_quang_can"), threshold=thresholds.get("b1_quang_can", 0.70))
            m2 = self.engine.matcher.match_any(full_frame, templates.get("b2_giat_can"), threshold=thresholds.get("b2_giat_can", 0.70))
            m3 = self.engine.matcher.match_any(full_frame, templates.get("b3_bao_quan"), threshold=thresholds.get("b3_bao_quan", 0.70))
            m4 = self.engine.matcher.match_any(full_frame, templates.get("b4_sua_can_title", "assets/B4_SuaCan/title"), threshold=thresholds.get("b4_sua_can", 0.70))
            loc = self.engine.name_locator.locate(full_frame, offset)

            self.on_engine_log("=" * 45)
            self.on_engine_log(f"📸 [SOI MẮT BOT] Đã chụp màn hình ({w}x{h})!")
            self.on_engine_log(f"   - Nút Quăng Cần: {'✅ KHỚP' if m1.found else '❌ KHÔNG KHỚP'} (Độ khớp: {m1.confidence:.2f} [{m1.template_name}])")
            self.on_engine_log(f"   - Nút Giật Cần : {'✅ KHỚP' if m2.found else '❌ KHÔNG KHỚP'} (Độ khớp: {m2.confidence:.2f} [{m2.template_name}])")
            self.on_engine_log(f"   - Nút Bảo Quản : {'✅ KHỚP' if m3.found else '❌ KHÔNG KHỚP'} (Độ khớp: {m3.confidence:.2f} [{m3.template_name}])")
            self.on_engine_log(f"   - Bảng Sửa Cần : {'⚠️ KHỚP (CẦN HỎNG)' if m4.found else '❌ KHÔNG KHỚP'} (Độ khớp: {m4.confidence:.2f} [{m4.template_name}])")
            if loc is not None:
                self.on_engine_log(f"   - Tên Nhân Vật : ✅ KHỚP ({loc.name}, độ khớp: {loc.score:.2f})")
            else:
                self.on_engine_log("   - Tên Nhân Vật : ❌ KHÔNG KHỚP (Chưa thấy tên trên màn hình)")
            self.on_engine_log(f"   👉 Ảnh chụp đã lưu vào: {save_path}")
            self.on_engine_log("=" * 45)
        except Exception as e:
            self.on_engine_log(f"❌ [LỖI SOI MẮT BOT]: {e}")

    def start_bot(self):
        self.engine.start()

    def stop_bot(self):
        self.engine.stop()

    def on_engine_log(self, msg: str):
        self.log_queue.put(msg)

    def on_engine_state_change(self, new_state: BotState):
        self.root.after(0, self._update_state_ui, new_state)

    def on_fish_caught(self):
        self.fish_count += 1
        self.root.after(0, self._update_fish_count)

    def _update_fish_count(self):
        self.counter_lbl.config(text=f"🐟 Đã câu: {self.fish_count} con")

    def _update_state_ui(self, state: BotState):
        if state == BotState.IDLE or state == BotState.STOPPED:
            self.status_lbl.config(text="🔴 ĐÃ DỪNG (Nhấn F1)", fg="#ed8796")
        elif state == BotState.STAGE_1:
            self.status_lbl.config(text="🎣 BƯỚC 1: Quăng Cần", fg="#89b4fa")
        elif state == BotState.STAGE_2:
            self.status_lbl.config(text="🎯 BƯỚC 2: Canh Cá Cắn...", fg="#f9e2af")
        elif state == BotState.STAGE_3:
            self.status_lbl.config(text="📦 BƯỚC 3: Lưu Cá Vào Balo", fg="#a6da95")
        elif state == BotState.STAGE_REPAIR:
            self.status_lbl.config(text="🔧 BƯỚC PHỤ: Sửa Cần Hỏng", fg="#cba6f7")

    def _format_simple_log(self, msg: str):
        """Translates raw engine logs into a clean, concise, human-friendly summary."""
        # Skip purely technical/telemetry lines
        if any(skip in msg for skip in ("⏱ [ĐO ĐẠC]", "└ tốn nhiều nhất", "[HIỆU NĂNG]", "record_full", "corpus/stuck", "tự động dọn dẹp")):
            return None

        # Format user-facing steps
        if "Đã thấy nút Quăng Cần" in msg:
            m = re.search(r"Click tại \((\d+),\s*(\d+)\)", msg)
            coords = f" tại ({m.group(1)}, {m.group(2)})" if m else ""
            return (f"🎣 [BƯỚC 1] Đã quăng cần câu{coords}", "STEP_CAST")

        if "Chờ Dấu Chấm Than -> Giật Cần" in msg and "==>" in msg:
            return ("🎯 [BƯỚC 2] Cần đã ở dưới nước, đang chờ cá cắn...", "STEP_WAIT")

        if "Định vị nhân vật" in msg:
            m = re.search(r"vùng đầu = \((\d+),\s*(\d+)", msg)
            coords = f" tại ({m.group(1)}, {m.group(2)})" if m else ""
            return (f"👤 Đã khóa vị trí nhân vật{coords}", "STEP_INFO")

        if "PHÁT HIỆN DẤU CHẤM THAN" in msg:
            m = re.search(r"tại (\d+),\s*(\d+)", msg)
            coords = f" tại ({m.group(1)}, {m.group(2)})" if m else ""
            return (f"⚡ [BƯỚC 2] CÁ CẮN! Phát hiện dấu '!'{coords}", "STEP_BITE")

        if "KÍCH HOẠT: Giật cần" in msg:
            m = re.search(r"tại nút bấm \((\d+),\s*(\d+)\)", msg)
            coords = f" tại ({m.group(1)}, {m.group(2)})" if m else ""
            return (f"💥 Đã bấm giật cần{coords}!", "STEP_BITE")

        if "Lưu Cá Vào Balo" in msg and "==>" in msg:
            return ("📦 [BƯỚC 3] Đang chờ kéo cá lên...", "STEP_WAIT")

        if "Đã thấy nút Bảo Quản" in msg:
            m = re.search(r"Click tại \((\d+),\s*(\d+)\)", msg)
            coords = f" tại ({m.group(1)}, {m.group(2)})" if m else ""
            return (f"📦 [BƯỚC 3] ĐÃ ẤN NÚT BẢO QUẢN{coords}! (+1 cá vào balo 🐟)", "STEP_KEEP")

        if "HOÀN THÀNH 1 VÒNG CÂU" in msg:
            sep = "━" * 46
            return (f"{sep}\n✨ HOÀN THÀNH 1 VÒNG CÂU (Tổng câu: {self.fish_count} con cá 🐟)\n{sep}", "ROUND_COMPLETE")

        if "TỰ SỬA TRẠNG THÁI" in msg:
            if "Nút Giật Cần vẫn còn" in msg:
                return ("🔄 [TỰ NẮN CHỈNH] Nút Giật Cần vẫn còn -> Tiếp tục canh cá...", "STEP_ADJUST")
            if "Bảo quản" in msg:
                return ("🔄 [TỰ NẮN CHỈNH] Đã thấy popup Bảo Quản -> Chuyển sang nhận cá...", "STEP_ADJUST")
            return ("🔄 [TỰ NẮN CHỈNH] Nhận diện màn hình -> Bot tự thích ứng!", "STEP_ADJUST")

        if "ĐỨT DÂY / HỤT CÁ" in msg:
            return ("⚠️ Đứt dây câu / hụt cá -> Tự quăng cần mới.", "STEP_WARN")

        if "CHẶN BONG BÓNG" in msg:
            return ("🛡️ Đã phát hiện & bỏ qua bóng nhiệm vụ/mong muốn trên đầu.", "STEP_INFO")

        if "Sửa Cần" in msg or "sửa cần" in msg.lower():
            if "Đã định vị nút Sửa Cần" in msg:
                return ("🔧 [SỬA CẦN] Đã bấm nút trả tiền sửa cần...", "STEP_REPAIR")
            if "Đang chờ nút xanh OK" in msg:
                return ("🔧 [SỬA CẦN] Đang chờ hoàn thành sửa chữa...", "STEP_REPAIR")
            if "Đã định vị nút OK" in msg:
                return ("🔧 [SỬA CẦN] Đã bấm nút OK xác nhận!", "STEP_REPAIR")
            if "SỬA CẦN HOÀN THÀNH" in msg:
                return ("✅ [SỬA CẦN] Cần câu đã sửa chữa hoàn tất! Tiếp tục câu.", "STEP_REPAIR")

        if "BOT ĐÃ KHỞI ĐỘNG" in msg:
            return ("🚀 BOT ĐÃ BẮT ĐẦU CHẠY (Nhấn F2 để dừng).", "ROUND_COMPLETE")

        if "BOT ĐÃ DỪNG" in msg:
            return ("⏹ BOT ĐÃ DỪNG LẠI (Nhấn F1 để bắt đầu lại).", "STEP_WARN")

        if "KẸT >30s" in msg:
            return ("⚠️ Cảnh báo: Kẹt quá 30s! Bot đã lưu ảnh bằng chứng.", "STEP_WARN")

        return None

    def _process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()

            # 1. Always append to Debug Log (Tab 2)
            tag = None
            if "PHÁT HIỆN" in msg or "HOÀN THÀNH" in msg or "✅" in msg:
                tag = "SUCCESS"
            elif "CẢNH BÁO" in msg or "HẾT GIỜ" in msg or "⚠️" in msg:
                tag = "WARN"
            elif "LỖI" in msg or "⏹" in msg:
                tag = "ALERT"
            elif "📊" in msg:
                tag = "STATS"
            elif "🔍" in msg or "🚀" in msg or "🎣" in msg or "🔧" in msg or "🔄" in msg:
                tag = "INFO"

            self.debug_text.insert(tk.END, msg + "\n", tag)
            self.debug_text.see(tk.END)

            # 2. Parse & append to Simplified Status Log (Tab 1)
            parsed = self._format_simple_log(msg)
            if parsed is not None:
                sim_text, sim_tag = parsed
                self.simple_text.insert(tk.END, sim_text + "\n", sim_tag)
                self.simple_text.see(tk.END)

        # Buffer limits
        try:
            line_count_sim = int(self.simple_text.index("end-1c").split(".")[0])
            if line_count_sim > 350:
                self.simple_text.delete("1.0", f"{line_count_sim - 250}.0")

            line_count_dbg = int(self.debug_text.index("end-1c").split(".")[0])
            if line_count_dbg > 600:
                self.debug_text.delete("1.0", f"{line_count_dbg - 400}.0")
        except Exception:
            pass

        self.root.after(80, self._process_log_queue)

    def on_close(self):
        self.engine.stop()
        self.hotkey_mgr.unregister_all()
        self.root.destroy()
        sys.exit(0)


def main():
    try:
        root = tk.Tk()
        app = AutoFishingApp(root)
        root.mainloop()
    except Exception as e:
        import traceback
        with open("crash_log.txt", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        try:
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Lỗi Khởi Động", f"Ứng dụng gặp lỗi khi khởi động:\n{e}\n\nXem chi tiết tại file crash_log.txt")
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
