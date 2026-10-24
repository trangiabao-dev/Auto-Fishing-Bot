import os
import sys
import json
import time
import queue
import threading
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
        self.root.title("Auto Fishing Bot - Play Together")
        self.root.geometry("580x620")
        self.root.minsize(500, 500)
        self.root.configure(bg="#181825")

        # Window and option states
        self.is_topmost = tk.BooleanVar(value=True)
        self.auto_clear_log = tk.BooleanVar(value=True)
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

        # Build UI
        self._build_ui()

        # Schedule auto-clear log timer (every 60s)
        self.root.after(60000, self._schedule_auto_clear)

        # Start periodic queue polling
        self.root.after(100, self._process_log_queue)

        # Cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_hotkeys(self):
        start_key = self.config.get("hotkeys", {}).get("start", "f1").lower()
        stop_key = self.config.get("hotkeys", {}).get("stop", "f2").lower()

        # Global system hooks
        self.hotkey_mgr.register(start_key, self.start_bot)
        self.hotkey_mgr.register(stop_key, self.stop_bot)
        self.hotkey_mgr.register("f3", self.engine.recorder.request_dump)  # Step 0: nhấn khi thấy '!' trên đầu

    def _build_ui(self):
        # 1. Header Frame
        header = tk.Frame(self.root, bg="#181825", pady=10)
        header.pack(fill="x", padx=15)

        title_lbl = tk.Label(
            header,
            text="🎣 AUTO FISHING BOT",
            font=("Segoe UI", 16, "bold"),
            fg="#8aadf4",
            bg="#181825"
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            header,
            text="Nhận diện thị giác máy tính • Tự bám đầu nhân vật • Chống giật nhầm",
            font=("Segoe UI", 9),
            fg="#a5adcb",
            bg="#181825"
        )
        sub_lbl.pack(anchor="w")

        hint_lbl = tk.Label(
            header,
            text="💡 Gợi ý: Có thể thu nhỏ cửa sổ bot [-] xuống Taskbar, bot vẫn tự câu ngầm!",
            font=("Segoe UI", 8, "italic"),
            fg="#94e2d5",
            bg="#181825"
        )
        hint_lbl.pack(anchor="w", pady=(2, 0))

        # 2. Status Card
        card = tk.Frame(self.root, bg="#24273a", padx=15, pady=12, relief="flat", bd=0)
        card.pack(fill="x", padx=15, pady=5)

        # Status text
        status_box = tk.Frame(card, bg="#24273a")
        status_box.pack(fill="x")

        tk.Label(status_box, text="Trạng thái:", font=("Segoe UI", 10, "bold"), fg="#cad3f5", bg="#24273a").pack(side="left")
        self.status_lbl = tk.Label(status_box, text="🔴 ĐÃ DỪNG (Sẵn sàng)", font=("Segoe UI", 10, "bold"), fg="#ed8796", bg="#24273a")
        self.status_lbl.pack(side="left", padx=10)

        # Fish counter
        self.counter_lbl = tk.Label(card, text="🐟 Số cá đã câu: 0 con", font=("Segoe UI", 11, "bold"), fg="#a6da95", bg="#24273a")
        self.counter_lbl.pack(anchor="w", pady=(8, 0))

        # 3. Control Buttons
        btn_frame = tk.Frame(self.root, bg="#181825", pady=10)
        btn_frame.pack(fill="x", padx=15)

        self.btn_start = tk.Button(
            btn_frame,
            text="▶ BẮT ĐẦU (F1)",
            font=("Segoe UI", 11, "bold"),
            bg="#a6da95",
            fg="#181825",
            activebackground="#8bd57e",
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
            relief="flat",
            cursor="hand2",
            padx=15,
            pady=8,
            command=self.stop_bot
        )
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=(6, 0))

        # 4. Checkbox Options
        opts_frame = tk.Frame(self.root, bg="#181825")
        opts_frame.pack(fill="x", padx=15, pady=(0, 5))

        top_check = tk.Checkbutton(
            opts_frame,
            text="📌 Luôn nổi (Top)",
            variable=self.is_topmost,
            font=("Segoe UI", 9),
            fg="#cad3f5",
            bg="#181825",
            activebackground="#181825",
            activeforeground="#cad3f5",
            selectcolor="#24273a",
            command=self._toggle_topmost
        )
        top_check.pack(side="left")

        auto_clear_check = tk.Checkbutton(
            opts_frame,
            text="🧹 Tự xóa log (60s)",
            variable=self.auto_clear_log,
            font=("Segoe UI", 9),
            fg="#cad3f5",
            bg="#181825",
            activebackground="#181825",
            activeforeground="#cad3f5",
            selectcolor="#24273a"
        )
        auto_clear_check.pack(side="left", padx=(8, 0))

        test_btn = tk.Button(
            opts_frame,
            text="📸 Soi Mắt Bot",
            font=("Segoe UI", 8, "bold"),
            bg="#8aadf4",
            fg="#181825",
            activebackground="#7dc4e4",
            relief="flat",
            cursor="hand2",
            padx=8,
            command=self._test_bot_vision
        )
        test_btn.pack(side="right", padx=(4, 0))

        clear_btn = tk.Button(
            opts_frame,
            text="🧹 Xóa Log",
            font=("Segoe UI", 8),
            bg="#363a4f",
            fg="#cad3f5",
            activebackground="#494d64",
            relief="flat",
            cursor="hand2",
            command=self._clear_log
        )
        clear_btn.pack(side="right")

        # 5. Real-time Log Console
        log_frame = tk.Frame(self.root, bg="#181825")
        log_frame.pack(fill="both", expand=True, padx=15, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            bg="#1e1e2e",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            font=("Consolas", 9),
            wrap="word",
            relief="flat",
            bd=0
        )
        self.log_text.pack(fill="both", expand=True)

        # Text tags for colorful logging
        self.log_text.tag_config("SUCCESS", foreground="#a6da95")
        self.log_text.tag_config("WARN", foreground="#f9e2af")
        self.log_text.tag_config("ALERT", foreground="#ed8796")
        self.log_text.tag_config("INFO", foreground="#89b4fa")

        # Initial Welcome Log
        self.on_engine_log("✨ Ứng dụng đã khởi động sẵn sàng!")
        self.on_engine_log("💡 Bấm [F1] để bắt đầu, [F2] để dừng.")
        self.on_engine_log("🎯 [BƯỚC 0 CHẨN ĐOÁN] Thấy dấu '!' trên đầu hãy bấm [F3] ngay để lưu dữ liệu!")

    def _toggle_topmost(self):
        self.root.attributes("-topmost", self.is_topmost.get())

    def _schedule_auto_clear(self):
        if self.auto_clear_log.get():
            # Tránh xóa log đúng khoảnh khắc đang chờ cá cắn (STAGE_2) để ưu tiên 100% tài nguyên CPU & phản xạ
            if self.engine.state == BotState.STAGE_2:
                self.root.after(10000, self._schedule_auto_clear)
                return
            self._clear_log(auto=True)
        self.root.after(60000, self._schedule_auto_clear)

    def _clear_log(self, auto: bool = False):
        self.log_text.delete("1.0", tk.END)
        if auto:
            now = datetime.now().strftime("%H:%M:%S")
            self.on_engine_log(f"[{now}] 🧹 [TỰ ĐỘNG DỌN DẸP] Đã xóa log định kỳ 60s. Bot vẫn đang chạy ngầm an toàn.")

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
        self.counter_lbl.config(text=f"🐟 Số cá đã câu: {self.fish_count} con")

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

    def _process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            tag = None
            if "PHÁT HIỆN" in msg or "HOÀN THÀNH" in msg or "✅" in msg:
                tag = "SUCCESS"
            elif "CẢNH BÁO" in msg or "HẾT GIỜ" in msg or "⚠️" in msg:
                tag = "WARN"
            elif "LỖI" in msg or "⏹" in msg:
                tag = "ALERT"
            elif "🔍" in msg or "🚀" in msg or "🎣" in msg or "🔧" in msg:
                tag = "INFO"

            self.log_text.insert(tk.END, msg + "\n", tag)
            self.log_text.see(tk.END)

        # Buffer cap: Keep at most 200 lines to prevent memory bloat & sluggish GUI
        try:
            line_count = int(self.log_text.index("end-1c").split(".")[0])
            if line_count > 250:
                self.log_text.delete("1.0", f"{line_count - 150}.0")
        except Exception:
            pass

        self.root.after(100, self._process_log_queue)

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
