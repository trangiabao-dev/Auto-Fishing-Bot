import os
import sys
import json
import time
import queue
import re
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog

from bot_engine import BotEngine, BotState
from core.hotkey_manager import HotkeyManager
from core.farm_config_manager import FarmConfigManager, FruitRule
from core.farm_engine import FarmEngine, FarmState

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
        
        # Dimensions: compact by default, fits comfortably on 1080p without blocking game action
        self.expanded_w = 520
        self.expanded_h = 580
        self.mini_h = 175
        self.is_mini = False
        
        self.root.geometry(f"{self.expanded_w}x{self.expanded_h}")
        self.root.minsize(400, self.mini_h)
        self.root.configure(bg="#181825")

        # Window options
        self.is_topmost = tk.BooleanVar(value=True)
        self.is_transparent = tk.BooleanVar(value=False)
        self.auto_clear_simple = tk.BooleanVar(value=False)
        self.auto_clear_debug = tk.BooleanVar(value=True)
        self.auto_clear_farm = tk.BooleanVar(value=False)
        self.root.attributes("-topmost", True)

        # Thread-safe log queues
        self.log_queue = queue.Queue()
        self.farm_log_queue = queue.Queue()
        self.fish_count = 0
        self.farm_harvest_count = 0
        self.farm_sell_count = 0

        # Active Mode: "FISHING" or "FARM"
        self.active_mode = "FISHING"

        # Load fishing engine configuration
        self.config = load_config()

        # Initialize Fishing Bot Engine
        self.engine = BotEngine(
            self.config,
            on_log=self.on_engine_log,
            on_state_change=self.on_engine_state_change,
            on_fish_caught=self.on_fish_caught
        )

        # Initialize Farm Configuration & Engine
        self.farm_cfg_mgr = FarmConfigManager()
        self.farm_engine = FarmEngine(
            config_manager=self.farm_cfg_mgr,
            log_callback=self.on_farm_log,
            state_callback=self.on_farm_state_change
        )

        # Fruit Table UI mapping
        self.fruit_vars = {}
        self.fruit_entries = {}

        # Initialize Hotkey Manager
        self.hotkey_mgr = HotkeyManager()
        self._setup_hotkeys()

        # Build Dual-Mode Modern UI
        self._build_ui()

        # Schedule auto-clear log timer (every 60s)
        self.root.after(60000, self._schedule_auto_clear)

        # Start periodic queue polling
        self.root.after(80, self._process_log_queue)
        self.root.after(80, self._process_farm_log_queue)

        # Cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_hotkeys(self):
        start_key = self.config.get("hotkeys", {}).get("start", "f1").lower()
        stop_key = self.config.get("hotkeys", {}).get("stop", "f2").lower()

        # Fishing hotkeys
        self.hotkey_mgr.register(start_key, self.start_bot)
        self.hotkey_mgr.register(stop_key, self.stop_bot)
        self.hotkey_mgr.register("f3", self.engine.recorder.request_dump)

        # Farm hotkeys
        self.hotkey_mgr.register("f5", self.start_farm_bot)
        self.hotkey_mgr.register("f6", self.stop_farm_bot)

    def _build_ui(self):
        # 1. Top Mode Selector Bar: [ 🎣 CÂU CÁ ] [ 🌾 NÔNG TRẠI ]
        mode_bar = tk.Frame(self.root, bg="#11111b", pady=3)
        mode_bar.pack(fill="x", padx=10, pady=(6, 2))

        self.btn_tab_fishing = tk.Button(
            mode_bar,
            text="🎣 CÂU CÁ (F1/F2)",
            font=("Segoe UI", 9, "bold"),
            bg="#313244",
            fg="#8aadf4",
            activebackground="#45475a",
            relief="flat",
            cursor="hand2",
            padx=12,
            pady=3,
            command=lambda: self.switch_mode("FISHING")
        )
        self.btn_tab_fishing.pack(side="left", padx=(0, 6))

        self.btn_tab_farm = tk.Button(
            mode_bar,
            text="🌾 NÔNG TRẠI (F5/F6)",
            font=("Segoe UI", 9, "bold"),
            bg="#1e1e2e",
            fg="#a6adc8",
            activebackground="#313244",
            relief="flat",
            cursor="hand2",
            padx=12,
            pady=3,
            command=lambda: self.switch_mode("FARM")
        )
        self.btn_tab_farm.pack(side="left")

        # Mini Mode & Window Quick Controls on the right of mode bar
        trans_check = tk.Checkbutton(
            mode_bar,
            text="🌓 Mờ",
            variable=self.is_transparent,
            font=("Segoe UI", 8),
            fg="#cad3f5",
            bg="#11111b",
            activebackground="#11111b",
            activeforeground="#cad3f5",
            selectcolor="#1e1e2e",
            command=self._toggle_transparency
        )
        trans_check.pack(side="right", padx=2)

        top_check = tk.Checkbutton(
            mode_bar,
            text="📌 Top",
            variable=self.is_topmost,
            font=("Segoe UI", 8),
            fg="#cad3f5",
            bg="#11111b",
            activebackground="#11111b",
            activeforeground="#cad3f5",
            selectcolor="#1e1e2e",
            command=self._toggle_topmost
        )
        top_check.pack(side="right", padx=2)

        # Containers for both modes
        self.container_fishing = tk.Frame(self.root, bg="#181825")
        self.container_farm = tk.Frame(self.root, bg="#181825")

        self.container_fishing.pack(fill="both", expand=True)

        self._build_fishing_ui()
        self._build_farm_ui()

    def switch_mode(self, mode: str):
        if mode == "FISHING":
            self.container_farm.pack_forget()
            self.container_fishing.pack(fill="both", expand=True)
            self.btn_tab_fishing.config(bg="#313244", fg="#8aadf4")
            self.btn_tab_farm.config(bg="#1e1e2e", fg="#a6adc8")
            self.active_mode = "FISHING"
            if self.is_mini:
                self.notebook_frame.pack_forget()
            else:
                self.notebook_frame.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        else:
            self.container_fishing.pack_forget()
            self.container_farm.pack(fill="both", expand=True)
            self.btn_tab_farm.config(bg="#313244", fg="#a6da95")
            self.btn_tab_fishing.config(bg="#1e1e2e", fg="#a6adc8")
            self.active_mode = "FARM"
            if self.is_mini:
                if hasattr(self, "farm_table_card"):
                    self.farm_table_card.pack_forget()
                if hasattr(self, "farm_log_frame"):
                    self.farm_log_frame.pack_forget()
            else:
                if hasattr(self, "farm_table_card"):
                    self.farm_table_card.pack(fill="x", padx=10, pady=(2, 4))
                if hasattr(self, "farm_log_frame"):
                    self.farm_log_frame.pack(fill="both", expand=True, padx=10, pady=(2, 6))

    def _build_fishing_ui(self):
        header = tk.Frame(self.container_fishing, bg="#181825")
        header.pack(fill="x", padx=10, pady=(4, 2))

        title_lbl = tk.Label(
            header,
            text="🎣 AUTO FISHING BOT",
            font=("Segoe UI", 12, "bold"),
            fg="#8aadf4",
            bg="#181825"
        )
        title_lbl.pack(side="left")

        ver_pill = tk.Label(
            header,
            text="PRO v2.5",
            font=("Segoe UI", 8, "bold"),
            bg="#313244",
            fg="#a6da95",
            padx=5,
            pady=1
        )
        ver_pill.pack(side="left", padx=6)

        self.counter_lbl = tk.Label(
            header,
            text="🐟 0 con",
            font=("Segoe UI", 9, "bold"),
            fg="#a6da95",
            bg="#1e1e2e",
            padx=8,
            pady=1,
            highlightbackground="#313244",
            highlightthickness=1
        )
        self.counter_lbl.pack(side="right")

        card = tk.Frame(self.container_fishing, bg="#1e1e2e", padx=10, pady=4, highlightbackground="#313244", highlightthickness=1)
        card.pack(fill="x", padx=10, pady=(2, 4))

        self.status_lbl = tk.Label(
            card,
            text="🔴 ĐÃ DỪNG (Nhấn F1)",
            font=("Segoe UI", 9, "bold"),
            fg="#ed8796",
            bg="#1e1e2e"
        )
        self.status_lbl.pack(side="left")

        self.ticker_lbl = tk.Label(
            card,
            text="Sẵn sàng bắt đầu",
            font=("Segoe UI", 8),
            fg="#a5adcb",
            bg="#1e1e2e",
            anchor="e"
        )
        self.ticker_lbl.pack(side="right", fill="x", expand=True)

        btn_frame = tk.Frame(self.container_fishing, bg="#181825")
        btn_frame.pack(fill="x", padx=10, pady=2)

        self.btn_start = tk.Button(
            btn_frame,
            text="▶ BẮT ĐẦU CÂU (F1)",
            font=("Segoe UI", 10, "bold"),
            bg="#a6da95",
            fg="#181825",
            activebackground="#8bd57e",
            activeforeground="#181825",
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=4,
            command=self.start_bot
        )
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.btn_stop = tk.Button(
            btn_frame,
            text="⏹ DỪNG LẠI (F2)",
            font=("Segoe UI", 10, "bold"),
            bg="#ed8796",
            fg="#181825",
            activebackground="#e0697a",
            activeforeground="#181825",
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=4,
            command=self.stop_bot
        )
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=(4, 0))

        opts_bar = tk.Frame(self.container_fishing, bg="#181825")
        opts_bar.pack(fill="x", padx=10, pady=(2, 4))

        self.toggle_mini_btn = tk.Button(
            opts_bar,
            text="🗕 Thu Gọn",
            font=("Segoe UI", 8, "bold"),
            bg="#8aadf4",
            fg="#181825",
            activebackground="#7dc4e4",
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=1,
            command=self._toggle_mini_mode
        )
        self.toggle_mini_btn.pack(side="right")

        test_btn = tk.Button(
            opts_bar,
            text="📸 Soi Mắt Bot",
            font=("Segoe UI", 8),
            bg="#313244",
            fg="#cad3f5",
            activebackground="#45475a",
            relief="flat",
            cursor="hand2",
            padx=6,
            pady=1,
            command=self._test_bot_vision
        )
        test_btn.pack(side="right", padx=4)

        self.notebook_frame = tk.Frame(self.container_fishing, bg="#181825")
        self.notebook_frame.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Custom.TNotebook', background='#181825', borderwidth=0)
        style.configure('Custom.TNotebook.Tab', background='#1e1e2e', foreground='#cad3f5', padding=[10, 4], font=('Segoe UI', 8, 'bold'), borderwidth=0)
        style.map('Custom.TNotebook.Tab',
                  background=[('selected', '#313244'), ('active', '#24273a')],
                  foreground=[('selected', '#8aadf4'), ('active', '#b4befe')])

        self.notebook = ttk.Notebook(self.notebook_frame, style='Custom.TNotebook')
        self.notebook.pack(fill='both', expand=True)

        self.tab_simple = tk.Frame(self.notebook, bg="#181825")
        self.notebook.add(self.tab_simple, text="📋 Rút Gọn")

        t1_toolbar = tk.Frame(self.tab_simple, bg="#181825", pady=2)
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
            padx=6,
            pady=1,
            command=self._clear_simple_log
        ).pack(side="right")

        self.simple_text = scrolledtext.ScrolledText(
            self.tab_simple,
            bg="#1e1e2e",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            font=("Segoe UI", 8),
            wrap="word",
            relief="flat",
            bd=0,
            padx=6,
            pady=4
        )
        self.simple_text.pack(fill="both", expand=True)

        self.simple_text.tag_config("ROUND_COMPLETE", foreground="#a6da95", font=("Segoe UI", 8, "bold"))
        self.simple_text.tag_config("STEP_CAST", foreground="#89b4fa", font=("Segoe UI", 8, "bold"))
        self.simple_text.tag_config("STEP_WAIT", foreground="#f9e2af", font=("Segoe UI", 8))
        self.simple_text.tag_config("STEP_BITE", foreground="#f38ba8", font=("Segoe UI", 9, "bold"))
        self.simple_text.tag_config("STEP_KEEP", foreground="#a6e3a1", font=("Segoe UI", 8, "bold"))
        self.simple_text.tag_config("STEP_REPAIR", foreground="#cba6f7", font=("Segoe UI", 8, "bold"))
        self.simple_text.tag_config("STEP_ADJUST", foreground="#89dceb", font=("Segoe UI", 8, "italic"))
        self.simple_text.tag_config("STEP_WARN", foreground="#fab387", font=("Segoe UI", 8))
        self.simple_text.tag_config("STEP_INFO", foreground="#cad3f5", font=("Segoe UI", 8))
        self.simple_text.tag_config("SEPARATOR", foreground="#fab387", font=("Segoe UI", 8, "bold"))

        self.tab_debug = tk.Frame(self.notebook, bg="#181825")
        self.notebook.add(self.tab_debug, text="🛠 Chi Tiết (Debug)")

        t2_toolbar = tk.Frame(self.tab_debug, bg="#181825", pady=2)
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
        self.copy_status_lbl.pack(side="left", padx=6)

        tk.Button(
            t2_toolbar,
            text="📋 Copy Toàn Bộ Log",
            font=("Segoe UI", 8, "bold"),
            bg="#8aadf4",
            fg="#181825",
            activebackground="#7dc4e4",
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=1,
            command=self._copy_debug_log
        ).pack(side="right", padx=(4, 0))

        tk.Button(
            t2_toolbar,
            text="🧹 Xóa Log",
            font=("Segoe UI", 8),
            bg="#313244",
            fg="#cad3f5",
            activebackground="#45475a",
            relief="flat",
            cursor="hand2",
            padx=6,
            pady=1,
            command=self._clear_debug_log
        ).pack(side="right")

        self.debug_text = scrolledtext.ScrolledText(
            self.tab_debug,
            bg="#1e1e2e",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            font=("Consolas", 8),
            wrap="word",
            relief="flat",
            bd=0,
            padx=6,
            pady=4
        )
        self.debug_text.pack(fill="both", expand=True)

        self.debug_text.tag_config("SUCCESS", foreground="#a6da95")
        self.debug_text.tag_config("WARN", foreground="#f9e2af")
        self.debug_text.tag_config("ALERT", foreground="#ed8796")
        self.debug_text.tag_config("INFO", foreground="#89b4fa")
        self.debug_text.tag_config("STATS", foreground="#cba6f7")

        self.on_engine_log("✨ Ứng dụng đã sẵn sàng! Bấm [F1] để bắt đầu câu cá, [F2] để dừng.")

    def _build_farm_ui(self):
        # 1. Header Frame: Title + Badge + Farm Counter
        header = tk.Frame(self.container_farm, bg="#181825")
        header.pack(fill="x", padx=10, pady=(4, 2))

        title_lbl = tk.Label(
            header,
            text="🌾 AUTO FARM",
            font=("Segoe UI", 12, "bold"),
            fg="#a6da95",
            bg="#181825"
        )
        title_lbl.pack(side="left")

        ver_pill = tk.Label(
            header,
            text="PRO v2.5",
            font=("Segoe UI", 8, "bold"),
            bg="#313244",
            fg="#8aadf4",
            padx=5,
            pady=1
        )
        ver_pill.pack(side="left", padx=6)

        self.farm_counter_lbl = tk.Label(
            header,
            text="🧺 0 quả  |  💰 0 đợt",
            font=("Segoe UI", 9, "bold"),
            fg="#a6da95",
            bg="#1e1e2e",
            padx=8,
            pady=2,
            highlightbackground="#313244",
            highlightthickness=1
        )
        self.farm_counter_lbl.pack(side="right")

        # 2. Status & Live Ticker Card
        card = tk.Frame(self.container_farm, bg="#1e1e2e", padx=10, pady=4, highlightbackground="#313244", highlightthickness=1)
        card.pack(fill="x", padx=10, pady=(2, 4))

        self.farm_status_lbl = tk.Label(
            card,
            text="🔴 ĐÃ DỪNG (Nhấn F5)",
            font=("Segoe UI", 9, "bold"),
            fg="#ed8796",
            bg="#1e1e2e"
        )
        self.farm_status_lbl.pack(side="left")

        self.farm_ticker_lbl = tk.Label(
            card,
            text="Sẵn sàng thu hoạch",
            font=("Segoe UI", 8),
            fg="#a5adcb",
            bg="#1e1e2e",
            anchor="e"
        )
        self.farm_ticker_lbl.pack(side="right", fill="x", expand=True)

        # 3. Compact Control Buttons (F5 / F6)
        btn_frame = tk.Frame(self.container_farm, bg="#181825")
        btn_frame.pack(fill="x", padx=10, pady=2)

        self.btn_farm_start = tk.Button(
            btn_frame,
            text="▶ BẮT ĐẦU THU HOẠCH (F5)",
            font=("Segoe UI", 10, "bold"),
            bg="#a6da95",
            fg="#181825",
            activebackground="#8bd57e",
            activeforeground="#181825",
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=4,
            command=self.start_farm_bot
        )
        self.btn_farm_start.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.btn_farm_stop = tk.Button(
            btn_frame,
            text="⏹ DỪNG LẠI (F6)",
            font=("Segoe UI", 10, "bold"),
            bg="#ed8796",
            fg="#181825",
            activebackground="#e0697a",
            activeforeground="#181825",
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=4,
            command=self.stop_farm_bot
        )
        self.btn_farm_stop.pack(side="left", expand=True, fill="x", padx=(4, 0))

        # 4. Multi-Fruit Checklist Table Card
        self.farm_table_card = tk.Frame(self.container_farm, bg="#1e1e2e", padx=6, pady=4, highlightbackground="#313244", highlightthickness=1)
        self.farm_table_card.pack(fill="x", padx=10, pady=(2, 4))

        # Header Row (5-Column Grid with exact alignment + scrollbar spacer)
        hdr_bg = "#24273a"
        hdr = tk.Frame(self.farm_table_card, bg=hdr_bg, pady=3)
        hdr.pack(fill="x", pady=(0, 2))
        hdr.columnconfigure(0, weight=0, minsize=42)
        hdr.columnconfigure(1, weight=1, minsize=100)
        hdr.columnconfigure(2, weight=0, minsize=80)
        hdr.columnconfigure(3, weight=0, minsize=130)
        hdr.columnconfigure(4, weight=0, minsize=40)
        hdr.columnconfigure(5, weight=0, minsize=14)

        tk.Label(hdr, text="BẬT", font=("Segoe UI", 8, "bold"), fg="#8aadf4", bg=hdr_bg).grid(row=0, column=0, padx=(6, 0))
        tk.Label(hdr, text="TÊN QUẢ/HẠT", font=("Segoe UI", 8, "bold"), fg="#8aadf4", bg=hdr_bg).grid(row=0, column=1, sticky="w", padx=6)
        tk.Label(hdr, text="NHẬP KG", font=("Segoe UI", 8, "bold"), fg="#89b4fa", bg=hdr_bg).grid(row=0, column=2, padx=4)
        tk.Label(hdr, text="BỎ QUA NẾU ≥", font=("Segoe UI", 8, "bold"), fg="#fab387", bg=hdr_bg).grid(row=0, column=3, padx=4)
        tk.Label(hdr, text="XÓA", font=("Segoe UI", 8, "bold"), fg="#ed8796", bg=hdr_bg).grid(row=0, column=4, sticky="e", padx=(0, 4))

        # Dynamic scrollable rows container (Canvas + Scrollbar)
        scroll_container = tk.Frame(self.farm_table_card, bg="#1e1e2e")
        scroll_container.pack(fill="x")

        self.farm_rows_canvas = tk.Canvas(scroll_container, bg="#1e1e2e", height=160, highlightthickness=0, bd=0)
        self.farm_rows_sb = tk.Scrollbar(scroll_container, orient="vertical", command=self.farm_rows_canvas.yview, bg="#1e1e2e", troughcolor="#181825", width=12)

        self.farm_rows_frame = tk.Frame(self.farm_rows_canvas, bg="#1e1e2e")
        self.farm_rows_win = self.farm_rows_canvas.create_window((0, 0), window=self.farm_rows_frame, anchor="nw")

        self.farm_rows_frame.bind("<Configure>", lambda e: self.farm_rows_canvas.configure(scrollregion=self.farm_rows_canvas.bbox("all")))
        self.farm_rows_canvas.bind("<Configure>", lambda e: self.farm_rows_canvas.itemconfig(self.farm_rows_win, width=e.width))
        self.farm_rows_canvas.configure(yscrollcommand=self.farm_rows_sb.set)

        self.farm_rows_canvas.pack(side="left", fill="both", expand=True)
        self.farm_rows_sb.pack(side="right", fill="y")

        self.farm_rows_canvas.bind("<MouseWheel>", self._on_fruit_table_scroll)
        self.farm_rows_frame.bind("<MouseWheel>", self._on_fruit_table_scroll)
        self._render_fruit_rows()

        # Table footer: Add Fruit button + Help text + Auto sell checkbox
        ftr = tk.Frame(self.farm_table_card, bg="#1e1e2e", pady=3)
        ftr.pack(fill="x")

        tk.Button(
            ftr,
            text="➕ Thêm Quả Mới",
            font=("Segoe UI", 8, "bold"),
            bg="#313244",
            fg="#a6da95",
            activebackground="#45475a",
            relief="flat",
            cursor="hand2",
            padx=6,
            pady=1,
            command=self._add_new_fruit_dialog
        ).pack(side="left")

        self.var_auto_sell = tk.BooleanVar(value=self.farm_cfg_mgr.auto_sell)
        sell_frame = tk.Frame(ftr, bg="#1e1e2e", cursor="hand2")
        sell_frame.pack(side="right")

        sell_sym = "☑" if self.farm_cfg_mgr.auto_sell else "☐"
        sell_fg = "#a6da95" if self.farm_cfg_mgr.auto_sell else "#6c7086"
        self.lbl_sell_chk = tk.Label(
            sell_frame,
            text=sell_sym,
            font=("Segoe UI", 11, "bold"),
            fg=sell_fg,
            bg="#1e1e2e"
        )
        self.lbl_sell_chk.pack(side="left", padx=(0, 4))

        lbl_sell_text = tk.Label(
            sell_frame,
            text="💰 Tự bán khi đầy túi",
            font=("Segoe UI", 8),
            fg="#cad3f5",
            bg="#1e1e2e"
        )
        lbl_sell_text.pack(side="left")

        def _on_sell_click(event=None):
            new_val = not self.farm_cfg_mgr.auto_sell
            self.farm_cfg_mgr.auto_sell = new_val
            self.farm_cfg_mgr.save()
            self.var_auto_sell.set(new_val)
            s_sym = "☑" if new_val else "☐"
            s_fg = "#a6da95" if new_val else "#6c7086"
            self.lbl_sell_chk.config(text=s_sym, fg=s_fg)
            st = "BẬT" if new_val else "TẮT"
            self.on_farm_log(f"💾 Đã {st} tính năng tự động bán khi đầy túi.")

        sell_frame.bind("<Button-1>", _on_sell_click)
        self.lbl_sell_chk.bind("<Button-1>", _on_sell_click)
        lbl_sell_text.bind("<Button-1>", _on_sell_click)

        # 5. Farm Log Box
        self.farm_log_frame = tk.Frame(self.container_farm, bg="#181825")
        self.farm_log_frame.pack(fill="both", expand=True, padx=10, pady=(2, 6))

        t_bar = tk.Frame(self.farm_log_frame, bg="#181825", pady=2)
        t_bar.pack(fill="x")

        tk.Label(t_bar, text="📋 NHẬT KÝ NÔNG TRẠI", font=("Segoe UI", 8, "bold"), fg="#8aadf4", bg="#181825").pack(side="left")

        tk.Checkbutton(
            t_bar,
            text="🧹 Tự xóa log (60s)",
            variable=self.auto_clear_farm,
            font=("Segoe UI", 8),
            fg="#a6adc8",
            bg="#181825",
            activebackground="#181825",
            selectcolor="#1e1e2e"
        ).pack(side="left", padx=(10, 0))

        tk.Button(
            t_bar,
            text="🧹 Xóa Log",
            font=("Segoe UI", 8),
            bg="#313244",
            fg="#cad3f5",
            relief="flat",
            cursor="hand2",
            padx=6,
            pady=1,
            command=self._clear_farm_log
        ).pack(side="right")

        self.farm_text = scrolledtext.ScrolledText(
            self.farm_log_frame,
            bg="#1e1e2e",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            font=("Segoe UI", 8),
            wrap="word",
            relief="flat",
            bd=0,
            padx=6,
            pady=4
        )
        self.farm_text.pack(fill="both", expand=True)

        self.farm_text.tag_config("HARVEST", foreground="#a6da95", font=("Segoe UI", 8, "bold"))
        self.farm_text.tag_config("SKIP_HEAVY", foreground="#fab387", font=("Segoe UI", 8))
        self.farm_text.tag_config("BAG_FULL", foreground="#f38ba8", font=("Segoe UI", 8, "bold"))
        self.farm_text.tag_config("SELL", foreground="#89dceb", font=("Segoe UI", 8, "bold"))
        self.farm_text.tag_config("INFO", foreground="#89b4fa", font=("Segoe UI", 8))
        self.farm_text.tag_config("WARN", foreground="#f9e2af", font=("Segoe UI", 8))

        self.on_farm_log("🌾 Hệ thống Nông trại đã sẵn sàng! Tích chọn quả và bấm [F5] để chạy.")

    def _render_fruit_rows(self):
        for widget in self.farm_rows_frame.winfo_children():
            widget.destroy()

        self.fruit_vars.clear()
        self.fruit_entries.clear()

        rules = self.farm_cfg_mgr.list_rules()
        for idx, rule in enumerate(rules):
            row_bg = "#1e1e2e" if idx % 2 == 0 else "#25273a"
            row = tk.Frame(self.farm_rows_frame, bg=row_bg, pady=2)
            row.pack(fill="x", pady=1)
            row.columnconfigure(0, weight=0, minsize=42)
            row.columnconfigure(1, weight=1, minsize=100)
            row.columnconfigure(2, weight=0, minsize=80)
            row.columnconfigure(3, weight=0, minsize=130)
            row.columnconfigure(4, weight=0, minsize=40)

            # Modern High-Contrast Checkbox (☑ vs ☐) - 100% visible on dark theme
            var = tk.BooleanVar(value=rule.enabled)
            self.fruit_vars[rule.name] = var
            sym = "☑" if rule.enabled else "☐"
            fg_col = "#a6da95" if rule.enabled else "#6c7086"
            chk = tk.Label(
                row,
                text=sym,
                font=("Segoe UI", 12, "bold"),
                fg=fg_col,
                bg=row_bg,
                cursor="hand2"
            )
            chk.grid(row=0, column=0, padx=(6, 0))
            chk.bind("<Button-1>", lambda e, name=rule.name, label=chk: self._toggle_fruit_custom(name, label))

            # Fruit Name
            lbl_name = tk.Label(row, text=rule.name, font=("Segoe UI", 8, "bold"), fg="#cad3f5", bg=row_bg)
            lbl_name.grid(row=0, column=1, sticky="w", padx=6)

            # Entry for Max kg (Centered over 'NHẬP KG')
            ent = tk.Entry(row, width=6, font=("Segoe UI", 8, "bold"), bg="#313244", fg="#a6da95", insertbackground="#cad3f5", relief="flat", justify="center")
            val_str = f"{rule.max_kg:.2f}" if rule.max_kg > 0.0 else ""
            ent.insert(0, val_str)
            ent.grid(row=0, column=2, padx=4)
            ent.bind("<FocusOut>", lambda e, name=rule.name: self._on_fruit_kg_change(name))
            ent.bind("<Return>", lambda e, name=rule.name: self._on_fruit_kg_change(name))
            self.fruit_entries[rule.name] = ent

            # Badge indicator (Centered directly under 'BỎ QUA NẾU ≥' with matching orange color)
            badge_text = f"< {rule.max_kg:.2f} kg" if rule.max_kg > 0.0 else "⚡ Thu hết"
            badge_fg = "#fab387" if rule.max_kg > 0.0 else "#a6da95"
            badge = tk.Label(row, text=badge_text, font=("Segoe UI", 8, "bold"), fg=badge_fg, bg="#313244", width=12, pady=1)
            badge.grid(row=0, column=3, padx=4)

            # Delete button
            btn_del = tk.Button(
                row,
                text="✕",
                font=("Segoe UI", 8, "bold"),
                bg="#313244",
                fg="#ed8796",
                activebackground="#45475a",
                relief="flat",
                cursor="hand2",
                padx=5,
                pady=0,
                command=lambda name=rule.name: self._delete_fruit(name)
            )
            btn_del.grid(row=0, column=4, sticky="e", padx=(0, 4))

            # Bind mousewheel to row and all its components
            for w in (row, chk, lbl_name, ent, badge, btn_del):
                w.bind("<MouseWheel>", self._on_fruit_table_scroll)

    def _on_fruit_table_scroll(self, event):
        if hasattr(self, 'farm_rows_canvas'):
            self.farm_rows_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _toggle_fruit_custom(self, name: str, label: tk.Label):
        rule = self.farm_cfg_mgr.get_rule(name)
        if not rule:
            return
        rule.enabled = not rule.enabled
        self.farm_cfg_mgr.save()
        if name in self.fruit_vars:
            self.fruit_vars[name].set(rule.enabled)

        sym = "☑" if rule.enabled else "☐"
        fg_col = "#a6da95" if rule.enabled else "#6c7086"
        label.config(text=sym, fg=fg_col)

        status = "ĐÃ CHỌN" if rule.enabled else "ĐÃ BỎ"
        self.on_farm_log(f"💾 {status} quả '{name}' vào danh sách thu hoạch.")

    def _on_fruit_toggle(self, name: str):
        rule = self.farm_cfg_mgr.get_rule(name)
        if rule:
            rule.enabled = self.fruit_vars[name].get()
            self.farm_cfg_mgr.save()
            status = "ĐÃ CHỌN" if rule.enabled else "ĐÃ BỎ"
            self.on_farm_log(f"💾 {status} quả '{name}' vào danh sách thu hoạch.")

    def _on_fruit_kg_change(self, name: str):
        ent = self.fruit_entries.get(name)
        if not ent:
            return
        val_str = ent.get().strip()
        try:
            val = float(val_str) if val_str else 0.0
        except ValueError:
            val = 0.0
        rule = self.farm_cfg_mgr.get_rule(name)
        if rule:
            rule.max_kg = val
            self.farm_cfg_mgr.save()
            if val > 0:
                self.on_farm_log(f"💾 Đã lưu: '{name}' -> Chỉ thu quả < {val:.2f} kg (Bỏ qua quả khổng lồ).")
            else:
                self.on_farm_log(f"💾 Đã lưu: '{name}' -> Thu hoạch tất cả không giới hạn số kg.")
        self._render_fruit_rows()

    def _delete_fruit(self, name: str):
        if messagebox.askyesno("Xác Nhận Xóa", f"Bạn có chắc muốn xóa loại quả '{name}' khỏi danh sách?"):
            self.farm_cfg_mgr.remove_rule(name)
            self._render_fruit_rows()
            self.on_farm_log(f"🗑️ Đã xóa loại quả '{name}'.")

    def _on_auto_sell_toggle(self):
        self.farm_cfg_mgr.auto_sell = self.var_auto_sell.get()
        self.farm_cfg_mgr.save()
        status = "BẬT" if self.farm_cfg_mgr.auto_sell else "TẮT"
        self.on_farm_log(f"💾 Đã {status} chế độ tự động bán tại Oliver khi đầy túi.")

    def _add_new_fruit_dialog(self):
        name = simpledialog.askstring("Thêm Quả Mới", "Nhập tên loại quả/hạt trong game:")
        if not name or not name.strip():
            return
        name = name.strip()
        kg_str = simpledialog.askstring("Ngưỡng Khổng Lồ", f"Nhập ngưỡng kg khổng lồ cho '{name}' (bỏ trống để thu tất cả):", initialvalue="3.0")
        try:
            val = float(kg_str) if kg_str and kg_str.strip() else 0.0
        except ValueError:
            val = 0.0
        self.farm_cfg_mgr.set_rule(name, max_kg=val, enabled=True)
        self._render_fruit_rows()
        if val > 0:
            self.on_farm_log(f"➕ Đã thêm quả '{name}' (Chỉ thu < {val:.2f} kg)!")
        else:
            self.on_farm_log(f"➕ Đã thêm quả '{name}' (Thu hoạch tất cả)!")

    def _clear_farm_log(self):
        self.farm_text.delete("1.0", tk.END)

    def _collapse_farm_ui(self):
        """Auto-collapses Farm UI to mini bar (Header + Buttons only) when harvesting starts."""
        if hasattr(self, "farm_table_card") and self.farm_table_card.winfo_ismapped():
            self.farm_table_card.pack_forget()
        if hasattr(self, "farm_log_frame") and self.farm_log_frame.winfo_ismapped():
            self.farm_log_frame.pack_forget()
        self.root.geometry(f"{self.expanded_w}x{self.mini_h}")
        self.toggle_mini_btn.config(text="🗖 Mở Rộng", bg="#a6da95")
        self.is_mini = True

    def _expand_farm_ui(self):
        """Auto-expands Farm UI back to full table & log view when harvesting stops."""
        if hasattr(self, "farm_table_card") and not self.farm_table_card.winfo_ismapped():
            self.farm_table_card.pack(fill="x", padx=10, pady=(2, 4))
        if hasattr(self, "farm_log_frame") and not self.farm_log_frame.winfo_ismapped():
            self.farm_log_frame.pack(fill="both", expand=True, padx=10, pady=(2, 6))
        self.root.geometry(f"{self.expanded_w}x{self.expanded_h}")
        self.toggle_mini_btn.config(text="🗕 Thu Gọn", bg="#8aadf4")
        self.is_mini = False

    def start_farm_bot(self):
        if self.engine.state != BotState.STOPPED and self.engine.state != BotState.IDLE:
            messagebox.showwarning("Cảnh Báo", "Bot Câu Cá đang chạy! Vui lòng bấm [F2] để dừng câu cá trước khi bật Nông Trại.")
            return

        enabled_rules = self.farm_cfg_mgr.get_enabled_rules()
        if not enabled_rules:
            messagebox.showwarning("Chưa Chọn Quả", "Vui lòng tích chọn ít nhất 1 loại quả trong bảng nông sản trước khi bắt đầu!")
            return

        # Auto collapse Farm tab to mini mode upon starting
        self._collapse_farm_ui()
        self.farm_engine.start()

    def stop_farm_bot(self):
        self.farm_engine.stop()
        # Auto expand Farm tab back to normal upon stopping
        self._expand_farm_ui()

    def on_farm_log(self, msg: str):
        self.farm_log_queue.put(msg)

    def on_farm_state_change(self, new_state: FarmState):
        self.root.after(0, self._update_farm_state_ui, new_state)

    def _update_farm_state_ui(self, state: FarmState):
        if state == FarmState.STOPPED:
            self.farm_status_lbl.config(text="🔴 ĐÃ DỪNG (F5)", fg="#ed8796")
            self.farm_ticker_lbl.config(text="Sẵn sàng")
            if self.is_mini:
                self._expand_farm_ui()
        elif state == FarmState.TELEPORTING:
            self.farm_status_lbl.config(text="🚀 Đến Oliver (B1)", fg="#89b4fa")
            self.farm_ticker_lbl.config(text="Bấm 'Bán' để dịch chuyển tới ông bán quả...")
        elif state == FarmState.OPENING_HARVEST:
            self.farm_status_lbl.config(text="🌱 Mở Vườn (B2)", fg="#89b4fa")
            self.farm_ticker_lbl.config(text="Bấm 'Thu hoạch' mở bảng nông sản...")
        elif state == FarmState.APPLYING_FILTER:
            self.farm_status_lbl.config(text="🔍 Cài Đặt Lọc", fg="#cba6f7")
            self.farm_ticker_lbl.config(text="Đang chọn loại quả & loại trừ đột biến...")
        elif state == FarmState.HARVESTING:
            self.farm_status_lbl.config(text="🧺 Đang Thu Hoạch", fg="#a6da95")
            self.farm_ticker_lbl.config(text="Đang quét thẻ quả & thu hoạch...")
        elif state == FarmState.BAG_FULL_DETECTED or "SELLING" in state.value:
            self.farm_status_lbl.config(text="💰 Đang Đi Bán", fg="#f9e2af")
            self.farm_ticker_lbl.config(text="Đang bán nông sản tại Oliver...")
        elif state == FarmState.COMPLETED:
            self.farm_status_lbl.config(text="🎉 Hoàn Thành", fg="#a6da95")
            self.farm_ticker_lbl.config(text="Đã quét sạch toàn bộ quả đã chọn!")
            if self.is_mini:
                self._expand_farm_ui()

    def _process_farm_log_queue(self):
        while not self.farm_log_queue.empty():
            msg = self.farm_log_queue.get_nowait()
            tag = "INFO"
            if "ĐÃ THU HOẠCH" in msg or "THU HOẠCH THÀNH CÔNG" in msg or "🎉" in msg or "✨" in msg:
                tag = "HARVEST"
                if "Đang thu" in msg or "Đã bấm" in msg:
                    self.farm_harvest_count += 1
            elif "BỎ QUA" in msg or "khổng lồ" in msg.lower() or "vượt ngưỡng" in msg.lower() or ">= " in msg:
                tag = "SKIP_HEAVY"
            elif "ĐẦY TÚI" in msg or "Túi đã đầy" in msg:
                tag = "BAG_FULL"
            elif "BÁN" in msg or "Oliver" in msg:
                tag = "SELL"
                if "BÁN THÀNH CÔNG" in msg:
                    self.farm_sell_count += 1
            elif "⚠️" in msg or "LỖI" in msg:
                tag = "WARN"

            self.farm_text.insert(tk.END, msg + "\n", tag)
            self.farm_text.see(tk.END)

            self.farm_counter_lbl.config(text=f"🧺 {self.farm_harvest_count} quả | 💰 {self.farm_sell_count} đợt")

        self.root.after(80, self._process_farm_log_queue)

    def _toggle_topmost(self):
        self.root.attributes("-topmost", self.is_topmost.get())

    def _toggle_transparency(self):
        alpha = 0.88 if self.is_transparent.get() else 1.0
        self.root.attributes("-alpha", alpha)

    def _toggle_mini_mode(self):
        if not self.is_mini:
            if self.active_mode == "FARM":
                self._collapse_farm_ui()
            else:
                self.notebook_frame.pack_forget()
                self.root.geometry(f"{self.expanded_w}x{self.mini_h}")
                self.toggle_mini_btn.config(text="🗖 Mở Rộng", bg="#a6da95")
                self.is_mini = True
        else:
            if self.active_mode == "FARM":
                self._expand_farm_ui()
            else:
                self.notebook_frame.pack(fill="both", expand=True, padx=10, pady=(0, 6))
                self.root.geometry(f"{self.expanded_w}x{self.expanded_h}")
                self.toggle_mini_btn.config(text="🗕 Thu Gọn", bg="#8aadf4")
                self.is_mini = False

    def _copy_debug_log(self):
        content = self.debug_text.get("1.0", tk.END).strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.root.update()
            self.copy_status_lbl.config(text="✅ Đã copy!")
            self.root.after(2500, lambda: self.copy_status_lbl.config(text=""))
        else:
            self.copy_status_lbl.config(text="⚠️ Trống!")
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
        if hasattr(self, "auto_clear_farm") and self.auto_clear_farm.get():
            self.farm_text.delete("1.0", tk.END)
            now = datetime.now().strftime("%H:%M:%S")
            self.farm_text.insert(tk.END, f"[{now}] 🧹 [TỰ ĐỘNG DỌN DẸP] Đã xóa nhật ký nông trại định kỳ 60s.\n", "INFO")
        self.root.after(60000, self._schedule_auto_clear)

    def _test_bot_vision(self):
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
        if self.farm_engine.state != FarmState.STOPPED and self.farm_engine.state != FarmState.COMPLETED:
            messagebox.showwarning("Cảnh Báo", "Bot Nông Trại đang chạy! Vui lòng bấm [F6] để dừng Nông Trại trước khi bật Câu Cá.")
            return
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
        self.counter_lbl.config(text=f"🐟 {self.fish_count} con")

    def _update_state_ui(self, state: BotState):
        if state == BotState.IDLE or state == BotState.STOPPED:
            self.status_lbl.config(text="🔴 ĐÃ DỪNG (F1)", fg="#ed8796")
            self.ticker_lbl.config(text="Sẵn sàng")
        elif state == BotState.STAGE_1:
            self.status_lbl.config(text="🎣 B1: Quăng Cần", fg="#89b4fa")
            self.ticker_lbl.config(text="Đang quăng cần...")
        elif state == BotState.STAGE_2:
            self.status_lbl.config(text="🎯 B2: Canh Cá", fg="#f9e2af")
            self.ticker_lbl.config(text="Đang canh chừng dấu '!'...")
        elif state == BotState.STAGE_3:
            self.status_lbl.config(text="📦 B3: Lưu Balo", fg="#a6da95")
            self.ticker_lbl.config(text="Đang lưu cá vào balo...")
        elif state == BotState.STAGE_REPAIR:
            self.status_lbl.config(text="🔧 Sửa Cần", fg="#cba6f7")
            self.ticker_lbl.config(text="Đang sửa cần câu...")

    def _format_simple_log(self, msg: str):
        if any(skip in msg for skip in ("⏱ [ĐO ĐẠC]", "└ tốn nhiều nhất", "[HIỆU NĂNG]", "record_full", "corpus/stuck", "tự động dọn dẹp")):
            return None

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
            return (f"⚡ [BƯỚC 2] CÁ CẮN! Dấu '!'{coords}", "STEP_BITE")

        if "KÍCH HOẠT: Giật cần" in msg:
            m = re.search(r"tại nút bấm \((\d+),\s*(\d+)\)", msg)
            coords = f" tại ({m.group(1)}, {m.group(2)})" if m else ""
            return (f"💥 Đã bấm giật cần{coords}!", "STEP_BITE")

        if "Lưu Cá Vào Balo" in msg and "==>" in msg:
            return ("📦 [BƯỚC 3] Đang chờ kéo cá lên...", "STEP_WAIT")

        if "Đã thấy nút Bảo Quản" in msg:
            m = re.search(r"Click tại \((\d+),\s*(\d+)\)", msg)
            coords = f" tại ({m.group(1)}, {m.group(2)})" if m else ""
            return (f"📦 [BƯỚC 3] ĐÃ ẤN BẢO QUẢN{coords}! (+1 cá 🐟)", "STEP_KEEP")

        if "HOÀN THÀNH 1 VÒNG CÂU" in msg:
            sep = "━" * 38
            return (f"{sep}\n✨ XONG 1 VÒNG CÂU (Tổng: {self.fish_count} con 🐟)\n{sep}", "ROUND_COMPLETE")

        if "TỰ SỬA TRẠNG THÁI" in msg:
            if "Nút Giật Cần vẫn còn" in msg:
                return ("🔄 [TỰ SỬA] Nút Giật Cần còn -> Canh cá tiếp...", "STEP_ADJUST")
            if "Bảo quản" in msg:
                return ("🔄 [TỰ SỬA] Thấy Bảo Quản -> Nhận cá...", "STEP_ADJUST")
            return ("🔄 [TỰ SỬA] Tự nắn chỉnh theo màn hình!", "STEP_ADJUST")

        if "ĐỨT DÂY / HỤT CÁ" in msg:
            return ("⚠️ Đứt dây / hụt cá -> Quăng cần mới.", "STEP_WARN")

        if "CHẶN BONG BÓNG" in msg:
            return ("🛡️ Bỏ qua bóng nhiệm vụ/mong muốn.", "STEP_INFO")

        if "Sửa Cần" in msg or "sửa cần" in msg.lower():
            if "Đã định vị nút Sửa Cần" in msg:
                return ("🔧 [SỬA CẦN] Đã bấm trả tiền sửa...", "STEP_REPAIR")
            if "Đang chờ nút xanh OK" in msg:
                return ("🔧 [SỬA CẦN] Đang chờ sửa chữa...", "STEP_REPAIR")
            if "Đã định vị nút OK" in msg:
                return ("🔧 [SỬA CẦN] Đã bấm nút OK xác nhận!", "STEP_REPAIR")
            if "SỬA CẦN HOÀN THÀNH" in msg:
                return ("✅ [SỬA CẦN] Sửa chữa hoàn tất! Tiếp tục.", "STEP_REPAIR")

        if "BOT ĐÃ KHỞI ĐỘNG" in msg:
            return ("🚀 BOT ĐÃ BẮT ĐẦU (F2 để dừng).", "ROUND_COMPLETE")

        if "BOT ĐÃ DỪNG" in msg:
            return ("⏹ BOT ĐÃ DỪNG (F1 để tiếp tục).", "STEP_WARN")

        if "KẸT >30s" in msg:
            return ("⚠️ Cảnh báo: Kẹt >30s! Đã lưu ảnh.", "STEP_WARN")

        return None

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
            elif "📊" in msg:
                tag = "STATS"
            elif "🔍" in msg or "🚀" in msg or "🎣" in msg or "🔧" in msg or "🔄" in msg:
                tag = "INFO"

            self.debug_text.insert(tk.END, msg + "\n", tag)
            self.debug_text.see(tk.END)

            parsed = self._format_simple_log(msg)
            if parsed is not None:
                sim_text, sim_tag = parsed
                self.simple_text.insert(tk.END, sim_text + "\n", sim_tag)
                self.simple_text.see(tk.END)

                clean_ticker = sim_text.replace("\n", " ").strip()
                if "━" in clean_ticker:
                    clean_ticker = f"✨ Xong 1 vòng ({self.fish_count} cá)"
                self.ticker_lbl.config(text=clean_ticker[:32])

        try:
            line_count_sim = int(self.simple_text.index("end-1c").split(".")[0])
            if line_count_sim > 300:
                self.simple_text.delete("1.0", f"{line_count_sim - 200}.0")

            line_count_dbg = int(self.debug_text.index("end-1c").split(".")[0])
            if line_count_dbg > 500:
                self.debug_text.delete("1.0", f"{line_count_dbg - 350}.0")
        except Exception:
            pass

        self.root.after(80, self._process_log_queue)

    def on_close(self):
        self.engine.stop()
        self.farm_engine.stop()
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
