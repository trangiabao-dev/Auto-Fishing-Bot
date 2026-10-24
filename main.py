import os
import sys
import json
import time
from typing import Dict, Any

from bot_engine import BotEngine
from core.hotkey_manager import HotkeyManager


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Loads JSON configuration with sensible fallback defaults."""
    if not os.path.exists(config_path):
        print(f"[LỖI] Không tìm thấy file '{config_path}'. Đang tạo file mặc định...")
        default_config = {
            "templates": {"img1": "assets/img1.png", "img2": "assets/img2.png", "img3": "assets/img3.png"},
            "thresholds": {"img1": 0.8, "img2": 0.75, "img3": 0.8},
            "regions": {"img1_region": None, "img2_region": None, "img3_region": None},
            "delays": {
                "scan_interval_ms": 100,
                "stage2_wait_timeout_sec": 120,
                "stage3_wait_delay_sec": 3.0,
                "click_hold_duration_min_ms": 40,
                "click_hold_duration_max_ms": 90,
                "click_margin_px": 5
            },
            "hotkeys": {"start": "f1", "stop": "f2"}
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        return default_config

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_assets_status(config: Dict[str, Any]):
    """Displays visual checkmarks for required template assets."""
    templates = config.get("templates", {})
    labels = {
        "b1_quang_can": "B1: QUĂNG CẦN",
        "b2_giat_can": "B2: GIẬT CẦN",
        "b2_cham_than": "B2: CHẤM THAN",
        "b2_vat_neo": "B2: VẬT NEO TÊN",
        "b3_bao_quan": "B3: BẢO QUẢN",
        "b4_sua_can_title": "B4: TIÊU ĐỀ SỬA CẦN",
        "b4_sua_can_btn": "B4: NÚT BẤM SỬA CẦN",
        "img1": "B1: QUĂNG CẦN",
        "img2": "B2: CHẤM THAN",
        "img2_btn": "B2: GIẬT CẦN",
        "img3": "B3: BẢO QUẢN",
        "anchor": "B2: VẬT NEO TÊN"
    }

    print("\n" + "=" * 70)
    print("  TRẠNG THÁI FILE ẢNH MẪU THEO TỪNG BƯỚC (TEMPLATES):")
    print("=" * 70)
    all_ready = True
    for key, path in templates.items():
        tag = labels.get(key, key.upper())
        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if len(files) > 0:
                status = f"✅ SẴN SÀNG ({len(files)} file: {', '.join(files[:2])}{'...' if len(files)>2 else ''})"
            else:
                status = "❌ THƯ MỤC TRỐNG"
                all_ready = False
        else:
            exists = os.path.exists(path)
            status = "✅ SẴN SÀNG" if exists else "❌ CHƯA CÓ FILE"
            if not exists:
                all_ready = False
        print(f"  - [{tag:<15}]: {path:<38} -> {status}")
    print("=" * 70)
    if not all_ready:
        print("  💡 LỜI KHUYÊN: Hãy chạy 'python helper_roi.py' để chụp ảnh!")
    print("=" * 70 + "\n")


def print_banner(config: Dict[str, Any]):
    start_key = config.get("hotkeys", {}).get("start", "f1").upper()
    stop_key = config.get("hotkeys", {}).get("stop", "f2").upper()

    print("\n" + "#" * 65)
    print("   TOOL QUÉT HÌNH ẢNH & CLICK TỰ ĐỘNG CHO GAME (VISION BOT)")
    print("#" * 65)
    print(f"   ▶ Nhấn phím [{start_key}] : BẮT ĐẦU CHẠY / TIẾP TỤC")
    print(f"   ⏹ Nhấn phím [{stop_key}] : DỪNG KHẨN CẤP / TẠM DỪNG")
    print("   ⛔ Nhấn [Ctrl + C] : Thoát hoàn toàn ứng dụng")
    print("#" * 65)


def main():
    config = load_config("config.json")
    print_banner(config)
    check_assets_status(config)

    engine = BotEngine(config)
    hotkey_mgr = HotkeyManager()

    start_key = config.get("hotkeys", {}).get("start", "f1").lower()
    stop_key = config.get("hotkeys", {}).get("stop", "f2").lower()

    # Register global hotkeys
    hotkey_mgr.register(start_key, engine.start)
    hotkey_mgr.register(stop_key, engine.stop)
    hotkey_mgr.register("f3", engine.recorder.request_dump)  # Step 0: nhấn khi thấy '!' trên đầu

    print(f"[*] Tool đã sẵn sàng ở chế độ ngầm. Hãy chuyển sang game và nhấn [{start_key.upper()}] để bắt đầu!\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Đang đóng ứng dụng...")
        engine.stop()
        hotkey_mgr.unregister_all()
        print("[✔] Đã thoát an toàn.")


if __name__ == "__main__":
    main()
