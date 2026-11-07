import os
import cv2
import numpy as np
import pytest
from core.farm_engine import FarmEngine, FarmState
from core.farm_config_manager import FarmConfigManager, FruitRule

def test_farm_engine_uses_path_utils_assets_dir():
    engine = FarmEngine()
    assert os.path.exists(engine.assets_dir)
    assert engine.assets_dir.endswith(os.path.join("assets", "farm"))
    assert getattr(engine, "_templates_valid", False) is True
    assert len(engine._templates) > 0

def test_farm_engine_fail_fast_when_templates_missing(tmp_path):
    empty_dir = tmp_path / "empty_assets"
    empty_dir.mkdir()
    engine = FarmEngine(assets_dir=str(empty_dir))
    logs = []
    engine.log_callback = logs.append
    
    assert getattr(engine, "_templates_valid", True) is False
    assert len(engine._templates) == 0

    # Attempting to start must abort immediately
    cfg = FarmConfigManager()
    cfg.rules = {"Hạt dẻ": FruitRule("Hạt dẻ", 3.0, True)}
    engine.cfg_mgr = cfg
    engine.start()

    assert engine.state == FarmState.STOPPED
    assert any("KHỞI ĐỘNG THẤT BẠI" in l or "LỖI TẢI ẢNH MẪU" in l for l in logs)

def test_hud_detection_supports_2k_scale():
    engine = FarmEngine()
    tpl_core = engine._templates.get("core_b1_ban")
    assert tpl_core is not None, "core_b1_ban must be loaded"

    # Create a 2K resolution blank frame (2560x1440)
    frame_2k = np.zeros((1440, 2560, 3), dtype=np.uint8)

    # Scale template by 1.33x (typical 1080p -> 1440p scaling)
    scaled_w = int(tpl_core.shape[1] * 1.33)
    scaled_h = int(tpl_core.shape[0] * 1.33)
    scaled_tpl = cv2.resize(tpl_core, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    # Paste scaled template onto frame at (1800, 1100)
    frame_2k[1100:1100+scaled_h, 1800:1800+scaled_w] = scaled_tpl

    # Detect with find_hud_sell_button
    hit = engine.find_hud_sell_button(frame_2k, threshold=0.70)
    assert hit is not None, "find_hud_sell_button must detect button at 1.33x (2K scale)"
    rx, ry, rw, rh = hit
    assert abs(rx - 1800) <= 5
    assert abs(ry - 1100) <= 5


def test_hud_harvest_button_detected_on_real_screen_photo():
    engine = FarmEngine()
    img_path = r'C:/Users/ACER/.gemini/antigravity/brain/a7c3f462-8ce1-4d55-9dda-0561f4999588/.user_uploaded/media_1793893762631_6ff731d7.jpg'
    if os.path.exists(img_path):
        frame = cv2.imread(img_path)
        hit = engine.find_hud_harvest_button(frame)
        assert hit is not None, "find_hud_harvest_button must detect green harvest button on real-world sample!"
