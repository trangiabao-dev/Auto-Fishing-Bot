import os
import cv2
import numpy as np
import pytest
from bot_engine import BotEngine, BotState
from core.region_scan import ScanHit, RegionScanner, ScanTarget


def test_auto_detect_state_prioritizes_repair_over_cast():
    """Checks that if repair dialog and cast button are both visible, repair wins."""
    # Build synthetic targets
    scanner = RegionScanner([
        ScanTarget("repair_title", "assets/B4_SuaCan/title", (0.15, 0.05, 0.85, 0.85), 0.70),
        ScanTarget("repair_btn",   "assets/B4_SuaCan/button", (0.20, 0.40, 0.80, 0.90), 0.70),
        ScanTarget("done_title",   "assets/B4_SuaCan/title_hoan_thanh", (0.15, 0.05, 0.85, 0.85), 0.70),
        ScanTarget("ok",           "assets/B4_SuaCan/btn_ok", (0.20, 0.40, 0.80, 0.90), 0.70),
        ScanTarget("keep",         "assets/B3_BaoQuan", (0.65, 0.65, 0.95, 0.95), 0.70),
        ScanTarget("reel",         "assets/B2_DangCau/nut_giat_can.png", (0.65, 0.55, 1.00, 1.00), 0.70),
        ScanTarget("cast",         "assets/B1_QuangCan", (0.65, 0.55, 1.00, 1.00), 0.70),
    ])

    user_img_path = r"C:\Users\ACER\.gemini\antigravity\brain\6bf98727-89dc-4060-8a7f-c4d9602d9ce2\.user_uploaded\media_1793071871176_c6f7d4d3.png"
    if not os.path.exists(user_img_path):
        pytest.skip("Test image not available in local environment")

    frame = cv2.imread(user_img_path)
    order = ["repair_title", "repair_btn", "done_title", "ok", "keep", "reel", "cast"]
    first = scanner.first_found(frame, order)
    assert first is not None
    assert first[0] in ("repair_title", "repair_btn")


def test_find_repair_button_color_segmentation():
    """Validates that cyan capsule repair button is detected regardless of dynamic price."""
    user_img_path = r"C:\Users\ACER\.gemini\antigravity\brain\6bf98727-89dc-4060-8a7f-c4d9602d9ce2\.user_uploaded\media_1793071871176_c6f7d4d3.png"
    if not os.path.exists(user_img_path):
        pytest.skip("Test image not available in local environment")

    frame = cv2.imread(user_img_path)
    engine = BotEngine({"thresholds": {"b4_sua_can": 0.70}})
    btn = engine._find_repair_button(frame)
    assert btn is not None
    assert btn.found
    assert btn.w > 50 and btn.h > 20
    # Click point must fall safely inside the button
    cx, cy = engine.matcher.get_random_click_point(btn, margin=5)
    assert btn.x <= cx <= btn.x + btn.w
    assert btn.y <= cy <= btn.y + btn.h


def test_post_cast_reaction_detects_repair():
    """Validates that if repair modal pops up after casting, scanner identifies it immediately."""
    user_img_path = r"C:\Users\ACER\.gemini\antigravity\brain\6bf98727-89dc-4060-8a7f-c4d9602d9ce2\.user_uploaded\media_1793071871176_c6f7d4d3.png"
    if not os.path.exists(user_img_path):
        pytest.skip("Test image not available in local environment")

    frame = cv2.imread(user_img_path)
    engine = BotEngine({"thresholds": {"b4_sua_can": 0.70}})
    sc = engine.scanner.scan(frame)
    # Even if cast is in background, modal detection must trigger
    has_repair_modal = sc["repair_title"].found or sc["repair_btn"].found
    assert has_repair_modal, "Repair modal must be recognized on post-cast frame"
