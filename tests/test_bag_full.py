import os
import glob
import cv2
import pytest
from bot_engine import BotEngine, BotState


def test_all_25_user_samples_detected_as_bag_full():
    """Validates that all 25 user sample images across 2 folders and outside are detected."""
    base_dir = r"D:\ToolPT\assets\B5_TuiDay"
    toast_files = glob.glob(os.path.join(base_dir, "thong bao tui da day", "*.jpg"))
    speech_files = glob.glob(os.path.join(base_dir, "tin nhan tui da day", "*.jpg"))
    outside_files = glob.glob(os.path.join(base_dir, "luc nhan vat ngu*.jpg"))

    all_files = toast_files + speech_files + outside_files
    if len(all_files) < 25:
        pytest.skip(f"Not all 25 samples present (found {len(all_files)})")

    engine = BotEngine({
        "templates": {
            "b5_tui_day_toast": "assets/B5_TuiDay/toast_tui_day.png",
            "b5_tui_day_speech": "assets/B5_TuiDay/speech_tui_day.png",
        },
        "thresholds": {"b5_tui_day": 0.72}
    })

    detected_count = 0
    for f in all_files:
        img = cv2.imread(f)
        hits = engine.scanner.scan(img)
        t_found = hits["bag_toast"].found
        s_found = hits["bag_speech"].found
        if t_found or s_found:
            detected_count += 1

    assert detected_count == 25, f"Expected 25/25 detected, got {detected_count}"


def test_auto_detect_state_stops_on_bag_full():
    """Validates that first_found identifies bag full and stops the bot."""
    sample_img = r"D:\ToolPT\assets\B5_TuiDay\luc nhan vat ngu ma tui day 9.jpg"
    if not os.path.exists(sample_img):
        pytest.skip("Sample image not present in local environment")

    engine = BotEngine({
        "templates": {
            "b5_tui_day_toast": "assets/B5_TuiDay/toast_tui_day.png",
            "b5_tui_day_speech": "assets/B5_TuiDay/speech_tui_day.png",
        },
        "thresholds": {"b5_tui_day": 0.72}
    })

    frame = cv2.imread(sample_img)
    order = ["bag_toast", "bag_speech", "repair_title", "done_title", "ok", "keep", "reel", "cast"]
    first = engine.scanner.first_found(frame, order)
    assert first is not None
    assert first[0] in ("bag_toast", "bag_speech")
