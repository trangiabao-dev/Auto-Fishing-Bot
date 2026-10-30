import os
from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest

from core.farm_config_manager import FarmConfigManager, FruitRule
from core.farm_engine import FarmEngine, FarmState


@pytest.fixture
def farm_engine(tmp_path):
    cfg_file = str(tmp_path / "test_farm_cfg.json")
    mgr = FarmConfigManager(config_path=cfg_file)
    mgr.set_rule("Hạt dẻ", max_kg=3.0, enabled=True)
    mgr.shift_delay_s = 0.01  # Fast test delay

    input_sim = MagicMock()
    capturer = MagicMock()

    engine = FarmEngine(
        config_manager=mgr,
        capturer=capturer,
        input_sim=input_sim,
    )
    return engine


def test_farm_engine_detects_bag_full_toast(farm_engine):
    toast_screen = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\khung anh vi du cho thong bao day tui.jpg")
    clean_screen = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\B8.jpg")

    assert farm_engine.is_bag_full(toast_screen) is True
    assert farm_engine.is_bag_full(clean_screen) is False


def test_farm_engine_harvest_round_clicks_only_eligible_cards(farm_engine):
    b8 = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\B8.jpg")
    farm_engine.capturer.capture.return_value = (b8, (0, 0))

    # In B8: weights are 5.08, 4.98, 4.86, 4.75, 4.46, 4.02, 1.06, 1.03...
    # Case 1: max_kg = 2.0 -> Cards with 1.06kg and 1.03kg are eligible
    harvested, should_scroll = farm_engine.execute_harvest_round(b8, max_kg=2.0)
    assert harvested is True
    assert should_scroll is False
    assert farm_engine.stats["harvested_count"] == 1
    assert farm_engine.input_sim.click.called

    # Reset mock
    farm_engine.input_sim.click.reset_mock()

    # Case 2: max_kg = 0.5 -> No cards in B8 are < 0.5kg (smallest is 1.03)
    harvested, should_scroll = farm_engine.execute_harvest_round(b8, max_kg=0.5)
    assert harvested is False
    assert should_scroll is True
    assert not farm_engine.input_sim.click.called


def test_farm_engine_sell_cycle_executes_all_steps(farm_engine):
    # Mock capture returning appropriate frames for each sell step
    f_shop = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\anh tong the man hinh luc ban.jpg")
    f_b3 = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\khung hinh anh ban.jpg")
    f_b4 = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\khung hinh anh ban B4.jpg")
    f_b5 = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\khung hinh anh ban B5.jpg")
    f_b6 = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\khung hinh anh ban B6.jpg")

    # Sequence of frames captured during step_sell_cycle
    frames = [f_b3, f_shop, f_b3, f_b3, f_b4, f_b5, f_b6, f_b3]
    frame_iter = iter(frames)

    def mock_capture(region=None):
        try:
            return next(frame_iter), (0, 0)
        except StopIteration:
            return f_b3, (0, 0)

    farm_engine.capturer.capture.side_effect = mock_capture

    res = farm_engine.step_sell_cycle()
    assert res is True
    assert farm_engine.stats["sell_cycles"] == 1
    # Check that input simulator pressed space to exit Oliver dialog
    assert farm_engine.input_sim.click.called


def test_farm_engine_detects_empty_filter(farm_engine):
    empty_screen = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\man_hinh_het_trai.png")
    empty_1080 = cv2.resize(empty_screen, (1920, 1080))
    clean_screen = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\B8.jpg")

    assert farm_engine.is_filter_empty(empty_1080) is True
    assert farm_engine.is_filter_empty(clean_screen) is False


def test_farm_engine_fast_harvest_round_clicks_cards(farm_engine):
    b8 = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\B8.jpg")
    harvested, should_scroll = farm_engine.execute_fast_harvest_round(b8)
    assert harvested is True
    assert should_scroll is False
    assert farm_engine.stats["harvested_count"] == 1
    assert farm_engine.input_sim.click.called


def test_farm_engine_step_open_harvest_window_detects_already_open(farm_engine):
    b8 = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\B8.jpg")
    res = farm_engine.step_open_harvest_window(b8)
    assert res is True
    # Should not click anything because harvest window is already open
    assert not farm_engine.input_sim.click.called


def test_farm_engine_step_open_harvest_window_executes_b1_b2_sequence(farm_engine):
    b1_tpl = cv2.imread(r"D:\ToolPT\assets\farm\thu_hoach\B1.png")
    b2_tpl = cv2.imread(r"D:\ToolPT\assets\farm\thu_hoach\B2.png")
    b8 = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\B8.jpg")

    # Synthetic 1080p frame with B1 pasted
    f_b1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    h1, w1 = b1_tpl.shape[:2]
    f_b1[100:100+h1, 100:100+w1] = b1_tpl

    # Synthetic frame with B2 pasted
    f_b2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    h2, w2 = b2_tpl.shape[:2]
    f_b2[200:200+h2, 200:200+w2] = b2_tpl

    # Setup capturer sequence: f_b2 (after clicking B1), then b8 (after clicking B2)
    frames = [f_b2, f_b2, b8, b8]
    frame_iter = iter(frames)
    def mock_cap(region=None):
        try:
            return next(frame_iter), (0, 0)
        except StopIteration:
            return b8, (0, 0)
    farm_engine.capturer.capture.side_effect = mock_cap

    res = farm_engine.step_open_harvest_window(f_b1)
    assert res is True
    assert farm_engine.input_sim.click.call_count >= 2


def test_farm_engine_step_setup_filter_wide_random_click(farm_engine):
    b3 = cv2.imread(r"D:\ToolPT\assets\farm\thu_hoach\B3.jpg")
    b4 = cv2.imread(r"D:\ToolPT\assets\farm\thu_hoach\B4.jpg")
    b5 = cv2.imread(r"D:\ToolPT\assets\farm\thu_hoach\B5.jpg")
    b6 = cv2.imread(r"D:\ToolPT\assets\farm\thu_hoach\item_hat_de.png")
    b7 = cv2.imread(r"D:\ToolPT\assets\farm\thu_hoach\B7.jpg")

    f_b3 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    f_b3[50:50+b3.shape[0], 50:50+b3.shape[1]] = b3

    f_b4 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    f_b4[100:100+b4.shape[0], 100:100+b4.shape[1]] = b4

    f_b5 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    f_b5[150:150+b5.shape[0], 150:150+b5.shape[1]] = b5

    f_b6 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    f_b6[200:200+b6.shape[0], 200:200+b6.shape[1]] = b6

    f_b7 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    f_b7[250:250+b7.shape[0], 250:250+b7.shape[1]] = b7

    frames = [f_b4, f_b5, f_b6, f_b7]
    frame_iter = iter(frames)
    farm_engine.capturer.capture.side_effect = lambda region=None: (next(frame_iter), (0, 0))

    clicks = []
    farm_engine.input_sim.click.side_effect = lambda x, y: clicks.append((x, y))

    res = farm_engine.step_setup_filter(f_b3, "Hạt dẻ")
    assert res is True
    # Verify B6 click was recorded and had wide x-span
    # clicks[0] = B3, clicks[1] = B4, clicks[2] = B5, clicks[3] = B6, clicks[4] = B7
    assert len(clicks) >= 4
    b6_click_x, b6_click_y = clicks[3]
    assert 200 <= b6_click_x <= 200 + 250
    assert 200 <= b6_click_y <= 200 + b6.shape[0]


def test_farm_engine_bottom_up_and_pre_click_veto(farm_engine):
    b8 = cv2.imread(r"D:\ToolPT\tests\fixtures\farm\B8.jpg")
    farm_engine.capturer.capture.return_value = (b8, (0, 0))

    clicks = []
    farm_engine.input_sim.click.side_effect = lambda x, y, **kwargs: clicks.append((x, y))

    # Case 1: Normal harvest with Bottom-Up strategy
    # In B8, lowest card < 2.0kg is harvested first
    h1, s1 = farm_engine.execute_harvest_round(b8, max_kg=2.0)
    assert h1 is True
    assert len(clicks) == 1

    # Case 2: Pre-Click VETO
    # If the frame suddenly shifts right before click to a giant crop (e.g. 5.08kg at the cursor)
    # The pre-click verification MUST detect weight >= max_kg and abort the click!
    clicks.clear()
    cards = farm_engine.digit_reader.scan_cards(b8)
    # Mock a shifted frame where all cards are giant (> 5.0kg)
    # Simulate fresh capture having only heavy card
    giant_frame = b8.copy()
    farm_engine.capturer.capture.return_value = (giant_frame, (0, 0))

    # Try harvesting with max_kg=1.04 (only the 1.03kg card would match from initial scan)
    # But if verify frame is treated as heavy, veto triggers:
    mock_heavy_card = MagicMock()
    mock_heavy_card.btn_x, mock_heavy_card.btn_w = 0, 1920
    mock_heavy_card.btn_y, mock_heavy_card.btn_h = 0, 1080
    mock_heavy_card.weight = 4.85  # Heavy shifted crop!
    
    orig_scan = farm_engine.digit_reader.scan_cards
    farm_engine.digit_reader.scan_cards = MagicMock(side_effect=[
        orig_scan(b8),           # Initial scan (finds eligible card)
        [mock_heavy_card],        # Verification capture right before click (detects heavy shifted card!)
    ])

    h_veto, s_veto = farm_engine.execute_harvest_round(b8, max_kg=2.0)
    assert h_veto is False, "Must abort harvest when pre-click verification detects shifted heavy crop!"
    assert len(clicks) == 0, "Mouse click MUST be vetoed and never fired!"


def test_farm_engine_get_scrollbar_info(farm_engine):
    im_bot = cv2.resize(cv2.imread(r"C:\Users\ACER\.gemini\antigravity\brain\6bf98727-89dc-4060-8a7f-c4d9602d9ce2\.user_uploaded\media_1793323706119_95ed675a.png"), (1920, 1080))
    thumb_ratio, bottom_ratio, is_at_bottom = farm_engine.get_scrollbar_info(im_bot)
    assert 0.5 <= thumb_ratio <= 0.9
    assert bottom_ratio >= 0.95
    assert is_at_bottom is True


def test_count_sellable_crops_in_shop_detects_unlocked_and_ignores_empty_or_locked(farm_engine):
    # Construct a synthetic 1080p frame
    frame = np.full((1080, 1920, 3), 180, dtype=np.uint8)

    # Empty frame should have 0 sellable crops
    assert farm_engine.count_sellable_crops_in_shop(frame) == 0

    # Draw an unlocked card (white card > 240, size 180x230 at grid slot 1)
    # Shop grid ROI is: X: 780..1640, Y: 240..760
    # Card 1 at (800, 260) with size 180x240
    frame[260:500, 800:980] = 250  # White card
    assert farm_engine.count_sellable_crops_in_shop(frame) == 1

    # Now add a lock icon in top-left of Card 1 (dark pixels < 175)
    frame[260:300, 800:840] = 50  # Dark lock
    assert farm_engine.count_sellable_crops_in_shop(frame) == 0

    # Add Card 2 (unlocked)
    frame[260:500, 1010:1190] = 250  # White card 2
    assert farm_engine.count_sellable_crops_in_shop(frame) == 1


def test_handle_space_prompt_presses_space_without_mouse_click_on_error_popup(farm_engine):
    popup_img = cv2.imread(r"C:\Users\ACER\.gemini\antigravity\brain\6bf98727-89dc-4060-8a7f-c4d9602d9ce2\.user_uploaded\media_1793275514061_8dcca536.png")
    assert popup_img is not None

    farm_engine.input_sim.click.reset_mock()
    farm_engine.input_sim.press_key.reset_mock()

    handled = farm_engine.handle_space_prompt_if_present(popup_img)
    assert handled is True
    # STRICT COMPLIANCE: 100% Space key, ZERO mouse click!
    assert farm_engine.input_sim.press_key.called
    farm_engine.input_sim.press_key.assert_called_with("space")
    assert not farm_engine.input_sim.click.called, "Must NOT use mouse click on Space prompts!"


def test_scroll_harvest_list_down_calls_drag_and_detects_shift(farm_engine):
    im1 = cv2.resize(cv2.imread(r"C:\Users\ACER\.gemini\antigravity\brain\6bf98727-89dc-4060-8a7f-c4d9602d9ce2\.user_uploaded\media_1793287819965_f56904b4.png"), (1920, 1080))
    im2 = cv2.resize(cv2.imread(r"C:\Users\ACER\.gemini\antigravity\brain\6bf98727-89dc-4060-8a7f-c4d9602d9ce2\.user_uploaded\media_1793287879880_51a16a10.png"), (1920, 1080))

    # Mock capturer to return im2 after scroll
    farm_engine.capturer.capture.return_value = (im2, (0, 0))

    moved = farm_engine.scroll_harvest_list_down(im1)
    assert moved is True
    assert farm_engine.input_sim.drag.called
    # Verify drag was called with swipe-up parameters (y0 > y1)
    drag_args = farm_engine.input_sim.drag.call_args[0]
    assert drag_args[1] > drag_args[3], f"Drag must swipe up (y0={drag_args[1]} > y1={drag_args[3]})"

    # If frame doesn't change, returns False (stationary / bottom reached)
    farm_engine.capturer.capture.return_value = (im2, (0, 0))
    moved_stationary = farm_engine.scroll_harvest_list_down(im2)
    assert moved_stationary is False
