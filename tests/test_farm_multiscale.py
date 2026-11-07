import os
import sys

# Ensure ToolPT is first in import path
TOOLPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TOOLPT_DIR not in sys.path:
    sys.path.insert(0, TOOLPT_DIR)

import cv2
import numpy as np
import pytest

from core.farm_engine import FarmEngine
from core.digit_reader import DigitReader

USER_IMG2_PATH = r"C:\Users\ACER\.gemini\antigravity\brain\a7c3f462-8ce1-4d55-9dda-0561f4999588\.user_uploaded\media_1793932716885_06ae690a.jpg"
USER_IMG4_PATH = r"C:\Users\ACER\.gemini\antigravity\brain\a7c3f462-8ce1-4d55-9dda-0561f4999588\.user_uploaded\media_1793932748633_143de7b0.png"
USER_IMG_SHOP_PATH = r"C:\Users\ACER\.gemini\antigravity\brain\a7c3f462-8ce1-4d55-9dda-0561f4999588\.user_uploaded\media_1793940673468_7abb7c40.jpg"


@pytest.fixture
def img2():
    if os.path.exists(USER_IMG2_PATH):
        return cv2.imread(USER_IMG2_PATH)
    pytest.skip("User test image 2 (harvest window) not found")


@pytest.fixture
def img4():
    if os.path.exists(USER_IMG4_PATH):
        return cv2.imread(USER_IMG4_PATH)
    pytest.skip("User test image 4 (filter modal) not found")


@pytest.fixture
def img_shop():
    if os.path.exists(USER_IMG_SHOP_PATH):
        return cv2.imread(USER_IMG_SHOP_PATH)
    pytest.skip("User test image shop modal not found")


def test_multiscale_find_template_on_img2_and_img4(img2, img4):
    engine = FarmEngine()
    
    # On img2 (1024x570 Harvest Window):
    loc_btn = engine.find_template(img2, "btn_loc", threshold=0.60)
    assert loc_btn is not None, "Failed to find 'Lọc' button on img2"
    assert abs(loc_btn[0] - 829) < 20
    assert abs(loc_btn[1] - 17) < 15

    # On img4 (1024x571 Filter Modal):
    btn_cai_dat = engine.find_template(img4, "btn_cai_dat_loc", threshold=0.65)
    assert btn_cai_dat is not None, "Failed to find 'Cài đặt bộ lọc' on img4"
    assert abs(btn_cai_dat[0] - 512) < 25
    assert abs(btn_cai_dat[1] - 471) < 20

    tab_ns = engine.find_tab_nong_san(img4)
    assert tab_ns is not None, "Failed to find tab 'Nông sản' on img4"
    assert abs(tab_ns[0] - 184) < 20
    assert abs(tab_ns[1] - 73) < 15

    chk = engine.find_dot_bien(img4)
    assert chk is not None, "Failed to find 'Ngoại trừ trái đột biến' on img4"
    assert abs(chk[0] - 36) < 20
    assert abs(chk[1] - 404) < 20


def test_filter_modal_never_misidentifies_as_harvest_window(img4):
    engine = FarmEngine()
    
    # In img4 (Filter Modal):
    # 1. in_filter_modal must be True
    in_filter_modal = (
        engine.find_template(img4, "btn_cai_dat_loc", threshold=0.65) is not None or
        engine.find_dot_bien(img4, threshold=0.65) is not None
    )
    assert in_filter_modal is True, "Filter modal should be detected as True"

    # 2. is_harvest_window_open must be strictly False
    assert engine.is_harvest_window_open(img4) is False, "Filter modal should NOT be seen as harvest window"

    # 3. find_filter_button MUST return None (Zero clicking on blank white space!)
    assert engine.find_filter_button(img4) is None, "find_filter_button should return None in Filter Modal"


def test_harvest_window_detects_filter_button_accurately(img2):
    engine = FarmEngine()
    
    # In img2 (Harvest Window):
    # 1. in_filter_modal must be False
    in_filter_modal = (
        engine.find_template(img2, "btn_cai_dat_loc", threshold=0.65) is not None or
        engine.find_dot_bien(img2, threshold=0.65) is not None
    )
    assert in_filter_modal is False, "Harvest window should NOT be seen as filter modal"

    # 2. is_harvest_window_open must be True
    assert engine.is_harvest_window_open(img2) is True, "Harvest window should be detected as True"

    # 3. find_filter_button must accurately return the 'Lọc' button (NOT white space!)
    loc_btn = engine.find_filter_button(img2)
    assert loc_btn is not None, "Should find 'Lọc' button in harvest window"
    assert abs(loc_btn[0] - 829) < 20
    assert abs(loc_btn[1] - 17) < 15


def test_has_blue_checkbox_adaptive_detection(img4):
    engine = FarmEngine()
    chk = engine.find_dot_bien(img4)
    assert chk is not None
    fx, fy, fw, fh = chk
    box_w = min(fw, max(20, int(fh * 1.1)))
    roi_chk = img4[fy:fy+fh, fx:fx+box_w]

    # In img4, the checkbox is unchecked
    assert engine.has_blue_checkbox(roi_chk) is False, "Unchecked box should be False"

    # Synthetic active checkbox (cyan-blue square)
    active_box = np.full((fh, box_w, 3), (255, 180, 0), dtype=np.uint8)  # BGR cyan
    assert engine.has_blue_checkbox(active_box) is True, "Cyan checked box should be True"


def test_digit_reader_multiscale_scan_cards(img2):
    reader = DigitReader()
    cards = reader.scan_cards(img2)
    assert len(cards) >= 8, f"Expected at least 8 cards detected on img2, got {len(cards)}"
    # Check button bounds align with cyan buttons (around x=390 and x=864)
    x_coords = [c.btn_x for c in cards]
    assert any(abs(x - 390) < 30 for x in x_coords), "Should detect left column buttons"
    assert any(abs(x - 864) < 30 for x in x_coords), "Should detect right column buttons"


def test_startup_skips_b1_when_hinh2_oliver_basket_visible():
    """Validates user requirement: If Hinh 2 (Oliver basket) is visible, NEVER click B1 (Bán)."""
    engine = FarmEngine()
    logs = []
    engine.log_callback = logs.append
    clicked_coords = []
    engine.input_sim = type("MockInput", (), {
        "click": lambda s, x, y, **kw: clicked_coords.append((x, y)),
        "move_to": lambda s, x, y: None,
        "press_key": lambda s, k: None,
    })()

    # Create synthetic frame with Oliver basket (Hinh 2) and HUD Harvest (Hinh 6)
    oliver_tpl = engine._templates["oliver_basket"]
    b2_core = engine._templates["core_b2_harvest"]
    b1_core = engine._templates["core_b1_ban"]

    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    # Place Oliver bubble at center-top
    ho, wo = oliver_tpl.shape[:2]
    frame[200:200+ho, 800:800+wo] = oliver_tpl
    # Place B1 (Bán) at HUD (x=1650, y=100)
    hb1, wb1 = b1_core.shape[:2]
    frame[100:100+hb1, 1650:1650+wb1] = b1_core
    # Place B2 (Thu hoạch) at HUD (x=1550, y=100)
    hb2, wb2 = b2_core.shape[:2]
    frame[100:100+hb2, 1550:1550+wb2] = b2_core

    engine.capturer = type("MockCap", (), {
        "capture": lambda s, reg=None: (frame, (0, 0))
    })()

    engine._teleported_to_oliver = False
    engine.step_open_harvest_window(frame)

    # Must NOT click B1 (x around 1650..1700)
    # Must click B2 (x around 1550..1600)
    assert len(clicked_coords) == 1, f"Expected exactly 1 click on Thu hoạch, got: {clicked_coords}"
    cx, cy = clicked_coords[0]
    assert 1540 <= cx <= 1620, f"Click must be within B2 (Thu hoạch) bounds, got {cx}"
    assert any("ĐÃ Ở TRƯỚC CỬA HÀNG" in l for l in logs)
    assert any("BƯỚC 2" in l for l in logs)
    assert not any("BƯỚC 1" in l for l in logs)


def test_startup_clicks_b1_when_hinh2_not_visible():
    """Validates user requirement: If Hinh 2 is NOT visible, MUST click B1 (Bán) to teleport, then B2."""
    engine = FarmEngine()
    logs = []
    engine.log_callback = logs.append
    clicked_coords = []
    engine.input_sim = type("MockInput", (), {
        "click": lambda s, x, y, **kw: clicked_coords.append((x, y)),
        "move_to": lambda s, x, y: None,
        "press_key": lambda s, k: None,
    })()

    b2_core = engine._templates["core_b2_harvest"]
    b1_core = engine._templates["core_b1_ban"]

    # Frame WITHOUT Oliver bubble (only B1 and B2 on HUD)
    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    hb1, wb1 = b1_core.shape[:2]
    frame[100:100+hb1, 1650:1650+wb1] = b1_core
    hb2, wb2 = b2_core.shape[:2]
    frame[100:100+hb2, 1550:1550+wb2] = b2_core

    engine.capturer = type("MockCap", (), {
        "capture": lambda s, reg=None: (frame, (0, 0))
    })()

    engine._teleported_to_oliver = False
    engine.step_open_harvest_window(frame)

    # Must click B1 first, then B2
    assert len(clicked_coords) == 2, f"Expected 2 clicks (B1 then B2), got: {clicked_coords}"
    # Click 1: B1 (Bán) around 1650
    assert 1640 <= clicked_coords[0][0] <= 1720, f"Click 1 must be within B1 (Bán), got {clicked_coords[0]}"
    # Click 2: B2 (Thu hoạch) around 1550
    assert 1540 <= clicked_coords[1][0] <= 1620, f"Click 2 must be within B2 (Thu hoạch), got {clicked_coords[1]}"
    assert any("BƯỚC 1" in l for l in logs)
    assert any("BƯỚC 2" in l for l in logs)


def test_shop_modal_detection(img_shop):
    """Validates accurate shop modal detection on user windowed screen (1024x575)."""
    engine = FarmEngine()
    assert engine.is_shop_open(img_shop) is True, "Shop modal should be detected as open"

    close_btn = engine.find_close_button(img_shop)
    assert close_btn is not None, "Should locate close [X] button on shop modal"
    assert abs(close_btn[0] - 846) < 20
    assert abs(close_btn[1] - 82) < 20

    coin_btn = engine.find_shop_coin_button(img_shop)
    assert coin_btn is not None, "Should detect active cyan coin button on shop modal"
    assert abs(coin_btn[0] - 696) < 25 or abs(coin_btn[0] - 691) < 25

    auto_btn = engine.find_shop_auto_select_button(img_shop)
    assert auto_btn is not None, "Should detect yellow auto select button on shop modal"
    assert abs(auto_btn[0] - 576) < 25 or abs(auto_btn[0] - 571) < 25


def test_shop_modal_does_not_false_positive_on_other_screens(img2, img4):
    """Guarantees shop modal detection returns False on Harvest Window and Filter Modal."""
    engine = FarmEngine()
    assert engine.is_shop_open(img2) is False, "Harvest window must NOT be identified as shop modal"
    assert engine.is_shop_open(img4) is False, "Filter modal must NOT be identified as shop modal"


def test_reactive_priority_closes_shop_when_opening_harvest(img_shop):
    """Validates user requirement: If screen is on Shop Modal when opening harvest, close shop cleanly first."""
    engine = FarmEngine()
    logs = []
    engine.log_callback = logs.append
    clicked_coords = []
    engine.input_sim = type("MockInput", (), {
        "click": lambda s, x, y, **kw: clicked_coords.append((x, y)),
        "move_to": lambda s, x, y: None,
        "press_key": lambda s, k: None,
    })()

    res = engine.step_open_harvest_window(img_shop)
    assert res is False, "Should return False to re-check screen after closing shop"
    assert len(clicked_coords) == 1, "Should click close [X] button once"
    # Close button is near (846, 82)
    cx, cy = clicked_coords[0]
    assert abs(cx - 846) < 30 and abs(cy - 82) < 30, f"Must click close button, got ({cx}, {cy})"
    assert any("ƯU TIÊN MÀN HÌNH" in l and "Cửa hàng Oliver đang mở" in l for l in logs)

