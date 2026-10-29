import os
import cv2
import pytest
from core.digit_reader import DigitReader, HarvestCard
from core.farm_config_manager import FarmConfigManager, FruitRule


def test_digit_reader_parses_all_10_cards_from_b8():
    b8_path = r"D:\ToolPT\tests\fixtures\farm\B8.jpg"
    assert os.path.exists(b8_path), "B8.jpg must exist in assets/farm/thu_hoach"

    reader = DigitReader()
    b8 = cv2.imread(b8_path)
    cards = reader.scan_cards(b8, origin=(0, 0))

    assert len(cards) == 10, f"Expected 10 harvest cards, found {len(cards)}"

    expected_weights = [5.08, 4.98, 4.86, 4.75, 4.46, 4.02, 1.06, 1.03, 1.03, 1.03]
    for idx, (card, exp) in enumerate(zip(cards, expected_weights)):
        assert card.card_index == idx + 1
        assert abs(card.weight - exp) < 0.01, f"Card {idx+1}: expected {exp}, got {card.weight}"
        assert card.btn_w > 0 and card.btn_h > 0
        cx, cy = card.get_click_point()
        assert card.btn_x <= cx <= card.btn_x + card.btn_w
        assert card.btn_y <= cy <= card.btn_y + card.btn_h


def test_farm_config_persistence(tmp_path):
    cfg_file = str(tmp_path / "farm_config.json")
    mgr = FarmConfigManager(config_path=cfg_file)

    # Initial defaults
    assert mgr.get_rule("Hạt dẻ") is not None
    assert mgr.get_rule("Hạt dẻ").max_kg == 3.0
    assert mgr.get_rule("Hạt dẻ").enabled is True

    # Add a new fruit rule
    mgr.set_rule("Táo", max_kg=15.5, enabled=True, template_path="assets/farm/tao.png")
    assert mgr.get_rule("Táo").max_kg == 15.5

    # Reload from disk to verify persistence
    mgr2 = FarmConfigManager(config_path=cfg_file)
    tao = mgr2.get_rule("Táo")
    assert tao is not None
    assert tao.max_kg == 15.5
    assert tao.enabled is True

    # Remove rule
    assert mgr2.remove_rule("Táo") is True
    assert mgr2.get_rule("Táo") is None


def test_digit_reader_handles_small_and_large_weights():
    """Verifies that DigitReader correctly parses small weights (0.1, 0.2, 0.3) and large giant weights (18.0, 25.5, 99.9)."""
    import numpy as np
    reader = DigitReader()

    def synth_number(num_str):
        boxes = []
        for ch in num_str:
            if ch == '.':
                box = np.zeros((16, 4), dtype=np.uint8)
                box[12:15, 1:4] = 255
                boxes.append(box)
            else:
                tpl = reader.digits[ch]
                box = np.zeros((16, tpl.shape[1]), dtype=np.uint8)
                h = min(16, tpl.shape[0])
                box[0:h, 0:tpl.shape[1]] = tpl[0:h, :]
                boxes.append(box)
            boxes.append(np.zeros((16, 2), dtype=np.uint8))
        strip = np.hstack(boxes[:-1])
        full_strip = np.zeros((16, max(65, strip.shape[1] + 5)), dtype=np.uint8)
        w = strip.shape[1]
        full_strip[:, full_strip.shape[1] - w - 2:full_strip.shape[1] - 2] = strip
        return full_strip

    test_cases = {
        "0.10": 0.1,
        "0.20": 0.2,
        "0.30": 0.3,
        "1.03": 1.03,
        "3.50": 3.5,
        "5.08": 5.08,
        "18.00": 18.0,
        "18.50": 18.5,
        "25.00": 25.0,
        "99.90": 99.9,
    }

    for str_val, expected_float in test_cases.items():
        mask = synth_number(str_val)
        parsed = reader.read_weight_from_strip(mask)
        assert abs(parsed - expected_float) < 0.01, f"Failed for {str_val}: expected {expected_float}, got {parsed}"



def test_digit_reader_fail_safe_rejects_missing_integer_part():
    """
    CRITICAL SAFETY TEST:
    Verifies that if OCR loses the integer digit before the decimal dot (e.g. '.17', '.15'),
    it returns 9999.0 (FAIL-SAFE) instead of 0.17, strictly protecting giant crops (e.g. 3.17kg)
    from being mistakenly harvested.
    """
    import numpy as np
    reader = DigitReader()

    # Synthetic corrupted strip: dot followed by '1' and '7' without an integer digit
    d1 = reader.digits['1']
    d7 = reader.digits['7']

    strip = np.zeros((24, 50), dtype=np.uint8)
    # Dot at x=10, y=18
    strip[18:21, 10:13] = 255
    # Digit 1 at x=16
    h1 = min(24, d1.shape[0])
    strip[2:2+h1, 16:16+d1.shape[1]] = d1[:h1, :]
    # Digit 7 at x=26
    h7 = min(24, d7.shape[0])
    strip[3:3+h7, 26:26+d7.shape[1]] = d7[:h7, :]

    parsed = reader.read_weight_from_strip(strip)
    assert parsed >= 9000.0, f"Expected corrupted strip without integer to return >= 9000.0, got {parsed}"

    # Also verify that a valid 3.17 is parsed accurately as 3.17
    d3 = reader.digits['3']
    strip_317 = np.zeros((24, 60), dtype=np.uint8)
    h3 = min(24, d3.shape[0])
    strip_317[1:1+h3, 5:5+d3.shape[1]] = d3[:h3, :]
    strip_317[18:21, 18:21] = 255
    strip_317[2:2+h1, 24:24+d1.shape[1]] = d1[:h1, :]
    strip_317[3:3+h7, 34:34+d7.shape[1]] = d7[:h7, :]

    parsed_317 = reader.read_weight_from_strip(strip_317)
    assert abs(parsed_317 - 3.17) < 0.01, f"Expected 3.17, got {parsed_317}"
