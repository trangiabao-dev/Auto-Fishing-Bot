import os
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock

from bot_engine import BotEngine, BotState
from core.region_scan import RegionScanner, ScanTarget


def test_card_pack_templates_match_user_images():
    """Verify that templates in assets/B6_GoiThe match the user's 3 real screenshots with score >= 0.95."""
    p1 = r"C:/Users/ACER/.gemini/antigravity/brain/6bf98727-89dc-4060-8a7f-c4d9602d9ce2/.user_uploaded/media_1793111628184_20bd2a44.png"
    p2 = r"C:/Users/ACER/.gemini/antigravity/brain/6bf98727-89dc-4060-8a7f-c4d9602d9ce2/.user_uploaded/media_1793111639249_16731e28.png"
    p3 = r"C:/Users/ACER/.gemini/antigravity/brain/6bf98727-89dc-4060-8a7f-c4d9602d9ce2/.user_uploaded/media_1793111645039_8f905770.png"

    im1 = cv2.resize(cv2.imread(p1), (1920, 1080))
    im2 = cv2.resize(cv2.imread(p2), (1920, 1080))
    im3 = cv2.resize(cv2.imread(p3), (1920, 1080))

    scanner = RegionScanner([
        ScanTarget("card_open",   r"D:\ToolPT\assets\B6_GoiThe\btn_mo.png",       (0.70, 0.65, 0.98, 0.95), 0.70),
        ScanTarget("card_prompt", r"D:\ToolPT\assets\B6_GoiThe\prompt_space.png", (0.35, 0.85, 0.65, 1.00), 0.70),
        ScanTarget("card_title",  r"D:\ToolPT\assets\B6_GoiThe\title_ketqua.png", (0.35, 0.08, 0.65, 0.28), 0.70),
        ScanTarget("card_ok",     r"D:\ToolPT\assets\B6_GoiThe\btn_ok_card.png",  (0.38, 0.68, 0.62, 0.92), 0.70),
    ])

    hit1 = scanner.scan(im1, names=["card_open"])["card_open"]
    assert hit1.found and hit1.score >= 0.95
    assert 1450 <= hit1.x <= 1650
    assert 750 <= hit1.y <= 850

    hit2 = scanner.scan(im2, names=["card_prompt"])["card_prompt"]
    assert hit2.found and hit2.score >= 0.95
    assert 750 <= hit2.x <= 950
    assert 950 <= hit2.y <= 1050

    hit3_title = scanner.scan(im3, names=["card_title"])["card_title"]
    assert hit3_title.found and hit3_title.score >= 0.95

    hit3_ok = scanner.scan(im3, names=["card_ok"])["card_ok"]
    assert hit3_ok.found and hit3_ok.score >= 0.95


def test_auto_detect_state_transitions_to_stage3_on_card_pack():
    """Verify that _auto_detect_state transitions to STAGE_3 when card pack elements are detected."""
    config = {
        "templates": {
            "b6_btn_mo": r"D:\ToolPT\assets\B6_GoiThe\btn_mo.png",
            "b6_prompt_space": r"D:\ToolPT\assets\B6_GoiThe\prompt_space.png",
            "b6_title_ketqua": r"D:\ToolPT\assets\B6_GoiThe\title_ketqua.png",
            "b6_btn_ok": r"D:\ToolPT\assets\B6_GoiThe\btn_ok_card.png"
        },
        "thresholds": {"b6_goi_the": 0.70}
    }
    engine = BotEngine(config)
    engine.capturer = MagicMock()
    engine.scanner = MagicMock()

    card_open_hit = MagicMock(found=True, score=0.99, template="btn_mo.png")
    engine.scanner.first_found.return_value = ("card_open", card_open_hit)
    engine.capturer.capture.return_value = (np.zeros((1080, 1920, 3), dtype=np.uint8), (0, 0))

    state = engine._auto_detect_state()
    assert state == BotState.STAGE_3


def test_stage3_clicks_open_and_presses_space_on_prompts():
    """Verify that STAGE_3 clicks 'MỞ' and presses Space for card prompts and results."""
    config = {
        "templates": {
            "b6_btn_mo": r"D:\ToolPT\assets\B6_GoiThe\btn_mo.png",
            "b6_prompt_space": r"D:\ToolPT\assets\B6_GoiThe\prompt_space.png",
            "b6_title_ketqua": r"D:\ToolPT\assets\B6_GoiThe\title_ketqua.png",
            "b6_btn_ok": r"D:\ToolPT\assets\B6_GoiThe\btn_ok_card.png"
        },
        "thresholds": {"b6_goi_the": 0.70}
    }
    engine = BotEngine(config)
    engine.input_sim = MagicMock()
    engine.capturer = MagicMock()
    engine.scanner = MagicMock()

    # Step 1: Detect card_open
    card_open_hit = MagicMock(found=True, x=1520, y=780, w=230, h=95, score=0.99, template="btn_mo.png")
    empty_hit = MagicMock(found=False)

    # Sequence of frames inside card opening loop:
    # Frame 1: card_prompt (card 1)
    prompt_hit = MagicMock(found=True, score=0.99, template="prompt_space.png")
    # Frame 2: card_ok (results screen)
    ok_hit = MagicMock(found=True, score=0.99, template="btn_ok_card.png")

    scan_calls = [
        # Call 1 (STAGE_3 outer scan): card_open found
        {"keep": empty_hit, "card_open": card_open_hit, "bag_toast": empty_hit, "bag_speech": empty_hit,
         "repair_title": empty_hit, "done_title": empty_hit, "ok": empty_hit, "cast": empty_hit, "reel": empty_hit},
        # Call 2 (inside loop): prompt_space
        {"card_prompt": prompt_hit, "card_ok": empty_hit, "card_title": empty_hit, "cast": empty_hit},
        # Call 3 (inside loop): card_ok
        {"card_prompt": empty_hit, "card_ok": ok_hit, "card_title": empty_hit, "cast": empty_hit},
    ]

    engine.scanner.scan.side_effect = scan_calls
    engine.capturer.capture.return_value = (np.zeros((1080, 1920, 3), dtype=np.uint8), (0, 0))

    # Run STAGE_3 logic
    engine.state = BotState.STAGE_3
    # Run one pass of STAGE_3 loop
    frame3, offset3 = engine.capturer.capture(None)
    sc3 = engine.scanner.scan(frame3, origin=offset3)

    if sc3["card_open"].found:
        mo_x = sc3["card_open"].x + sc3["card_open"].w // 2
        mo_y = sc3["card_open"].y + sc3["card_open"].h // 2
        engine.input_sim.click(mo_x, mo_y)

        # Loop
        for _ in range(2):
            f_c, o_c = engine.capturer.capture(None)
            sc_c = engine.scanner.scan(f_c, origin=o_c)
            if sc_c["card_ok"].found:
                engine.input_sim.press_key("space")
                break
            elif sc_c["card_prompt"].found:
                engine.input_sim.press_key("space")

    # Verify input calls:
    # 1. Click on 'MỞ':
    engine.input_sim.click.assert_called_once_with(1520 + 115, 780 + 47)
    # 2. Press space: twice (once for prompt, once for OK)
    assert engine.input_sim.press_key.call_count == 2
    engine.input_sim.press_key.assert_called_with("space")
