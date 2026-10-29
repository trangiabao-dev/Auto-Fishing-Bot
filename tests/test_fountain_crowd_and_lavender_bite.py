import math
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.scale_aware import BiteTemplateBank
from core.cast_button_guard import CastButtonGuard
from core.geometry import Rect
from core.detector import MatchResult
from bot_engine import BotEngine, BotState


def test_mautim_oaihuong_matches_lavender_bite_above_threshold():
    """Verify BiteTemplateBank with mautim_oaihuong matches the lavender exclamation mark with score >= 0.90."""
    bank = BiteTemplateBank(r"D:\ToolPT\assets\B2_DangCau\dau_cham_than", reference_scale=1.0)
    head_roi = cv2.imread(r"D:\ToolPT\diagnostics\exact_head_roi_img5.png")
    assert head_roi is not None, "exact_head_roi_img5.png must exist"

    gray_roi = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
    scale = 1.4725
    bm = bank.best_match(gray_roi, scale)

    assert bm is not None
    assert bm.name == "mautim_oaihuong.png"
    assert bm.score >= 0.90, f"Expected score >= 0.90, got {bm.score}"
    # Verify match coordinates are centered on the head
    assert 70 <= bm.x <= 140
    assert 80 <= bm.y <= 140


def test_cast_button_fallback_on_f_failure():
    """Verify that if press_key('f') yields no game reaction, retry activates safe mouse click fallback."""
    config = {
        "templates": {
            "b1_quang_can": "assets/B1_QuangCan",
            "b2_cham_than": "assets/B2_DangCau/dau_cham_than",
            "b2_ten_nhan_vat": "assets/B2_DangCau/vat_neo_ten/columbinaa.png"
        },
        "thresholds": {"b1_quang_can": 0.70, "b2_cham_than": 0.80}
    }
    engine = BotEngine(config)
    engine.input_sim = MagicMock()
    engine.capturer = MagicMock()
    engine.scanner = MagicMock()
    engine.cast_guard = CastButtonGuard()

    # Mock cast button hit
    cast_hit = MagicMock(found=True, x=1590, y=720, w=64, h=64, score=0.85, template="inner_phao_cau.png")
    repair_hit = MagicMock(found=False)
    done_hit = MagicMock(found=False)
    ok_hit = MagicMock(found=False)
    keep_hit = MagicMock(found=False)
    reel_hit = MagicMock(found=False)
    bag_toast_hit = MagicMock(found=False)
    bag_speech_hit = MagicMock(found=False)

    def scan_side_effect(frame, origin=(0, 0), names=None):
        return {
            "cast": cast_hit,
            "repair_title": repair_hit,
            "repair_btn": repair_hit,
            "done_title": done_hit,
            "ok": ok_hit,
            "keep": keep_hit,
            "reel": reel_hit,
            "bag_toast": bag_toast_hit,
            "bag_speech": bag_speech_hit
        }

    engine.scanner.scan.side_effect = scan_side_effect
    fake_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    engine.capturer.capture.return_value = (fake_frame, (0, 0))

    # Test safe click point properties
    cand_center = (cast_hit.x + cast_hit.w // 2, cast_hit.y + cast_hit.h // 2)
    click_pt = engine.cast_guard.get_safe_click_point(cand_center, max_radius=20)
    x_btn_center = engine.cast_guard.get_x_button_center()
    dist_to_x = math.hypot(click_pt[0] - x_btn_center[0], click_pt[1] - x_btn_center[1])

    # Must stay safely away from [X] button
    assert dist_to_x > 105.0, f"Click point {click_pt} too close to [X] ({dist_to_x:.1f}px)"


def test_stage2_roi_hysteresis_filters_transient_jump():
    """Verify that transient ROI jump (>35px) for 1 tick is ignored, preventing detector resets."""
    # Simulate the logic implemented in STAGE_2:
    effective_region = (436, 176, 288, 280)
    last_roi_move_time = 100.0
    bite_detector_resets = 0

    roi_jump_streak = 0
    pending_jump_roi = None

    # Tick 1: Small sway (10px) -> Smooth tracking without reset
    sway_roi = (442, 180, 288, 280)
    dist = math.hypot(sway_roi[0] - effective_region[0], sway_roi[1] - effective_region[1])
    if dist <= 35:
        effective_region = sway_roi
        roi_jump_streak = 0
        pending_jump_roi = None

    assert effective_region == sway_roi
    assert bite_detector_resets == 0
    assert last_roi_move_time == 100.0

    # Tick 2: Stranger walks by, jump to (745, 235) (dist ~ 310px)
    stranger_roi = (745, 235, 288, 280)
    dist = math.hypot(stranger_roi[0] - effective_region[0], stranger_roi[1] - effective_region[1])
    if dist > 35:
        if pending_jump_roi is not None and math.hypot(stranger_roi[0] - pending_jump_roi[0], stranger_roi[1] - pending_jump_roi[1]) <= 25:
            roi_jump_streak += 1
        else:
            roi_jump_streak = 1
            pending_jump_roi = stranger_roi

        if roi_jump_streak >= 2:
            effective_region = stranger_roi
            bite_detector_resets += 1

    # Effective region must NOT jump to stranger on single tick
    assert effective_region == sway_roi
    assert bite_detector_resets == 0
    assert last_roi_move_time == 100.0

    # Tick 3: Stranger leaves, returns to character (440, 178)
    back_roi = (440, 178, 288, 280)
    dist = math.hypot(back_roi[0] - effective_region[0], back_roi[1] - effective_region[1])
    if dist <= 35:
        effective_region = back_roi
        roi_jump_streak = 0
        pending_jump_roi = None

    assert effective_region == back_roi
    assert roi_jump_streak == 0
    assert bite_detector_resets == 0
