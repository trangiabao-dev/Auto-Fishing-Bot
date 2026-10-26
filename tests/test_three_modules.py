"""
Tests for the 3 upgraded modules:
1. PlateTracker Cold Start Center-X preference & Sticky Anchor persistence.
2. InputSimulator Space key DirectInput scancode & Safe Core Zone clicking.
3. CastButtonGuard radial clicking R <= 25px, weather drift rejection & [X] veto.
4. Engine resilience under unexpected exceptions and TrackResult.motion safety.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.geometry import Rect
from core.detector import MatchResult, TemplateMatcher
from core.plate_locator import PlateHit, PlateLocator, PlateTracker, TrackResult
from core.cast_button_guard import CastButtonGuard
from core.input_sim import InputSimulator, SCANCODE_MAP


class MockLocator:
    """Mock PlateLocator for testing tracking decisions."""
    def __init__(self, candidates_sequence: Optional[List[List[PlateHit]]] = None):
        self._seq = list(candidates_sequence or [])
        self._idx = 0

    def candidates(self, frame_bgr: np.ndarray, origin: Tuple[int, int] = (0, 0),
                   window: Optional[Rect] = None) -> List[PlateHit]:
        if not self._seq:
            return []
        hits = self._seq[min(self._idx, len(self._seq) - 1)]
        self._idx += 1
        return hits


# ==============================================================================
# BÀI TOÁN 1: KHÓA ĐÚNG NHÂN VẬT CHÍNH CHỦ TRONG ĐÁM ĐÔNG
# ==============================================================================

def test_cold_start_prioritizes_candidate_closest_to_x_960():
    """
    Cold start with multiple candidates:
    Stranger on the left (x=450) has higher score (0.91).
    Player in the center (x=965) has lower score (0.75).
    Cold start MUST pick the player closest to X=960.
    """
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    stranger_hit = PlateHit(
        plate=Rect(400, 300, 100, 20),
        text=Rect(420, 300, 60, 15),  # center x = 450
        score=0.91,
        name="stranger.png",
        scale=1.0,
    )
    player_hit = PlateHit(
        plate=Rect(920, 300, 100, 20),
        text=Rect(935, 300, 60, 15),  # center x = 965
        score=0.75,
        name="columbinaa.png",
        scale=1.0,
    )

    mock_loc = MockLocator([[stranger_hit, player_hit]])
    tracker = PlateTracker(mock_loc, center_x=960.0)

    res = tracker.update(dummy_frame)

    assert res.state == "TRACKING"
    assert res.hit is not None
    assert res.hit.name == "columbinaa.png"
    assert abs(res.hit.text.center.x - 960) < abs(stranger_hit.text.center.x - 960)


def test_sticky_anchor_persists_across_multiple_rounds():
    """
    Once locked, the anchor must remain sticky across updates.
    Even if a stranger with a higher score appears nearby, anti-hijack prevents jump.
    """
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    player_round1 = PlateHit(Rect(920, 300, 100, 20), Rect(935, 300, 60, 15), 0.85, "columbinaa.png", 1.0)
    # Round 2: player moved slightly (camera rotation)
    player_round2 = PlateHit(Rect(930, 305, 100, 20), Rect(945, 305, 60, 15), 0.84, "columbinaa.png", 1.0)
    # Round 3: player stands still, stranger walks by with score 0.88 (not exceeding jump_margin 0.12)
    stranger = PlateHit(Rect(700, 300, 100, 20), Rect(720, 300, 60, 15), 0.88, "stranger.png", 1.0)

    mock_loc = MockLocator([
        [player_round1],
        [player_round2],
        [player_round2, stranger],
    ])
    tracker = PlateTracker(mock_loc, center_x=960.0)

    # Round 1 (Cold start)
    r1 = tracker.update(dummy_frame)
    assert r1.hit == player_round1

    # Round 2 (Sticky tracking)
    r2 = tracker.update(dummy_frame)
    assert r2.hit == player_round2
    assert r2.motion == (10, 5)  # Moved dx=10, dy=5

    # Round 3 (Stranger present)
    r3 = tracker.update(dummy_frame)
    assert r3.hit == player_round2  # Lock maintained on player!
    assert r3.hit.name == "columbinaa.png"


def test_track_result_motion_attribute_safety():
    """
    Ensure TrackResult always has the 'motion' property with (dx, dy)
    to prevent AttributeError (such as track_res.motion crashes).
    """
    dummy_hit = PlateHit(Rect(100, 100, 50, 10), Rect(110, 100, 30, 10), 0.8, "columbinaa.png", 1.0)
    res_track = TrackResult(hit=dummy_hit, state="TRACKING", motion=(5, -2))
    assert res_track.motion == (5, -2)

    res_stale = TrackResult(hit=dummy_hit, state="STALE")
    assert res_stale.motion == (0, 0)

    res_lost = TrackResult(hit=None, state="LOST")
    assert res_lost.motion == (0, 0)


# ==============================================================================
# BÀI TOÁN 2: NÉ NÚT KHOE VÀ BÁN - SPACE KEY & SAFE CORE ZONE
# ==============================================================================

def test_space_key_press_and_scancode_mapping():
    """Verify DirectInput scancode mapping for SPACE and F."""
    assert SCANCODE_MAP["space"] == 0x39
    assert SCANCODE_MAP["f"] == 0x21

    sim = InputSimulator()
    with patch("ctypes.windll.user32.SendInput") as mock_send_input:
        mock_send_input.return_value = 1
        success = sim.press_key("space", hold_min_ms=1, hold_max_ms=2)
        assert success is True
        # Both KeyDown and KeyUp must be sent
        assert mock_send_input.call_count == 2


def test_safe_core_zone_click_bounds():
    """
    'Bảo Quản' button on native 1080p (scaled from actual measurement in Image 8):
    btn_x=1396, btn_y=776, btn_w=231, btn_h=96.
    'Khoe' button left edge on 1080p: 1641.
    Safe Core Zone: x ∈ [0.15W, 0.50W], y ∈ [0.30H, 0.70H].
    Right edge of safe zone must strictly maintain > 80px distance from 'Khoe'.
    """
    btn_x, btn_y, btn_w, btn_h = 1396, 776, 231, 96
    match = MatchResult(found=True, confidence=0.97, x=btn_x, y=btn_y, w=btn_w, h=btn_h, template_name="baoquan")

    khoe_left_edge = 1641

    for _ in range(100):
        cx, cy = TemplateMatcher.get_safe_core_click_point(match, x_ratio_range=(0.15, 0.50))
        # Inside button bounds
        assert btn_x <= cx <= btn_x + btn_w
        assert btn_y <= cy <= btn_y + btn_h

        # Strictly in the safe core zone
        assert btn_x + int(btn_w * 0.15) <= cx <= btn_x + int(btn_w * 0.50)
        assert btn_y + int(btn_h * 0.30) <= cy <= btn_y + int(btn_h * 0.70)

        # Distance to 'Khoe' button must strictly exceed 80px
        dist_to_khoe = khoe_left_edge - cx
        assert dist_to_khoe > 80, f"Click point {cx} too close to Khoe ({dist_to_khoe}px)"


# ==============================================================================
# BÀI TOÁN 3: CHẶN CLICK NHẦM NÚT [X], [+], [^] KHI QUĂNG CẦN
# ==============================================================================

def test_cast_button_guard_safe_radial_click_zone():
    """
    Phao center on 1080p: (1622, 747).
    Button [X] center on 1080p: (1571, 879) [distance ~141.5px].
    All clicks within R <= 25px must remain > 115px away from [X]!
    """
    guard = CastButtonGuard(default_center=(1622, 747), safe_radius=25.0)
    phao_center = (1622, 747)
    x_btn_center = guard.get_x_button_center()

    assert x_btn_center == (1622 - 51, 747 + 132)  # (1571, 879)

    for _ in range(100):
        cx, cy = guard.get_safe_click_point(phao_center, max_radius=25.0)

        # Must be within 25px radius of phao center
        radial_dist = math.hypot(cx - phao_center[0], cy - phao_center[1])
        assert radial_dist <= 25.0, f"Radial distance {radial_dist:.1f}px exceeds 25px limit!"

        # Must remain safely separated from [X]
        dist_to_x = math.hypot(cx - x_btn_center[0], cy - x_btn_center[1])
        assert dist_to_x > 115.0, f"Click point ({cx}, {cy}) too close to [X]: {dist_to_x:.1f}px <= 115px"


def test_cast_button_guard_weather_drift_rejection():
    """
    In rain/foam weather, score drops to ~0.60.
    1. A match close to anchor (dist <= 30px) is VALID.
    2. An argmax drift towards [X] (dist > 30px) is REJECTED.
    3. A match inside [X] keep-out zone is VETOED immediately.
    """
    guard = CastButtonGuard(default_center=(1622, 747), max_anchor_drift=30.0)

    # Legitimate phao match in rain (dist = 5px <= 30px)
    valid_res = guard.verify_candidate(center=(1625, 750), score=0.62)
    assert valid_res.valid is True
    assert valid_res.dist_to_anchor < 10.0

    # Argmax drift towards bottom-left (dist = 45px > 30px)
    drift_res = guard.verify_candidate(center=(1590, 775), score=0.62)
    assert drift_res.valid is False
    assert "Argmax drift in weather" in (drift_res.veto_reason or "")

    # False positive near [X] button
    x_center = guard.get_x_button_center()
    veto_x_res = guard.verify_candidate(center=x_center, score=0.65)
    assert veto_x_res.valid is False
    assert "too close to [X]" in (veto_x_res.veto_reason or "")


# ==============================================================================
# QUY TRÌNH & KIẾN TRÚC: ENGINE EXCEPTION RESILIENCE
# ==============================================================================

def test_engine_run_loop_handles_unexpected_exception_gracefully():
    """
    Verify that an unhandled exception inside _run_loop is caught by the top-level
    try/except, logs traceback, calls on_error callback, and sets state to STOPPED.
    """
    from bot_engine import BotEngine, BotState

    config = {
        "templates": {},
        "thresholds": {},
        "delays": {"scan_interval_ms": 10},
    }

    engine = BotEngine(config=config)
    engine._log = MagicMock()
    error_cb = MagicMock()
    engine.on_error = error_cb

    # Force auto_detect_state to raise an unexpected RuntimeError
    with patch.object(engine, "_auto_detect_state", side_effect=RuntimeError("Simulated hardware crash")):
        engine._run_loop()

    # Bot must catch the error cleanly
    assert engine.state == BotState.STOPPED
    error_cb.assert_called_once()
    assert "Simulated hardware crash" in error_cb.call_args[0][0]
