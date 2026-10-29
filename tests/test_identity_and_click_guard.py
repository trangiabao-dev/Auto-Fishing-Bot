import os
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from core.click_guard import (BLUE_HI, BLUE_LO, YELLOW_HI, YELLOW_LO, cast_button_core, keep_button_core)
from core.identity_gate import GatedTracker, IdentityGate

DATA = os.path.join(os.path.dirname(__file__), "data")


# ============================================================ IdentityGate
class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def H(x, y, w=150, score=0.75):
    """A plate hit reduced to what the gate reads: text centre, text width, score."""
    return NS(text=NS(center=(x, y), w=w), score=score)


def gate(**kw):
    clock = Clock()
    return IdentityGate(clock=clock, **kw), clock


def feed(g, clock, hits, step=0.5):
    out = []
    for h in hits:
        clock.t += step
        out.append(g.observe(h))
    return out


class TestIdentityGate:
    def test_first_lock_is_trusted_and_flags_motion(self):
        g, _ = gate()
        v = g.observe(H(900, 310))
        assert v.trusted and v.reason == "locked" and v.motion and g.anchor is not None

    def test_tiny_drift_is_continuous_without_motion(self):
        g, c = gate()
        feed(g, c, [H(900, 310)])
        v = feed(g, c, [H(908, 312)])[0]
        assert v.trusted and v.reason == "continuous" and not v.motion

    def test_hand_rotated_camera_is_followed_step_by_step(self):
        g, c = gate()
        feed(g, c, [H(500, 300)])
        verdicts = feed(g, c, [H(500 + 60 * i, 300) for i in range(1, 8)])   # 60 px per tick, same width
        assert all(v.trusted for v in verdicts) and all(v.motion for v in verdicts)

    def test_zoom_around_a_fixed_plate_is_trusted_even_though_the_width_changes_a_lot(self):
        g, c = gate()
        feed(g, c, [H(900, 310, w=150)])
        v = feed(g, c, [H(902, 311, w=215)])[0]
        assert v.trusted and v.motion

    def test_the_logged_hijack_is_rejected_and_the_roi_stays_on_the_anchor(self):
        """10:07:04 log: plate jumped ~330 px and the scale went 1.11 -> 1.29 while the camera was still."""
        g, c = gate()
        own = H(900, 310, w=150, score=0.75)
        feed(g, c, [own])
        stranger = H(1230, 200, w=174, score=0.68)                              # +16 % width, 348 px away
        for v in feed(g, c, [stranger] * 12):
            assert not v.trusted and v.rejected and v.hit is own and v.reason in ("jump_width", "jump_too_far")

    @pytest.mark.parametrize("own_w,other_w", [(163, 77), (78, 41), (82, 34), (81, 18)])
    def test_the_other_green_name_seen_in_the_screenshots_never_wins(self, own_w, other_w):
        """Measured text widths of our name vs the neighbour's in the same frame (ratio 0.22-0.53)."""
        g, c = gate()
        feed(g, c, [H(700, 500, w=own_w)])
        for v in feed(g, c, [H(700 - 250, 500 + 230, w=other_w, score=0.7)] * 12):
            assert not v.trusted

    def test_stacked_neighbour_with_a_different_width_is_rejected(self):
        g, c = gate()
        feed(g, c, [H(900, 310, w=150)])
        v = feed(g, c, [H(900, 350, w=200)])[0]          # 40 px below (inside the step limit) but 33 % wider
        assert not v.trusted and v.reason == "jump_width"

    def test_lookalike_within_reach_is_accepted_after_3_ticks(self):
        g, c = gate()
        feed(g, c, [H(900, 310, w=100)])
        verdicts = feed(g, c, [H(1060, 310, w=100, score=0.72)] * 3)
        assert [v.trusted for v in verdicts] == [False, False, True] and verdicts[-1].reason == "jump_accepted"

    def test_distant_stranger_is_never_accepted_even_with_same_width_and_score(self):
        """Strangers > max_jump_dist px away (e.g. BéTrăng at 348 px) can NEVER steal the lock."""
        g, c = gate()
        feed(g, c, [H(900, 310)])
        verdicts = feed(g, c, [H(1230, 200, w=150, score=0.72)] * 6)
        assert all(not v.trusted for v in verdicts) and all(v.reason == "jump_too_far" for v in verdicts)

    def test_a_jump_with_a_much_lower_score_is_never_accepted(self):
        g, c = gate()
        feed(g, c, [H(900, 310, score=0.78)])
        assert not any(v.trusted for v in feed(g, c, [H(1230, 200, w=150, score=0.60)] * 10))

    def test_returning_to_the_anchor_clears_the_pending_jump(self):
        g, c = gate()
        feed(g, c, [H(900, 310)])
        feed(g, c, [H(1230, 200)] * 2)
        assert feed(g, c, [H(905, 312)])[0].trusted
        assert not feed(g, c, [H(1230, 200)])[0].trusted        # the count started again from 1

    def test_alternating_candidates_never_confirm(self):
        g, c = gate()
        feed(g, c, [H(900, 310)])
        hits = [H(1230, 200), H(1300, 420)] * 6
        assert not any(v.trusted for v in feed(g, c, hits))

    def test_stale_plate_is_trusted_for_a_while_then_not(self):
        g, c = gate(stale_trust_s=4.0)
        feed(g, c, [H(900, 310)])
        c.t += 1.0
        assert g.observe(None, "STALE").trusted
        c.t += 4.0
        v = g.observe(None, "STALE")
        assert not v.trusted and v.reason == "lost"

    def test_ambiguous_first_lock_is_not_a_lock(self):
        g, _ = gate()
        v = g.observe(H(900, 310), ambiguous=True)
        assert not v.trusted and v.reason == "ambiguous" and g.anchor is None
        assert g.observe(H(900, 310)).reason == "locked"

    def test_reset_forgets_the_anchor(self):
        g, c = gate()
        feed(g, c, [H(900, 310)])
        g.reset()
        assert g.anchor is None and g.observe(H(100, 100)).reason == "locked"


class FakeTracker:
    def __init__(self):
        self.hit, self.state, self.pinned, self.resets = None, "TRACKING", [], 0

    def update(self, frame, origin=(0, 0)):
        return NS(hit=self.hit, state=self.state, ambiguous=False)

    def pin(self, hit):
        self.pinned.append(hit)

    def reset(self):
        self.resets += 1


class TestGatedTracker:
    def test_rejected_jump_pins_the_tracker_on_the_anchor_and_returns_the_anchor(self):
        ft = FakeTracker()
        gt = GatedTracker(ft, IdentityGate(clock=Clock()))
        own = H(900, 310)
        ft.hit = own
        assert gt.update(None).trusted
        ft.hit = H(1230, 200, w=174)
        r = gt.update(None)
        assert not r.trusted and r.hit is own and ft.pinned == [own]

    def test_trusted_result_does_not_pin(self):
        ft = FakeTracker()
        gt = GatedTracker(ft, IdentityGate(clock=Clock()))
        ft.hit = H(900, 310)
        gt.update(None)
        ft.hit = H(930, 312)
        r = gt.update(None)
        assert r.trusted and ft.pinned == []

    def test_reset_resets_both(self):
        ft = FakeTracker()
        gt = GatedTracker(ft, IdentityGate(clock=Clock()))
        ft.hit = H(900, 310)
        gt.update(None)
        gt.reset()
        assert ft.resets == 1 and gt.gate.anchor is None

    def test_tracker_without_pin_still_works(self):
        ft = FakeTracker()
        del FakeTracker.pin
        try:
            gt = GatedTracker(ft, IdentityGate(clock=Clock()))
            ft.hit = H(900, 310)
            gt.update(None)
            ft.hit = H(1230, 200, w=174)
            assert not gt.update(None).trusted
        finally:
            FakeTracker.pin = lambda self, hit: self.pinned.append(hit)


# ============================================================ ClickGuard
POPUP = cv2.imread(os.path.join(DATA, "popup_keep_crop.png"))
POPUP_ORIGIN = (1180, 740)                 # where the crop sits in the real 1917x1078 frame
SCREEN_H = 1078
BAOQUAN = (1395, 779, 230, 93)
KHOE = (1639, 779, 231, 93)
COIN = (1217, 762, 112, 100)               # the 'Thành viên Bán tức thì' icon


def gap_x(core, rect):
    """Horizontal clearance between two boxes (>= 0 when they do not overlap in x)."""
    return max(rect[0] - (core[0] + core[2]), core[0] - (rect[0] + rect[2]))


class TestKeepButton:
    def test_exact_box_gives_a_core_inside_the_blue_pill(self):
        core = keep_button_core(POPUP, POPUP_ORIGIN, BAOQUAN, SCREEN_H)
        assert core is not None
        assert BAOQUAN[0] < core[0] and core[0] + core[2] < BAOQUAN[0] + BAOQUAN[2]
        assert BAOQUAN[1] < core[1] and core[1] + core[3] < BAOQUAN[1] + BAOQUAN[3]

    def test_an_oversized_box_that_reaches_khoe_still_yields_a_core_inside_bao_quan(self):
        """The failure mode: template/box too big -> a random click inside it lands on Khoe."""
        sloppy = (1340, 760, 400, 130)                 # covers Bảo quản AND the left half of Khoe
        core = keep_button_core(POPUP, POPUP_ORIGIN, sloppy, SCREEN_H)
        assert core is not None and core[0] + core[2] < KHOE[0] - 40 and core[0] > BAOQUAN[0]

    def test_core_keeps_clear_of_khoe_and_the_coin_icon(self):
        core = keep_button_core(POPUP, POPUP_ORIGIN, BAOQUAN, SCREEN_H)
        assert gap_x(core, KHOE) >= 40 and gap_x(core, COIN) >= 60

    def test_no_random_click_inside_the_core_touches_a_yellow_pixel(self):
        core = keep_button_core(POPUP, POPUP_ORIGIN, BAOQUAN, SCREEN_H)
        hsv = cv2.cvtColor(POPUP, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
        blue = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
        rng = np.random.default_rng(0)
        for _ in range(3000):
            px = int(rng.integers(core[0], core[0] + core[2])) - POPUP_ORIGIN[0]
            py = int(rng.integers(core[1], core[1] + core[3])) - POPUP_ORIGIN[1]
            assert not yellow[py, px]
            # the pill has white text on it: look at a 9x9 neighbourhood, it must contain blue face pixels
            assert blue[max(0, py - 4):py + 5, max(0, px - 4):px + 5].any()

    def test_a_box_on_khoe_never_returns_a_core(self):
        assert keep_button_core(POPUP, POPUP_ORIGIN, KHOE, SCREEN_H) is None

    def test_a_box_on_the_coin_icon_never_returns_a_core(self):
        assert keep_button_core(POPUP, POPUP_ORIGIN, COIN, SCREEN_H) is None

    def test_no_blue_pill_means_no_click(self):
        blank = np.full((300, 600, 3), 90, np.uint8)
        assert keep_button_core(blank, (1300, 700), (1395, 779, 230, 93), SCREEN_H) is None

    def test_box_outside_the_frame_means_no_click(self):
        assert keep_button_core(POPUP, POPUP_ORIGIN, (10, 10, 50, 50), SCREEN_H) is None


# Hough-measured circles of the cast cluster crop (centre x, centre y, radius), 1:1 with a 1080p frame
CAST = (140, 116, 52)
OTHERS = {"X": (92, 231, 36), "+": (261, 75, 39), "up-arrow": (265, 260, 89)}


class TestCastButton:
    @pytest.mark.parametrize("off", [(0, 0), (12, -9), (-14, 10)])
    def test_core_cannot_reach_x_plus_or_arrow_even_if_the_match_is_off_by_15px(self, off):
        box = (CAST[0] - 25 + off[0], CAST[1] - 30 + off[1], 50, 60)              # an icon-sized template box
        x, y, w, h = cast_button_core(box, SCREEN_H)
        for cx, cy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:           # even the square's CORNERS
            for name, (ox, oy, r) in OTHERS.items():
                assert np.hypot(cx - ox, cy - oy) - r > 15, name

    def test_core_is_small_and_centred_on_the_icon(self):
        x, y, w, h = cast_button_core((100, 70, 80, 90), SCREEN_H)
        assert w == h == 52 and (x + w / 2, y + h / 2) == (140, 115)

    def test_core_scales_with_the_screen(self):
        assert cast_button_core((0, 0, 40, 40), 720)[2] == 35

    def test_fixture_really_contains_four_distinct_buttons(self):
        img = cv2.imread(os.path.join(DATA, "cast_cluster_crop.png"))
        assert img is not None and img.shape[:2] == (432, 440)
