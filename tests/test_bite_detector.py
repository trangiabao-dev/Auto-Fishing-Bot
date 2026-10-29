import json
import os

import cv2
import numpy as np
import pytest

from core.bite_detector import BiteDetector
from core.bite_policy import TriggerPolicy
from tools.replay_bite_detector import evaluate, summarize

WATER_PURPLE = (199, 109, 70)     # BGR ~ HSV(111,165,199): the purple/blue water of the real caught clips
SKY_PINK = (200, 150, 240)
PINK_PASTEL = (230, 170, 250)
DEEP_PINK = (156, 112, 168)       # the badge-like deep pink that the OLD absolute gate (min channel >= 130) rejects
DT = 0.03


def background(bg, size=(190, 237), seed=0):
    rng = np.random.default_rng(seed)
    img = np.empty((*size, 3), np.uint8)
    img[:] = bg
    return np.clip(img.astype(int) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)


def draw_bang(img, color, k=1.0, cx=None, top=None, alpha=0.9):
    """Semi-transparent capsule + dot, ~19x48 px at k=1, drawn over `img` in place."""
    h, w = img.shape[:2]
    cx = w // 2 if cx is None else cx
    top = int(60 * k) if top is None else top
    ov = img.copy()
    cw, ch = int(9 * k), int(34 * k)
    cv2.rectangle(ov, (cx - cw, top), (cx + cw, top + ch), color, -1)
    cv2.circle(ov, (cx, top), cw, color, -1)
    cv2.circle(ov, (cx, top + ch + int(7 * k)), max(2, int(5 * k)), color, -1)
    img[:] = cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)


def run(frames, det=None, scale=None, t0=0.0):
    """Feeds frames 30 ms apart. Returns (detector, index of the first frame that returned a hit or None)."""
    det = det or BiteDetector()
    first = None
    for i, f in enumerate(frames):
        if det.update(f, t0 + i * DT, scale=scale) is not None and first is None:
            first = i
    return det, first


def scene(bg, bang_color, k=1.0, warm=30, bang_frames=15, size=(190, 237), static=None, **kw):
    frames = []
    for i in range(warm + bang_frames):
        f = background(bg, size, seed=i)
        if static:
            static(f)
        if i >= warm:
            draw_bang(f, bang_color, k=k, **kw)
        frames.append(f)
    return frames, warm


class TestRealBitesOnEveryBackground:
    @pytest.mark.parametrize("bg,color", [
        (WATER_PURPLE, PINK_PASTEL),          # the recorded caught clips: pink '!' on purple/blue water
        (SKY_PINK, (255, 255, 255)),          # white '!' on a pink sky
        (WATER_PURPLE, DEEP_PINK),            # deep pink: rejected by the old absolute gate
        ((70, 70, 90), (120, 230, 255)),      # night: dark background, yellowish '!'
    ])
    def test_detected_within_a_few_scans_of_appearing(self, bg, color):
        frames, onset = scene(bg, color)
        _, first = run(frames)
        assert first is not None, "bite not detected"
        assert onset <= first <= onset + 3

    def test_hit_carries_calibration_features(self):
        frames, _ = scene(WATER_PURPLE, PINK_PASTEL)
        det = BiteDetector()
        hit = None
        for i, f in enumerate(frames):
            hit = det.update(f, i * DT) or hit
        assert hit.contrast > 20 and hit.aspect >= 1.6 and 0.45 <= hit.fill <= 1.0

    def test_invisible_bite_with_no_colour_difference_is_not_detectable(self):
        """Documented limit: no signal, no detection (the fix is data-driven thresholds, not a smarter filter)."""
        bg = (120, 120, 120)
        frames, _ = scene(bg, (126, 126, 126))
        _, first = run(frames)
        assert first is None


class TestEverythingElseIsRejected:
    def test_static_decoration_already_there_never_triggers(self):
        def badge(f):                       # a pink title badge + a capsule-shaped static prop on the axis
            draw_bang(f, DEEP_PINK, alpha=1.0)
            cv2.rectangle(f, (95, 20), (140, 34), (200, 120, 230), -1)
        frames = [background(WATER_PURPLE, seed=i) for i in range(60)]
        for f in frames:
            badge(f)
        _, first = run(frames)
        assert first is None

    def test_thought_bubble_appearing_suddenly_is_rejected(self):
        def bubble(f):
            cv2.ellipse(f, (118, 90), (53, 50), 0, 0, 360, (252, 252, 252), -1)
            cv2.rectangle(f, (105, 65), (125, 115), (170, 170, 175), -1)
        frames = [background(WATER_PURPLE, seed=i) for i in range(45)]
        for f in frames[30:]:
            bubble(f)
        _, first = run(frames)
        assert first is None

    def test_drifting_white_mist_never_triggers(self):
        rng = np.random.default_rng(5)
        frames = []
        for i in range(110):
            f = background(WATER_PURPLE, seed=i)
            for j in range(3):                                   # soft translucent blobs drifting sideways
                cx = int((40 + 70 * j + i * 2.2) % 237)
                cy = 70 + 25 * j
                ov = f.copy()
                cv2.ellipse(ov, (cx, cy), (28 + rng.integers(0, 6), 22), 0, 0, 360, (255, 255, 255), -1)
                f = cv2.addWeighted(ov, 0.55, f, 0.45, 0)
            frames.append(f)
        _, first = run(frames)
        assert first is None

    def test_round_sparkles_and_bobbers_are_rejected(self):
        frames = [background(WATER_PURPLE, seed=i) for i in range(45)]
        for f in frames[30:]:
            cv2.circle(f, (118, 100), 9, (120, 240, 255), -1)    # round, ~19 px: not an upright capsule
        _, first = run(frames)
        assert first is None

    def test_global_flicker_is_rejected(self):
        frames = [background(WATER_PURPLE, seed=i) for i in range(45)]
        for i in range(30, 36):
            frames[i] = np.clip(frames[i].astype(int) + 45, 0, 255).astype(np.uint8)
        _, first = run(frames)
        assert first is None

    def test_neighbours_capsule_off_the_head_axis_is_rejected(self):
        frames = []
        for i in range(45):
            f = background(WATER_PURPLE, seed=i)
            if i >= 30:
                draw_bang(f, PINK_PASTEL, cx=int(237 * 0.5 + 237 * 0.3))   # 30 % of the ROI width to the right
            frames.append(f)
        # the strip is only +-22 % wide, so such a capsule is never even looked at
        _, first = run(frames)
        assert first is None

    def test_one_frame_flash_is_not_enough(self):
        frames = [background(WATER_PURPLE, seed=i) for i in range(45)]
        draw_bang(frames[35], PINK_PASTEL)
        _, first = run(frames)
        assert first is None

    def test_candidate_that_jumps_every_scan_is_not_persistent(self):
        frames = [background(WATER_PURPLE, seed=i) for i in range(45)]
        for i in range(30, 45):
            draw_bang(frames[i], PINK_PASTEL, top=30 if i % 2 else 110)   # alternates 80 px apart
        _, first = run(frames)
        assert first is None


class TestZoom:
    def test_scale_argument_fixes_a_roi_that_is_wide_because_of_a_long_name(self):
        """Regression for the old `k = roi_w / 240`: a 360 px wide ROI (long name) at zoom 0.7 made every limit 50 % too big.
        The ROI height follows the zoom (scaled_head_roi: 190 * scale), only its width also depends on the name."""
        size = (133, 360)
        frames, onset = scene(WATER_PURPLE, PINK_PASTEL, k=0.7, size=size)
        _, with_scale = run(frames, scale=0.7)
        _, legacy = run(frames, scale=None)
        assert with_scale is not None and with_scale <= onset + 3
        assert legacy is None                       # documents the bug that `scale` removes

    @pytest.mark.parametrize("k", [0.5, 1.0, 1.6])
    def test_works_at_other_zoom_levels(self, k):
        size = (int(190 * k), int(237 * k))
        frames, onset = scene(WATER_PURPLE, PINK_PASTEL, k=k, size=size)
        _, first = run(frames, scale=k)
        assert first is not None and first <= onset + 3

    def test_reset_forgets_history(self):
        frames, onset = scene(WATER_PURPLE, PINK_PASTEL, warm=30, bang_frames=1)
        det, _ = run(frames[:onset])
        det.reset()
        assert det.update(frames[onset], onset * DT) is None   # no references any more -> no evidence, no hit


class TestTriggerPolicy:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            TriggerPolicy("shadw")

    @pytest.mark.parametrize("mode", ["off", "shadow"])
    def test_template_alone_reels_in_shadow_modes(self, mode):
        p = TriggerPolicy(mode)
        p.note_detector(1.0, False)
        p.note_template(1.0, True)
        assert p.should_reel(1.0)

    def test_shadow_never_reels_on_detector_alone(self):
        p = TriggerPolicy("shadow")
        p.note_detector(1.0, True)
        p.note_template(1.0, False)
        assert not p.should_reel(1.0) and p.detector_recent(1.0)

    def test_confirm_needs_both_within_the_window_in_either_order(self):
        p = TriggerPolicy("confirm", window_s=0.6)
        p.note_template(1.0, True)
        p.note_detector(1.0, False)
        assert not p.should_reel(1.0)
        p.note_template(1.2, False)
        p.note_detector(1.3, True)
        assert p.should_reel(1.3)                 # template came first, detector 0.3 s later
        p.reset()
        p.note_detector(2.0, True)
        p.note_template(3.0, True)                # detector is 1.0 s old: outside the window
        assert not p.should_reel(3.0)

    def test_primary_reels_on_detector_alone(self):
        p = TriggerPolicy("primary")
        p.note_detector(1.0, True)
        p.note_template(1.0, False)
        assert p.should_reel(1.0)

    def test_either_reels_on_detector_alone_or_template_alone(self):
        p = TriggerPolicy("either")
        # Neither sees it -> False
        assert not p.should_reel(1.0)

        # Detector sees it -> True
        p.note_detector(1.0, True)
        assert p.should_reel(1.0)

        # Template alone sees it -> True
        p.reset()
        p.note_template(1.0, True)
        assert p.should_reel(1.0)



class TestReplayTool:
    def _clip(self, root, outcome, name, with_bang):
        folder = os.path.join(root, outcome, name)
        os.makedirs(folder)
        rows = []
        n = 90                                                   # 2.7 s at 30 ms
        for i in range(n):
            f = background(WATER_PURPLE, seed=i)
            if with_bang and i >= n - 12:
                draw_bang(f, PINK_PASTEL)
            fname = f"roi_{i:03d}.jpg"
            cv2.imwrite(os.path.join(folder, fname), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({"file": fname, "t_before_trigger_s": round((i - (n - 1)) * DT, 3), "confidence": 0.0,
                         "template": ""})
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"outcome": outcome, "frames": rows}, fh)

    def test_recall_and_false_trigger_report(self, tmp_path):
        root = str(tmp_path)
        self._clip(root, "caught", "a", True)
        self._clip(root, "caught", "b", True)
        self._clip(root, "missed", "c", False)                   # template false trigger: nothing real on screen
        self._clip(root, "rod_retracted", "d", True)             # a real bite the template never saw
        res = evaluate(root)
        text = "\n".join(summarize(res))
        assert all(c.fired_near_trigger for c in res["caught"])
        assert not res["missed"][0].fired
        assert res["rod_retracted"][0].fired
        assert "RECALL near trigger 2/2" in text and "ADDED a catch in 1/1" in text
        assert "contrast of hits in caught clips" in text
