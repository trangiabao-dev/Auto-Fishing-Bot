import os

import cv2
import numpy as np

from core.geometry import Rect
from core.shape_veto import is_cloud_like

DATA = os.path.join(os.path.dirname(__file__), "data")


def load(name):
    img = cv2.imread(os.path.join(DATA, name))
    assert img is not None
    return img


class TestRealScreenshots:
    """Crops of the user's own screenshots."""

    def test_thought_bubble_with_pin_icon_is_vetoed_whatever_part_was_matched(self):
        img = load("bubble_pin.png")                        # bubble: cloud ~106x102, pin icon ~33x70
        for box in (Rect(113, 45, 40, 80),                  # the pin icon
                    Rect(80, 40, 100, 90),                  # the whole cloud
                    Rect(95, 30, 60, 50)):                  # the cloud's upper part
            r = is_cloud_like(img, box)
            assert r.cloud_like, (box, r)
            assert 0.8 <= r.aspect <= 1.4 and r.blob.w > 90

    def test_real_green_bang_beside_the_badge_is_not_vetoed(self):
        r = is_cloud_like(load("bang_green.png"), Rect(53, 28, 15, 64))
        assert not r.cloud_like

    def test_real_pale_top_down_bang_over_water_is_not_vetoed(self):
        r = is_cloud_like(load("bang_pale_topdown.png"), Rect(45, 38, 10, 18))
        assert not r.cloud_like


def scene(bg=(60, 120, 200), size=(260, 300)):
    return np.full((*size, 3), bg, np.uint8)


class TestSynthetic:
    def test_white_capsule_bang_is_not_a_cloud(self):
        img = scene()
        cv2.rectangle(img, (140, 60), (170, 150), (250, 250, 250), -1)
        cv2.circle(img, (155, 60), 15, (250, 250, 250), -1)
        cv2.circle(img, (155, 175), 8, (250, 250, 250), -1)
        r = is_cloud_like(img, Rect(138, 45, 34, 100))
        assert not r.cloud_like and r.aspect > 2.0

    def test_white_filled_blob_is_a_cloud(self):
        img = scene()
        cv2.ellipse(img, (150, 130), (55, 50), 0, 0, 360, (252, 252, 252), -1)
        cv2.rectangle(img, (140, 100), (160, 160), (170, 170, 175), -1)   # darker icon inside
        assert is_cloud_like(img, Rect(135, 95, 30, 70)).cloud_like

    def test_coloured_capsule_never_enters_the_mask(self):
        img = scene()
        cv2.rectangle(img, (140, 60), (170, 150), (200, 120, 230), -1)    # lavender/pink '!'
        r = is_cloud_like(img, Rect(138, 58, 34, 94))
        assert not r.cloud_like and r.blob is None

    def test_box_without_any_white_underneath(self):
        assert not is_cloud_like(scene(), Rect(50, 50, 30, 30)).cloud_like

    def test_box_partly_outside_the_image_does_not_crash(self):
        r = is_cloud_like(scene(), Rect(-20, -20, 40, 40))
        assert not r.cloud_like
