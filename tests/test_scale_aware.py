import cv2
import numpy as np
import pytest

from core.geometry import Rect
from core.scale_aware import BiteTemplateBank, clamp_scale, scaled_head_roi


def render_bang(scale, size=(150, 200)):
    """A pale capsule + dot on a smooth background, drawn at `scale`. Returns (image, capsule patch box)."""
    h, w = int(size[1] * scale), int(size[0] * scale)
    img = np.tile(np.linspace(70, 140, w, dtype=np.uint8), (h, 1))
    cx, cy = w // 2, h // 2
    cw, ch = int(10 * scale), int(26 * scale)
    cv2.rectangle(img, (cx - cw, cy - ch), (cx + cw, cy + ch), 235, -1)
    cv2.circle(img, (cx, cy + ch + int(14 * scale)), max(2, int(6 * scale)), 235, -1)
    cv2.circle(img, (cx, cy - ch), cw, 235, -1)
    m = int(8 * scale)
    return cv2.GaussianBlur(img, (3, 3), 0), (cx - cw - m, cy - ch - cw - m, 2 * cw + 2 * m, 2 * ch + cw + int(26 * scale) + 2 * m)


@pytest.fixture()
def bank(tmp_path):
    img, (x, y, w, h) = render_bang(1.0)
    cv2.imwrite(str(tmp_path / "chamthan_test.png"), img[y:y + h, x:x + w])
    return BiteTemplateBank(str(tmp_path))


class TestHeadRoi:
    def test_unit_scale_is_the_old_fixed_geometry(self):
        plate = Rect(600, 400, 137, 40)
        assert scaled_head_roi(plate, 1.0).to_tuple() == (550, 220, 237, 190)

    def test_zoomed_out_camera_shrinks_the_band(self):
        plate = Rect(600, 400, 70, 20)                     # plate at 0.5x
        roi = scaled_head_roi(plate, 0.5)
        assert roi.to_tuple() == (575, 310, 120, 95)       # 180, 10, 50 all halved

    def test_zoomed_in_camera_enlarges_it(self):
        assert scaled_head_roi(Rect(0, 400, 274, 80), 2.0).h == 380

    def test_scale_is_clamped(self):
        assert clamp_scale(0.01) == 0.35 and clamp_scale(99) == 3.0


class TestTemplateBank:
    @pytest.mark.parametrize("zoom", [0.5, 0.7, 1.0, 1.4, 2.0])
    def test_finds_the_bang_at_any_zoom_when_told_the_zoom(self, bank, zoom):
        roi, _ = render_bang(zoom)
        m = bank.best_match(roi, zoom)
        assert m is not None and m.score > 0.85

    def test_native_size_matching_fails_when_zoomed_out(self, bank):
        """This is the reported bug: a 0.5x '!' against 1.0x templates (multi_scale=False in the engine)."""
        roi, _ = render_bang(0.5)
        native = bank.best_match(roi, 1.0)
        aware = bank.best_match(roi, 0.5)
        assert aware.score > 0.85
        assert native is None or native.score < aware.score - 0.15

    def test_cache_is_bounded_by_quantisation(self, bank):
        roi, _ = render_bang(1.0)
        for s in np.linspace(0.90, 1.10, 200):
            bank.best_match(roi, float(s))
        assert len(bank._cache) <= 6                       # 0.05 steps over a 0.2 range

    def test_too_small_scale_returns_none_instead_of_crashing(self, bank):
        roi, _ = render_bang(1.0)
        m = bank.best_match(roi, 0.35)                     # must not raise; a 0.35x template may be below min_side
        assert m is None or m.w >= 8

    def test_roi_smaller_than_template_is_skipped(self, bank):
        assert bank.best_match(np.zeros((5, 5), np.uint8), 1.0) is None

    def test_reference_scale_maps_templates_cut_at_another_zoom(self, tmp_path):
        img, (x, y, w, h) = render_bang(0.5)               # templates were captured with the camera zoomed out
        cv2.imwrite(str(tmp_path / "t.png"), img[y:y + h, x:x + w])
        b = BiteTemplateBank(str(tmp_path), reference_scale=0.5)
        roi, _ = render_bang(1.0)                          # now the plate says 1.0 -> templates must be doubled
        assert b.best_match(roi, 1.0).score > 0.85
