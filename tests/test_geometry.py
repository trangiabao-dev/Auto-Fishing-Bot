import numpy as np
import pytest

from core.geometry import Frame, Point, Rect


class TestRect:
    def test_edges_and_center(self):
        r = Rect(10, 20, 30, 40)
        assert (r.right, r.bottom) == (40, 60)
        assert r.center == Point(25, 40)
        assert r.to_tuple() == (10, 20, 30, 40)
        assert Rect.from_tuple((10, 20, 30, 40)) == r

    def test_negative_size_rejected(self):
        with pytest.raises(ValueError):
            Rect(0, 0, -1, 5)

    def test_intersect_overlap(self):
        assert Rect(0, 0, 10, 10).intersect(Rect(5, 5, 10, 10)) == Rect(5, 5, 5, 5)

    def test_intersect_touching_edges_is_none(self):
        assert Rect(0, 0, 10, 10).intersect(Rect(10, 0, 5, 5)) is None

    def test_intersect_disjoint_is_none(self):
        assert Rect(0, 0, 10, 10).intersect(Rect(50, 50, 5, 5)) is None

    def test_inflate_both_sides(self):
        assert Rect(100, 100, 20, 10).inflate(60, 40) == Rect(40, 60, 140, 90)

    def test_inflate_negative_never_goes_below_zero(self):
        r = Rect(10, 10, 4, 4).inflate(-10)
        assert r.w == 0 and r.h == 0 and r.is_empty

    def test_contains_is_half_open(self):
        r = Rect(0, 0, 10, 10)
        assert r.contains(Point(0, 0)) and r.contains(Point(9, 9))
        assert not r.contains(Point(10, 5)) and not r.contains(Point(5, 10))

    def test_translate(self):
        assert Rect(1, 2, 3, 4).translate(10, 20) == Rect(11, 22, 3, 4)


class TestFrame:
    def _frame(self, w=200, h=100, origin=(1000, 500)):
        img = np.arange(w * h * 3, dtype=np.uint8).reshape(h, w, 3)
        return Frame(img, Point(*origin))

    def test_bounds_are_in_screen_coordinates(self):
        f = self._frame()
        assert f.bounds == Rect(1000, 500, 200, 100)

    def test_local_global_roundtrip(self):
        f = self._frame()
        local = Rect(10, 20, 30, 40)
        g = f.to_global(local)
        assert g == Rect(1010, 520, 30, 40)
        assert f.to_local(g) == local

    def test_crop_returns_view_with_correct_origin(self):
        f = self._frame()
        sub = f.crop(Rect(1050, 520, 40, 30))
        assert sub is not None
        assert sub.origin == Point(1050, 520)
        assert (sub.width, sub.height) == (40, 30)
        # same pixels as the matching slice of the parent, and no copy was made
        assert np.array_equal(sub.image, f.image[20:50, 50:90])
        assert np.shares_memory(sub.image, f.image)

    def test_crop_is_clipped_to_frame(self):
        f = self._frame()
        sub = f.crop(Rect(1150, 560, 500, 500))
        assert sub is not None
        assert sub.bounds == Rect(1150, 560, 50, 40)

    def test_crop_without_overlap_is_none(self):
        assert self._frame().crop(Rect(0, 0, 50, 50)) is None

    def test_crop_keeps_screen_coordinates_consistent(self):
        # the property that kills the local/global bug: a box found in a crop converts back to the same
        # place on the screen no matter how many crops deep it came from
        f = self._frame()
        sub = f.crop(Rect(1050, 520, 100, 60)).crop(Rect(1070, 530, 30, 30))
        found_inside_sub = Rect(5, 5, 10, 10)
        assert sub.to_global(found_inside_sub) == Rect(1075, 535, 10, 10)

    def test_rejects_bad_ndim(self):
        with pytest.raises(ValueError):
            Frame(np.zeros(5, dtype=np.uint8))
