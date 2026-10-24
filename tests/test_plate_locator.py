import os

import cv2
import numpy as np
import pytest

from core.geometry import Rect
from core.plate_locator import PlateLocator, PlateHit, PlateTracker, hue_mask


# ------------------------------------------------------------------ synthetic plates
def render_plate(text="columbinaa", scale=1.0, green=(0, 255, 0), bg=(40, 60, 90)):
    """A dark pill with neon text, drawn at `scale` (the same plate seen at a different camera zoom)."""
    fs, th = 0.9 * scale, max(1, int(round(2 * scale)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = int(14 * scale)
    img = np.full((tht + base + 2 * pad, tw + 2 * pad + int(30 * scale), 3), bg, np.uint8)
    cv2.rectangle(img, (2, 2), (img.shape[1] - 3, img.shape[0] - 3), (25, 25, 25), -1)
    cv2.putText(img, text, (pad + int(30 * scale), pad + tht), cv2.FONT_HERSHEY_SIMPLEX, fs, green, th, cv2.LINE_AA)
    return img


def scene(plates, size=(1080, 1920), seed=1):
    """plates: [(img, x, y)] pasted on a noisy, mostly bluish background."""
    rng = np.random.default_rng(seed)
    frame = np.clip(rng.normal((120, 90, 60), 18, (*size, 3)), 0, 255).astype(np.uint8)
    for img, x, y in plates:
        frame[y:y + img.shape[0], x:x + img.shape[1]] = img
    return frame


@pytest.fixture(scope="module")
def own_template(tmp_path_factory):
    d = tmp_path_factory.mktemp("tpl")
    cv2.imwrite(str(d / "columbinaa.png"), render_plate("columbinaa", 1.0))
    return str(d)


class TestMask:
    @pytest.mark.parametrize("bgr", [(0, 255, 0), (60, 200, 70), (50, 150, 60)])   # neon -> dimmed by fog / night
    def test_keeps_green_text_in_all_brightness_levels(self, bgr):
        img = np.zeros((20, 20, 3), np.uint8)
        img[:] = bgr
        assert hue_mask(img).all()

    @pytest.mark.parametrize("bgr", [(200, 190, 60), (90, 60, 220), (60, 200, 255), (20, 90, 20)])  # teal foliage, red, yellow, too dark (V<110)
    def test_rejects_foliage_and_other_colours(self, bgr):
        img = np.zeros((20, 20, 3), np.uint8)
        img[:] = bgr
        assert not hue_mask(img).any()


class TestLocator:
    @pytest.mark.parametrize("zoom", [0.5, 0.75, 1.0, 1.5, 2.0, 2.4])
    def test_finds_the_plate_at_any_zoom(self, own_template, zoom):
        """The old NameLocator stopped at 1.25x and returned None for a zoomed-in camera."""
        plate = render_plate("columbinaa", zoom)
        frame = scene([(plate, 700, 400)])
        hit = PlateLocator(own_template).locate(frame)
        assert hit is not None, f"lost the plate at zoom {zoom}"
        assert hit.scale == pytest.approx(zoom, rel=0.18)
        assert abs(hit.plate.x - 700) <= max(6, 0.08 * plate.shape[1])
        assert abs(hit.plate.y - 400) <= max(6, 0.12 * plate.shape[0])

    def test_dim_text_in_bad_weather_is_still_found(self, own_template):
        plate = render_plate("columbinaa", 1.0, green=(60, 190, 75))
        assert PlateLocator(own_template).locate(scene([(plate, 500, 300)])) is not None

    def test_picks_own_plate_among_other_players(self, own_template):
        frame = scene([(render_plate("columbinaa", 1.0), 900, 500),
                       (render_plate("BeTrang", 1.0), 300, 300),
                       (render_plate("Yud", 1.0), 1500, 700)])
        hit = PlateLocator(own_template).locate(frame)
        assert hit is not None and abs(hit.plate.x - 900) < 10 and abs(hit.plate.y - 500) < 10

    def test_cost_does_not_depend_on_where_the_other_players_are(self, own_template):
        """Old cost ~ area of the box around ALL green pixels. Here: 3 plates in the corners of the frame."""
        import time
        loc = PlateLocator(own_template)
        near = scene([(render_plate("columbinaa", 1.0), 900, 500)])
        far = scene([(render_plate("columbinaa", 1.0), 900, 500), (render_plate("Yud", 1.0), 30, 40),
                     (render_plate("BeTrang", 1.0), 1650, 1000)])
        def best_ms(fr):
            ts = []
            for _ in range(8):
                t = time.perf_counter()
                loc.locate(fr)
                ts.append(time.perf_counter() - t)
            return min(ts) * 1000
        assert best_ms(far) < best_ms(near) * 2.5 + 5

    def test_chat_zone_is_ignored(self, own_template):
        frame = scene([(render_plate("columbinaa", 1.0), 40, 800)])     # inside the chat zone (bottom-left)
        assert PlateLocator(own_template).locate(frame) is None

    def test_window_restricts_the_search(self, own_template):
        frame = scene([(render_plate("columbinaa", 1.0), 900, 500)])
        loc = PlateLocator(own_template)
        assert loc.locate(frame, window=Rect(0, 0, 400, 300)) is None
        assert loc.locate(frame, window=Rect(700, 400, 600, 400)) is not None

    def test_respects_origin_offset(self, own_template):
        frame = scene([(render_plate("columbinaa", 1.0), 100, 100)], size=(500, 800))
        hit = PlateLocator(own_template).locate(frame, origin=(2000, 300))
        assert hit is not None and abs(hit.plate.x - 2100) < 8 and abs(hit.plate.y - 400) < 8

    def test_no_green_means_none(self, own_template):
        assert PlateLocator(own_template).locate(scene([])) is None


# ------------------------------------------------------------------ tracker (pure logic, fake locator)
def hit(x, y, score, w=100):
    return PlateHit(plate=Rect(x - 20, y - 6, w + 40, 30), text=Rect(x, y, w, 18), score=score, name="t.png", scale=1.0)


class FakeLocator:
    """Returns its fixed plates that fall inside the requested window, like the real one does."""
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def candidates(self, frame, origin=(0, 0), window=None):
        self.calls.append(window)
        out = [h for h in self.hits if window is None or window.intersect(h.text) is not None]
        return sorted(out, key=lambda h: h.score, reverse=True)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make_tracker(hits, **kw):
    clock = Clock()
    loc = FakeLocator(hits)
    return PlateTracker(loc, clock=clock, **kw), loc, clock


FRAME = np.zeros((10, 10, 3), np.uint8)


class TestTracker:
    def test_global_acquire_then_windowed_tracking(self):
        own = hit(500, 300, 0.7)
        tr, loc, clock = make_tracker([own])
        r1 = tr.update(FRAME)
        assert r1.state == "TRACKING" and loc.calls[0] is None          # first call: global
        clock.t += 0.03
        r2 = tr.update(FRAME)
        assert r2.state == "TRACKING" and loc.calls[1] is not None      # afterwards: window only

    def test_far_stranger_cannot_steal_the_lock(self):
        own, stranger = hit(500, 300, 0.70), hit(1500, 300, 0.95)       # stranger scores higher but is 1000 px away
        tr, loc, clock = make_tracker([own, stranger])
        tr.update(FRAME)                                                 # global acquire picks the stranger! ...
        tr.reset()
        tr2, _, clk2 = make_tracker([own])                               # ... so acquire with only the own plate visible,
        tr2.update(FRAME)
        tr2._loc.hits = [own, stranger]                                  # then the stranger appears
        clk2.t += 0.03
        r = tr2.update(FRAME)
        assert r.hit is not None and r.hit.text.x == 500                 # window search never even sees it

    def test_near_stranger_needs_a_clear_margin(self):
        own = hit(500, 300, 0.70)
        close_slightly_better = hit(700, 300, 0.78)                      # inside the window, only +0.08
        tr, loc, clock = make_tracker([own])
        tr.update(FRAME)
        loc.hits = [own, close_slightly_better]
        clock.t += 0.03
        assert tr.update(FRAME).hit.text.x == 500
        loc.hits = [own, hit(700, 300, 0.90)]                            # +0.20: clearly better -> allowed
        clock.t += 0.03
        assert tr.update(FRAME).hit.text.x == 700

    def test_camera_pan_is_followed(self):
        tr, loc, clock = make_tracker([hit(500, 300, 0.7)])
        tr.update(FRAME)
        for x in (620, 740, 860, 980):                                   # plate slides 120 px per tick
            loc.hits = [hit(x, 300, 0.7)]
            clock.t += 0.03
            assert tr.update(FRAME).hit.text.x == x

    def test_stale_then_lost_then_global_reacquire_is_rate_limited(self):
        tr, loc, clock = make_tracker([hit(500, 300, 0.7)], lost_after_s=1.0, global_interval_s=0.5)
        tr.update(FRAME)
        loc.hits = []
        clock.t += 0.3
        r = tr.update(FRAME)
        assert r.state == "STALE" and r.hit.text.x == 500                # keeps the last known position
        clock.t += 1.0
        assert tr.update(FRAME).state == "LOST"                          # global search tried (nothing found)
        n = len(loc.calls)
        clock.t += 0.1
        assert tr.update(FRAME).state == "LOST" and len(loc.calls) == n  # rate limit: no new search yet
        loc.hits = [hit(1200, 400, 0.7)]
        clock.t += 0.6
        r = tr.update(FRAME)
        assert r.state == "TRACKING" and r.hit.text.x == 1200

    def test_ambiguous_global_acquire_is_flagged(self):
        tr, _, _ = make_tracker([hit(500, 300, 0.74), hit(1300, 320, 0.72)])
        assert tr.update(FRAME).ambiguous is True
        tr2, _, _ = make_tracker([hit(500, 300, 0.80), hit(1300, 320, 0.60)])
        assert tr2.update(FRAME).ambiguous is False


# ------------------------------------------------------------------ real screenshots (optional)
DATASET = os.environ.get("PTG_DATASET", os.path.join("tests", "data", "real"))
BIENSAU = os.path.join(DATASET, "2_Bien_Sau", "BienSau_01_Phai_Ngay_KetVongLap2.4s.png")


@pytest.fixture(scope="module")
def locator(tmp_path_factory):
    """Template cut from one real frame (normal zoom, daylight); every other frame is a different view."""
    src = cv2.imread(BIENSAU)
    m = hue_mask(src)
    m[int(src.shape[0] * .5):, :int(src.shape[1] * .3)] = 0
    ys, xs = np.where(m > 0)
    d = tmp_path_factory.mktemp("real_tpl")
    cv2.imwrite(str(d / "columbinaa.png"), src[ys.min() - 6:ys.max() + 7, xs.min() - 22:xs.max() + 11])
    return PlateLocator(str(d))


@pytest.mark.skipif(not os.path.isfile(BIENSAU), reason="real screenshot dataset not present (set PTG_DATASET)")
class TestRealScreenshots:

    def _all(self):
        out = []
        for root, _, files in os.walk(DATASET):
            out += [os.path.join(root, f) for f in files if f.lower().endswith(".png")]
        return sorted(out)

    def test_every_real_frame_has_the_plate(self, locator):
        missing = [os.path.basename(p) for p in self._all() if locator.locate(cv2.imread(p)) is None]
        assert missing == []

    def test_zoomed_in_camera_is_found_with_scale_over_two(self, locator):
        zoom = [p for p in self._all() if "ZoomCanCanh" in p][0]
        hit_ = locator.locate(cv2.imread(zoom))
        assert hit_ is not None and hit_.scale > 1.8   # the old locator's table ended at 1.25
