import threading
from enum import Enum
from types import SimpleNamespace

import pytest

from core.instrumentation import attach_telemetry


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt


class State(Enum):
    IDLE = 0
    STAGE_1 = 1
    STAGE_2 = 2
    STOPPED = 9


class ListSink:
    def __init__(self):
        self.items = []

    def write(self, obj):
        self.items.append(obj)


def build_engine(clock, scenario):
    """A fake with exactly the attributes attach_telemetry wraps; no OpenCV/mss/Win32 needed."""

    class Capturer:
        def capture(self, region=None):
            clock.advance(0.020 if region is None else 0.004)
            return ("frame", (0, 0))

    class Matcher:
        def match(self, screen, template_path, threshold=0.8, screen_offset=(0, 0), grayscale=True,
                  multi_scale=True):
            clock.advance(0.005)
            return SimpleNamespace(found=False, confidence=0.5)

        def match_any(self, screen, template_target, threshold=0.75, screen_offset=(0, 0), grayscale=True,
                      multi_scale=True):
            self.match(screen, template_target + "/a.png", multi_scale=multi_scale)  # nested call
            self.match(screen, template_target + "/b.png", multi_scale=multi_scale)
            return SimpleNamespace(found=False, confidence=0.61)

    class Locator:
        def locate(self, frame, offset=(0, 0)):
            clock.advance(0.64)
            return None

    class Recorder:
        def begin_cast(self, roi, meta=None):
            self.began = True

        def record(self, frame, confidence, name=""):
            pass

        def record_full(self, frame):
            pass

        def dump(self, reason, mark_time=None):
            return None

    class Sim:
        def click(self, x, y, hold_min_ms=40, hold_max_ms=90):
            clock.advance(0.06)

    eng = SimpleNamespace(
        capturer=Capturer(), matcher=Matcher(), name_locator=Locator(), recorder=Recorder(),
        input_sim=Sim(), state=State.IDLE, logs=[],
    )
    eng._log = eng.logs.append
    eng._auto_detect_state = lambda: State.STAGE_1

    def set_state(s):
        eng.state = s

    eng._set_state = set_state
    eng._run_loop = lambda: scenario(eng, clock)
    return eng


def stage2_scenario(eng, clock):
    eng._set_state(State.STAGE_2)
    eng.recorder.begin_cast((0, 0, 230, 190), {})
    for _ in range(10):                       # healthy scans
        clock.advance(0.03)
        eng.capturer.capture((0, 0, 230, 190))  # ROI grab = the tick marker
        eng.recorder.record("roi", 0.3, "x")
    eng.capturer.capture(None)                # the "relocate" block that blinds the loop
    eng.name_locator.locate("full")
    eng.matcher.match_any("roi", "assets/B2_DangCau/dau_cham_than", multi_scale=False)
    clock.advance(0.03)
    eng.capturer.capture((0, 0, 230, 190))
    eng.recorder.record("roi", 0.3, "x")
    eng._set_state(State.STOPPED)


def attach(eng, clock):
    sink = ListSink()
    tel = attach_telemetry(eng, jsonl_path=None, clock=clock)
    tel._sink = sink
    return tel, sink


def test_behaviour_and_return_values_are_untouched():
    clock = FakeClock()
    eng = build_engine(clock, stage2_scenario)
    attach(eng, clock)
    frame, offset = eng.capturer.capture(None)
    assert (frame, offset) == ("frame", (0, 0))
    assert eng.matcher.match_any("s", "d").confidence == 0.61
    assert eng.name_locator.locate("f") is None
    assert eng._auto_detect_state() is State.STAGE_1


def test_exceptions_pass_through_and_depth_recovers():
    clock = FakeClock()

    def scenario(eng, clk):
        with pytest.raises(RuntimeError, match="grab failed"):
            eng.capturer.capture(None)          # the wrapped call raises: the exception must pass through
        # depth bookkeeping survived the exception, so this call is still "outermost" (counted in the breakdown)
        eng.matcher.match_any("s", "assets/X")

    eng = build_engine(clock, scenario)
    def boom(region=None):
        clock.advance(0.01)
        raise RuntimeError("grab failed")
    eng.capturer.capture = boom                  # replaced BEFORE attach, like a real failing mss.grab
    tel, sink = attach(eng, clock)
    eng._run_loop()
    names = sink.items[0]["sections_ms"]
    assert names["capture:full"]["n"] == 1       # the failed call was still timed
    assert "match_any[X]+ms" in names


def test_scan_gap_is_attributed_to_locate_and_nested_calls_are_not_double_counted():
    clock = FakeClock()
    eng = build_engine(clock, stage2_scenario)
    tel, sink = attach(eng, clock)
    eng._run_loop()
    s = [i for i in sink.items if i["label"] == "STAGE_2"][0]
    top = dict(s["worst_gap"]["top"])
    assert top["locate"] == pytest.approx(640, abs=1)
    assert top["capture:full"] == pytest.approx(20, abs=1)
    assert top["match_any[dau_cham_than]"] == pytest.approx(10, abs=1)  # 2 x 5 ms, counted once
    assert not any(n.startswith("match[") for n in top)
    assert s["sections_ms"]["match[a.png]"]["n"] == 1                   # nested ones keep their own stats
    assert s["scores"]["match_any[dau_cham_than]"]["max"] == 0.61
    # ticks are ROI grabs (4 ms each). First gap 30 ms, then 9 gaps of 34 ms, then the blind gap:
    #   4 (previous grab) + 20 (full grab) + 640 (locate) + 10 (match_any) + 30 (sleep) = 704 ms
    blind = 0.004 + 0.020 + 0.640 + 0.010 + 0.030
    span = 0.030 + 9 * 0.034 + blind
    assert s["miss_prob_400ms"] == pytest.approx((blind - 0.4) / span, abs=1e-3)
    assert s["worst_gap"]["ms"] == pytest.approx(blind * 1000, abs=1)
    assert any("xác suất lọt" in line for line in eng.logs)             # GUI line produced for STAGE_2


def test_multi_scale_is_visible_in_the_section_name():
    clock = FakeClock()

    def scenario(eng, clk):
        eng.matcher.match_any("s", "assets/B1_QuangCan")
        eng.matcher.match_any("s", "assets/B1_QuangCan", multi_scale=False)

    eng = build_engine(clock, scenario)
    tel, sink = attach(eng, clock)
    eng._run_loop()
    names = sink.items[0]["sections_ms"]
    assert "match_any[B1_QuangCan]+ms" in names and "match_any[B1_QuangCan]" in names


def test_calls_from_other_threads_are_ignored():
    clock = FakeClock()

    def scenario(eng, clk):
        eng.capturer.capture(None)            # engine thread: measured
        t = threading.Thread(target=lambda: eng.capturer.capture(None))  # e.g. "Soi Mắt Bot" on the Tk thread
        t.start()
        t.join()

    eng = build_engine(clock, scenario)
    tel, sink = attach(eng, clock)
    eng._run_loop()
    assert sink.items[0]["sections_ms"]["capture:full"]["n"] == 1


def test_attach_is_idempotent():
    clock = FakeClock()
    eng = build_engine(clock, stage2_scenario)
    assert attach_telemetry(eng, jsonl_path=None, clock=clock) is attach_telemetry(eng, jsonl_path=None, clock=clock)


def test_engine_without_a_recorder_is_still_measured():
    """The restored baseline may not have StageTwoRecorder: telemetry must degrade, not crash."""
    clock = FakeClock()

    def scenario(eng, clk):
        eng._set_state(State.STAGE_2)
        for _ in range(5):
            clk.advance(0.03)
            eng.capturer.capture((0, 0, 230, 190))

    eng = build_engine(clock, scenario)
    del eng.recorder
    tel, sink = attach(eng, clock)
    assert "recorder.record" in eng.telemetry_missing and "recorder.begin_cast" in eng.telemetry_missing
    eng._run_loop()
    s = [i for i in sink.items if i["label"] == "STAGE_2"][0]
    assert s["scans"] == 5 and s["gap_ms"]["n"] == 4
