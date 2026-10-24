import json

import pytest

from core.telemetry import JsonlSink, Telemetry, _Series, format_log_lines, percentile


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt


class ListSink:
    def __init__(self):
        self.items = []

    def write(self, obj):
        self.items.append(obj)


def make(**kw):
    clock = FakeClock()
    sink = ListSink()
    lines = []
    tel = Telemetry(sink=sink, log=lines.append, clock=clock, wall=lambda: 1000.0, **kw)
    return tel, clock, sink, lines


class TestPercentile:
    def test_empty_and_single(self):
        assert percentile([], 50) == 0.0
        assert percentile([7.0], 99) == 7.0

    def test_interpolates(self):
        assert percentile([0.0, 10.0], 50) == 5.0
        assert percentile([1, 2, 3, 4, 5], 50) == 3
        assert percentile([1, 2, 3, 4, 5], 100) == 5


class TestSeries:
    def test_memory_is_bounded_but_totals_are_exact(self):
        s = _Series(window=100)
        for i in range(10_000):
            s.add(float(i))
        st = s.stats()
        assert st["n"] == 10_000
        assert st["max"] == 9999.0 and st["min"] == 0.0
        assert len(s._recent) == 100
        assert st["total"] == sum(range(10_000))


class TestGapMetrics:
    def _run_blind_cast(self, tel, clock):
        """10 ticks 30 ms apart, one 0.69 s blind gap, then 10 more ticks."""
        tel.begin_episode("STAGE_2")
        tel.begin_cast()
        for _ in range(10):
            clock.advance(0.03)
            tel.tick()
        tel.add_section("capture:full", 0.02)
        tel.add_section("locate", 0.64)
        tel.add_section("match[x]", 0.05, outermost=False)  # nested: must not enter the breakdown
        clock.advance(0.02 + 0.64 + 0.03)
        tel.tick()
        for _ in range(10):
            clock.advance(0.03)
            tel.tick()

    def test_miss_probability_is_exact(self):
        tel, clock, sink, _ = make()
        self._run_blind_cast(tel, clock)
        tel.end_episode("test")
        s = sink.items[0]
        span = 0.30 + 0.69 + 0.30
        assert s["scan_span_s"] == pytest.approx(span, abs=1e-3)
        assert s["miss_prob_400ms"] == pytest.approx((0.69 - 0.4) / span, abs=1e-3)

    def test_worst_gap_names_the_culprit(self):
        tel, clock, sink, _ = make()
        self._run_blind_cast(tel, clock)
        tel.end_episode("test")
        w = sink.items[0]["worst_gap"]
        assert w["ms"] == pytest.approx(690, abs=1)
        assert w["top"][0][0] == "locate" and w["top"][0][1] == pytest.approx(640, abs=1)
        assert [n for n, _ in w["top"]] == ["locate", "capture:full"]  # nested match[x] not double-counted
        assert w["untracked_ms"] == pytest.approx(30, abs=1)  # the loop's own sleep
        assert w["at_s"] == pytest.approx(0.99, abs=0.01)

    def test_gap_buckets(self):
        tel, clock, sink, _ = make()
        self._run_blind_cast(tel, clock)
        tel.end_episode("test")
        s = sink.items[0]
        assert s["gaps_over_ms"] == {"100": 1, "250": 1, "500": 1}
        assert s["gap_ms"]["max"] == pytest.approx(690, abs=1)

    def test_steady_loop_has_zero_miss_probability(self):
        tel, clock, sink, _ = make()
        tel.begin_episode("STAGE_2")
        tel.begin_cast()
        for _ in range(100):
            clock.advance(0.033)
            tel.tick()
        tel.end_episode()
        assert sink.items[0]["miss_prob_400ms"] == 0.0


class TestEpisodes:
    def test_empty_episode_emits_nothing(self):
        tel, clock, sink, lines = make()
        tel.begin_episode("STOPPED")
        tel.begin_episode("STAGE_1")
        assert sink.items == [] and lines == []

    def test_gui_log_only_for_stage2_but_sink_gets_everything(self):
        tel, clock, sink, lines = make()
        tel.begin_episode("STAGE_1")
        tel.add_section("match_any[B1]", 0.02)
        tel.begin_episode("STAGE_2")
        tel.begin_cast()
        clock.advance(0.03)
        tel.tick()
        clock.advance(0.03)
        tel.tick()
        tel.begin_episode("STAGE_3")
        assert [i["label"] for i in sink.items] == ["STAGE_1", "STAGE_2"]
        assert len(lines) >= 1 and "STAGE_2" in lines[0]
        assert all("STAGE_1" not in l for l in lines)

    def test_sections_before_any_state_go_to_startup_episode(self):
        tel, clock, sink, _ = make()
        tel.add_section("auto_detect", 0.5)
        tel.begin_episode("STAGE_1")
        assert sink.items[0]["label"] == "startup"

    def test_scores_are_summarised(self):
        tel, clock, sink, _ = make()
        tel.begin_episode("STAGE_2")
        for v in (0.47, 0.50, 0.54):
            tel.add_section("match[nut_giat_can.png]", 0.01, score=v)
        tel.end_episode()
        sc = sink.items[0]["scores"]["match[nut_giat_can.png]"]
        assert sc["min"] == 0.47 and sc["max"] == 0.54 and sc["n"] == 3

    def test_sink_failure_does_not_propagate(self):
        class Boom:
            def write(self, obj):
                raise RuntimeError("disk full")
        tel = Telemetry(sink=Boom(), clock=FakeClock())
        tel.begin_episode("STAGE_1")
        tel.add_section("x", 0.1)
        tel.end_episode()  # must not raise


class TestFormatting:
    def test_log_line_contains_the_headline_numbers(self):
        tel, clock, sink, lines = make()
        TestGapMetrics()._run_blind_cast(tel, clock)
        tel.end_episode()
        text = "\n".join(lines)
        assert "xác suất lọt cú cắn 400ms" in text
        assert "locate 640ms" in text
        assert "tốn nhiều nhất" in text


class TestJsonlSink:
    def test_writes_valid_json_lines_and_rotates(self, tmp_path):
        path = tmp_path / "t" / "telemetry.jsonl"
        sink = JsonlSink(str(path), max_bytes=300, backups=2)
        for i in range(40):
            sink.write({"i": i, "text": "tiếng Việt"})
        sink.close()
        files = sorted(p.name for p in path.parent.iterdir())
        assert files == ["telemetry.jsonl", "telemetry.jsonl.1", "telemetry.jsonl.2"]  # bounded retention
        for name in files:
            for line in (path.parent / name).read_text(encoding="utf-8").splitlines():
                assert "i" in json.loads(line)
