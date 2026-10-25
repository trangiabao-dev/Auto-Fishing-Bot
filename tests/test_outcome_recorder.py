import json
import os
import time
from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

from core.outcome_recorder import OutcomeRecorder, attach_outcome_recorder


class State(Enum):
    IDLE = 0
    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3


def frame(i):
    return np.full((40, 60, 3), i % 255, np.uint8)


def build(scenario, **kw):
    eng = SimpleNamespace(state=State.IDLE, stats={"caught": 0, "missed_after_click": 0, "lost_before_bite": 0},
                          _last_good_head_roi=(1, 2, 3, 4))
    eng.recorder = SimpleNamespace(record=lambda f, c=0.0, n="": None)
    eng.matcher = SimpleNamespace(match_any=lambda screen, target, *a, **k: SimpleNamespace(
        found=True, x=1127, y=363, w=40, h=30, confidence=0.74, template_name="chamthan_vang.png"))

    def set_state(s):
        eng.state = s
    eng._set_state = set_state
    eng._run_loop = lambda: scenario(eng)
    return eng


def go(state, eng):
    eng._set_state(state)


def cast(eng, n=10):
    go(State.STAGE_2, eng)
    for i in range(n):
        eng.recorder.record(frame(i), 0.3 + i * 0.01, "x.png")
    eng.matcher.match_any("roi", "assets/B2_DangCau/dau_cham_than")        # the trigger hit


def clips(root, outcome):
    d = os.path.join(root, outcome)
    return [os.path.join(d, x) for x in sorted(os.listdir(d))] if os.path.isdir(d) else []


def finish(rec):
    rec.close()


def test_caught_clip_is_written_with_trigger_metadata(tmp_path):
    def scenario(eng):
        cast(eng)
        go(State.STAGE_3, eng)
        eng.stats["caught"] += 1
        go(State.STAGE_1, eng)
    eng = build(scenario)
    rec = attach_outcome_recorder(eng, str(tmp_path))
    eng._run_loop()
    finish(rec)
    (folder,) = clips(str(tmp_path), "caught")
    meta = json.load(open(os.path.join(folder, "meta.json"), encoding="utf-8"))
    assert meta["outcome"] == "caught"
    assert meta["trigger"] == {"x": 1127, "y": 363, "w": 40, "h": 30, "score": 0.74, "template": "chamthan_vang.png"}
    assert meta["roi"] == [1, 2, 3, 4]
    assert len(meta["frames"]) == 10 and os.path.isfile(os.path.join(folder, "roi_009.jpg"))
    assert "s0.74_chamthan_vang" in os.path.basename(folder)


def test_false_trigger_is_labelled_missed(tmp_path):
    def scenario(eng):
        cast(eng)
        go(State.STAGE_3, eng)
        eng.stats["missed_after_click"] += 1
        go(State.STAGE_1, eng)
    eng = build(scenario)
    rec = attach_outcome_recorder(eng, str(tmp_path))
    eng._run_loop()
    finish(rec)
    assert len(clips(str(tmp_path), "missed")) == 1 and not clips(str(tmp_path), "caught")


def test_rod_pulled_without_a_trigger_is_recorded_as_a_missed_bite(tmp_path):
    def scenario(eng):
        go(State.STAGE_2, eng)
        for i in range(6):
            eng.recorder.record(frame(i), 0.2, "")
        eng.stats["lost_before_bite"] += 1
        go(State.STAGE_1, eng)
    eng = build(scenario)
    rec = attach_outcome_recorder(eng, str(tmp_path))
    eng._run_loop()
    finish(rec)
    assert len(clips(str(tmp_path), "rod_retracted")) == 1


def test_timeout_without_outcome_writes_nothing(tmp_path):
    def scenario(eng):
        cast(eng)
        go(State.STAGE_1, eng)                     # e.g. 120 s timeout: no stats changed
    eng = build(scenario)
    rec = attach_outcome_recorder(eng, str(tmp_path))
    eng._run_loop()
    finish(rec)
    assert not os.path.isdir(str(tmp_path)) or os.listdir(str(tmp_path)) == []


def test_frames_from_other_threads_are_ignored(tmp_path):
    import threading

    def scenario(eng):
        go(State.STAGE_2, eng)
        t = threading.Thread(target=lambda: eng.recorder.record(frame(1), 0.9, "gui"))
        t.start(); t.join()
        eng.recorder.record(frame(2), 0.3, "engine")
        eng.matcher.match_any("roi", "assets/dau_cham_than")
        go(State.STAGE_3, eng)
        eng.stats["caught"] += 1
        go(State.STAGE_1, eng)
    eng = build(scenario)
    rec = attach_outcome_recorder(eng, str(tmp_path))
    eng._run_loop()
    finish(rec)
    (folder,) = clips(str(tmp_path), "caught")
    meta = json.load(open(os.path.join(folder, "meta.json"), encoding="utf-8"))
    assert [f["template"] for f in meta["frames"]] == ["engine"]


def test_retention_deletes_the_oldest_clips(tmp_path):
    def scenario(eng):
        for _ in range(6):
            cast(eng, n=8)
            go(State.STAGE_3, eng)
            eng.stats["missed_after_click"] += 1
            go(State.STAGE_1, eng)
            time.sleep(0.03)                      # distinct mtimes
    eng = build(scenario)
    rec = attach_outcome_recorder(eng, str(tmp_path), max_bytes=20000, queue_size=32)
    eng._run_loop()
    finish(rec)
    left = clips(str(tmp_path), "missed")
    size = sum(os.path.getsize(os.path.join(c, f)) for c in left for f in os.listdir(c))
    assert 0 < len(left) < 6 and size <= 20000
    assert os.path.basename(left[-1]) == sorted(os.path.basename(c) for c in left)[-1]   # newest survives


def test_scan_loop_is_never_blocked_by_a_slow_disk(tmp_path):
    rec = OutcomeRecorder(str(tmp_path), queue_size=1)
    rec._write = lambda clip: time.sleep(0.5)     # a disk that stalls
    t0 = time.perf_counter()
    for _ in range(6):
        rec.on_state_change("STAGE_2", "STAGE_3", {}, None)
        rec.add_frame(frame(0))
        rec._pending.frames = [(0.0, frame(0), 0.5, "x")]
        rec.on_state_change("STAGE_3", "STAGE_1", {"caught": 1}, None)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2 and rec.dropped >= 3
    rec._q.put(None)


def test_attach_is_idempotent(tmp_path):
    eng = build(lambda e: None)
    a = attach_outcome_recorder(eng, str(tmp_path))
    assert attach_outcome_recorder(eng, str(tmp_path)) is a
    a.close()
