"""
Outcome-labelled clip recorder: turns every trigger into a labelled example, without a human.

Why: after three rounds of analysis the false triggers (7 of 8 in the morning log, scores 0.72-0.74) are still
only NUMBERS. Nobody has looked at what the template matched. The game itself labels each cast:

    STAGE_2 -> STAGE_3 (bot reeled)  then  "Bảo Quản" clicked      -> caught          (true positive)
    STAGE_2 -> STAGE_3 (bot reeled)  then  rod re-appears          -> missed          (false trigger OR late reel)
    STAGE_2 -> STAGE_1 (rod pulled, bot never reeled)              -> rod_retracted   (a bite was missed)

For each one this writes the last `window_s` seconds of ROI frames + meta.json to
    <out_dir>/<outcome>/<timestamp>_s<score>_<template>/

Design rules
  * Observer, like telemetry: wraps bound methods on the engine instance, changes no behaviour, every call guarded.
  * NEVER blocks the scan loop: frames are only referenced in RAM during the cast; JPEG encoding and disk writes
    happen on a writer thread behind a bounded queue (a full queue drops the clip and counts it).
  * Bounded disk: `max_bytes` retention, oldest clip deleted first (the old diagnostics/ folder had no cap).
  * Only the engine thread feeds the buffer; the keyboard-hook thread (F2) can still trigger state changes.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2


def _label(state: Any) -> str:
    return getattr(state, "name", str(state))


class _Clip:
    def __init__(self, outcome: str, frames: List[tuple], meta: Dict[str, Any]):
        self.outcome, self.frames, self.meta = outcome, frames, meta


class OutcomeRecorder:
    def __init__(self, out_dir: str = "corpus", window_s: float = 3.0, max_bytes: int = 600_000_000,
                 jpeg_quality: int = 92, queue_size: int = 8,
                 clock: Callable[[], float] = time.perf_counter, wall: Callable[[], datetime] = datetime.now):
        self.out_dir, self.window_s, self.max_bytes, self.jpeg_quality = out_dir, window_s, max_bytes, jpeg_quality
        self._clock, self._wall = clock, wall
        self._buf: Deque[tuple] = deque()                    # (t, frame, confidence, template_name)
        self._pending: Optional[_Clip] = None                # reeled, waiting for the STAGE_3 verdict
        self._stats_at_trigger: Dict[str, int] = {}
        self.last_hit: Dict[str, Any] = {}
        self.dropped = 0
        self.written = 0
        self._q: "queue.Queue[Optional[_Clip]]" = queue.Queue(maxsize=queue_size)
        self._writer = threading.Thread(target=self._write_loop, name="outcome-writer", daemon=True)
        self._writer.start()

    # ------------------------------------------------------------ feeding (engine thread)
    def add_frame(self, frame, confidence: float = 0.0, template: str = "") -> None:
        now = self._clock()
        self._buf.append((now, frame, float(confidence), template))
        while self._buf and now - self._buf[0][0] > self.window_s:
            self._buf.popleft()

    def note_hit(self, **hit: Any) -> None:
        self.last_hit = hit

    def on_state_change(self, old: str, new: str, stats: Dict[str, int], roi: Any = None) -> None:
        if old == "STAGE_2" and new == "STAGE_3":                       # the bot reeled
            self._pending = _Clip("pending", list(self._buf), {"trigger": dict(self.last_hit), "roi": roi,
                                                              "at": self._wall().isoformat(timespec="seconds")})
            self._stats_at_trigger = dict(stats)
        elif old == "STAGE_2" and new == "STAGE_1" and stats.get("lost_before_bite", 0) > \
                self._stats_at_trigger.get("lost_before_bite", 0):
            self._submit(_Clip("rod_retracted", list(self._buf), {"roi": roi, "at": self._wall().isoformat(timespec="seconds")}))
            self._stats_at_trigger = dict(stats)
        elif old == "STAGE_3" and self._pending is not None:
            if stats.get("caught", 0) > self._stats_at_trigger.get("caught", 0):
                outcome = "caught"
            elif stats.get("missed_after_click", 0) > self._stats_at_trigger.get("missed_after_click", 0):
                outcome = "missed"
            else:
                outcome = "unknown"
            clip, self._pending = self._pending, None
            clip.outcome = outcome
            self._submit(clip)
        elif old != "STAGE_3" and new != "STAGE_3":
            self._pending = None
        if new == "STAGE_1":
            self._buf.clear()                                           # the next cast starts a fresh window

    def close(self, timeout: float = 5.0) -> None:
        try:
            self._q.put(None, timeout=timeout)
        except queue.Full:
            pass
        self._writer.join(timeout)

    # ------------------------------------------------------------ writer thread
    def _submit(self, clip: _Clip) -> None:
        if not clip.frames:
            return
        try:
            self._q.put_nowait(clip)
        except queue.Full:
            self.dropped += 1

    def _write_loop(self) -> None:
        while True:
            clip = self._q.get()
            if clip is None:
                return
            try:
                self._write(clip)
                self._enforce_retention()
            except Exception:
                pass

    def _write(self, clip: _Clip) -> None:
        trig = clip.meta.get("trigger") or {}
        score = trig.get("score")
        name = os.path.splitext(str(trig.get("template", "none")))[0]
        stamp = self._wall().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        folder = os.path.join(self.out_dir, clip.outcome,
                              f"{stamp}_s{score:.2f}_{name}" if isinstance(score, (int, float)) else f"{stamp}_{name}")
        os.makedirs(folder, exist_ok=True)
        t_last = clip.frames[-1][0]
        rows = []
        for i, (t, frame, conf, tpl) in enumerate(clip.frames):
            fname = f"roi_{i:03d}.jpg"
            cv2.imwrite(os.path.join(folder, fname), frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            rows.append({"file": fname, "t_before_trigger_s": round(t - t_last, 3), "confidence": round(conf, 3), "template": tpl})
        meta = dict(clip.meta, outcome=clip.outcome, frames=rows)
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1, default=str)
        self.written += 1

    def _enforce_retention(self) -> None:
        clips: List[Tuple[float, int, str]] = []
        for outcome in os.listdir(self.out_dir) if os.path.isdir(self.out_dir) else []:
            od = os.path.join(self.out_dir, outcome)
            for d in os.listdir(od) if os.path.isdir(od) else []:
                p = os.path.join(od, d)
                size = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p))
                clips.append((os.path.getmtime(p), size, p))
        total = sum(s for _, s, _ in clips)
        for _, size, p in sorted(clips)[:-1]:          # the newest clip is always kept, even if it alone exceeds the cap
            if total <= self.max_bytes:
                break
            shutil.rmtree(p, ignore_errors=True)
            total -= size


def _safe(fn: Callable, *a, **k) -> None:
    try:
        fn(*a, **k)
    except Exception:
        pass


def attach_outcome_recorder(engine: Any, out_dir: str = "corpus", **kw: Any) -> OutcomeRecorder:
    """Wraps engine.recorder.record, engine.matcher.match_any and engine._set_state. Idempotent."""
    existing = getattr(engine, "outcome_recorder", None)
    if existing is not None:
        return existing
    rec = OutcomeRecorder(out_dir, **kw)
    owner = {"ident": None}

    rr = getattr(getattr(engine, "recorder", None), "record", None)
    if rr is not None:
        def record(frame, confidence=0.0, template_name="", *a, **k):
            if threading.get_ident() == owner["ident"]:
                _safe(rec.add_frame, frame, confidence, template_name)
            return rr(frame, confidence, template_name, *a, **k)
        engine.recorder.record = record

    ma = getattr(getattr(engine, "matcher", None), "match_any", None)
    if ma is not None:
        def match_any(screen, template_target, *a, **k):
            res = ma(screen, template_target, *a, **k)
            if threading.get_ident() == owner["ident"] and isinstance(template_target, str) \
                    and os.path.basename(os.path.normpath(template_target)) == "dau_cham_than" \
                    and getattr(res, "found", False):
                _safe(rec.note_hit, x=res.x, y=res.y, w=res.w, h=res.h, score=round(res.confidence, 3),
                      template=res.template_name)
            return res
        engine.matcher.match_any = match_any

    ss = engine._set_state

    def set_state(new_state, *a, **k):
        _safe(rec.on_state_change, _label(engine.state), _label(new_state), dict(getattr(engine, "stats", {})),
              getattr(engine, "_last_good_head_roi", None))
        return ss(new_state, *a, **k)
    engine._set_state = set_state

    rl = engine._run_loop

    def run_loop(*a, **k):
        owner["ident"] = threading.get_ident()
        try:
            return rl(*a, **k)
        finally:
            owner["ident"] = None
    engine._run_loop = run_loop

    engine.outcome_recorder = rec
    return rec
