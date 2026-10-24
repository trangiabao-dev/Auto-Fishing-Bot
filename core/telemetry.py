"""
Telemetry for the bot's hot loops - turns "the bot misses bites" into numbers.

This module only COLLECTS and AGGREGATES. It knows nothing about BotEngine, OpenCV or mss, so it can
be unit-tested with a fake clock. The glue that feeds it from the real engine lives in
`core/instrumentation.py` (single responsibility; removing telemetry = deleting one call).

Vocabulary
----------
episode : the time between two state changes (STAGE_1, STAGE_2, ...). One summary per episode.
section : a named piece of work (e.g. "locate", "capture:full", "match_any[dau_cham_than]").
tick    : one pass of the STAGE_2 scan loop. The gap between ticks is how long the bot was BLIND.
gap     : time between two consecutive ticks.

The headline metric: miss probability
-------------------------------------
A bite shows the '!' for ~`event_s` seconds (0.4 s is the shortest we have seen). Between two scans
that are `g` seconds apart, a bite that starts at time s is missed completely when no scan falls inside
[s, s + event_s], i.e. for a window of length max(0, g - event_s). Summing over all gaps and dividing by
the observed span gives the exact chance that a bite at a random moment is never seen:

    P(miss) = sum(max(0, g_i - event_s)) / span

Dependencies: standard library only. Memory is bounded: every series keeps exact count/min/max/total
plus a sliding window of recent samples for percentiles, so telemetry cannot itself leak on a long run.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_EVENT_S = 0.4
_GAP_BUCKETS_MS = (100, 250, 500)
_SERIES_WINDOW = 5000
_TOP_N_IN_WORST_GAP = 4
_TOP_N_IN_LOG = 5


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already SORTED sequence (q in 0..100)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


class _Series:
    """Exact count/total/min/max + a bounded window of recent samples for percentiles."""
    __slots__ = ("count", "total", "min", "max", "_recent")

    def __init__(self, window: int = _SERIES_WINDOW):
        self.count = 0
        self.total = 0.0
        self.min = math.inf
        self.max = -math.inf
        self._recent: Deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value
        self._recent.append(value)

    def stats(self, scale: float = 1.0, ndigits: int = 2) -> Dict[str, float]:
        if not self.count:
            return {"n": 0}
        s = sorted(self._recent)
        r = lambda v: round(v * scale, ndigits)  # noqa: E731
        return {
            "n": self.count,
            "total": r(self.total),
            "min": r(self.min),
            "p50": r(percentile(s, 50)),
            "p95": r(percentile(s, 95)),
            "p99": r(percentile(s, 99)),
            "max": r(self.max),
        }


class _Episode:
    def __init__(self, label: str, t0: float, wall0: float):
        self.label = label
        self.t0 = t0
        self.wall0 = wall0
        self.sections: Dict[str, _Series] = {}
        self.scores: Dict[str, _Series] = {}
        self.gaps = _Series()
        self.ticks = 0
        self.gap_over = {b: 0 for b in _GAP_BUCKETS_MS}
        self.miss_sum = 0.0
        self.cast_t0: Optional[float] = None
        self.last_tick: Optional[float] = None
        self.interval: Dict[str, float] = {}  # tracked work since the previous tick
        self.worst_gap = 0.0
        self.worst_at = 0.0
        self.worst_top: List[Tuple[str, float]] = []
        self.worst_untracked = 0.0


class JsonlSink:
    """
    Append-only JSON-lines file with size-based rotation (retention built in: the corpus work in the
    audit found `diagnostics/` growing without a cap - telemetry must not repeat that).
    """

    def __init__(self, path: str, max_bytes: int = 5_000_000, backups: int = 2):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self._handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
        self._handler.setFormatter(logging.Formatter("%(message)s"))

    def write(self, obj: dict) -> None:
        record = logging.makeLogRecord({"msg": json.dumps(obj, ensure_ascii=False)})
        self._handler.handle(record)

    def close(self) -> None:
        self._handler.close()


class Telemetry:
    def __init__(
        self,
        *,
        sink=None,
        log: Optional[Callable[[str], None]] = None,
        log_labels: Iterable[str] = ("STAGE_2",),
        event_s: float = DEFAULT_EVENT_S,
        clock: Callable[[], float] = time.perf_counter,
        wall: Callable[[], float] = time.time,
    ):
        self._sink = sink
        self._log = log
        self._log_labels = frozenset(log_labels)
        self._event_s = event_s
        self._clock = clock
        self._wall = wall
        self._lock = threading.RLock()  # state changes arrive from the keyboard-hook thread too
        self._ep: Optional[_Episode] = None

    # ------------------------------------------------------------------ episodes
    def begin_episode(self, label: str) -> None:
        """Ends the running episode (emitting its summary) and starts a new one."""
        with self._lock:
            self._end_locked("transition")
            self._ep = _Episode(label, self._clock(), self._wall())

    def end_episode(self, reason: str = "") -> None:
        with self._lock:
            self._end_locked(reason)

    def begin_cast(self) -> None:
        """Marks the start of the STAGE_2 wait: gap tracking starts from now."""
        with self._lock:
            ep = self._episode()
            now = self._clock()
            ep.cast_t0 = ep.last_tick = now
            ep.interval.clear()

    # ------------------------------------------------------------------ feeding
    def add_section(self, name: str, duration: float, *, outermost: bool = True,
                    score: Optional[float] = None) -> None:
        """
        `outermost=False` is for calls nested inside another timed call (match inside match_any):
        they get their own statistics but are not added to the gap breakdown, so time is never counted twice.
        """
        with self._lock:
            ep = self._episode()
            ep.sections.setdefault(name, _Series()).add(duration)
            if outermost:
                ep.interval[name] = ep.interval.get(name, 0.0) + duration
            if score is not None:
                ep.scores.setdefault(name, _Series()).add(score)

    def tick(self) -> None:
        """One pass of the scan loop. Call it at a fixed point of the loop (we use recorder.record)."""
        with self._lock:
            ep = self._episode()
            now = self._clock()
            ep.ticks += 1
            if ep.cast_t0 is None:
                ep.cast_t0 = now
            elif ep.last_tick is not None:
                gap = now - ep.last_tick
                ep.gaps.add(gap)
                for bucket in _GAP_BUCKETS_MS:
                    if gap * 1000.0 > bucket:
                        ep.gap_over[bucket] += 1
                ep.miss_sum += max(0.0, gap - self._event_s)
                if gap > ep.worst_gap:
                    ranked = sorted(ep.interval.items(), key=lambda kv: kv[1], reverse=True)
                    ep.worst_gap = gap
                    ep.worst_at = now - ep.cast_t0
                    ep.worst_top = ranked[:_TOP_N_IN_WORST_GAP]
                    ep.worst_untracked = max(0.0, gap - sum(ep.interval.values()))
            ep.last_tick = now
            ep.interval.clear()

    # ------------------------------------------------------------------ internals
    def _episode(self) -> _Episode:
        if self._ep is None:
            self._ep = _Episode("startup", self._clock(), self._wall())
        return self._ep

    def _end_locked(self, reason: str) -> None:
        ep, self._ep = self._ep, None
        if ep is None or (not ep.sections and not ep.ticks):
            return  # nothing was measured (e.g. IDLE/STOPPED): do not emit noise
        summary = self._summarize(ep, self._clock(), reason)
        if self._sink is not None:
            try:
                self._sink.write(summary)
            except Exception:
                pass  # observing must never break the bot
        if self._log is not None and ep.label in self._log_labels and ep.ticks:
            try:
                for line in format_log_lines(summary):
                    self._log(line)
            except Exception:
                pass

    def _summarize(self, ep: _Episode, t_end: float, reason: str) -> dict:
        span = (ep.last_tick - ep.cast_t0) if (ep.last_tick is not None and ep.cast_t0 is not None) else 0.0
        miss = min(1.0, ep.miss_sum / span) if span > 0 else 0.0
        return {
            "type": "episode",
            "label": ep.label,
            "reason": reason,
            "t_wall": round(ep.wall0, 3),
            "duration_s": round(t_end - ep.t0, 3),
            "scans": ep.ticks,
            "scan_span_s": round(span, 3),
            "event_s": self._event_s,
            f"miss_prob_{int(round(self._event_s * 1000))}ms": round(miss, 4),
            "gap_ms": ep.gaps.stats(1000.0),
            "gaps_over_ms": {str(b): n for b, n in ep.gap_over.items()},
            "worst_gap": {
                "ms": round(ep.worst_gap * 1000.0, 1),
                "at_s": round(ep.worst_at, 2),
                "top": [[name, round(d * 1000.0, 1)] for name, d in ep.worst_top],
                "untracked_ms": round(ep.worst_untracked * 1000.0, 1),
            },
            "sections_ms": {name: s.stats(1000.0) for name, s in sorted(ep.sections.items())},
            "scores": {name: s.stats(1.0, 3) for name, s in sorted(ep.scores.items())},
        }


def format_log_lines(summary: dict) -> List[str]:
    """Human-readable lines for the GUI log (Vietnamese, matching the rest of the app)."""
    gap = summary.get("gap_ms", {})
    miss_key = next((k for k in summary if k.startswith("miss_prob_")), None)
    miss = summary.get(miss_key, 0.0) if miss_key else 0.0
    event_ms = int(round(summary.get("event_s", DEFAULT_EVENT_S) * 1000))
    worst = summary.get("worst_gap", {})
    parts = [f"{name} {ms:.0f}ms" for name, ms in worst.get("top", [])]
    if worst.get("untracked_ms", 0) >= 1:
        parts.append(f"(không đo được/sleep) {worst['untracked_ms']:.0f}ms")
    lines = [
        f"⏱ [ĐO ĐẠC] {summary['label']} {summary['duration_s']:.1f}s | quét {summary['scans']} lần | "
        f"gap p50/p95/max = {gap.get('p50', 0):.0f}/{gap.get('p95', 0):.0f}/{gap.get('max', 0):.0f}ms | "
        f"xác suất lọt cú cắn {event_ms}ms ≈ {miss * 100:.0f}% | "
        f"gap xấu nhất {worst.get('ms', 0):.0f}ms tại +{worst.get('at_s', 0):.1f}s: "
        + (", ".join(parts) if parts else "—")
    ]
    sections = summary.get("sections_ms", {})
    top = sorted(sections.items(), key=lambda kv: kv[1].get("total", 0), reverse=True)[:_TOP_N_IN_LOG]
    if top:
        lines.append("   └ tốn nhiều nhất: " + " | ".join(
            f"{name} n={s['n']} p95={s['p95']:.0f}ms max={s['max']:.0f}ms" for name, s in top))
    return lines
