"""
Attaches `Telemetry` to a live BotEngine by wrapping a handful of bound methods on the INSTANCE.

Why wrap instead of editing bot_engine.py (Open-Closed): the engine's logic stays byte-for-byte the
same, so the numbers describe the real program; switching telemetry off removes the wrappers with no
trace. It is the poor man's aspect-oriented programming: cross-cutting concern (timing) kept out of the
business code.

Rules this module follows
-------------------------
1. Observing must never change behaviour. Every telemetry call is wrapped in `_safe`, return values and
   exceptions of the wrapped call pass through untouched.
2. Only the engine thread is measured. The GUI "Soi Mắt Bot" button calls the same capturer/matcher from
   the Tk thread; those calls are ignored (otherwise they would pollute the scan-gap numbers).
3. Nested timed calls (match_any -> match) are timed separately but only the OUTERMOST one feeds the
   gap breakdown, so time is not counted twice.

What is measured
----------------
capture:full / capture:roi, match[...] / match_any[...] (suffix "+ms" = multi-scale), locate,
record, record_full, dump, click, auto_detect.  The TICK marker is the ROI grab (`capture(region)` with a
region): the STAGE_2 loop does it exactly once per scan iteration, in every version of the engine, so the
time between two such grabs is the scan gap. (Not `recorder.record`: older/restored baselines may not have
a recorder at all.) Episodes (one summary each) are delimited by `engine._set_state`.

Hooks that do not exist on the engine are skipped, not fatal: `engine.telemetry_missing` lists them.
"""
from __future__ import annotations

import functools
import os
import threading
import time
from typing import Any, Callable, Optional

from core.telemetry import DEFAULT_EVENT_S, JsonlSink, Telemetry


def _safe(fn: Callable, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception:
        pass


def _state_label(state: Any) -> str:
    return getattr(state, "name", str(state))


def _target_label(target: Any) -> str:
    if isinstance(target, str):
        return os.path.basename(os.path.normpath(target)) or target
    if isinstance(target, (list, tuple)):
        return f"list[{len(target)}]"
    return type(target).__name__


def _capture_namer(args: tuple, kwargs: dict) -> str:
    region = args[0] if args else kwargs.get("region")
    return "capture:full" if region is None else "capture:roi"


def _match_namer(func_name: str, target_kwarg: str) -> Callable[[tuple, dict], str]:
    def namer(args: tuple, kwargs: dict) -> str:
        target = args[1] if len(args) > 1 else kwargs.get(target_kwarg)
        multi_scale = args[5] if len(args) > 5 else kwargs.get("multi_scale", True)
        return f"{func_name}[{_target_label(target)}]" + ("+ms" if multi_scale else "")
    return namer


def _fixed(name: str) -> Callable[[tuple, dict], str]:
    return lambda args, kwargs: name


class _Probe:
    def __init__(self, telemetry: Telemetry, clock: Callable[[], float]):
        self.telemetry = telemetry
        self.clock = clock
        self.owner: Optional[int] = None  # ident of the engine thread
        self.depth = 0                    # only ever touched by the owner thread

    def wrap(self, obj: Any, attr: str, namer: Callable[[tuple, dict], str], *,
             tick: Optional[Callable[[tuple, dict], bool]] = None, score: bool = False) -> bool:
        """Returns False (and wraps nothing) when `obj.attr` does not exist."""
        original = getattr(obj, attr, None) if obj is not None else None
        if original is None:
            return False
        probe = self

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            if threading.get_ident() != probe.owner:
                return original(*args, **kwargs)
            try:
                name = namer(args, kwargs)
            except Exception:
                name = attr
            outermost = probe.depth == 0
            try:
                is_tick = bool(tick and tick(args, kwargs))
            except Exception:
                is_tick = False
            if is_tick:
                _safe(probe.telemetry.tick)
            probe.depth += 1
            start = probe.clock()
            result = None
            ok = False
            try:
                result = original(*args, **kwargs)
                ok = True
                return result
            finally:
                probe.depth -= 1
                duration = probe.clock() - start
                confidence = getattr(result, "confidence", None) if (score and ok) else None
                _safe(probe.telemetry.add_section, name, duration, outermost=outermost, score=confidence)

        setattr(obj, attr, wrapper)
        return True


def _has_region(args: tuple, kwargs: dict) -> bool:
    return (args[0] if args else kwargs.get("region")) is not None


def attach_telemetry(
    engine: Any,
    *,
    jsonl_path: Optional[str] = "telemetry/telemetry.jsonl",
    event_s: float = DEFAULT_EVENT_S,
    log: Optional[Callable[[str], None]] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Telemetry:
    """Idempotent: attaching twice returns the existing Telemetry."""
    existing = getattr(engine, "telemetry", None)
    if existing is not None:
        return existing

    sink = JsonlSink(jsonl_path) if jsonl_path else None
    telemetry = Telemetry(sink=sink, log=log or getattr(engine, "_log", None), event_s=event_s, clock=clock)
    probe = _Probe(telemetry, clock)

    missing = []

    def hook(obj_name: str, attr: str, namer, **kw) -> None:
        obj = engine if obj_name == "engine" else getattr(engine, obj_name, None)
        if not probe.wrap(obj, attr, namer, **kw):
            missing.append(f"{obj_name}.{attr}")

    hook("capturer", "capture", _capture_namer, tick=_has_region)
    hook("matcher", "match", _match_namer("match", "template_path"), score=True)
    hook("matcher", "match_any", _match_namer("match_any", "template_target"), score=True)
    hook("name_locator", "locate", _fixed("locate"))
    hook("plate_tracker", "update", _fixed("locate"))
    hook("recorder", "record", _fixed("record"))
    hook("recorder", "record_full", _fixed("record_full"))
    hook("recorder", "dump", _fixed("dump"))
    hook("input_sim", "click", _fixed("click"))
    hook("engine", "_auto_detect_state", _fixed("auto_detect"))

    recorder = getattr(engine, "recorder", None)
    original_begin_cast = getattr(recorder, "begin_cast", None)
    if original_begin_cast is not None:
        @functools.wraps(original_begin_cast)
        def begin_cast(*args, **kwargs):
            if threading.get_ident() == probe.owner:
                _safe(telemetry.begin_cast)
            return original_begin_cast(*args, **kwargs)

        recorder.begin_cast = begin_cast
    else:
        missing.append("recorder.begin_cast")  # gap tracking then starts at the first scan of each cast

    original_set_state = engine._set_state

    @functools.wraps(original_set_state)
    def set_state(new_state, *args, **kwargs):
        # May run on the keyboard-hook thread (F2 -> stop). Telemetry is lock-protected for that reason.
        _safe(telemetry.begin_episode, _state_label(new_state))
        return original_set_state(new_state, *args, **kwargs)

    engine._set_state = set_state

    original_run_loop = engine._run_loop

    @functools.wraps(original_run_loop)
    def run_loop(*args, **kwargs):
        probe.owner = threading.get_ident()
        try:
            return original_run_loop(*args, **kwargs)
        finally:
            probe.owner = None
            _safe(telemetry.end_episode, "loop_end")

    engine._run_loop = run_loop  # BotEngine.start() resolves self._run_loop at thread start

    engine.telemetry = telemetry
    engine.telemetry_missing = missing
    if missing and telemetry._log is not None:
        _safe(telemetry._log, "⏱ [ĐO ĐẠC] đã gắn; không tìm thấy hook: " + ", ".join(missing))
    return telemetry
