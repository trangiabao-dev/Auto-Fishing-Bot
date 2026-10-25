"""
Progress watchdog: the bot must never loop silently.

Evidence: after a rod repair the bot sat in STAGE_1 for 70+ s, logging "Khởi động tuần tự" every 10 s and finding
nothing; the only way out was a manual restart. A loop that makes no progress should be LOUD and, after a
timeout, stop itself, not burn time (or keep clicking) unattended.

"Progress" = a cast click, a reel click, or a catch. The engine calls note_progress() at those points and polls
stalled() inside its loops. Pure logic with an injected clock, so it is unit-testable.
"""
from __future__ import annotations

import time
from typing import Callable


class ProgressWatchdog:
    def __init__(self, timeout_s: float = 90.0, clock: Callable[[], float] = time.perf_counter):
        self.timeout_s = timeout_s
        self._clock = clock
        self._last = clock()
        self._warned = False
        self.last_event = "start"

    def note_progress(self, what: str = "") -> None:
        self._last = self._clock()
        self._warned = False
        self.last_event = what or "progress"

    def idle_s(self) -> float:
        return self._clock() - self._last

    def stalled(self) -> bool:
        return self.idle_s() > self.timeout_s

    def warn_due(self, after_s: float = 30.0) -> bool:
        """True ONCE per stall, after `after_s` without progress: the moment to save evidence (a screenshot of what
        the bot is looking at) before the stall ends the run. Re-armed by note_progress()."""
        if not self._warned and self.idle_s() > after_s:
            self._warned = True
            return True
        return False

    def reset(self) -> None:
        self.note_progress("reset")
