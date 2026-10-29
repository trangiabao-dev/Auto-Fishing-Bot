"""
Who is allowed to pull the trigger? A tiny pure policy so the rollout of BiteDetector is a config switch, not a rewrite.

  off / shadow : only the template matcher reels (today's behaviour). `shadow` additionally lets the engine LOG and COUNT
                 what the detector would have done, so recall / false positives are measured on real casts first.
  confirm      : reel only when BOTH agree within `window_s` (false-positive reel ~ 0, may miss bites the detector
                 does not see).
  primary      : the detector alone reels (use once recall >= 95 % and false positives <= 1/hour were measured).
  either       : reels if EITHER the detector OR the template matcher detects '!' (maximum resilience across dynamic weathers).

Pure logic, no OpenCV, no clock of its own (the caller passes `now`), so it is trivially unit-testable.
"""
from __future__ import annotations

MODES = ("off", "shadow", "confirm", "primary", "either")


class TriggerPolicy:
    def __init__(self, mode: str = "shadow", window_s: float = 0.6):
        if mode not in MODES:
            raise ValueError(f"bite_detector.mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self._window = window_s
        self.reset()

    def reset(self) -> None:
        """Call at the start of every STAGE_2 wait."""
        self._det_t = self._tpl_t = -1e9
        self._det_now = self._tpl_now = False

    def note_detector(self, now: float, hit: bool) -> None:
        self._det_now = hit
        if hit:
            self._det_t = now

    def note_template(self, now: float, hit: bool) -> None:
        self._tpl_now = hit
        if hit:
            self._tpl_t = now

    def detector_recent(self, now: float) -> bool:
        return now - self._det_t <= self._window

    def template_recent(self, now: float) -> bool:
        return now - self._tpl_t <= self._window

    def should_reel(self, now: float) -> bool:
        if self.mode == "primary":
            return self.detector_recent(now)
        if self.mode == "either":
            return self.detector_recent(now) or self._tpl_now
        if self.mode == "confirm":
            return self.detector_recent(now) and self.template_recent(now)
        return self._tpl_now          # off / shadow: exactly the pre-detector behaviour
