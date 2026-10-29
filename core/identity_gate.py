"""
Identity gate: "is the plate we are tracking still OURS?"

Measured on the crowd screenshots (moonlight event, 7 frames, own name 'columbinaa'): the other neon-green name
that keeps showing up near the character ('BéTrăng') is 0.34-0.66 of OUR name's width in the same frame
(own/other text width: 163/77, 78/41, 82/34, 81/18 px), and it sits 300-380 px away. The hijack in the log
(plate jumped ~330 px, scale 1.11 -> 1.29 within 2 s, camera not moving) is exactly that kind of event.

Why the tracker alone cannot fix it: PlateTracker._choose only compares the candidates that are PRESENT. When our
plate is hidden by the crowd (or scores below min_score) the stranger is the only candidate inside the search
window, so it is `best` and `nearest` at the same time and is adopted. Nothing asks "is this the same object as
the last time?". The camera can legitimately be rotated or zoomed by hand during the wait, so "the position must
not change" is NOT the invariant. What is invariant is CONTINUITY:

  R1 continuous  small step (<= max_step_ratio * text width) AND the text width changed by <= width_tol, or the
                 step is tiny (zoom around a fixed plate changes the width a lot but not the position) -> trusted.
  R2 jump        anything else is a JUMP. It is not followed. The tracker is told to keep searching around the
                 last trusted position (`pin`), and bite scanning is suspended (trusted=False).
  R3 confirm     a jump is accepted only after `confirm_ticks` consecutive observations at the same place AND a
                 name score not far below ours AND a similar text width (a hand-rotated camera ends up here).
  R4 stale       tracker returned the last position without new evidence: trusted for `stale_trust_s`, then not.

Fail-safe by design: when unsure the bot does NOT reel (a missed bite costs one cast; a false reel costs the rod
position and a cycle). Known limit: a stranger with the same name width AND a similar score who stays put for
`confirm_ticks` observations IS accepted (tests document it). Distinguishing look-alike names needs more evidence
(score distribution of strangers: the engine logs `reason` and scores so it can be measured).

Pure logic, duck-typed on `hit.text.center`, `hit.text.w`, `hit.score`; no OpenCV, injectable clock.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class GateVerdict:
    trusted: bool            # may the engine scan for '!' / reel on this plate?
    reason: str              # for the log and for calibration: locked | continuous | stale | lost | jump_* | ambiguous
    hit: Optional[Any]       # the plate to build the head ROI from (the trusted anchor, never a rejected stranger)
    motion: bool = False     # position / zoom changed: frame history (StaticVeto, BiteDetector) is no longer valid
    rejected: bool = False   # a jump candidate was refused: tell the tracker to keep looking near the anchor


def _xy(hit: Any):
    c = hit.text.center
    return float(c[0]), float(c[1])


def _dist(a: Any, b: Any) -> float:
    (ax, ay), (bx, by) = _xy(a), _xy(b)
    return math.hypot(ax - bx, ay - by)


def _width_change(hit: Any, ref: Any) -> float:
    scale_h = float(getattr(hit, 'scale', 1.0) or 1.0)
    scale_r = float(getattr(ref, 'scale', 1.0) or 1.0)
    w_h = float(hit.text.w) / max(0.1, scale_h)
    w_r = float(ref.text.w) / max(0.1, scale_r)
    return abs(w_h / max(1.0, w_r) - 1.0)


class IdentityGate:
    def __init__(self, *, max_step_ratio: float = 1.15, tiny_step_ratio: float = 0.15, width_tol: float = 0.14,
                 max_jump_dist: float = 200.0, confirm_ticks: int = 3, score_slack: float = 0.10,
                 stale_trust_s: float = 4.0, ema: float = 0.3,
                 clock: Callable[[], float] = time.perf_counter):
        self._max_step, self._tiny, self._tol = max_step_ratio, tiny_step_ratio, width_tol
        self._max_jump_dist = max_jump_dist
        self._confirm, self._slack, self._stale, self._ema = confirm_ticks, score_slack, stale_trust_s, ema
        self._clock = clock
        self.reset()

    @property
    def anchor(self) -> Optional[Any]:
        return self._anchor

    def reset(self) -> None:
        """New cast: forget the anchor (the character may have moved between casts)."""
        self._anchor: Optional[Any] = None
        self._seen_t = -1e9
        self._ref_score = 0.0
        self._pending: Optional[Any] = None
        self._pending_n = 0

    def observe(self, hit: Optional[Any], state: str = "TRACKING", ambiguous: bool = False) -> GateVerdict:
        now = self._clock()
        if hit is None or state != "TRACKING":
            return self._no_evidence(now, hit)
        if self._anchor is None:
            if ambiguous:                       # two look-alike plates at the first lock: do not guess
                return GateVerdict(False, "ambiguous", hit)
            self._accept(hit, now)
            return GateVerdict(True, "locked", hit, motion=True)

        a = self._anchor
        unit = max(1.0, float(a.text.w))
        dist, dw = _dist(hit, a), _width_change(hit, a)
        tiny = dist <= self._tiny * unit
        if tiny or (dist <= self._max_step * unit and dw <= self._tol):
            self._accept(hit, now)
            return GateVerdict(True, "continuous", hit, motion=(not tiny) or dw > 0.05)

        # Distant candidates (e.g. neighbors in crowd > max_jump_dist px away) can never hijack our anchor
        if dist > self._max_jump_dist:
            self._pending = None
            self._pending_n = 0
            return GateVerdict(False, "jump_too_far", a, rejected=True)

        # ---- a jump: do not follow it unless it proves itself over several observations
        if self._pending is not None and _dist(hit, self._pending) <= self._tiny * unit:
            self._pending_n += 1
        else:
            self._pending_n = 1
        self._pending = hit
        score_ok = hit.score >= self._ref_score - self._slack
        if self._pending_n >= self._confirm and score_ok and dw <= self._tol:
            self._accept(hit, now)
            return GateVerdict(True, "jump_accepted", hit, motion=True)
        reason = "jump_width" if dw > self._tol else ("jump_score" if not score_ok else "jump_wait")
        return GateVerdict(False, reason, a, rejected=True)

    # ------------------------------------------------------------------ internals
    def _accept(self, hit: Any, now: float) -> None:
        self._ref_score = float(hit.score) if self._anchor is None else \
            (1 - self._ema) * self._ref_score + self._ema * float(hit.score)
        self._anchor, self._seen_t = hit, now
        self._pending, self._pending_n = None, 0

    def _no_evidence(self, now: float, hit: Optional[Any]) -> GateVerdict:
        if self._anchor is None:
            return GateVerdict(False, "no_plate", hit)
        if self._pending is not None:
            return GateVerdict(False, "jump_wait", self._anchor)
        if now - self._seen_t <= self._stale:
            return GateVerdict(True, "stale", self._anchor)
        return GateVerdict(False, "lost", self._anchor)


@dataclass(frozen=True)
class GatedResult:
    """Drop-in for TrackResult (hit, state, ambiguous) plus the gate's verdict."""
    hit: Optional[Any]
    state: str
    ambiguous: bool
    trusted: bool
    reason: str
    motion: bool


class GatedTracker:
    """PlateTracker + IdentityGate. The engine uses it exactly like the tracker (reset / update)."""

    def __init__(self, tracker: Any, gate: Optional[IdentityGate] = None):
        self._tracker = tracker
        self._gate = gate or IdentityGate()

    @property
    def gate(self) -> IdentityGate:
        return self._gate

    def reset(self) -> None:
        self._tracker.reset()
        self._gate.reset()

    def update(self, frame_bgr: Any, origin=(0, 0)) -> GatedResult:
        res = self._tracker.update(frame_bgr, origin)
        amb = bool(getattr(res, "ambiguous", False))
        v = self._gate.observe(res.hit, res.state, amb)
        if v.rejected and self._gate.anchor is not None:
            pin = getattr(self._tracker, "pin", None)
            if pin is not None:
                pin(self._gate.anchor)      # keep the search window on OUR plate, not on the stranger
        return GatedResult(hit=v.hit if v.hit is not None else res.hit, state=res.state, ambiguous=amb,
                           trusted=v.trusted, reason=v.reason, motion=v.motion)
