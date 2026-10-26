"""
Replay the BiteDetector over the clips the bot already recorded (corpus/<outcome>/<clip>/roi_*.jpg + meta.json).

    python tools/replay_bite_detector.py corpus
    python tools/replay_bite_detector.py corpus --min-contrast 16 --csv hits.csv

How to read it (the game labels each clip for us):
  caught         the template reeled AND the fish was kept -> a REAL bite. The detector should fire near t=0.
                 => `recall` = share of caught clips where it did.
  missed         the template reeled but no fish -> mostly FALSE triggers (or a late reel).
                 => `also fired` = share where the detector would have reeled too (false positive it does NOT remove).
  rod_retracted  the rod was pulled and the bot never saw a '!' -> a REAL bite the template missed.
                 => any detector hit here is a catch the detector would have ADDED.
Frames are only the last ~3 s before the trigger, the same window the live detector needs (history 1.6 s).
The corpus does not store the zoom, so `scale` is derived from the ROI width like the legacy fallback.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.bite_detector import BiteDetector  # noqa: E402

NEAR_WINDOW = (-1.0, 0.3)   # seconds relative to the trigger (t=0 is the last recorded frame)


@dataclass
class ClipResult:
    outcome: str
    clip: str
    n_frames: int
    hits: List[Tuple[float, object]] = field(default_factory=list)   # (t_before_trigger_s, BiteHit)

    @property
    def fired(self) -> bool:
        return bool(self.hits)

    @property
    def fired_near_trigger(self) -> bool:
        return any(NEAR_WINDOW[0] <= t <= NEAR_WINDOW[1] for t, _ in self.hits)

    @property
    def first_t(self) -> Optional[float]:
        return min((t for t, _ in self.hits), default=None)


def replay_clip(folder: str, detector: BiteDetector) -> Optional[ClipResult]:
    meta_path = os.path.join(folder, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    res = ClipResult(os.path.basename(os.path.dirname(folder)), os.path.basename(folder), 0)
    detector.reset()
    for row in meta.get("frames", []):
        img = cv2.imread(os.path.join(folder, row["file"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        t = float(row["t_before_trigger_s"])
        res.n_frames += 1
        hit = detector.update(img, t)          # t grows towards 0, which is all the detector needs
        if hit is not None:
            res.hits.append((t, hit))
    return res


def evaluate(corpus_dir: str, **detector_kwargs) -> Dict[str, List[ClipResult]]:
    out: Dict[str, List[ClipResult]] = {}
    for outcome in sorted(os.listdir(corpus_dir)):
        od = os.path.join(corpus_dir, outcome)
        if not os.path.isdir(od) or outcome == "stuck":
            continue
        for clip in sorted(os.listdir(od)):
            r = replay_clip(os.path.join(od, clip), BiteDetector(**detector_kwargs))
            if r is not None:
                out.setdefault(outcome, []).append(r)
    return out


def _pct(values: List[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def summarize(results: Dict[str, List[ClipResult]]) -> List[str]:
    lines: List[str] = []
    for outcome, clips in results.items():
        n = len(clips)
        near = sum(c.fired_near_trigger for c in clips)
        anyhit = sum(c.fired for c in clips)
        if outcome == "caught":
            lines.append(f"[caught]        {n} clips | RECALL near trigger {near}/{n} = {100 * near / max(1, n):.0f}% "
                         f"| fired at all {anyhit}/{n}")
        elif outcome == "missed":
            lines.append(f"[missed]        {n} clips | detector ALSO fired near trigger {near}/{n} "
                         f"(false triggers it would not remove) | fired at all {anyhit}/{n}")
        elif outcome == "rod_retracted":
            lines.append(f"[rod_retracted] {n} clips | detector would have ADDED a catch in {anyhit}/{n}")
        else:
            lines.append(f"[{outcome}] {n} clips | fired {anyhit}/{n}")
    caught_contrast = [h.contrast for c in results.get("caught", []) for _, h in c.hits]
    if caught_contrast:
        p5, p50 = _pct(caught_contrast, 5), _pct(caught_contrast, 50)
        lines.append(f"contrast of hits in caught clips: min {min(caught_contrast):.1f} | p5 {p5:.1f} | p50 {p50:.1f}  "
                     f"-> a safe min_contrast is roughly half of p5 (~{p5 / 2:.0f})")
    other_contrast = [h.contrast for o in ("missed", "rod_retracted") for c in results.get(o, []) for _, h in c.hits]
    if other_contrast:
        lines.append(f"contrast of hits in missed/rod_retracted clips: p50 {_pct(other_contrast, 50):.1f} "
                     f"| max {max(other_contrast):.1f}")
    return lines


def write_csv(results: Dict[str, List[ClipResult]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["outcome", "clip", "t_before_trigger_s", "x", "y", "w", "h", "aspect", "fill", "contrast",
                    "mean_b", "mean_g", "mean_r"])
        for clips in results.values():
            for c in clips:
                for t, h in c.hits:
                    w.writerow([c.outcome, c.clip, f"{t:.3f}", h.x, h.y, h.w, h.h, f"{h.aspect:.2f}",
                                f"{h.fill:.2f}", f"{h.contrast:.1f}", *h.mean_bgr])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="path to the corpus folder (contains caught/, missed/, rod_retracted/)")
    ap.add_argument("--min-contrast", type=float, default=20.0)
    ap.add_argument("--diff-thresh", type=int, default=24)
    ap.add_argument("--persist", type=int, default=2)
    ap.add_argument("--csv", help="write every detector hit (with its features) to this CSV")
    args = ap.parse_args(argv)
    results = evaluate(args.corpus, min_contrast=args.min_contrast, diff_thresh=args.diff_thresh,
                       persist_frames=args.persist)
    if not results:
        print("No clips found (expected corpus/<outcome>/<clip>/meta.json).")
        return 1
    print("\n".join(summarize(results)))
    if args.csv:
        write_csv(results, args.csv)
        print(f"hits written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
