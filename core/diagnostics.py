"""
Diagnostics for STAGE 2 (bite detection).

Purpose: turn "the bot misses the bite" from a guess into data. It keeps a short
rolling history of what the bot SAW (head-ROI frames + best template confidence per
frame + low-rate full-screen frames) and writes it to disk on demand:
  - F3 pressed by the human the moment they see the "!"  -> ground truth
  - STAGE 2 timeout                                      -> "what did the bot see for nothing?"

Design notes (why it is built this way):
  * Single responsibility: this class knows nothing about OpenCV matching or the FSM.
    The engine feeds it; it only buffers and persists.
  * Time-based ring buffer: we keep the last N seconds, not the last N frames, so the
    window stays meaningful even if the scan FPS changes.
  * Thread safety: the engine thread writes, the keyboard-hook thread requests a dump.
    The hook thread only sets a timestamp (cheap, never blocks on disk). The engine
    thread performs the actual disk I/O at its next loop iteration.
"""
import csv
import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

Region = Tuple[int, int, int, int]


class StageTwoRecorder:
    def __init__(
        self,
        out_dir: str = "diagnostics",
        roi_window_sec: float = 4.0,
        full_window_sec: float = 6.0,
        jpeg_quality: int = 85,
    ):
        self._out_dir = out_dir
        self._roi_window = roi_window_sec
        self._full_window = full_window_sec
        self._jpeg_quality = jpeg_quality

        self._roi: Deque[Tuple[float, np.ndarray, float, str]] = deque()
        self._full: Deque[Tuple[float, bytes]] = deque()
        self._meta: dict = {}
        self._manual_mark: Optional[float] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ feeding
    def begin_cast(self, roi: Optional[Region], meta: Optional[dict] = None) -> None:
        """Call once when STAGE 2 starts a new wait. Clears the previous cast's history."""
        with self._lock:
            self._roi.clear()
            self._full.clear()
            self._meta = {"roi": list(roi) if roi else None, **(meta or {})}

    def record(self, roi_frame: np.ndarray, confidence: float, template_name: str = "") -> None:
        """Call every scan iteration with the ROI frame and the best match confidence."""
        now = time.time()
        with self._lock:
            self._roi.append((now, roi_frame, float(confidence), template_name))
            while self._roi and now - self._roi[0][0] > self._roi_window:
                self._roi.popleft()

    def record_full(self, full_frame: np.ndarray) -> None:
        """Call at a LOW rate (e.g. every 8th iteration). Stored JPEG-encoded to save RAM."""
        ok, buf = cv2.imencode(".jpg", full_frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            return
        now = time.time()
        with self._lock:
            self._full.append((now, buf.tobytes()))
            while self._full and now - self._full[0][0] > self._full_window:
                self._full.popleft()

    # --------------------------------------------------------- manual (F3) trigger
    def request_dump(self) -> None:
        """Hotkey callback. Runs on the keyboard thread: only stores a timestamp."""
        with self._lock:
            self._manual_mark = time.time()

    def consume_manual_request(self) -> Optional[float]:
        """Engine thread polls this. Returns the press timestamp once, then clears it."""
        with self._lock:
            mark, self._manual_mark = self._manual_mark, None
            return mark

    # ---------------------------------------------------------------- persistence
    def dump(self, reason: str, mark_time: Optional[float] = None) -> Optional[str]:
        """
        Writes the buffered history to diagnostics/<timestamp>_<reason>/ and returns the folder.
        frames.csv columns: file, t_before_dump_sec, confidence, template, t_vs_f3_sec
        (t_vs_f3_sec < 0 means the frame was captured BEFORE you pressed F3.)
        """
        with self._lock:
            roi_snapshot = list(self._roi)
            full_snapshot = list(self._full)
            meta = dict(self._meta)

        if not roi_snapshot:
            return None

        t_dump = time.time()
        folder = os.path.join(self._out_dir, f"{datetime.now():%Y%m%d_%H%M%S}_{reason}")
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, "frames.csv"), "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file", "t_before_dump_sec", "confidence", "template", "t_vs_f3_sec"])
            for i, (t, frame, conf, name) in enumerate(roi_snapshot):
                fname = f"roi_{i:03d}.png"
                cv2.imwrite(os.path.join(folder, fname), frame)
                vs_mark = f"{t - mark_time:+.2f}" if mark_time else ""
                writer.writerow([fname, f"{t - t_dump:.2f}", f"{conf:.3f}", name, vs_mark])

        for j, (t, jpg) in enumerate(full_snapshot):
            vs_mark = f"_f3{t - mark_time:+.2f}" if mark_time else ""
            with open(os.path.join(folder, f"full_{j:02d}_t{t - t_dump:.2f}{vs_mark}.jpg"), "wb") as fh:
                fh.write(jpg)

        meta.update({"reason": reason, "roi_frames": len(roi_snapshot), "full_frames": len(full_snapshot)})
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        return folder
