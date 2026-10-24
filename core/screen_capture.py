import ctypes
import numpy as np
import mss
from typing import Optional, Tuple, Dict, Any

# Ensure process is Per-Monitor DPI Aware to prevent scaling issues on Windows (125%, 150%)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class ScreenCapturer:
    """
    High-performance screen capturer utilizing 'mss'.
    Captures either full screen or a specified Region of Interest (ROI).
    """

    def __init__(self):
        self._sct = mss.mss()
        # Default to primary monitor
        self.primary_monitor = self._sct.monitors[1] if len(self._sct.monitors) > 1 else self._sct.monitors[0]

    def capture(self, region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Captures the screen or specified region.

        Args:
            region: Optional tuple (x, y, width, height). If None, captures primary monitor.

        Returns:
            Tuple of:
                - frame: np.ndarray in BGR format (ready for OpenCV)
                - offset: (origin_x, origin_y) relative to the screen, used to convert local coords to global coords.
        """
        if region:
            x, y, w, h = region
            monitor_box = {"top": int(y), "left": int(x), "width": int(w), "height": int(h)}
            offset = (int(x), int(y))
        else:
            monitor_box = self.primary_monitor
            offset = (monitor_box["left"], monitor_box["top"])

        # Capture raw image (BGRA)
        sct_img = self._sct.grab(monitor_box)
        # Convert to numpy array and drop alpha channel (BGRA -> BGR)
        frame = np.array(sct_img, dtype=np.uint8)[:, :, :3]
        return frame, offset

    def close(self):
        """Release mss instance."""
        if hasattr(self, "_sct") and self._sct:
            self._sct.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
