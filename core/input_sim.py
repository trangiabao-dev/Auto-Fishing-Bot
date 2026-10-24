import time
import random
import ctypes
from typing import Tuple

# Windows SendInput structures for low-level hardware mouse events
PUL = ctypes.POINTER(ctypes.c_ulong)


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class INPUT_I(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", INPUT_I)]


# Constants
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000


class InputSimulator:
    """
    Simulates mouse clicks and movements using DirectInput / Win32 SendInput
    to ensure full compatibility with 3D games and anti-macro checks.
    """

    def __init__(self):
        # Cache screen metrics for absolute coordinate conversions
        self.screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        self.screen_height = ctypes.windll.user32.GetSystemMetrics(1)

    def move_to(self, x: int, y: int):
        """Moves cursor to exact screen coordinate (x, y)."""
        ctypes.windll.user32.SetCursorPos(int(x), int(y))

    def click(self, x: int, y: int, hold_min_ms: int = 40, hold_max_ms: int = 90):
        """
        Moves to (x, y), presses left button, holds for random duration, and releases.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            hold_min_ms: Minimum mouse hold duration in milliseconds
            hold_max_ms: Maximum mouse hold duration in milliseconds
        """
        self.move_to(x, y)
        time.sleep(random.uniform(0.010, 0.025))  # Organic micro-pause before press

        # Send Mouse Down
        extra = ctypes.c_ulong(0)
        ii_down = INPUT_I()
        ii_down.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
        input_down = INPUT(INPUT_MOUSE, ii_down)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(input_down), ctypes.sizeof(input_down))

        # Human-like press hold time
        hold_time = random.uniform(hold_min_ms / 1000.0, hold_max_ms / 1000.0)
        time.sleep(hold_time)

        # Send Mouse Up
        ii_up = INPUT_I()
        ii_up.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
        input_up = INPUT(INPUT_MOUSE, ii_up)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(input_up), ctypes.sizeof(input_up))
