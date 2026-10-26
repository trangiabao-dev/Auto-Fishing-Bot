import time
import random
import ctypes
from typing import Tuple

# Windows SendInput structures for low-level hardware mouse and keyboard events
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


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class INPUT_I(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", INPUT_I)]


# Constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

# Hardware DirectInput ScanCodes (for 3D DirectX / Unreal / Unity PC games)
SCANCODE_MAP = {
    "space": 0x39,
    "f": 0x21,
    "enter": 0x1C,
    "esc": 0x01,
    "tab": 0x0F,
    "w": 0x11,
    "a": 0x1E,
    "s": 0x1F,
    "d": 0x20,
}

VK_MAP = {
    "space": 0x20,
    "f": 0x46,
    "enter": 0x0D,
    "esc": 0x1B,
    "tab": 0x09,
}


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

    def press_key(self, key: str, hold_min_ms: int = 40, hold_max_ms: int = 80) -> bool:
        """
        Simulates a hardware-level key press using DirectInput scancodes.
        Guarantees 100% recognition by 3D games (such as Play Together PC dialogs).

        Args:
            key: Name of key (e.g. 'space', 'f', 'enter')
            hold_min_ms: Minimum key hold time in ms
            hold_max_ms: Maximum key hold time in ms
        """
        key_lower = key.strip().lower()
        scancode = SCANCODE_MAP.get(key_lower)
        vk = VK_MAP.get(key_lower, 0)
        if scancode is None:
            raise ValueError(f"Unsupported key: {key}. Supported: {list(SCANCODE_MAP.keys())}")

        extra = ctypes.c_ulong(0)

        # Send Key Down with SCANCODE flag for DirectInput compatibility
        ki_down = KEYBDINPUT(wVk=vk, wScan=scancode, dwFlags=KEYEVENTF_SCANCODE, time=0, dwExtraInfo=ctypes.pointer(extra))
        inp_down = INPUT(INPUT_KEYBOARD, INPUT_I(ki=ki_down))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(inp_down), ctypes.sizeof(inp_down))

        # Hold time
        hold_time = random.uniform(hold_min_ms / 1000.0, hold_max_ms / 1000.0)
        time.sleep(hold_time)

        # Send Key Up
        ki_up = KEYBDINPUT(wVk=vk, wScan=scancode, dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=ctypes.pointer(extra))
        inp_up = INPUT(INPUT_KEYBOARD, INPUT_I(ki=ki_up))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(inp_up), ctypes.sizeof(inp_up))
        return True
