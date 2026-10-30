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
MOUSEEVENTF_WHEEL = 0x0800
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

    def move_to(self, x: int, y: int, human_curve: bool = True):
        """Moves cursor to screen coordinate (x, y) with human-like Bezier curve trajectory."""
        if not human_curve:
            ctypes.windll.user32.SetCursorPos(int(x), int(y))
            return

        from ctypes import wintypes
        pt = wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        x0, y0 = pt.x, pt.y

        dist = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
        if dist < 25:
            ctypes.windll.user32.SetCursorPos(int(x), int(y))
            return

        # Cubic Bezier interpolation with organic micro-control points
        ctrl1_x = x0 + (x - x0) * random.uniform(0.25, 0.45) + random.randint(-12, 12)
        ctrl1_y = y0 + (y - y0) * random.uniform(0.25, 0.45) + random.randint(-12, 12)
        ctrl2_x = x0 + (x - x0) * random.uniform(0.55, 0.75) + random.randint(-12, 12)
        ctrl2_y = y0 + (y - y0) * random.uniform(0.55, 0.75) + random.randint(-12, 12)

        steps = random.randint(5, 7)
        for i in range(1, steps + 1):
            t = i / float(steps)
            bx = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * ctrl1_x + 3 * (1 - t) * t ** 2 * ctrl2_x + t ** 3 * x
            by = (1 - t) ** 3 * y0 + 3 * (1 - t) ** 2 * t * ctrl1_y + 3 * (1 - t) * t ** 2 * ctrl2_y + t ** 3 * y
            ctypes.windll.user32.SetCursorPos(int(bx), int(by))
            time.sleep(random.uniform(0.005, 0.010))

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
        time.sleep(random.uniform(0.015, 0.030))  # Organic micro-pause before press

        # Send Mouse Down with SendInput and mouse_event fallback
        extra = ctypes.c_ulong(0)
        ii_down = INPUT_I()
        ii_down.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
        input_down = INPUT(INPUT_MOUSE, ii_down)
        ret_down = ctypes.windll.user32.SendInput(1, ctypes.pointer(input_down), ctypes.sizeof(input_down))
        if ret_down == 0:
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

        # Human-like press hold time
        hold_time = random.uniform(hold_min_ms / 1000.0, hold_max_ms / 1000.0)
        time.sleep(hold_time)

        # Send Mouse Up with SendInput and mouse_event fallback
        ii_up = INPUT_I()
        ii_up.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
        input_up = INPUT(INPUT_MOUSE, ii_up)
        ret_up = ctypes.windll.user32.SendInput(1, ctypes.pointer(input_up), ctypes.sizeof(input_up))
        if ret_up == 0:
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def press_key(self, key: str, hold_min_ms: int = 40, hold_max_ms: int = 80, ensure_focus: bool = True) -> bool:
        """
        Simulates a hardware-level key press using DirectInput scancodes and universal Win32 keybd_event.
        Guarantees 100% recognition by 3D games, modal popups, and UI dialogues.

        Args:
            key: Name of key (e.g. 'space', 'f', 'enter')
            hold_min_ms: Minimum key hold time in ms
            hold_max_ms: Maximum key hold time in ms
            ensure_focus: Automatically focuses game window before sending key
        """
        if ensure_focus:
            self.focus_game_window()

        key_lower = key.strip().lower()
        scancode = SCANCODE_MAP.get(key_lower)
        vk = VK_MAP.get(key_lower, 0)
        if scancode is None:
            raise ValueError(f"Unsupported key: {key}. Supported: {list(SCANCODE_MAP.keys())}")

        extra = ctypes.c_ulong(0)

        # 1. SendInput with SCANCODE flag (DirectInput for 3D engine)
        ki_down = KEYBDINPUT(wVk=vk, wScan=scancode, dwFlags=KEYEVENTF_SCANCODE, time=0, dwExtraInfo=ctypes.pointer(extra))
        inp_down = INPUT(INPUT_KEYBOARD, INPUT_I(ki=ki_down))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(inp_down), ctypes.sizeof(inp_down))

        # 2. Universal Win32 keybd_event (guarantees WM_KEYDOWN for UI dialogs and modal popups)
        ctypes.windll.user32.keybd_event(vk, scancode, 0, 0)

        # Hold time
        hold_time = random.uniform(hold_min_ms / 1000.0, hold_max_ms / 1000.0)
        time.sleep(hold_time)

        # Send Key Up
        ki_up = KEYBDINPUT(wVk=vk, wScan=scancode, dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=ctypes.pointer(extra))
        inp_up = INPUT(INPUT_KEYBOARD, INPUT_I(ki=ki_up))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(inp_up), ctypes.sizeof(inp_up))

        ctypes.windll.user32.keybd_event(vk, scancode, 2, 0)  # 2 = KEYEVENTF_KEYUP
        return True

    def drag(self, x0: int, y0: int, x1: int, y1: int, duration_s: float = 0.22, steps: int = 12, ensure_focus: bool = True) -> None:
        """
        Simulates a mouse drag gesture from (x0, y0) to (x1, y1).
        Essential for scrolling mobile-ported Unity ScrollRect lists reliably.
        """
        if ensure_focus:
            self.focus_game_window()

        ctypes.windll.user32.SetCursorPos(int(x0), int(y0))
        time.sleep(0.04)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.04)

        for step in range(1, steps + 1):
            t = step / float(steps)
            cur_x = int(x0 + (x1 - x0) * t)
            cur_y = int(y0 + (y1 - y0) * t)
            ctypes.windll.user32.SetCursorPos(cur_x, cur_y)
            time.sleep(duration_s / float(steps))

        ctypes.windll.user32.SetCursorPos(int(x1), int(y1))
        time.sleep(0.04)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.15)

    def scroll(self, delta: int = -120, ensure_focus: bool = True) -> None:
        """
        Simulates mouse wheel scrolling.
        delta: positive scrolls up, negative scrolls down (standard notch = -120).
        """
        if ensure_focus:
            self.focus_game_window()
        extra = ctypes.c_ulong(0)
        ii = INPUT_I()
        ii.mi = MOUSEINPUT(0, 0, ctypes.c_ulong(delta), MOUSEEVENTF_WHEEL, 0, ctypes.pointer(extra))
        input_s = INPUT(INPUT_MOUSE, ii)
        ret = ctypes.windll.user32.SendInput(1, ctypes.pointer(input_s), ctypes.sizeof(input_s))
        if ret == 0:
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, delta, 0)

    def focus_game_window(self, titles: Tuple[str, ...] = ("Play Together", "PlayTogether")) -> bool:
        """
        Brings the game window to foreground to ensure SendInput keys are received.
        Excludes other apps (Antigravity, Chrome, IDE, Discord) that might have 'Play Together' in their title.
        """
        user32 = ctypes.windll.user32
        target_hwnd = None

        blacklist = (
            "antigravity", "vision bot", "toolpt", "auto fishing bot",
            "chrome", "firefox", "edge", "visual studio", "code", "github",
            "discord", "youtube", "chat"
        )

        def enum_cb(hwnd, lparam):
            nonlocal target_hwnd
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buff, length + 1)
                    raw_title = buff.value.strip()
                    title = raw_title.lower()

                    # Strictly ignore any non-game window containing blacklisted keywords
                    if any(b in title for b in blacklist):
                        return True

                    for t in titles:
                        # Match game window (blacklist already blocks non-game windows)
                        if t.lower() in title:
                            target_hwnd = hwnd
                            return False
            return True

        from ctypes import wintypes
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows(EnumWindowsProc(enum_cb), 0)

        if target_hwnd:
            fg_hwnd = user32.GetForegroundWindow()
            if fg_hwnd != target_hwnd:
                fg_thread = user32.GetWindowThreadProcessId(fg_hwnd, None)
                cur_thread = ctypes.windll.kernel32.GetCurrentThreadId()
                attached = False
                if fg_thread and fg_thread != cur_thread:
                    attached = bool(user32.AttachThreadInput(cur_thread, fg_thread, True))
                try:
                    user32.ShowWindow(target_hwnd, 9)  # SW_RESTORE
                    user32.BringWindowToTop(target_hwnd)
                    user32.SetForegroundWindow(target_hwnd)
                finally:
                    if attached:
                        user32.AttachThreadInput(cur_thread, fg_thread, False)
                time.sleep(0.05)
            return True
        return False

