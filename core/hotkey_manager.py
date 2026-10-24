import keyboard
from typing import Callable, Optional


class HotkeyManager:
    """
    Registers and listens for global hotkeys across the entire operating system.
    """

    def __init__(self):
        self._registered_hotkeys = []

    def register(self, hotkey: str, callback: Callable[[], None], suppress: bool = False):
        """
        Registers a global hotkey with a callback.

        Args:
            hotkey: Key name or combination, e.g., 'f1', 'f2'
            callback: Function to invoke when pressed
            suppress: Whether to suppress event from passing to other apps
        """
        try:
            keyboard.add_hotkey(hotkey, callback, suppress=suppress)
            self._registered_hotkeys.append(hotkey)
        except Exception as e:
            print(f"[LỖI HOTKEY] Không thể đăng ký phím tắt '{hotkey}': {e}")
            print("Gợi ý: Hãy thử chạy terminal bằng quyền Administrator nếu bị chặn.")

    def unregister_all(self):
        """Unregisters all hook listeners."""
        try:
            keyboard.unhook_all()
            self._registered_hotkeys.clear()
        except Exception:
            pass
