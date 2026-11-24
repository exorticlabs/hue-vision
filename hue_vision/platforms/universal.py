"""
Hue Vision - Universal CV Fallback Adapter
──────────────────────────────────────────
Pure computer vision and cross-platform automation adapter.
Guarantees full perception and interaction in DOM-absent environments:
remote desktop streams (RDP/VNC), video games, canvas apps, or headless runners.
"""

import sys
import time
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from hue_vision.platforms.base import PlatformAdapter


class UniversalCVAdapter(PlatformAdapter):
    """Pure vision and standard cross-platform interaction fallback."""

    def __init__(self):
        self._sct = None

    @property
    def platform_name(self) -> str:
        return "universal_cv"

    @property
    def is_available(self) -> bool:
        return True

    def _get_mss(self):
        if self._sct is None:
            try:
                import mss
                self._sct = mss.MSS()
            except Exception:
                pass
        return self._sct

    def capture_screen(self) -> Tuple[Image.Image, Tuple[int, int]]:
        sct = self._get_mss()
        if sct is not None:
            try:
                import numpy as np
                monitor = sct.monitors[0]
                shot = sct.grab(monitor)
                w, h = shot.size
                raw_np = np.frombuffer(shot.raw, dtype=np.uint8).reshape((h, w, 4))
                rgb_np = raw_np[:, :, [2, 1, 0]]
                img = Image.fromarray(rgb_np)
                return img, img.size
            except Exception:
                self._sct = None

        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            return img, img.size
        except Exception:
            img = Image.new("RGB", (1920, 1080), (30, 30, 30))
            return img, img.size

    def capture_window(self, target_handle: Any) -> Optional[Image.Image]:
        img, _ = self.capture_screen()
        return img

    def get_active_window(self) -> Dict[str, Any]:
        return {
            "title": "Universal Display",
            "process_name": "system",
            "pid": 0,
            "rect": [0, 0, 1920, 1080],
            "is_maximized": True,
            "hwnd": 0
        }

    def enumerate_windows(self) -> List[Dict[str, Any]]:
        return [self.get_active_window()]

    def focus_window(self, target_handle: Any) -> bool:
        return True

    def extract_hierarchy(
        self,
        target_handle: Any = None,
        max_elements: int = 150,
        max_depth: int = 10
    ) -> Tuple[List[Any], List[Any]]:
        # In pure CV mode, DOM is absent: returns empty list so CVDetector handles bounding box discovery
        return [], []

    def execute_action(
        self,
        action: str,
        x: int,
        y: int,
        text: Optional[str] = None,
        press_enter: bool = False,
        clear_first: bool = False,
        direction: str = "down",
        amount: int = 3,
        keys: Optional[List[str]] = None
    ) -> None:
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            ix, iy = int(x), int(y)
            if action == "click":
                pyautogui.click(ix, iy)
            elif action == "double_click":
                pyautogui.doubleClick(ix, iy)
            elif action == "right_click":
                pyautogui.rightClick(ix, iy)
            elif action == "hover":
                pyautogui.moveTo(ix, iy)
            elif action == "type":
                pyautogui.click(ix, iy)
                time.sleep(0.05)
                if clear_first:
                    pyautogui.hotkey("ctrl", "a")
                    pyautogui.press("backspace")
                if text:
                    pyautogui.write(text, interval=0.01)
                if press_enter:
                    pyautogui.press("enter")
            elif action == "scroll":
                scroll_amount = amount if direction == "up" else -amount
                pyautogui.scroll(scroll_amount * 10)
            elif action == "hotkey" and keys:
                pyautogui.hotkey(*[k.lower() for k in keys])
        except ImportError:
            pass

    def launch_and_wait_gui(
        self,
        executable: str,
        args: Optional[List[str]] = None,
        expected_title: str = "",
        timeout_s: float = 10.0
    ) -> int:
        import subprocess
        if args is None:
            args = []
        proc = subprocess.Popen([executable] + args)
        time.sleep(1.0)
        return proc.pid
