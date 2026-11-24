"""
Hue Vision - macOS Platform Adapter
───────────────────────────────────
macOS Accessibility (AXUIElement), AppleScript (osascript), and Quartz backend.
Enables full screen capture, window tracking, and GUI interaction on macOS.
"""

import os
import sys
import time
import subprocess
import json
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from hue_vision.platforms.base import PlatformAdapter


class MacOSAdapter(PlatformAdapter):
    """macOS-specific automation and perception backend."""

    def __init__(self):
        self._sct = None

    @property
    def platform_name(self) -> str:
        return "macos"

    @property
    def is_available(self) -> bool:
        return sys.platform == "darwin"

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

        # Fallback 1: PIL.ImageGrab
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            return img, img.size
        except Exception:
            pass

        # Fallback 2: native screencapture utility
        try:
            tmp_path = "/tmp/hue_vision_screen.png"
            subprocess.run(["screencapture", "-x", tmp_path], check=True, timeout=2.0)
            img = Image.open(tmp_path)
            size = img.size
            img.load()
            os.remove(tmp_path)
            return img, size
        except Exception:
            img = Image.new("RGB", (1920, 1080), (30, 30, 30))
            return img, img.size

    def capture_window(self, target_handle: Any) -> Optional[Image.Image]:
        # On macOS, window capture can be done with screencapture -l<windowid>
        if not target_handle:
            return None
        try:
            wid = str(target_handle)
            tmp_path = f"/tmp/hue_vision_win_{wid}.png"
            subprocess.run(["screencapture", "-x", "-l", wid, tmp_path], check=True, timeout=2.0)
            img = Image.open(tmp_path)
            img.load()
            os.remove(tmp_path)
            return img
        except Exception:
            return None

    def get_active_window(self) -> Dict[str, Any]:
        info = {
            "title": "Desktop Screen",
            "process_name": "Finder",
            "pid": 0,
            "rect": [0, 0, 1920, 1080],
            "is_maximized": False,
            "hwnd": 0
        }
        if sys.platform != "darwin":
            return info

        script = """
        tell application "System Events"
            set frontProc to first process whose frontmost is true
            set procName to name of frontProc
            set procPid to unix id of frontProc
            set winTitle to "Untitled Window"
            set winPos to {0, 0}
            set winSize to {1920, 1080}
            try
                set frontWin to front window of frontProc
                set winTitle to name of frontWin
                set winPos to position of frontWin
                set winSize to size of frontWin
            end try
            return procName & "|||" & procPid & "|||" & winTitle & "|||" & (item 1 of winPos) & "," & (item 2 of winPos) & "," & (item 1 of winSize) & "," & (item 2 of winSize)
        end tell
        """
        try:
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=1.5)
            if res.returncode == 0 and res.stdout.strip():
                parts = res.stdout.strip().split("|||")
                if len(parts) >= 4:
                    proc_name = parts[0]
                    pid_val = int(parts[1]) if parts[1].isdigit() else 0
                    win_title = parts[2]
                    coords = [int(float(x.strip())) for x in parts[3].split(",")]
                    x, y, w, h = coords[0], coords[1], coords[2], coords[3]
                    return {
                        "title": win_title,
                        "process_name": proc_name,
                        "pid": pid_val,
                        "rect": [x, y, x + w, y + h],
                        "is_maximized": False,
                        "hwnd": pid_val
                    }
        except Exception:
            pass
        return info

    def enumerate_windows(self) -> List[Dict[str, Any]]:
        windows = []
        if sys.platform != "darwin":
            return windows

        script = """
        tell application "System Events"
            set outList to {}
            set pList to every process whose visible is true
            repeat with p in pList
                set pName to name of p
                set pPid to unix id of p
                set isFront to frontmost of p
                repeat with w in (every window of p)
                    try
                        set wTitle to name of w
                        set wPos to position of w
                        set wSize to size of w
                        set end of outList to pName & ":::" & pPid & ":::" & wTitle & ":::" & (item 1 of wPos) & "," & (item 2 of wPos) & "," & (item 1 of wSize) & "," & (item 2 of wSize) & ":::" & isFront
                    end try
                end repeat
            end repeat
            return outList
        end tell
        """
        try:
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2.5)
            if res.returncode == 0:
                raw_lines = res.stdout.strip().split(", ")
                for item in raw_lines:
                    parts = item.split(":::")
                    if len(parts) >= 5:
                        p_name = parts[0].strip()
                        p_pid = int(parts[1].strip()) if parts[1].strip().isdigit() else 0
                        w_title = parts[2].strip()
                        coords = [int(float(x.strip())) for x in parts[3].split(",")]
                        x, y, w, h = coords[0], coords[1], coords[2], coords[3]
                        is_front = parts[4].strip().lower() == "true"
                        if w >= 50 and h >= 50 and w_title:
                            windows.append({
                                "hwnd": p_pid,
                                "title": w_title,
                                "process_name": p_name,
                                "pid": p_pid,
                                "rect": [x, y, x + w, y + h],
                                "is_focused": is_front,
                                "is_minimized": False
                            })
        except Exception:
            pass
        return windows

    def focus_window(self, target_handle: Any) -> bool:
        if not target_handle:
            return False
        # target_handle can be process name or PID
        try:
            script = f"""
            tell application "System Events"
                set procList to every process whose unix id is {int(target_handle)}
                if (count of procList) > 0 then
                    set frontmost of (item 1 of procList) to true
                    return true
                end if
            end tell
            return false
            """
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=1.5)
            return "true" in res.stdout.lower()
        except Exception:
            return False

    def extract_hierarchy(
        self,
        target_handle: Any = None,
        max_elements: int = 150,
        max_depth: int = 10
    ) -> Tuple[List[Any], List[Any]]:
        # Returns empty list so engine automatically triggers instant CV candidate detection,
        # or extracts standard AppleScript UI elements if accessibility permissions are granted.
        from hue_vision.dom_extractor import RawDOMGroup, RawDOMElement

        groups: List[RawDOMGroup] = []
        elements: List[RawDOMElement] = []

        if sys.platform != "darwin":
            return groups, elements

        # Query top-level front process UI elements via AppleScript
        script = """
        tell application "System Events"
            set frontProc to first process whose frontmost is true
            set uiList to {}
            try
                repeat with btn in (every button of front window of frontProc)
                    set bName to name of btn
                    set bPos to position of btn
                    set bSize to size of btn
                    set end of uiList to "ButtonControl:::" & bName & ":::" & (item 1 of bPos) & "," & (item 2 of bPos) & "," & (item 1 of bSize) & "," & (item 2 of bSize)
                end repeat
            end try
            try
                repeat with txt in (every text field of front window of frontProc)
                    set tName to name of txt
                    set tVal to ""
                    try
                        set tVal to value of txt
                    end try
                    set tPos to position of txt
                    set tSize to size of txt
                    set end of uiList to "EditControl:::" & (tName & " " & tVal) & ":::" & (item 1 of tPos) & "," & (item 2 of tPos) & "," & (item 1 of tSize) & "," & (item 2 of tSize)
                end repeat
            end try
            return uiList
        end tell
        """
        try:
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2.0)
            if res.returncode == 0 and res.stdout.strip():
                items = res.stdout.strip().split(", ")
                counter = 0
                for it in items:
                    parts = it.split(":::")
                    if len(parts) >= 3:
                        ctype = parts[0].strip()
                        cname = parts[1].strip()
                        coords = [int(float(x.strip())) for x in parts[2].split(",")]
                        x, y, w, h = coords[0], coords[1], coords[2], coords[3]
                        if w >= 8 and h >= 8:
                            counter += 1
                            elements.append(RawDOMElement(
                                element_id=counter,
                                name=cname,
                                text=cname,
                                control_type=ctype,
                                bbox=[x, y, x + w, y + h],
                                is_interactive=True,
                                is_focused=False,
                                is_enabled=True,
                                parent_group_id="grp_main"
                            ))
                if elements:
                    groups.append(RawDOMGroup(
                        group_id="grp_main",
                        group_type="MAIN_CONTENT",
                        title="Active Window",
                        bbox=[0, 0, 1920, 1080],
                        element_ids=[e.element_id for e in elements]
                    ))
        except Exception:
            pass

        return groups, elements

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
        # Cross-platform action execution using pynput, pyautogui, or native osascript
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            if action == "click":
                pyautogui.click(x, y)
            elif action == "double_click":
                pyautogui.doubleClick(x, y)
            elif action == "right_click":
                pyautogui.rightClick(x, y)
            elif action == "hover":
                pyautogui.moveTo(x, y)
            elif action == "type":
                pyautogui.click(x, y)
                time.sleep(0.05)
                if clear_first:
                    pyautogui.hotkey("command", "a")
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
            return
        except ImportError:
            pass

        # Fallback to AppleScript for keyboard and mouse
        if action == "type" and text:
            clean_text = text.replace('"', '\\"')
            s = f'tell application "System Events" to keystroke "{clean_text}"'
            if press_enter:
                s += '\ntell application "System Events" to key code 36'  # Enter key
            subprocess.run(["osascript", "-e", s], timeout=1.0)
        elif action == "hotkey" and keys:
            # Send keys via AppleScript
            pass

    def launch_and_wait_gui(
        self,
        executable: str,
        args: Optional[List[str]] = None,
        expected_title: str = "",
        timeout_s: float = 10.0
    ) -> int:
        if args is None:
            args = []
        if executable.endswith(".app") or "/" not in executable:
            subprocess.Popen(["open", "-a", executable] + args)
        else:
            subprocess.Popen([executable] + args)

        start = time.time()
        while time.time() - start < timeout_s:
            wins = self.enumerate_windows()
            for w in wins:
                if expected_title and expected_title.lower() in w["title"].lower():
                    self.focus_window(w["pid"])
                    return w["pid"]
                elif not expected_title and executable.lower() in w["process_name"].lower():
                    self.focus_window(w["pid"])
                    return w["pid"]
            time.sleep(0.1)
        return 0
