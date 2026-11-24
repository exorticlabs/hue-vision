"""
Hue Vision - Android Device & Emulator Adapter (ADB)
────────────────────────────────────────────────────
Enables autonomous AI perception, ground-truth UIAutomator DOM hierarchy extraction,
and high-speed causal interaction on physical Android phones, tablets, and emulators.
Uses the Android Debug Bridge (ADB) protocol.
"""

import os
import re
import sys
import io
import time
import shutil
import subprocess
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from hue_vision.platforms.base import PlatformAdapter


class AndroidAdapter(PlatformAdapter):
    """Android device automation adapter via ADB."""

    WIDGET_TO_CONTROL_TYPE = {
        "android.widget.Button": "ButtonControl",
        "android.widget.ImageButton": "ButtonControl",
        "android.widget.EditText": "EditControl",
        "android.widget.TextView": "TextControl",
        "android.widget.CheckBox": "CheckBoxControl",
        "android.widget.RadioButton": "RadioButtonControl",
        "android.widget.Switch": "CheckBoxControl",
        "android.widget.ToggleButton": "ButtonControl",
        "android.widget.ImageView": "ImageControl",
        "android.widget.ListView": "ListControl",
        "android.widget.RecyclerView": "ListControl",
        "android.widget.ScrollView": "PaneControl",
        "android.widget.FrameLayout": "PaneControl",
        "android.widget.LinearLayout": "GroupControl",
        "android.widget.RelativeLayout": "GroupControl",
    }

    def __init__(self, device_serial: Optional[str] = None):
        self.device_serial = device_serial
        self._adb_cmd = shutil.which("adb") or "adb"

    @property
    def platform_name(self) -> str:
        return "android"

    @property
    def is_available(self) -> bool:
        if not shutil.which("adb"):
            return False
        try:
            res = subprocess.run([self._adb_cmd, "devices"], capture_output=True, text=True, timeout=2.0)
            lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip() and not l.startswith("List of")]
            return len(lines) > 0
        except Exception:
            return False

    def _adb_prefix(self) -> List[str]:
        cmd = [self._adb_cmd]
        if self.device_serial:
            cmd.extend(["-s", self.device_serial])
        return cmd

    def _run_adb(self, args: List[str], timeout: float = 5.0) -> subprocess.CompletedProcess:
        full_cmd = self._adb_prefix() + args
        return subprocess.run(full_cmd, capture_output=True, timeout=timeout)

    def capture_screen(self) -> Tuple[Image.Image, Tuple[int, int]]:
        """Ultra-fast raw PNG screencap directly from device frame buffer."""
        try:
            res = self._run_adb(["exec-out", "screencap", "-p"], timeout=4.0)
            if res.returncode == 0 and res.stdout:
                img = Image.open(io.BytesIO(res.stdout))
                img.load()
                return img, img.size
        except Exception:
            pass

        # Fallback via pull
        try:
            self._run_adb(["shell", "screencap", "-p", "/sdcard/hue_cap.png"])
            res = self._run_adb(["exec-out", "cat", "/sdcard/hue_cap.png"])
            if res.returncode == 0 and res.stdout:
                img = Image.open(io.BytesIO(res.stdout))
                img.load()
                return img, img.size
        except Exception:
            pass

        img = Image.new("RGB", (1080, 2400), (20, 20, 20))
        return img, img.size

    def capture_window(self, target_handle: Any) -> Optional[Image.Image]:
        img, _ = self.capture_screen()
        return img

    def get_active_window(self) -> Dict[str, Any]:
        info = {
            "title": "Android Screen",
            "process_name": "com.android.launcher",
            "pid": 0,
            "rect": [0, 0, 1080, 2400],
            "is_maximized": True,
            "hwnd": 0
        }
        try:
            res = self._run_adb(["shell", "dumpsys", "window", "displays"], timeout=2.0)
            if res.returncode == 0:
                text = res.stdout.decode("utf-8", errors="ignore")
                match = re.search(r"mCurrentFocus=Window\{[^\s]+ [^\s]+ ([^\s/]+)/?([^\s\}]*)", text)
                if match:
                    pkg = match.group(1)
                    act = match.group(2)
                    info["process_name"] = pkg
                    info["title"] = f"{pkg} ({act})" if act else pkg

            # Read screen resolution
            sz_res = self._run_adb(["shell", "wm", "size"], timeout=1.5)
            if sz_res.returncode == 0:
                sz_match = re.search(r"(\d+)x(\d+)", sz_res.stdout.decode("utf-8", errors="ignore"))
                if sz_match:
                    w = int(sz_match.group(1))
                    h = int(sz_match.group(2))
                    info["rect"] = [0, 0, w, h]
        except Exception:
            pass
        return info

    def enumerate_windows(self) -> List[Dict[str, Any]]:
        active = self.get_active_window()
        return [active]

    def focus_window(self, target_handle: Any) -> bool:
        if not target_handle:
            return False
        pkg = str(target_handle)
        try:
            self._run_adb(["shell", "monkey", "-p", pkg, "-c", "android.intent.category.LAUNCHER", "1"])
            return True
        except Exception:
            return False

    def extract_hierarchy(
        self,
        target_handle: Any = None,
        max_elements: int = 150,
        max_depth: int = 10
    ) -> Tuple[List[Any], List[Any]]:
        """Dumps and parses the live Android UIAutomator XML hierarchy into standard UI elements."""
        from hue_vision.dom_extractor import RawDOMGroup, RawDOMElement

        groups: List[RawDOMGroup] = []
        elements: List[RawDOMElement] = []

        xml_data = None
        try:
            # Attempt 1: Direct stdout dump
            res = self._run_adb(["exec-out", "uiautomator", "dump", "/dev/tty"], timeout=3.5)
            if res.returncode == 0 and b"<?xml" in res.stdout:
                xml_data = res.stdout.decode("utf-8", errors="ignore")
        except Exception:
            pass

        if not xml_data:
            try:
                # Attempt 2: Device temp file
                self._run_adb(["shell", "uiautomator", "dump", "/sdcard/window_dump.xml"], timeout=4.0)
                res2 = self._run_adb(["exec-out", "cat", "/sdcard/window_dump.xml"], timeout=3.0)
                if res2.returncode == 0 and b"<?xml" in res2.stdout:
                    xml_data = res2.stdout.decode("utf-8", errors="ignore")
            except Exception:
                pass

        if not xml_data:
            return [], []

        try:
            # Strip preamble if present
            idx = xml_data.find("<?xml")
            if idx > 0:
                xml_data = xml_data[idx:]
            idx2 = xml_data.find("<hierarchy")
            if idx2 >= 0:
                xml_data = xml_data[idx2:]

            root = ET.fromstring(xml_data)
        except Exception:
            return [], []

        main_grp = RawDOMGroup("grp_android_main", "MAIN_CONTENT", "Android Screen", [0, 0, 1080, 2400])
        groups.append(main_grp)

        counter = [0]
        bounds_pattern = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")

        def walk(node, depth: int = 0):
            if depth > max_depth or counter[0] >= max_elements:
                return

            for child in node:
                try:
                    c_class = child.attrib.get("class", "")
                    c_text = child.attrib.get("text", "") or ""
                    c_desc = child.attrib.get("content-desc", "") or ""
                    c_bounds = child.attrib.get("bounds", "")
                    c_clickable = child.attrib.get("clickable", "false").lower() == "true"
                    c_focused = child.attrib.get("focused", "false").lower() == "true"
                    c_enabled = child.attrib.get("enabled", "true").lower() == "true"

                    m = bounds_pattern.match(c_bounds)
                    if not m:
                        continue
                    x1, y1, x2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                    w, h = x2 - x1, y2 - y1

                    if w < 10 or h < 10:
                        continue

                    label = c_text if c_text else c_desc
                    ctype = AndroidAdapter.WIDGET_TO_CONTROL_TYPE.get(c_class, "CustomControl")

                    is_interactive = c_clickable or ctype in ["ButtonControl", "EditControl", "CheckBoxControl"]

                    if is_interactive or (label and w < 1000):
                        counter[0] += 1
                        el = RawDOMElement(
                            element_id=counter[0],
                            name=label,
                            text=label,
                            control_type=ctype,
                            bbox=[x1, y1, x2, y2],
                            is_interactive=is_interactive,
                            is_focused=c_focused,
                            is_enabled=c_enabled,
                            parent_group_id="grp_android_main",
                            native_attributes={
                                "class": c_class,
                                "resource_id": child.attrib.get("resource-id", ""),
                                "package": child.attrib.get("package", "")
                            }
                        )
                        elements.append(el)
                        main_grp.element_ids.append(el.element_id)

                    walk(child, depth + 1)
                except Exception:
                    continue

        walk(root, 0)
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
        ix, iy = int(x), int(y)

        if action == "click":
            self._run_adb(["shell", "input", "tap", str(ix), str(iy)])
        elif action == "double_click":
            self._run_adb(["shell", "input", "tap", str(ix), str(iy)])
            time.sleep(0.06)
            self._run_adb(["shell", "input", "tap", str(ix), str(iy)])
        elif action == "right_click" or action == "long_press":
            # Android long press: swipe in place for 500ms
            self._run_adb(["shell", "input", "swipe", str(ix), str(iy), str(ix), str(iy), "500"])
        elif action == "hover":
            pass
        elif action == "type":
            self._run_adb(["shell", "input", "tap", str(ix), str(iy)])
            time.sleep(0.1)
            if clear_first:
                # Select all and delete
                self._run_adb(["shell", "input", "keyevent", "KEYCODE_MOVE_END"])
                for _ in range(30):
                    self._run_adb(["shell", "input", "keyevent", "67"])  # KEYCODE_DEL
            if text:
                # Escape spaces and special chars for adb shell
                escaped = text.replace(" ", "%s").replace("'", "\\'").replace('"', '\\"')
                self._run_adb(["shell", "input", "text", escaped])
            if press_enter:
                self._run_adb(["shell", "input", "keyevent", "66"])  # KEYCODE_ENTER
        elif action == "scroll":
            delta_y = amount * 180
            start_y = iy
            end_y = iy - delta_y if direction == "down" else iy + delta_y
            self._run_adb(["shell", "input", "swipe", str(ix), str(start_y), str(ix), str(end_y), "300"])
        elif action == "hotkey" and keys:
            KEY_MAP = {
                "back": "4",
                "home": "3",
                "enter": "66",
                "tab": "61",
                "escape": "111",
                "menu": "82",
                "app_switch": "187",
                "volume_up": "24",
                "volume_down": "25"
            }
            for k in keys:
                code = KEY_MAP.get(k.lower(), "66")
                self._run_adb(["shell", "input", "keyevent", code])

    def launch_and_wait_gui(
        self,
        executable: str,
        args: Optional[List[str]] = None,
        expected_title: str = "",
        timeout_s: float = 10.0
    ) -> int:
        pkg = executable
        self._run_adb(["shell", "monkey", "-p", pkg, "-c", "android.intent.category.LAUNCHER", "1"])
        time.sleep(1.0)
        return 1
