"""
Hue Vision - Windows Platform Adapter
─────────────────────────────────────
Full Windows UIAutomation, Win32 GDI/DWM, and native interaction backend.
Supports UWP, WinUI3, Win32, WPF, and Chromium accessibility trees.
"""

import os
import sys
import time
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from hue_vision.platforms.base import PlatformAdapter

# Bind WinSta0 at module load on Windows
if os.name == "nt":
    try:
        import ctypes
        _u32 = ctypes.windll.user32
        _h_ws = _u32.OpenWindowStationW("WinSta0", False, 0x37F)
        if _h_ws:
            _u32.SetProcessWindowStation(_h_ws)
            _h_dt = _u32.OpenDesktopW("default", 0, False, 0x1FF)
            if _h_dt:
                _u32.SetThreadDesktop(_h_dt)
    except Exception:
        pass


class WindowsAdapter(PlatformAdapter):
    """Windows-specific automation and accessibility adapter."""

    INTERACTIVE_CONTROL_TYPES = {
        "ButtonControl", "EditControl", "DocumentControl", "CheckBoxControl",
        "RadioButtonControl", "ComboBoxControl", "TabItemControl", "MenuItemControl",
        "TreeItemControl", "ListItemControl", "HyperlinkControl", "SplitButtonControl"
    }

    CONTAINER_CONTROL_TYPES = {
        "ToolBarControl", "MenuBarControl", "TabControl", "ListControl",
        "TableControl", "TreeControl", "GroupControl", "PaneControl", "WindowControl"
    }

    def __init__(self):
        self._sct = None
        self._uia_available = False
        if os.name == "nt":
            try:
                import uiautomation
                self._uia_available = True
            except ImportError:
                self._uia_available = False

    @property
    def platform_name(self) -> str:
        return "windows"

    @property
    def is_available(self) -> bool:
        return os.name == "nt"

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
        if os.name != "nt" or not target_handle:
            return None
        hwnd = int(target_handle)
        try:
            import win32gui
            import win32ui
            import ctypes
            import numpy as np

            user32 = ctypes.windll.user32
            if not user32.IsWindow(hwnd):
                return None

            rect = win32gui.GetWindowRect(hwnd)
            w = max(1, rect[2] - rect[0])
            h = max(1, rect[3] - rect[1])

            hwnd_dc = win32gui.GetWindowDC(hwnd)
            if not hwnd_dc:
                return None
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
            save_dc.SelectObject(save_bitmap)

            result = user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)

            bmpinfo = save_bitmap.GetInfo()
            bmpstr = save_bitmap.GetBitmapBits(True)

            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)

            if result != 1:
                return None

            img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
            rgb = img[:, :, [2, 1, 0]]
            return Image.fromarray(rgb)
        except Exception:
            return None

    def get_active_window(self) -> Dict[str, Any]:
        info = {
            "title": "Desktop Screen",
            "process_name": "explorer.exe",
            "pid": 0,
            "rect": [0, 0, 1920, 1080],
            "is_maximized": False,
            "hwnd": 0
        }
        if os.name != "nt":
            return info
        try:
            import uiautomation as auto
            fg = auto.GetForegroundControl()
            if fg:
                info["title"] = getattr(fg, "Name", "") or "Untitled Window"
                info["pid"] = getattr(fg, "ProcessId", 0)
                info["hwnd"] = getattr(fg, "NativeWindowHandle", 0)
                rect = fg.BoundingRectangle
                info["rect"] = [rect.left, rect.top, rect.right, rect.bottom]
                if info["pid"]:
                    try:
                        import psutil
                        info["process_name"] = psutil.Process(info["pid"]).name()
                    except Exception:
                        pass
        except Exception:
            pass
        return info

    def enumerate_windows(self) -> List[Dict[str, Any]]:
        if os.name != "nt":
            return []
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            psutil_ok = False
            try:
                import psutil
                psutil_ok = True
            except ImportError:
                pass

            fg_hwnd = user32.GetForegroundWindow()
            windows = []

            def enum_cb(hwnd, _):
                try:
                    if not user32.IsWindowVisible(hwnd):
                        return True

                    rect = wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top
                    if w < 50 or h < 50:
                        return True

                    length = user32.GetWindowTextLengthW(hwnd)
                    if length <= 0:
                        return True
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value.strip()
                    if not title or title in ("Default IME", "MSCTFIME UI", "Windows Input Experience"):
                        return True

                    is_minimized = bool(user32.IsIconic(hwnd))

                    pid_val_buf = ctypes.c_ulong(0)
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_val_buf))
                    pid_val = pid_val_buf.value

                    proc_name = "unknown"
                    if psutil_ok and pid_val:
                        try:
                            proc_name = psutil.Process(pid_val).name()
                        except Exception:
                            pass

                    windows.append({
                        "hwnd": hwnd,
                        "title": title,
                        "process_name": proc_name,
                        "pid": pid_val,
                        "rect": [rect.left, rect.top, rect.right, rect.bottom],
                        "is_focused": (hwnd == fg_hwnd),
                        "is_minimized": is_minimized,
                    })
                except Exception:
                    pass
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
            return windows
        except Exception:
            return []

    def focus_window(self, target_handle: Any) -> bool:
        if os.name != "nt" or not target_handle:
            return False
        hwnd = int(target_handle)
        try:
            import ctypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            if not user32.IsWindow(hwnd):
                return False

            if user32.GetForegroundWindow() == hwnd:
                return True

            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)
                time.sleep(0.05)

            fore_hwnd = user32.GetForegroundWindow()
            fore_thread = user32.GetWindowThreadProcessId(fore_hwnd, None) if fore_hwnd else 0
            app_thread = kernel32.GetCurrentThreadId()

            if fore_thread and fore_thread != app_thread:
                user32.AttachThreadInput(fore_thread, app_thread, True)

            user32.AllowSetForegroundWindow(0xFFFFFFFF)
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)

            if fore_thread and fore_thread != app_thread:
                user32.AttachThreadInput(fore_thread, app_thread, False)

            time.sleep(0.05)
            return user32.GetForegroundWindow() == hwnd
        except Exception:
            return False

    def _get_popup_hwnds(self, main_hwnd: int) -> List[int]:
        if os.name != "nt":
            return []
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32

            if not main_hwnd or not user32.IsWindow(main_hwnd):
                main_hwnd = user32.GetForegroundWindow()
            if not main_hwnd or not user32.IsWindow(main_hwnd):
                return []

            target_pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(main_hwnd, ctypes.byref(target_pid))
            t_pid = target_pid.value
            if not t_pid:
                return []

            popups = []
            POPUP_CLASSES = {
                "#32768", "Qt5QWindowPopupSaveBits", "ComboLBox",
                "ToolTips_Win32_v1", "CASCADEWND", "SunAwtPopupMenu",
                "Chrome_WidgetWin_1", "Windows.UI.Core.CoreWindow"
            }

            def enum_cb(hwnd, _):
                if hwnd == main_hwnd or not user32.IsWindowVisible(hwnd):
                    return True
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value == t_pid:
                    class_buff = ctypes.create_unicode_buffer(256)
                    user32.GetClassNameW(hwnd, class_buff, 256)
                    cname = class_buff.value

                    style = user32.GetWindowLongW(hwnd, -16)
                    is_popup = (style & 0x80000000) != 0

                    rect = wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top

                    if (is_popup or cname in POPUP_CLASSES) and w >= 10 and h >= 10:
                        popups.append(hwnd)
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
            return popups
        except Exception:
            return []

    def extract_hierarchy(
        self,
        target_handle: Any = None,
        max_elements: int = 150,
        max_depth: int = 10
    ) -> Tuple[List[Any], List[Any]]:
        if not self._uia_available:
            return [], []

        from hue_vision.dom_extractor import RawDOMGroup, RawDOMElement
        import uiautomation as auto

        target_hwnd = int(target_handle) if target_handle else 0
        groups: List[RawDOMGroup] = []
        elements: List[RawDOMElement] = []
        group_lookup: Dict[str, RawDOMGroup] = {}

        try:
            if target_hwnd is not None and target_hwnd != 0:
                self.focus_window(target_hwnd)
                root_ctrl = auto.ControlFromHandle(target_hwnd)
            else:
                root_ctrl = auto.GetForegroundControl()
            if not root_ctrl:
                return [], []

            win_rect = root_ctrl.BoundingRectangle
            wx, wy, ww, wh = win_rect.left, win_rect.top, win_rect.right - win_rect.left, win_rect.bottom - win_rect.top

            top_bar_grp = RawDOMGroup("grp_top_bar", "NAVBAR_HEADER", "Top Navigation Bar", [wx, wy, wx + ww, wy + min(110, wh)])
            groups.append(top_bar_grp)
            group_lookup["grp_top_bar"] = top_bar_grp

            sidebar_grp = RawDOMGroup("grp_sidebar", "SIDEBAR_PANEL", "Sidebar Panel", [wx, wy + 110, wx + min(320, ww), wy + wh])
            groups.append(sidebar_grp)
            group_lookup["grp_sidebar"] = sidebar_grp

            main_grp = RawDOMGroup("grp_main", "MAIN_CONTENT", "Main Content Canvas", [wx + min(320, ww), wy + 110, wx + ww, wy + wh])
            groups.append(main_grp)
            group_lookup["grp_main"] = main_grp

            counter = [0]
            group_counter = [0]

            def walk(ctrl, current_group_id: str, depth: int = 0, z_layer: int = 0):
                if depth > max_depth or counter[0] >= max_elements:
                    return

                try:
                    children = ctrl.GetChildren()
                except Exception:
                    return

                for child in children:
                    try:
                        crect = child.BoundingRectangle
                        cw = crect.right - crect.left
                        ch = crect.bottom - crect.top
                        cx, cy = crect.left, crect.top

                        if cw < 6 or ch < 6:
                            continue

                        if z_layer == 0 and (cx < wx - 60 or cy < wy - 60 or cx + cw > wx + ww + 60 or cy + ch > wy + wh + 60):
                            continue

                        ctype = getattr(child, "ControlTypeName", "CustomControl")
                        cname = (getattr(child, "Name", "") or "").strip()
                        assigned_group_id = current_group_id

                        if ctype == "WindowControl" or getattr(child, "ClassName", "").lower() in ["#32770", "dialog"]:
                            group_counter[0] += 1
                            m_id = f"grp_modal_{group_counter[0]}"
                            m_grp = RawDOMGroup(m_id, "MODAL_DIALOG", cname or "Dialog Window", [cx, cy, cx + cw, cy + ch])
                            groups.append(m_grp)
                            group_lookup[m_id] = m_grp
                            assigned_group_id = m_id
                        elif ctype == "ToolBarControl" and depth <= 3:
                            group_counter[0] += 1
                            tb_id = f"grp_toolbar_{group_counter[0]}"
                            tb_grp = RawDOMGroup(tb_id, "NAVBAR_HEADER", cname or "Toolbar Ribbon", [cx, cy, cx + cw, cy + ch])
                            groups.append(tb_grp)
                            group_lookup[tb_id] = tb_grp
                            assigned_group_id = tb_id
                        elif ctype == "GroupControl" and cname:
                            group_counter[0] += 1
                            fc_id = f"grp_form_{group_counter[0]}"
                            fc_grp = RawDOMGroup(fc_id, "FORM_GROUP", cname, [cx, cy, cx + cw, cy + ch])
                            groups.append(fc_grp)
                            group_lookup[fc_id] = fc_grp
                            assigned_group_id = fc_id

                        if assigned_group_id == "grp_main":
                            if cy + ch <= wy + 110:
                                assigned_group_id = "grp_top_bar"
                            elif cx + cw <= wx + 320:
                                assigned_group_id = "grp_sidebar"

                        extracted_text = cname
                        is_read_only = False
                        try:
                            vp = child.GetPattern(auto.PatternId.ValuePattern)
                            if vp:
                                if not extracted_text and vp.Value:
                                    extracted_text = str(vp.Value).strip()
                                is_read_only = bool(vp.IsReadOnly)
                        except Exception:
                            pass

                        if not is_read_only:
                            try:
                                rvp = child.GetPattern(auto.PatternId.RangeValuePattern)
                                if rvp:
                                    if not extracted_text:
                                        extracted_text = str(rvp.Value).strip()
                                    is_read_only = bool(rvp.IsReadOnly)
                            except Exception:
                                pass

                        live_setting = getattr(child, "LiveSetting", 0)
                        is_interactive = (ctype in WindowsAdapter.INTERACTIVE_CONTROL_TYPES) or (z_layer > 0)
                        is_focused = getattr(child, "HasKeyboardFocus", False)
                        is_enabled = getattr(child, "IsEnabled", True)

                        if is_interactive or (extracted_text and cw < ww * 0.9):
                            counter[0] += 1
                            el = RawDOMElement(
                                element_id=counter[0],
                                name=cname,
                                text=extracted_text,
                                control_type=ctype,
                                bbox=[cx, cy, cx + cw, cy + ch],
                                is_interactive=is_interactive,
                                is_focused=is_focused,
                                is_enabled=is_enabled,
                                parent_group_id=assigned_group_id,
                                z_layer=z_layer,
                                native_attributes={
                                    "is_read_only": is_read_only,
                                    "live_setting": live_setting,
                                    "z_layer": z_layer
                                }
                            )
                            elements.append(el)
                            if assigned_group_id in group_lookup:
                                group_lookup[assigned_group_id].element_ids.append(el.element_id)

                        walk(child, assigned_group_id, depth + 1, z_layer=z_layer)
                    except Exception:
                        continue

            walk(root_ctrl, "grp_main", depth=0, z_layer=0)

            effective_hwnd = target_hwnd if (target_hwnd and target_hwnd != 0) else getattr(root_ctrl, "NativeWindowHandle", 0)
            popup_hwnds = self._get_popup_hwnds(effective_hwnd)
            for p_idx, p_hwnd in enumerate(popup_hwnds, 1):
                try:
                    p_ctrl = auto.ControlFromHandle(p_hwnd)
                    if not p_ctrl:
                        continue
                    p_rect = p_ctrl.BoundingRectangle
                    pw = p_rect.right - p_rect.left
                    ph = p_rect.bottom - p_rect.top
                    if pw < 6 or ph < 6:
                        continue
                    p_grp_id = f"grp_popup_{p_idx}"
                    p_name = getattr(p_ctrl, "Name", "") or f"Menu Overlay #{p_idx}"
                    popup_group = RawDOMGroup(
                        group_id=p_grp_id,
                        group_type="MODAL_DIALOG",
                        title=f"Active Popup / Dropdown: {p_name}",
                        bbox=[p_rect.left, p_rect.top, p_rect.right, p_rect.bottom]
                    )
                    groups.append(popup_group)
                    group_lookup[p_grp_id] = popup_group
                    walk(p_ctrl, p_grp_id, depth=0, z_layer=1)
                except Exception:
                    continue

        except Exception:
            return [], []

        elements.sort(key=lambda el: (-el.z_layer, (el.bbox[2] - el.bbox[0]) * (el.bbox[3] - el.bbox[1])))
        active_groups = [g for g in groups if g.element_ids or g.group_type == "MODAL_DIALOG"]
        return active_groups, elements

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
        import uiautomation as auto

        ix, iy = int(x), int(y)
        auto.SetCursorPos(ix, iy)
        time.sleep(0.02)

        if action == "click":
            auto.Click(ix, iy)
        elif action == "double_click":
            auto.DoubleClick(ix, iy)
        elif action == "right_click":
            auto.RightClick(ix, iy)
        elif action == "hover":
            pass
        elif action == "type":
            auto.Click(ix, iy)
            time.sleep(0.05)
            if clear_first:
                auto.SendKeys("{Ctrl}a{Back}")
                time.sleep(0.03)
            if text:
                auto.SendKeys(text)
            if press_enter:
                auto.SendKeys("{Enter}")
        elif action == "scroll":
            auto.WheelDown(amount) if direction == "down" else auto.WheelUp(amount)
        elif action == "hotkey":
            if keys:
                key_str = "".join([f"{{{k.capitalize()}}}" if len(k) > 1 else k for k in keys])
                auto.SendKeys(key_str)

    def launch_and_wait_gui(
        self,
        executable: str,
        args: Optional[List[str]] = None,
        expected_title: str = "",
        timeout_s: float = 10.0
    ) -> int:
        if args is None:
            args = []
        cmd = [executable] + args
        proc = subprocess.Popen(cmd)
        start_time = time.time()

        base_name = os.path.basename(executable).lower()
        if base_name.endswith(".exe"):
            base_name = base_name[:-4]

        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32

        while time.time() - start_time < timeout_s:
            found_hwnd = [0]
            target_pid = proc.pid

            def enum_win(hwnd, _):
                try:
                    if not user32.IsWindowVisible(hwnd):
                        return True
                    rect = wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top
                    if w < 50 or h < 50:
                        return True

                    pid_val = wintypes.DWORD()
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_val))

                    length = user32.GetWindowTextLengthW(hwnd)
                    title = ""
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buf, length + 1)
                        title = buf.value.strip()

                    title_match = bool(expected_title and expected_title.lower() in title.lower())
                    pid_match = (pid_val.value == target_pid)
                    exe_title_match = bool(base_name and base_name in title.lower())

                    if title_match or pid_match or (not expected_title and exe_title_match):
                        found_hwnd[0] = hwnd
                        return False
                except Exception:
                    pass
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            user32.EnumWindows(WNDENUMPROC(enum_win), 0)

            if found_hwnd[0]:
                self.focus_window(found_hwnd[0])
                return found_hwnd[0]

            ret = proc.poll()
            if ret is not None:
                raise RuntimeError(f"Application terminated prematurely with exit code {ret}.")

            time.sleep(0.1)

        raise TimeoutError(f"Application window did not appear within {timeout_s}s.")
