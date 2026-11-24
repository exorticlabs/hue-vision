"""
Hue Vision - Platform Adapter Registry & Manager
─────────────────────────────────────────────────
Auto-detects the host operating system (Windows, macOS, Linux) or connected mobile
devices (Android via ADB), routing perception and interaction to the optimal adapter.
"""

import os
import sys
from typing import Optional, Dict

from hue_vision.platforms.base import PlatformAdapter
from hue_vision.platforms.windows import WindowsAdapter
from hue_vision.platforms.macos import MacOSAdapter
from hue_vision.platforms.linux import LinuxAdapter
from hue_vision.platforms.android import AndroidAdapter
from hue_vision.platforms.universal import UniversalCVAdapter


class PlatformManager:
    """Manages platform adapters for multi-device desktop and mobile automation."""

    _instance: Optional["PlatformManager"] = None

    def __init__(self):
        self._adapters: Dict[str, PlatformAdapter] = {}
        self._default_adapter: Optional[PlatformAdapter] = None
        self._init_adapters()

    def _init_adapters(self):
        # Always register all platform adapter classes
        win_adapter = WindowsAdapter()
        mac_adapter = MacOSAdapter()
        lin_adapter = LinuxAdapter()
        android_adapter = AndroidAdapter()
        cv_adapter = UniversalCVAdapter()

        self._adapters["windows"] = win_adapter
        self._adapters["win"] = win_adapter
        self._adapters["macos"] = mac_adapter
        self._adapters["mac"] = mac_adapter
        self._adapters["darwin"] = mac_adapter
        self._adapters["linux"] = lin_adapter
        self._adapters["android"] = android_adapter
        self._adapters["adb"] = android_adapter
        self._adapters["mobile"] = android_adapter
        self._adapters["universal"] = cv_adapter
        self._adapters["cv"] = cv_adapter

        # Set host default adapter
        if os.name == "nt":
            self._default_adapter = win_adapter
        elif sys.platform == "darwin":
            self._default_adapter = mac_adapter
        elif sys.platform.startswith("linux"):
            self._default_adapter = lin_adapter
        else:
            self._default_adapter = cv_adapter

    @classmethod
    def get_instance(cls) -> "PlatformManager":
        if cls._instance is None:
            cls._instance = PlatformManager()
        return cls._instance

    def get_adapter(self, target: Optional[str] = None) -> PlatformAdapter:
        """
        Resolves the appropriate PlatformAdapter for a given device target.
        target: 'device' (default host OS), 'windows', 'macos', 'linux', 'android', 'universal'
        """
        if not target or target in ("device", "host", "desktop", "active_window", "window"):
            return self._default_adapter

        t_lower = target.lower().strip()
        if t_lower in self._adapters:
            return self._adapters[t_lower]

        # Check if target is an Android device serial
        if t_lower.startswith("adb:") or t_lower.startswith("emulator-") or re_is_device_serial(t_lower):
            serial = t_lower.split(":", 1)[1] if t_lower.startswith("adb:") else t_lower
            return AndroidAdapter(device_serial=serial)

        return self._default_adapter


def re_is_device_serial(s: str) -> bool:
    # Typical Android device serials: e.g. emulator-5554, 192.168.1.10:5555, R58M45...
    return ":" in s and s.split(":")[-1].isdigit()


__all__ = [
    "PlatformAdapter",
    "WindowsAdapter",
    "MacOSAdapter",
    "LinuxAdapter",
    "AndroidAdapter",
    "UniversalCVAdapter",
    "PlatformManager",
]
