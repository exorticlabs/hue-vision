"""
Hue Vision - Native Desktop & Device Process / GUI Lifecycle Manager
───────────────────────────────────────────────────────────────────
Guarantees robust process launch, GUI readiness verification,
profile lock / early crash detection, and message pump settling.
Supports Windows, macOS, Linux, and Android.
"""

import os
import sys
import time
from typing import List, Optional, Dict, Any

from hue_vision.platforms import PlatformManager


class ProcessLifecycleManager:
    """Manages application lifecycles and guarantees GUI readiness before agent interaction."""

    @staticmethod
    def launch_and_wait_gui(
        executable: str,
        args: Optional[List[str]] = None,
        expected_title: str = "",
        timeout_s: float = 10.0,
        target: Optional[str] = None
    ) -> int:
        """
        Spawns an application on Windows, macOS, Linux, or Android and synchronously polls
        until an active, visible GUI window is registered with the desktop compositor or Android window manager.
        """
        adapter = PlatformManager.get_instance().get_adapter(target)
        return adapter.launch_and_wait_gui(
            executable=executable,
            args=args,
            expected_title=expected_title,
            timeout_s=timeout_s
        )
