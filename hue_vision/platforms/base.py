"""
Hue Vision - Abstract Platform & Device Adapter
───────────────────────────────────────────────
Defines the unified interface for multi-device screen capture,
window management, structural hierarchy extraction, and causal GUI interaction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image


class PlatformAdapter(ABC):
    """Abstract base class for all OS and device automation backends."""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Name of the platform (e.g. 'windows', 'macos', 'linux', 'android')."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if this platform adapter is usable on the current host/environment."""
        pass

    @abstractmethod
    def capture_screen(self) -> Tuple[Image.Image, Tuple[int, int]]:
        """Captures the full display. Returns (PIL.Image, (width, height))."""
        pass

    @abstractmethod
    def capture_window(self, target_handle: Any) -> Optional[Image.Image]:
        """Captures a specific window buffer, even if occluded."""
        pass

    @abstractmethod
    def get_active_window(self) -> Dict[str, Any]:
        """Returns metadata for the currently active/focused window."""
        pass

    @abstractmethod
    def enumerate_windows(self) -> List[Dict[str, Any]]:
        """Enumerates visible top-level windows/applications on the display."""
        pass

    @abstractmethod
    def focus_window(self, target_handle: Any) -> bool:
        """Brings the specified window to the foreground and focuses it."""
        pass

    @abstractmethod
    def extract_hierarchy(
        self,
        target_handle: Any = None,
        max_elements: int = 150,
        max_depth: int = 10
    ) -> Tuple[List[Any], List[Any]]:
        """
        Extracts structural groups and actionable UI leaf elements.
        Returns (groups, elements).
        """
        pass

    @abstractmethod
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
        """Executes a low-level native input event on the device."""
        pass

    @abstractmethod
    def launch_and_wait_gui(
        self,
        executable: str,
        args: Optional[List[str]] = None,
        expected_title: str = "",
        timeout_s: float = 10.0
    ) -> int:
        """Spawns an application and verifies GUI readiness."""
        pass
