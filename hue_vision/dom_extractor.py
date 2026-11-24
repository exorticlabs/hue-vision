"""
Hue Vision - Direct DOM, UIAutomation & Device Extractor
────────────────────────────────────────────────────────
Extracts ground-truth text, native controls, states, and structural groups
from live Windows, macOS, Linux, and Android viewports with zero OCR hallucination.
Designed for non-blocking, zero-retry instant execution with immediate CV fallback.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from hue_vision.platforms import PlatformManager


@dataclass
class RawDOMGroup:
    """Represents a structural container group in the UI."""
    group_id: str
    group_type: str  # MODAL_DIALOG, NAVBAR_HEADER, SIDEBAR_PANEL, FORM_GROUP, CONTENT_CARD, MAIN_CONTENT
    title: str
    bbox: List[int]  # [x1, y1, x2, y2]
    element_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "group_type": self.group_type,
            "title": self.title,
            "bbox": self.bbox,
            "element_ids": self.element_ids
        }


@dataclass
class RawDOMElement:
    """Represents an extracted UI leaf element with ground-truth DOM/UIA attributes."""
    element_id: int
    name: str
    text: str
    control_type: str
    bbox: List[int]  # [x1, y1, x2, y2]
    is_interactive: bool
    is_focused: bool
    is_enabled: bool
    parent_group_id: Optional[str] = None
    native_role: Optional[str] = None
    native_attributes: Dict[str, Any] = field(default_factory=dict)
    z_layer: int = 0

    @property
    def center(self) -> Tuple[int, int]:
        return int((self.bbox[0] + self.bbox[2]) / 2), int((self.bbox[1] + self.bbox[3]) / 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": self.element_id,
            "name": self.name,
            "text": self.text,
            "control_type": self.control_type,
            "bbox": self.bbox,
            "center": list(self.center),
            "is_interactive": self.is_interactive,
            "is_focused": self.is_focused,
            "is_enabled": self.is_enabled,
            "parent_group_id": self.parent_group_id,
            "native_role": self.native_role,
            "z_layer": self.z_layer,
            "native_attributes": self.native_attributes
        }


class DOMExtractor:
    """Fast, non-blocking structural hierarchy extractor across all devices."""

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
        self.platform_manager = PlatformManager.get_instance()

    def extract_hierarchy(
        self,
        target_hwnd: Any = 0,
        max_elements: int = 150,
        max_depth: int = 10,
        target: Optional[str] = None
    ) -> Tuple[List[RawDOMGroup], List[RawDOMElement]]:
        """
        Extracts structural groups and actionable leaf controls across Windows, macOS, Linux, or Android.
        Returns immediately with empty lists on error or CV mode so caller switches to pure CV.
        """
        adapter = self.platform_manager.get_adapter(target)
        try:
            return adapter.extract_hierarchy(
                target_handle=target_hwnd,
                max_elements=max_elements,
                max_depth=max_depth
            )
        except Exception:
            return [], []
