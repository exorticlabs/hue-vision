"""
Hue Vision: Live Multi-Device Display Understanding & Interaction Engine for AI Agents & LLMs.
───────────────────────────────────────────────────────────────────────────────────────────────
Provides real-time display perception, structured screen context synthesis,
continuous visual delta tracking, and causally-verified GUI interactions for any
agent framework across Windows, macOS, Linux, and Android.
"""

__version__ = "1.0.0"
__author__ = "Exortic Research Lab"

from hue_vision.model import VisionModelManager
from hue_vision.context_builder import ScreenContext, ContextBuilder, UIElement, UIGroup
from hue_vision.tracker import LiveScreenTracker, ScreenDelta
from hue_vision.interact import CausalInteractionEngine, InteractionResult
from hue_vision.engine import VisionEngine
from hue_vision.platforms import PlatformManager
from hue_vision.dom_extractor import DOMExtractor
from hue_vision.cv_detector import CVDetector
from hue_vision.lifecycle import ProcessLifecycleManager

__all__ = [
    "VisionEngine",
    "ScreenContext",
    "ContextBuilder",
    "UIElement",
    "UIGroup",
    "LiveScreenTracker",
    "ScreenDelta",
    "CausalInteractionEngine",
    "InteractionResult",
    "VisionModelManager",
    "PlatformManager",
    "DOMExtractor",
    "CVDetector",
    "ProcessLifecycleManager",
]
