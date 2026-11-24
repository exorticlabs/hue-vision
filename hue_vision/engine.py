"""
Hue Vision - Hybrid Live Display Understanding Engine (LDUE Runtime)
─────────────────────────────────────────────────────────────────────
Full-device system display perception and grounded agent interaction across
Windows, macOS, Linux, and Android.

Features:
  1. Multi-device support: Auto-adapts to host OS (Win/Mac/Linux) or Android ADB device.
  2. Sub-30ms screen perception cycles with persistent compositor capture.
  3. Structural DOM + UIAutomation hierarchy fusion with pure CV edge morphology fallback.
  4. UnifiedVisionNet deep multi-task inference for element classification & CTC text.
  5. Causal interaction engine with verified pre/post delta state tracking.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np

from hue_vision.model import VisionModelManager
from hue_vision.dom_extractor import DOMExtractor, RawDOMGroup, RawDOMElement
from hue_vision.cv_detector import CVDetector
from hue_vision.context_builder import ContextBuilder, ScreenContext, UIElement, UIGroup
from hue_vision.tracker import LiveScreenTracker, ScreenDelta
from hue_vision.interact import CausalInteractionEngine, InteractionResult, force_foreground_window
from hue_vision.platforms import PlatformManager


def _iou(a: List[int], b: List[int]) -> float:
    """Compute Intersection-over-Union of two [x1,y1,x2,y2] bounding boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + ab - inter)


class VisionEngine:
    """The central runtime for live hybrid screen understanding and agent interaction across all devices."""

    _instance: Optional["VisionEngine"] = None

    def __init__(self, weights_path: Optional[str] = None):
        self.platform_manager = PlatformManager.get_instance()
        self.model_manager = VisionModelManager.get_instance(weights_path=weights_path)
        self.dom_extractor = DOMExtractor()
        self.tracker = LiveScreenTracker()
        self.last_context: Optional[ScreenContext] = None
        self._perception_mode: str = "hybrid"
        self.interact_engine = CausalInteractionEngine(
            capture_fn=self.capture_screen_context,
            tracker=self.tracker,
            context_provider=lambda: self.last_context,
            platform_manager=self.platform_manager
        )

    @classmethod
    def get_instance(cls, weights_path: Optional[str] = None) -> "VisionEngine":
        if cls._instance is None:
            cls._instance = VisionEngine(weights_path=weights_path)
        return cls._instance

    # ── Display Capture & Window Management ───────────────────────────────────

    def capture_screenshot(self, target: Optional[str] = None) -> Tuple[Image.Image, Tuple[int, int]]:
        """Captures display through the appropriate platform adapter."""
        adapter = self.platform_manager.get_adapter(target)
        return adapter.capture_screen()

    def _enumerate_all_windows(self, target: Optional[str] = None) -> List[Dict[str, Any]]:
        """Enumerates visible top-level windows through the platform adapter."""
        adapter = self.platform_manager.get_adapter(target)
        return adapter.enumerate_windows()

    def get_active_window_info(self, target: Optional[str] = None) -> Dict[str, Any]:
        """Returns metadata for the current foreground window."""
        adapter = self.platform_manager.get_adapter(target)
        return adapter.get_active_window()

    def capture_window_buffer(self, hwnd: Any, target: Optional[str] = None) -> Optional[np.ndarray]:
        """Captures exact window visual contents directly from application buffer."""
        adapter = self.platform_manager.get_adapter(target)
        img = adapter.capture_window(hwnd)
        if img is not None:
            return np.array(img)
        return None

    def _find_hwnd_by(
        self,
        app_name: Optional[str] = None,
        window_title: Optional[str] = None,
        pid: Optional[int] = None,
        target: Optional[str] = None
    ) -> Optional[Any]:
        """Find an HWND / window handle for an app by name, title or pid."""
        if not any([app_name, window_title, pid]):
            return None
        candidates = []
        for w in self._enumerate_all_windows(target=target):
            if pid and w.get("pid") == pid:
                candidates.append(w)
            elif window_title and window_title.lower() in w.get("title", "").lower():
                candidates.append(w)
            elif app_name and app_name.lower() in w.get("process_name", "").lower():
                candidates.append(w)
            elif app_name and app_name.lower() in w.get("title", "").lower():
                candidates.append(w)

        if not candidates:
            return None

        for c in candidates:
            if "applicationframehost" in c.get("process_name", "").lower():
                return c["hwnd"]

        candidates.sort(
            key=lambda c: (c["rect"][2] - c["rect"][0]) * (c["rect"][3] - c["rect"][1]),
            reverse=True
        )
        return candidates[0]["hwnd"]

    # ── Hybrid Spatial Coverage Check with Schmitt Trigger Hysteresis ─────────

    def _decide_mode(
        self,
        raw_elements: list,
        win_rect: List[int],
        tau_low: float = 0.40,
        tau_high: float = 0.60,
        min_interactive: int = 8,
    ) -> str:
        """
        Schmitt-Trigger dual-threshold state machine to prevent mode oscillation.
        Hysteresis zone: [tau_low=0.40, tau_high=0.60].
        """
        if not raw_elements:
            self._perception_mode = "pure_cv"
            return "pure_cv"

        win_area = max(1, (win_rect[2] - win_rect[0]) * (win_rect[3] - win_rect[1]))
        uia_area = 0
        interactive_count = 0
        for el in raw_elements:
            b = el.bbox if hasattr(el, "bbox") else el.get("bbox", [0, 0, 0, 0])
            uia_area += max(0, (b[2] - b[0]) * (b[3] - b[1]))
            is_inter = el.is_interactive if hasattr(el, "is_interactive") else el.get("is_interactive", False)
            if is_inter:
                interactive_count += 1

        coverage = uia_area / win_area

        if coverage < tau_low or interactive_count < min_interactive:
            self._perception_mode = "hybrid_cv"
        elif coverage > tau_high and interactive_count >= min_interactive:
            self._perception_mode = "hybrid"

        return self._perception_mode

    # ── Core Perception ───────────────────────────────────────────────────────

    def capture_screen_context(
        self,
        target: str = "device",
        app_name: Optional[str] = None,
        window_title: Optional[str] = None,
        pid: Optional[int] = None,
    ) -> ScreenContext:
        """
        High-speed multi-device perception cycle across Windows, macOS, Linux, and Android.

        target:
          "device"        — full system desktop (default).
          "active_window" — foreground window only.
          "window"        — specific window identified by app_name / window_title / pid.
          "android"       — connected Android device / emulator.
          "macos"         — macOS desktop.
          "linux"         — Linux X11/Wayland desktop.
          "universal"     — pure computer vision fallback.
        """
        t_start = time.perf_counter()

        # Step 1: Desktop topology
        all_windows = self._enumerate_all_windows(target=target)

        # Step 2: Resolve target HWND and window metadata
        target_hwnd: Optional[Any] = None
        win_meta: Dict[str, Any] = {
            "title": "Full Desktop",
            "process_name": "system",
            "pid": 0,
            "rect": [0, 0, 1920, 1080],
            "is_maximized": True,
            "hwnd": 0,
            "desktop_windows": all_windows,
        }

        if target == "active_window":
            win_meta = self.get_active_window_info(target=target)
            win_meta["desktop_windows"] = all_windows
            target_hwnd = win_meta.get("hwnd") or None
            if target_hwnd == 0:
                target_hwnd = None

        elif target == "window" or any([app_name, window_title, pid]):
            hwnd = self._find_hwnd_by(app_name=app_name, window_title=window_title, pid=pid, target=target)
            if hwnd is not None and hwnd != 0:
                target_hwnd = hwnd
                for w in all_windows:
                    if w["hwnd"] == hwnd:
                        win_meta = {
                            "title": w["title"],
                            "process_name": w["process_name"],
                            "pid": w["pid"],
                            "rect": w["rect"],
                            "is_maximized": False,
                            "hwnd": hwnd,
                            "desktop_windows": all_windows,
                        }
                        break

        # Step 3: Surface Acquisition
        used_buffer = False
        if target_hwnd and target_hwnd != 0 and target not in ("android", "mobile"):
            activated = force_foreground_window(target_hwnd, target=target)
            if not activated:
                buf_rgb = self.capture_window_buffer(target_hwnd, target=target)
                if buf_rgb is not None:
                    img = Image.fromarray(buf_rgb)
                    screen_size = img.size
                    used_buffer = True

        if not used_buffer:
            img, screen_size = self.capture_screenshot(target=target)

        img_w, img_h = screen_size
        if win_meta["rect"] == [0, 0, 1920, 1080]:
            win_meta["rect"] = [0, 0, img_w, img_h]

        dom_hwnd = target_hwnd if (target_hwnd is not None and target_hwnd != 0) else 0

        # Step 4: Multi-surface hierarchy extraction
        raw_groups, raw_elements = self.dom_extractor.extract_hierarchy(
            target_hwnd=dom_hwnd, max_elements=150, max_depth=10, target=target
        )

        w_rect = win_meta["rect"]

        # Step 5: Schmitt-trigger spatial coverage check
        extraction_source = self._decide_mode(raw_elements, w_rect)
        if extraction_source in ("hybrid_cv", "pure_cv"):
            cv_boxes = CVDetector.detect_visual_candidates(img, w_rect, min_size=12, max_boxes=60)

            # Merge CV candidates that don't overlap existing DOM elements
            dom_bboxes = [
                (el.bbox if hasattr(el, "bbox") else el.get("bbox", [0, 0, 0, 0]))
                for el in raw_elements
            ]
            new_cv_elements = []
            start_id = len(raw_elements) + 1
            for b_idx, cb in enumerate(cv_boxes):
                cb_bbox = cb["bbox"]
                overlaps = any(_iou(cb_bbox, ub) > 0.35 for ub in dom_bboxes)
                if not overlaps:
                    new_cv_elements.append({
                        "element_id": start_id + b_idx,
                        "name": "",
                        "text": "",
                        "control_type": cb.get("control_type", "CustomControl"),
                        "bbox": cb_bbox,
                        "is_interactive": True,
                        "is_focused": False,
                        "is_enabled": True,
                        "parent_group_id": "grp_cv_main"
                    })

            if new_cv_elements:
                raw_elements = list(raw_elements) + new_cv_elements
                cv_grp_exists = any(
                    (g.group_id if hasattr(g, "group_id") else g.get("group_id", "")) == "grp_cv_main"
                    for g in raw_groups
                )
                if not cv_grp_exists:
                    raw_groups = list(raw_groups) + [{
                        "group_id": "grp_cv_main",
                        "group_type": "MAIN_CONTENT",
                        "title": "Visual Canvas (CV Detected)",
                        "bbox": w_rect,
                        "element_ids": [el["element_id"] for el in new_cv_elements]
                    }]

            if not raw_elements:
                extraction_source = "pure_cv"
                raw_elements = []
                for b_idx, cb in enumerate(cv_boxes, 1):
                    raw_elements.append({
                        "element_id": b_idx,
                        "name": "",
                        "text": "",
                        "control_type": cb.get("control_type", "CustomControl"),
                        "bbox": cb["bbox"],
                        "is_interactive": True,
                        "is_focused": False,
                        "is_enabled": True,
                        "parent_group_id": "grp_cv_main"
                    })
                raw_groups = [{
                    "group_id": "grp_cv_main",
                    "group_type": "MAIN_CONTENT",
                    "title": "Visual Canvas",
                    "bbox": w_rect,
                    "element_ids": [el["element_id"] for el in raw_elements]
                }]

        # Step 6: Extract visual crops for model inference
        crops: List[Image.Image] = []
        valid_indices: List[int] = []

        for idx, el in enumerate(raw_elements):
            b = el.bbox if hasattr(el, "bbox") else el.get("bbox", [0, 0, 0, 0])
            bx1 = max(0, b[0])
            by1 = max(0, b[1])
            bx2 = min(img_w, b[2])
            by2 = min(img_h, b[3])
            if bx2 - bx1 >= 6 and by2 - by1 >= 6:
                crop = img.crop((bx1, by1, bx2, by2))
                crops.append(crop)
                valid_indices.append(idx)

        # Step 7: Batched neural inference
        preds = self.model_manager.predict_crops(crops)
        full_preds: List[Dict] = [{} for _ in raw_elements]
        for v_idx, p in zip(valid_indices, preds):
            full_preds[v_idx] = p

        # Step 8: Assemble ScreenContext
        context = ContextBuilder.build_hybrid_context(
            window_meta=win_meta,
            screen_size=screen_size,
            raw_groups=raw_groups,
            raw_elements=raw_elements,
            model_predictions=full_preds,
            source=extraction_source
        )

        self.last_context = context
        return context

    def perceive(
        self,
        target: str = "device",
        grouped: bool = True,
        interactive_only: bool = True,
        max_elements: int = 50,
        app_name: Optional[str] = None,
        window_title: Optional[str] = None,
        pid: Optional[int] = None,
    ) -> str:
        """Returns ready-to-inject Markdown context for LLM agents."""
        ctx = self.capture_screen_context(
            target=target, app_name=app_name, window_title=window_title, pid=pid
        )
        return ctx.to_markdown(grouped=grouped, interactive_only=interactive_only, max_elements=max_elements)

    def track(self, timeout_s: float = 2.0, poll_interval_s: float = 0.25, target: str = "device") -> ScreenDelta:
        """Polls until a visual change occurs or timeout expires."""
        start = time.time()
        if self.last_context is None:
            self.last_context = self.capture_screen_context(target=target)
            self.tracker.update(self.last_context)

        while time.time() - start < timeout_s:
            time.sleep(poll_interval_s)
            curr = self.capture_screen_context(target=target)
            delta = self.tracker.update(curr)
            if delta.has_changed:
                return delta

        curr = self.capture_screen_context(target=target)
        return self.tracker.update(curr)

    def interact(
        self,
        action: str,
        element_id: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        text: Optional[str] = None,
        press_enter: bool = False,
        clear_first: bool = False,
        direction: str = "down",
        amount: int = 3,
        keys: Optional[List[str]] = None,
        target: Optional[str] = None
    ) -> InteractionResult:
        """Executes a UI interaction and causally verifies its visual impact on any device."""
        return self.interact_engine.execute_action(
            action=action,
            element_id=element_id,
            x=x,
            y=y,
            text=text,
            press_enter=press_enter,
            clear_first=clear_first,
            direction=direction,
            amount=amount,
            keys=keys,
            target=target
        )
