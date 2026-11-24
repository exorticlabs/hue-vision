"""
Hue Vision - Causal UI Interaction & Verification Engine
────────────────────────────────────────────────────────
Executes high-precision mouse, keyboard, and touch actions across Windows, macOS, Linux, and Android,
immediately observes the post-action display state, and returns a verified causal report
explaining exactly WHAT changed as a result of the agent's action.
"""

import time
import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Callable

from hue_vision.context_builder import ScreenContext, UIElement
from hue_vision.tracker import ScreenDelta, LiveScreenTracker
from hue_vision.platforms import PlatformManager


@dataclass
class InteractionResult:
    """Detailed causal outcome of a UI action executed by an agent."""
    ok: bool
    action: str
    target_element_id: Optional[int]
    target_label: Optional[str]
    position: Tuple[int, int]
    visual_change_detected: bool
    causal_summary: str
    delta: Optional[ScreenDelta] = None
    error: Optional[str] = None

    def to_markdown(self) -> str:
        status_icon = "[OK]" if self.ok else "[FAILED]"
        lines = [f"### Action Result: {status_icon} {self.action.upper()}"]
        if self.target_label:
            lines.append(f"- **Target Element**: #{self.target_element_id} (\"{self.target_label}\") @ {self.position}")
        else:
            lines.append(f"- **Target Coordinates**: {self.position}")

        lines.append(f"- **Visual Change Detected**: {'Yes' if self.visual_change_detected else 'No'}")
        lines.append(f"- **Causal Verification**: {self.causal_summary}")

        if self.delta and self.delta.has_changed:
            lines.append("")
            lines.append(self.delta.to_markdown())

        if self.error:
            lines.append(f"- **Error Details**: {self.error}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "target_element_id": self.target_element_id,
            "target_label": self.target_label,
            "position": list(self.position),
            "visual_change_detected": self.visual_change_detected,
            "causal_summary": self.causal_summary,
            "error": self.error,
            "delta": self.delta.to_dict() if self.delta else None
        }


def force_foreground_window(hwnd: Any, target: Optional[str] = None) -> bool:
    """Brings window to top of Z-order and focuses it via PlatformManager."""
    return PlatformManager.get_instance().get_adapter(target).focus_window(hwnd)


class CausalInteractionEngine:
    """Executes multi-device UI actions with pre- and post-condition delta verification."""

    def __init__(
        self,
        capture_fn: Callable[..., ScreenContext],
        tracker: Optional[LiveScreenTracker] = None,
        settle_delay_s: float = 0.20,
        context_provider: Optional[Callable[[], Optional[ScreenContext]]] = None,
        platform_manager: Optional[PlatformManager] = None
    ):
        self.capture_fn = capture_fn
        self.tracker = tracker or LiveScreenTracker()
        self.settle_delay_s = settle_delay_s
        self.context_provider = context_provider
        self.platform_manager = platform_manager or PlatformManager.get_instance()

    def _adjust_coordinates_for_window_movement(
        self,
        target_el: UIElement,
        pre_context: ScreenContext,
        default_x: int,
        default_y: int,
        target: Optional[str] = None
    ) -> Tuple[int, int]:
        """Sub-millisecond coordinate self-correction if target window shifted."""
        if os.name != "nt":
            return default_x, default_y
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            hwnd = getattr(pre_context.window, "hwnd", 0)
            if not hwnd or hwnd == 0:
                class POINT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
                pt = POINT(int(default_x), int(default_y))
                raw_hwnd = user32.WindowFromPoint(pt)
                hwnd = user32.GetAncestor(raw_hwnd, 2) or raw_hwnd

            if hwnd and user32.IsWindow(hwnd):
                curr_rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(curr_rect))
                cached_rect = pre_context.window.rect
                if cached_rect and len(cached_rect) == 4:
                    dx = curr_rect.left - cached_rect[0]
                    dy = curr_rect.top - cached_rect[1]
                    if dx != 0 or dy != 0:
                        return default_x + dx, default_y + dy
        except Exception:
            pass
        return default_x, default_y

    def execute_action(
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
        """
        Executes a UI interaction and performs causal verification across any device.
        Actions supported: 'click', 'double_click', 'right_click', 'type', 'scroll', 'hover', 'hotkey'.
        """
        # 1. Obtain Baseline Pre-Action Display State
        pre_context: Optional[ScreenContext] = None
        if self.context_provider:
            pre_context = self.context_provider()
        if pre_context is None:
            pre_context = self.capture_fn(target=target or "device")

        self.tracker.update(pre_context)

        # 2. Resolve Target Coordinates & Element
        target_el: Optional[UIElement] = None
        target_x, target_y = x, y
        target_lbl = None

        if element_id is not None:
            target_el = pre_context.get_element(element_id)
            if not target_el:
                # Refresh context once if element not found in current cached snapshot
                pre_context = self.capture_fn(target=target or "device")
                self.tracker.update(pre_context)
                target_el = pre_context.get_element(element_id)

            if not target_el:
                return InteractionResult(
                    ok=False,
                    action=action,
                    target_element_id=element_id,
                    target_label=None,
                    position=(0, 0),
                    visual_change_detected=False,
                    causal_summary=f"Element #{element_id} was not found on screen.",
                    error=f"Element #{element_id} not found."
                )

            target_lbl = target_el.label
            target_x, target_y = target_el.center

            # 3. Pre-action coordinate self-correction if window moved
            target_x, target_y = self._adjust_coordinates_for_window_movement(
                target_el=target_el,
                pre_context=pre_context,
                default_x=target_x,
                default_y=target_y,
                target=target
            )

        if target_x is None or target_y is None:
            if action in ["hotkey", "scroll"] and target_x is None:
                target_x = int(pre_context.screen_size[0] / 2)
                target_y = int(pre_context.screen_size[1] / 2)
            else:
                return InteractionResult(
                    ok=False,
                    action=action,
                    target_element_id=element_id,
                    target_label=target_lbl,
                    position=(0, 0),
                    visual_change_detected=False,
                    causal_summary="No target coordinates or element ID specified.",
                    error="Coordinates missing."
                )

        # 4. Ensure window focus if targeting a desktop window
        self._ensure_window_focus(target_x, target_y, target=target)

        # 5. Perform the native OS / Device Action via PlatformAdapter
        exec_err = None
        try:
            adapter = self.platform_manager.get_adapter(target)
            adapter.execute_action(
                action=action,
                x=target_x,
                y=target_y,
                text=text,
                press_enter=press_enter,
                clear_first=clear_first,
                direction=direction,
                amount=amount,
                keys=keys
            )
        except Exception as e:
            exec_err = str(e)

        if exec_err:
            return InteractionResult(
                ok=False,
                action=action,
                target_element_id=element_id,
                target_label=target_lbl,
                position=(target_x, target_y),
                visual_change_detected=False,
                causal_summary=f"Execution error while dispatching '{action}': {exec_err}",
                error=exec_err
            )

        # 6. Wait for UI thread / compositor to settle
        time.sleep(self.settle_delay_s)

        # 7. Capture Post-Action Display State matching pre_context target
        post_app = None
        if pre_context.window and pre_context.window.title not in ("Full Desktop", "Desktop Screen"):
            post_app = pre_context.window.title

        try:
            if post_app and target != "android":
                post_context = self.capture_fn(target="window", window_title=post_app)
            else:
                post_context = self.capture_fn(target=target or "device")
        except Exception:
            post_context = self.capture_fn(target=target or "device")

        delta = self.tracker.update(post_context)

        # 8. Evaluate Causal Feedback
        causal_summary = self._synthesize_causality(
            action=action,
            target_lbl=target_lbl,
            element_id=element_id,
            text=text,
            delta=delta
        )

        return InteractionResult(
            ok=True,
            action=action,
            target_element_id=element_id,
            target_label=target_lbl,
            position=(target_x, target_y),
            visual_change_detected=delta.has_changed,
            causal_summary=causal_summary,
            delta=delta
        )

    def _ensure_window_focus(self, x: int, y: int, target: Optional[str] = None) -> None:
        """Focuses the window containing (x, y) if running on Windows."""
        if os.name != "nt" or target == "android":
            return
        try:
            import ctypes
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            user32 = ctypes.windll.user32
            pt = POINT(int(x), int(y))
            target_hwnd = user32.WindowFromPoint(pt)
            if not target_hwnd:
                return

            root_hwnd = user32.GetAncestor(target_hwnd, 2) or target_hwnd
            force_foreground_window(root_hwnd, target=target)
        except Exception:
            pass

    def _synthesize_causality(
        self,
        action: str,
        target_lbl: Optional[str],
        element_id: Optional[int],
        text: Optional[str],
        delta: ScreenDelta
    ) -> str:
        """Constructs an explanatory explanation of the observed causal state transition."""
        target_desc = f"#{element_id} ('{target_lbl}')" if target_lbl else (f"#{element_id}" if element_id else "element")

        if not delta.has_changed:
            return f"Action '{action}' on {target_desc} executed, but produced NO detectable visual state changes after {int(self.settle_delay_s * 1000)}ms."

        explanations = []
        if delta.window_changed:
            explanations.append(f"Switched window to '{delta.new_window.title}'.")
        if delta.modal_opened:
            explanations.append("Modal dialog opened on screen.")
        if delta.modal_closed:
            explanations.append("Modal dialog was closed/dismissed.")
        if delta.focus_shifted:
            new_f = f"#{delta.new_focus.element_id} ('{delta.new_focus.label}')" if delta.new_focus else "None"
            explanations.append(f"Keyboard focus shifted to {new_f}.")
        if delta.appeared_elements:
            explanations.append(f"{len(delta.appeared_elements)} new control(s) rendered.")
        if delta.disappeared_elements:
            explanations.append(f"{len(delta.disappeared_elements)} control(s) disappeared.")
        if delta.mutations:
            explanations.append(f"{len(delta.mutations)} element(s) mutated.")

        return f"Action '{action}' on {target_desc} triggered visual changes: " + " ".join(explanations)
