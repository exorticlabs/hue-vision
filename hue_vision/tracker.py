"""
Hue Vision - Live Screen Tracker & Delta Observer
─────────────────────────────────────────────────
Tracks continuous display changes, window transitions, modal popups, and element mutations.
Provides agents with concise, causally-meaningful explanations of WHAT changed on screen.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from hue_vision.context_builder import ScreenContext, UIElement, WindowContext


@dataclass
class ElementMutation:
    """Represents a specific change in an existing UI element."""
    element_id: int
    element_type: str
    change_type: str  # TEXT_CHANGED, STATE_CHANGED, POSITION_CHANGED
    old_value: str
    new_value: str
    element: UIElement

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": self.element_id,
            "type": self.element_type,
            "change_type": self.change_type,
            "old_value": self.old_value,
            "new_value": self.new_value
        }


@dataclass
class ScreenDelta:
    """The computed difference between two consecutive display states."""
    timestamp: float
    has_changed: bool
    window_changed: bool
    old_window: Optional[WindowContext]
    new_window: WindowContext
    appeared_elements: List[UIElement] = field(default_factory=list)
    disappeared_elements: List[UIElement] = field(default_factory=list)
    mutations: List[ElementMutation] = field(default_factory=list)
    focus_shifted: bool = False
    old_focus: Optional[UIElement] = None
    new_focus: Optional[UIElement] = None
    modal_opened: bool = False
    modal_closed: bool = False
    summary: str = "No detectable visual changes."

    def to_markdown(self) -> str:
        """Renders an agent-ready summary of what changed on the screen."""
        if not self.has_changed:
            return "### Display State: Stable (No changes detected)"

        lines = [f"### Display Change Detected [{time.strftime('%H:%M:%S', time.localtime(self.timestamp))}]"]

        if self.window_changed:
            lines.append(f"- **Window Focus Changed**: \"{self.new_window.title}\" ({self.new_window.process_name})")

        if self.modal_opened:
            lines.append("- **New Modal / Dialog Opened**")
        elif self.modal_closed:
            lines.append("- **Modal / Dialog Dismissed**")

        if self.focus_shifted:
            new_f_desc = f"#{self.new_focus.element_id} (\"{self.new_focus.label}\")" if self.new_focus else "None"
            lines.append(f"- **Keyboard Focus Shifted To**: {new_f_desc}")

        if self.appeared_elements:
            lines.append(f"- **Appeared Elements ({len(self.appeared_elements)})**:")
            for el in self.appeared_elements[:8]:
                lbl = f"\"{el.label}\"" if el.label else el.element_type
                lines.append(f"  + #{el.element_id} [{el.element_type}]: {lbl} @ center {el.center}")
            if len(self.appeared_elements) > 8:
                lines.append(f"  + ... and {len(self.appeared_elements) - 8} more")

        if self.disappeared_elements:
            lines.append(f"- **Disappeared / Dismissed ({len(self.disappeared_elements)})**:")
            for el in self.disappeared_elements[:5]:
                lbl = f"\"{el.label}\"" if el.label else el.element_type
                lines.append(f"  - [{el.element_type}]: {lbl}")

        if self.mutations:
            lines.append(f"- **Mutated Elements ({len(self.mutations)})**:")
            for mut in self.mutations[:6]:
                lines.append(f"  * #{mut.element_id} [{mut.element_type}]: {mut.change_type} ('{mut.old_value}' -> '{mut.new_value}')")

        lines.append(f"\n**Summary**: {self.summary}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "has_changed": self.has_changed,
            "window_changed": self.window_changed,
            "modal_opened": self.modal_opened,
            "modal_closed": self.modal_closed,
            "focus_shifted": self.focus_shifted,
            "appeared_count": len(self.appeared_elements),
            "disappeared_count": len(self.disappeared_elements),
            "mutations_count": len(self.mutations),
            "summary": self.summary,
            "appeared": [el.to_dict() for el in self.appeared_elements],
            "disappeared": [el.to_dict() for el in self.disappeared_elements],
            "mutations": [m.to_dict() for m in self.mutations]
        }


class LiveScreenTracker:
    """Maintains display history and evaluates delta between consecutive observations."""

    def __init__(self, iou_match_threshold: float = 0.70):
        self.iou_threshold = iou_match_threshold
        self.previous_context: Optional[ScreenContext] = None

    def reset(self) -> None:
        """Clears tracking memory."""
        self.previous_context = None

    def update(self, current: ScreenContext) -> ScreenDelta:
        """
        Calculates the delta between previous_context and current.
        Updates internal history and returns a ScreenDelta.
        """
        now = time.time()
        if self.previous_context is None:
            self.previous_context = current
            return ScreenDelta(
                timestamp=now,
                has_changed=False,
                window_changed=False,
                old_window=None,
                new_window=current.window,
                summary="Initial screen baseline captured."
            )

        prev = self.previous_context
        self.previous_context = current

        # Check window change
        win_changed = (
            prev.window.title != current.window.title or
            prev.window.pid != current.window.pid
        )

        modal_opened = not prev.has_modal_dialog and current.has_modal_dialog
        modal_closed = prev.has_modal_dialog and not current.has_modal_dialog

        # Check focus shift
        focus_shifted = prev.focused_element_id != current.focused_element_id
        old_f = prev.get_element(prev.focused_element_id) if prev.focused_element_id else None
        new_f = current.get_element(current.focused_element_id) if current.focused_element_id else None

        # Element matching by IoU & label (Hysteresis Spatial Matching)
        matched_prev = set()
        matched_curr = set()
        mutations: List[ElementMutation] = []

        curr_interactive = current.get_interactive_elements()
        prev_interactive = prev.get_interactive_elements()

        # Phase 1: High-confidence identity matching (IoU >= threshold)
        for c_el in curr_interactive:
            best_iou = 0.0
            best_prev: Optional[UIElement] = None

            for p_el in prev_interactive:
                if p_el.element_id in matched_prev:
                    continue
                iou = self._calc_iou(c_el.bbox, p_el.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_prev = p_el

            if best_prev and best_iou >= self.iou_threshold:
                matched_prev.add(best_prev.element_id)
                matched_curr.add(c_el.element_id)

                # Check for mutations
                if best_prev.label != c_el.label and (best_prev.label or c_el.label):
                    mutations.append(ElementMutation(
                        element_id=c_el.element_id,
                        element_type=c_el.element_type,
                        change_type="TEXT_CHANGED",
                        old_value=best_prev.label,
                        new_value=c_el.label,
                        element=c_el
                    ))
                elif best_prev.dynamic_state != c_el.dynamic_state:
                    mutations.append(ElementMutation(
                        element_id=c_el.element_id,
                        element_type=c_el.element_type,
                        change_type="STATE_CHANGED",
                        old_value=best_prev.dynamic_state,
                        new_value=c_el.dynamic_state,
                        element=c_el
                    ))

        # Phase 2: Moderate overlap matching (0.20 <= IoU < threshold) -> POSITION_CHANGED
        for c_el in curr_interactive:
            if c_el.element_id in matched_curr:
                continue
            best_iou = 0.0
            best_prev: Optional[UIElement] = None
            for p_el in prev_interactive:
                if p_el.element_id in matched_prev:
                    continue
                iou = self._calc_iou(c_el.bbox, p_el.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_prev = p_el

            if best_prev and best_iou >= 0.20:
                matched_prev.add(best_prev.element_id)
                matched_curr.add(c_el.element_id)
                mutations.append(ElementMutation(
                    element_id=c_el.element_id,
                    element_type=c_el.element_type,
                    change_type="POSITION_CHANGED",
                    old_value=f"bbox:{best_prev.bbox}",
                    new_value=f"bbox:{c_el.bbox}",
                    element=c_el
                ))

        # Phase 3: Only elements with max IoU < 0.20 are APPEARED / DISAPPEARED
        appeared = []
        for c_el in curr_interactive:
            if c_el.element_id in matched_curr:
                continue
            # Check max overlap with ANY element in prev
            max_ov = max((self._calc_iou(c_el.bbox, p.bbox) for p in prev_interactive), default=0.0)
            if max_ov < 0.20:
                appeared.append(c_el)

        disappeared = []
        for p_el in prev_interactive:
            if p_el.element_id in matched_prev:
                continue
            # Check max overlap with ANY element in curr
            max_ov = max((self._calc_iou(p_el.bbox, c.bbox) for c in curr_interactive), default=0.0)
            if max_ov < 0.20:
                disappeared.append(p_el)

        has_changed = (
            win_changed or
            modal_opened or
            modal_closed or
            focus_shifted or
            bool(appeared) or
            bool(disappeared) or
            bool(mutations)
        )

        # Generate human-readable summary
        summary_parts = []
        if win_changed:
            summary_parts.append(f"Window changed to '{current.window.title}'.")
        if modal_opened:
            summary_parts.append("Modal dialog appeared.")
        if modal_closed:
            summary_parts.append("Modal dialog closed.")
        if appeared:
            summary_parts.append(f"{len(appeared)} new element(s) appeared.")
        if disappeared:
            summary_parts.append(f"{len(disappeared)} element(s) closed.")
        if mutations:
            summary_parts.append(f"{len(mutations)} element(s) mutated.")

        summary = " ".join(summary_parts) if summary_parts else "Display state remained unchanged."

        return ScreenDelta(
            timestamp=now,
            has_changed=has_changed,
            window_changed=win_changed,
            old_window=prev.window,
            new_window=current.window,
            appeared_elements=appeared,
            disappeared_elements=disappeared,
            mutations=mutations,
            focus_shifted=focus_shifted,
            old_focus=old_f,
            new_focus=new_f,
            modal_opened=modal_opened,
            modal_closed=modal_closed,
            summary=summary
        )

    @staticmethod
    def _calc_iou(boxA: List[int], boxB: List[int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        return inter / float(areaA + areaB - inter)
