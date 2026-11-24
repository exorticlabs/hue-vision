"""
Hue Vision - Hybrid Semantic Context Builder for AI Agents & LLMs
───────────────────────────────────────────────────────────────────
Fuses direct DOM / UIAutomation structural hierarchy and ground-truth text
with deep neural visual semantics (UnifiedVisionNet) into a token-efficient,
group-structured representation optimized for agent reasoning and causal interaction.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class UIElement:
    """Represents a discrete actionable or informational UI element."""
    element_id: int
    element_type: str
    semantic_role: str
    label: str
    bbox: List[int]       # [x1, y1, x2, y2]
    center: Tuple[int, int]
    is_interactive: bool
    dynamic_state: str = "NORMAL"  # NORMAL, HOVER, ACTIVE, FOCUSED, TYPING, DISABLED
    parent_region: str = "MAIN_CONTENT"  # TOP_BAR, SIDEBAR, MAIN_CONTENT, MODAL_DIALOG, BOTTOM_BAR
    parent_group_id: Optional[str] = None
    parent_group_title: Optional[str] = None
    raw_control_type: Optional[str] = None
    is_enabled: bool = True
    confidence: float = 1.0
    z_layer: int = 0
    is_primary_readout: bool = False

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.element_id,
            "type": self.element_type,
            "role": self.semantic_role,
            "label": self.label,
            "bbox": self.bbox,
            "center": list(self.center),
            "is_interactive": self.is_interactive,
            "dynamic_state": self.dynamic_state,
            "region": self.parent_region,
            "group_id": self.parent_group_id,
            "group_title": self.parent_group_title,
            "is_enabled": self.is_enabled,
            "confidence": round(self.confidence, 3),
            "z_layer": self.z_layer,
            "is_primary_readout": self.is_primary_readout
        }


@dataclass
class UIGroup:
    """Represents a structural logical container (Modal, Navbar, Sidebar, Form, Card)."""
    group_id: str
    group_type: str  # MODAL_DIALOG, NAVBAR_HEADER, SIDEBAR_PANEL, FORM_GROUP, CONTENT_CARD, MAIN_CONTENT
    title: str
    bbox: List[int]
    elements: List[UIElement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "group_type": self.group_type,
            "title": self.title,
            "bbox": self.bbox,
            "element_ids": [el.element_id for el in self.elements]
        }


@dataclass
class WindowContext:
    """Active window metadata including full desktop window topology."""
    title: str
    process_name: str
    pid: int
    rect: List[int]  # [left, top, right, bottom]
    is_maximized: bool = False
    hwnd: int = 0
    desktop_windows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "process_name": self.process_name,
            "pid": self.pid,
            "rect": self.rect,
            "is_maximized": self.is_maximized,
            "desktop_windows_count": len(self.desktop_windows),
        }


@dataclass
class ScreenContext:
    """Complete, self-contained semantic snapshot of the display for agents."""
    window: WindowContext
    screen_size: Tuple[int, int]  # (width, height)
    elements: List[UIElement] = field(default_factory=list)
    groups: List[UIGroup] = field(default_factory=list)
    has_modal_dialog: bool = False
    active_modal_title: Optional[str] = None
    focused_element_id: Optional[int] = None
    extraction_source: str = "hybrid"  # "hybrid" or "pure_cv"

    def __post_init__(self):
        self._lookup: Dict[int, UIElement] = {el.element_id: el for el in self.elements}
        self._group_lookup: Dict[str, UIGroup] = {g.group_id: g for g in self.groups}

    def get_element(self, element_id: int) -> Optional[UIElement]:
        """Fast O(1) lookup by element ID."""
        return self._lookup.get(element_id)

    def get_group(self, group_id: str) -> Optional[UIGroup]:
        """Fast O(1) lookup by group ID."""
        return self._group_lookup.get(group_id)

    def find_by_label(self, query: str, case_sensitive: bool = False) -> List[UIElement]:
        """Finds elements whose text or label contains query."""
        results = []
        q = query if case_sensitive else query.lower()
        for el in self.elements:
            lbl = el.label if case_sensitive else el.label.lower()
            if q in lbl:
                results.append(el)
        return results

    def get_interactive_elements(self) -> List[UIElement]:
        """Returns only actionable controls (buttons, inputs, links, tabs, etc.)."""
        return [el for el in self.elements if el.is_interactive]

    def to_markdown(
        self,
        grouped: bool = True,
        interactive_only: bool = True,
        max_elements: int = 60
    ) -> str:
        """
        Renders a token-efficient Markdown view specifically structured for LLM perception.
        Supports group-partitioned rendering and flat table rendering.
        Includes Desktop Window Topology header so agents know all running apps.
        """
        lines = []
        w = self.window
        src_tag = "[Hybrid: DOM + Model]" if self.extraction_source == "hybrid" else "[Pure Vision: Model]"
        lines.append(f"### Active Screen: {w.title} [{w.process_name} | PID: {w.pid}] {src_tag}")
        lines.append(f"- **Window Bounds**: {w.rect} (Screen: {self.screen_size[0]}x{self.screen_size[1]})")

        # Desktop window topology — agents see all open apps immediately
        desktop_wins = getattr(self.window, "desktop_windows", None)
        if desktop_wins:
            lines.append("\n**Open Windows on Desktop:**")
            lines.append("| App | Process | PID | Bounds | Focused |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for dw in desktop_wins[:20]:  # Cap at 20 to keep context tight
                title = str(dw.get("title", ""))[:40]
                proc = dw.get("process_name", "?")
                dpid = dw.get("pid", "?")
                rect = dw.get("rect", [])
                focused = "✓" if dw.get("is_focused") else ""
                lines.append(f"| {title} | {proc} | {dpid} | {rect} | {focused} |")
            lines.append("")

        if self.has_modal_dialog:
            lines.append(f"- **Alert / Modal Detected**: {self.active_modal_title or 'Dialog Box'}")

        if self.focused_element_id is not None:
            focused = self.get_element(self.focused_element_id)
            focus_desc = f"#{focused.element_id} ({focused.element_type}: \"{focused.label}\")" if focused else f"#{self.focused_element_id}"
            lines.append(f"- **Keyboard Focus**: {focus_desc}")

        # Active Application State / Primary Readouts (DEF-06)
        readouts = [el for el in self.elements if el.is_primary_readout]
        if readouts:
            lines.append("\n### Active Application State")
            for r in readouts[:4]:
                r_lbl = r.label or "(empty)"
                lines.append(f"- **Primary Readout**: \"{r_lbl}\" ({r.semantic_role} at {r.bbox})")
            lines.append("")

        targets = self.get_interactive_elements() if interactive_only else self.elements
        lines.append(f"- **Actionable Controls**: {len(targets)} displayed")

        if not targets:
            lines.append("\n_No actionable controls detected in this viewport._")
            return "\n".join(lines)

        # Render Group-Partitioned View
        if grouped and self.groups:
            active_grps = [g for g in self.groups if any(el in targets for el in g.elements)]
            if active_grps:
                lines.append(f"- **Semantic Groups**: {len(active_grps)} structural containers\n")

                rendered_count = 0
                for grp in active_grps:
                    grp_targets = [el for el in grp.elements if el in targets]
                    if not grp_targets:
                        continue

                    lines.append(f"#### [{grp.group_type}] {grp.title} ({len(grp_targets)} controls)")
                    lines.append("| ID | Type | Role | Label / Text | Bounds [x1, y1, x2, y2] | Center | State |")
                    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

                    for el in grp_targets:
                        if rendered_count >= max_elements:
                            break
                        lbl = el.label.replace("|", "\\|").replace("\n", " ").strip()
                        if len(lbl) > 40:
                            lbl = lbl[:37] + "..."
                        lbl_str = f"\"{lbl}\"" if lbl else "—"
                        state_badge = f"[{el.dynamic_state}]"

                        lines.append(
                            f"| #{el.element_id} | {el.element_type} | {el.semantic_role} | {lbl_str} | "
                            f"[{el.bbox[0]}, {el.bbox[1]}, {el.bbox[2]}, {el.bbox[3]}] | ({el.center[0]}, {el.center[1]}) | {state_badge} |"
                        )
                        rendered_count += 1

                    lines.append("")
                    if rendered_count >= max_elements:
                        break

                if len(targets) > rendered_count:
                    lines.append(f"_... and {len(targets) - rendered_count} more controls truncated for brevity._")
                return "\n".join(lines)

        # Flat Table Fallback
        lines.append("")
        lines.append("| ID | Type | Role | Label / Text | Bounds [x1, y1, x2, y2] | Center | State |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for el in targets[:max_elements]:
            lbl = el.label.replace("|", "\\|").replace("\n", " ").strip()
            if len(lbl) > 40:
                lbl = lbl[:37] + "..."
            lbl_str = f"\"{lbl}\"" if lbl else "—"
            state_badge = f"[{el.dynamic_state}]"

            lines.append(
                f"| #{el.element_id} | {el.element_type} | {el.semantic_role} | {lbl_str} | "
                f"[{el.bbox[0]}, {el.bbox[1]}, {el.bbox[2]}, {el.bbox[3]}] | ({el.center[0]}, {el.center[1]}) | {state_badge} |"
            )

        if len(targets) > max_elements:
            lines.append(f"\n_... and {len(targets) - max_elements} more controls truncated for brevity._")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes context to a dictionary."""
        return {
            "window": self.window.to_dict(),
            "screen_size": list(self.screen_size),
            "has_modal_dialog": self.has_modal_dialog,
            "active_modal_title": self.active_modal_title,
            "focused_element_id": self.focused_element_id,
            "extraction_source": self.extraction_source,
            "groups_count": len(self.groups),
            "elements_count": len(self.elements),
            "interactive_count": len(self.get_interactive_elements()),
            "groups": [g.to_dict() for g in self.groups],
            "elements": [el.to_dict() for el in self.elements]
        }


class ContextBuilder:
    """Builds clean ScreenContext objects by fusing DOM/UIA structures with UnifiedVisionNet."""

    @staticmethod
    def build_hybrid_context(
        window_meta: Dict[str, Any],
        screen_size: Tuple[int, int],
        raw_groups: List[Any],
        raw_elements: List[Any],
        model_predictions: Optional[List[Dict[str, Any]]] = None,
        source: str = "hybrid"
    ) -> ScreenContext:
        """
        Synthesizes direct DOM extraction and neural visual model predictions.
        """
        win_ctx = WindowContext(
            title=window_meta.get("title", "Desktop Screen"),
            process_name=window_meta.get("process_name", "explorer.exe"),
            pid=window_meta.get("pid", 0),
            rect=window_meta.get("rect", [0, 0, screen_size[0], screen_size[1]]),
            is_maximized=window_meta.get("is_maximized", False),
            hwnd=window_meta.get("hwnd", 0),
            desktop_windows=window_meta.get("desktop_windows", []),
        )

        elements: List[UIElement] = []
        element_lookup: Dict[int, UIElement] = {}
        has_modal = False
        modal_title = None
        focused_id = None

        has_preds = bool(model_predictions and len(model_predictions) == len(raw_elements))

        # 1. Instantiate Elements
        for idx, raw in enumerate(raw_elements, 1):
            pred = model_predictions[idx - 1] if has_preds else {}

            # Handle both RawDOMElement and dict
            if hasattr(raw, "bbox"):
                bbox = raw.bbox
                cname = raw.name
                ctext = raw.text
                ctype = raw.control_type
                is_inter = raw.is_interactive
                is_foc = raw.is_focused
                is_en = raw.is_enabled
                p_gid = raw.parent_group_id
            else:
                bbox = raw.get("bbox", [0, 0, 0, 0])
                cname = raw.get("name", "")
                ctext = raw.get("text", "")
                ctype = raw.get("control_type", "CustomControl")
                is_inter = raw.get("is_interactive", True)
                is_foc = raw.get("is_focused", False)
                is_en = raw.get("is_enabled", True)
                p_gid = raw.get("parent_group_id")

            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)

            # Type and Role Resolution
            # Direct DOM provides exact text; Model provides semantic role and visual dynamic state
            etype = pred.get("element_type") or ctype
            erole = pred.get("semantic_role") or "ACTION_BUTTON"
            state = pred.get("moment") or ("FOCUSED" if is_foc else "NORMAL")

            # Ground-truth text takes precedence; fallback to model CTC OCR or icon archetype
            label = ctext or cname or pred.get("ocr_text") or ""
            label = str(label).strip()
            if not label:
                icon_arch = pred.get("icon_archetype")
                if icon_arch:
                    label = f"[Icon: {icon_arch}]"

            # z_layer and native attributes (DEF-02, DEF-06)
            z_layer = getattr(raw, "z_layer", 0) if hasattr(raw, "z_layer") else (raw.get("z_layer", 0) if isinstance(raw, dict) else 0)
            n_attrs = getattr(raw, "native_attributes", {}) if hasattr(raw, "native_attributes") else (raw.get("native_attributes", {}) if isinstance(raw, dict) else {})
            is_read_only = n_attrs.get("is_read_only", False)
            live_setting = n_attrs.get("live_setting", 0)

            # Classify Primary Dynamic State Readout (DEF-06)
            win_w = max(1, win_ctx.rect[2] - win_ctx.rect[0])
            win_h = max(1, win_ctx.rect[3] - win_ctx.rect[1])
            win_area = win_w * win_h
            el_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            area_ratio = el_area / win_area
            name_lower = (cname + " " + ctext).lower()
            READOUT_KEYWORDS = {"display", "result", "output", "status", "total", "balance", "value", "readout", "address", "url"}
            name_match = any(kw in name_lower for kw in READOUT_KEYWORDS)

            is_readout = False
            if (is_read_only or live_setting > 0 or erole in ["STAT_COUNTER", "STATUS_INDICATOR"]) and (area_ratio > 0.02 or name_match):
                is_readout = True
                if erole in ["ACTION_BUTTON", "GENERIC_CONTAINER"]:
                    erole = "PRIMARY_READOUT"

            # Region detection
            region = "MAIN_CONTENT"
            if bbox[3] < 120:
                region = "TOP_BAR"
            elif bbox[0] < win_ctx.rect[0] + 320:
                region = "SIDEBAR"
            elif bbox[1] > win_ctx.rect[3] - 80:
                region = "BOTTOM_BAR"

            if p_gid and "modal" in p_gid:
                region = "MODAL_DIALOG"
                has_modal = True
                modal_title = label or "Dialog"

            if is_foc or state == "FOCUSED":
                focused_id = idx

            ui_el = UIElement(
                element_id=idx,
                element_type=etype,
                semantic_role=erole,
                label=label,
                bbox=bbox,
                center=(cx, cy),
                is_interactive=is_inter,
                dynamic_state=state,
                parent_region=region,
                parent_group_id=p_gid,
                raw_control_type=ctype,
                is_enabled=is_en,
                confidence=raw.get("confidence", 1.0) if isinstance(raw, dict) else 1.0,
                z_layer=z_layer,
                is_primary_readout=is_readout
            )
            elements.append(ui_el)
            element_lookup[idx] = ui_el

        # 2. Instantiate and Link Groups
        groups: List[UIGroup] = []
        for g in raw_groups:
            if hasattr(g, "group_id"):
                gid = g.group_id
                gtype = g.group_type
                gtitle = g.title
                gbbox = g.bbox
                eids = g.element_ids
            else:
                gid = g.get("group_id", "grp_0")
                gtype = g.get("group_type", "MAIN_CONTENT")
                gtitle = g.get("title", "Container")
                gbbox = g.get("bbox", [0, 0, 0, 0])
                eids = g.get("element_ids", [])

            grp_elements = [element_lookup[eid] for eid in eids if eid in element_lookup]

            # If element was not explicitly attached, check spatial containment
            if not grp_elements:
                for el in elements:
                    if (el.bbox[0] >= gbbox[0] - 5 and el.bbox[1] >= gbbox[1] - 5 and
                        el.bbox[2] <= gbbox[2] + 5 and el.bbox[3] <= gbbox[3] + 5):
                        grp_elements.append(el)
                        el.parent_group_id = gid
                        el.parent_group_title = gtitle

            for el in grp_elements:
                el.parent_group_id = gid
                el.parent_group_title = gtitle

            if gtype == "MODAL_DIALOG":
                has_modal = True
                modal_title = gtitle

            groups.append(UIGroup(
                group_id=gid,
                group_type=gtype,
                title=gtitle,
                bbox=gbbox,
                elements=grp_elements
            ))

        return ScreenContext(
            window=win_ctx,
            screen_size=screen_size,
            elements=elements,
            groups=groups,
            has_modal_dialog=has_modal,
            active_modal_title=modal_title,
            focused_element_id=focused_id,
            extraction_source=source
        )
