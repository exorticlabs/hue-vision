"""
Hue Vision - Anthropic Claude Tool Use Adapter
───────────────────────────────────────────────
Enables Claude 3.5 Sonnet / Claude 3 Opus agents to perceive screens,
track display state changes, and causally interact with GUI elements across all devices.
"""

from typing import List, Dict, Any
from hue_vision.engine import VisionEngine


def get_anthropic_tools() -> List[Dict[str, Any]]:
    """Returns Anthropic Claude tool use definitions for Hue Vision."""
    return [
        {
            "name": "perceive_screen",
            "description": "Captures the active screen across desktop or mobile and returns a structured markdown table of actionable UI controls with element IDs, types, roles, text labels, and coordinates.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Device target: 'device' (default), 'active_window', 'window', 'android', 'macos', 'linux', or 'universal'."
                    },
                    "interactive_only": {
                        "type": "boolean",
                        "description": "Whether to return only clickable/actionable controls (default: true)."
                    },
                    "max_elements": {
                        "type": "integer",
                        "description": "Maximum controls to display (default: 50)."
                    }
                }
            }
        },
        {
            "name": "track_screen",
            "description": "Observes live display changes, returning a delta report of appeared/disappeared elements, modal dialogs, and focus shifts.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "timeout_s": {
                        "type": "number",
                        "description": "Maximum seconds to wait for a visual state change (default: 2.0)."
                    },
                    "target": {
                        "type": "string",
                        "description": "Device target: 'device', 'android', etc."
                    }
                }
            }
        },
        {
            "name": "interact_ui",
            "description": "Executes a grounded GUI action (click by element ID, type text, scroll, hotkeys) and verifies whether the action produced the expected visual state change.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["click", "double_click", "right_click", "type", "scroll", "hover", "hotkey"],
                        "description": "UI action to perform."
                    },
                    "element_id": {
                        "type": "integer",
                        "description": "Numeric element ID (#1, #2, ...) from perceive_screen."
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type if action is 'type'."
                    },
                    "press_enter": {
                        "type": "boolean",
                        "description": "Whether to hit Enter after typing."
                    },
                    "clear_first": {
                        "type": "boolean",
                        "description": "Whether to clear input first."
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["down", "up"]
                    },
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keys to press for hotkey (e.g. ['ctrl', 's'])."
                    },
                    "target": {
                        "type": "string",
                        "description": "Target device ('device', 'android', etc.)"
                    }
                },
                "required": ["action"]
            }
        }
    ]


def handle_anthropic_tool_call(tool_name: str, tool_input: Dict[str, Any], engine: VisionEngine = None) -> str:
    """Dispatches a Claude tool use call and returns text content."""
    if engine is None:
        engine = VisionEngine.get_instance()

    if tool_name == "perceive_screen":
        return engine.perceive(
            target=tool_input.get("target", "device"),
            interactive_only=tool_input.get("interactive_only", True),
            max_elements=tool_input.get("max_elements", 50)
        )
    elif tool_name == "track_screen":
        delta = engine.track(
            timeout_s=float(tool_input.get("timeout_s", 2.0)),
            target=tool_input.get("target", "device")
        )
        return delta.to_markdown()
    elif tool_name == "interact_ui":
        result = engine.interact(
            action=tool_input.get("action"),
            element_id=tool_input.get("element_id"),
            text=tool_input.get("text"),
            press_enter=tool_input.get("press_enter", False),
            clear_first=tool_input.get("clear_first", False),
            direction=tool_input.get("direction", "down"),
            keys=tool_input.get("keys"),
            target=tool_input.get("target")
        )
        return result.to_markdown()
    else:
        return f"Unknown tool: {tool_name}"
