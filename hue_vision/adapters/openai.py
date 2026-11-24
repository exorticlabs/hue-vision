"""
Hue Vision - OpenAI Function Calling Adapter
─────────────────────────────────────────────
Enables any OpenAI-powered agent (GPT-4o, o1, Assistants API) to perceive screens,
track display changes, and causally interact with the desktop or mobile device.
"""

from typing import List, Dict, Any
from hue_vision.engine import VisionEngine


def get_openai_tools() -> List[Dict[str, Any]]:
    """Returns OpenAI tool calling schemas for Hue Vision."""
    return [
        {
            "type": "function",
            "function": {
                "name": "perceive_screen",
                "description": "Captures the active screen and returns a structured markdown table of actionable UI controls with element IDs, types, roles, text labels, and coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Target device ('device', 'active_window', 'android', 'macos', 'linux', etc.)",
                            "default": "device"
                        },
                        "interactive_only": {
                            "type": "boolean",
                            "description": "Whether to return only clickable/actionable controls (default: true).",
                            "default": True
                        },
                        "max_elements": {
                            "type": "integer",
                            "description": "Maximum controls to display (default: 50).",
                            "default": 50
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "track_screen",
                "description": "Observes live display changes, returning a delta report of appeared/disappeared elements, modal dialogs, and focus shifts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timeout_s": {
                            "type": "number",
                            "description": "Maximum seconds to wait for a visual state change.",
                            "default": 2.0
                        },
                        "target": {
                            "type": "string",
                            "description": "Target device ('device', 'android', etc.)",
                            "default": "device"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "interact_ui",
                "description": "Executes a grounded GUI action (click by element ID, type text, scroll, hotkeys) and verifies whether the action produced the expected visual state change.",
                "parameters": {
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
                            "description": "Whether to hit Enter after typing.",
                            "default": False
                        },
                        "clear_first": {
                            "type": "boolean",
                            "description": "Whether to clear input first.",
                            "default": False
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["down", "up"],
                            "default": "down"
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
        }
    ]


def handle_openai_tool_call(tool_name: str, arguments: Dict[str, Any], engine: VisionEngine = None) -> str:
    """Dispatches an OpenAI function tool call and returns text content."""
    if engine is None:
        engine = VisionEngine.get_instance()

    if tool_name == "perceive_screen":
        return engine.perceive(
            target=arguments.get("target", "device"),
            interactive_only=arguments.get("interactive_only", True),
            max_elements=arguments.get("max_elements", 50)
        )
    elif tool_name == "track_screen":
        delta = engine.track(
            timeout_s=float(arguments.get("timeout_s", 2.0)),
            target=arguments.get("target", "device")
        )
        return delta.to_markdown()
    elif tool_name == "interact_ui":
        result = engine.interact(
            action=arguments.get("action"),
            element_id=arguments.get("element_id"),
            text=arguments.get("text"),
            press_enter=arguments.get("press_enter", False),
            clear_first=arguments.get("clear_first", False),
            direction=arguments.get("direction", "down"),
            keys=arguments.get("keys"),
            target=arguments.get("target")
        )
        return result.to_markdown()
    else:
        return f"Unknown tool: {tool_name}"
