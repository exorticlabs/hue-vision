"""
Hue Vision - Universal MCP Server (JSON-RPC 2.0 Stdio)
───────────────────────────────────────────────────────
Standard Model Context Protocol server exposing real-time screen perception,
live display delta tracking, and causally-verified GUI interactions to ANY
agent or LLM (Claude, Cursor, Antigravity, Goose, Open-WebUI) across Windows, macOS, Linux, and Android.
"""

import sys
import os
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

from hue_vision.engine import VisionEngine

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {
    "name": "hue-vision",
    "version": "1.0.0"
}

TOOLS = [
    {
        "name": "perceive_screen",
        "description": "Perceives the full system desktop or mobile display across Windows, macOS, Linux, and Android. Returns structured summary and table of all actionable UI controls across visible apps. Includes Desktop Window Topology listing running apps. Supports targeting specific background apps or connected Android devices.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["device", "active_window", "window", "android", "macos", "linux", "universal"],
                    "default": "device",
                    "description": "'device' (default host display), 'active_window' (foreground window only), 'window' (specific app), 'android' (connected phone/tablet/emulator), 'macos', 'linux', or 'universal' (pure CV)."
                },
                "app_name": {
                    "type": "string",
                    "description": "Process or app name substring to target a specific background app (e.g. 'Calculator', 'Notepad', 'chrome'). Used when target='window'."
                },
                "window_title": {
                    "type": "string",
                    "description": "Window title substring to target a specific window."
                },
                "pid": {
                    "type": "integer",
                    "description": "Exact process ID to target a specific app instance."
                },
                "grouped": {
                    "type": "boolean",
                    "default": True,
                    "description": "When true, organizes controls into logical structural groups (Navbars, Sidebars, Modals, Forms, Cards)."
                },
                "interactive_only": {
                    "type": "boolean",
                    "default": True,
                    "description": "When true, filters out passive text/containers and returns only actionable controls (buttons, inputs, links, tabs)."
                },
                "max_elements": {
                    "type": "integer",
                    "default": 50,
                    "description": "Maximum number of controls to return in the table view."
                }
            }
        }
    },
    {
        "name": "track_screen",
        "description": "Monitors the live display for visual changes, newly appeared dialogs/modals, dismissed elements, or focus transitions across desktop or mobile devices.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "timeout_s": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Maximum seconds to observe for a visual change before returning current status."
                },
                "target": {
                    "type": "string",
                    "default": "device",
                    "description": "Device target: 'device' (default), 'android', 'macos', 'linux', or 'universal'."
                }
            }
        }
    },
    {
        "name": "interact_ui",
        "description": "Performs a grounded GUI interaction (click by element ID, type into input, scroll, hover, hotkey) across Windows, macOS, Linux, or Android, and immediately verifies whether the intended visual change occurred.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["click", "double_click", "right_click", "type", "scroll", "hover", "hotkey"],
                    "description": "The low-level interaction to perform."
                },
                "element_id": {
                    "type": "integer",
                    "description": "The numeric ID (#1, #2, ...) of the target element from the most recent perceive_screen call."
                },
                "x": {
                    "type": "integer",
                    "description": "Explicit X coordinate if not specifying an element_id."
                },
                "y": {
                    "type": "integer",
                    "description": "Explicit Y coordinate if not specifying an element_id."
                },
                "text": {
                    "type": "string",
                    "description": "Text to type if action is 'type'."
                },
                "press_enter": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to press Enter after typing text."
                },
                "clear_first": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to clear existing text in the input before typing."
                },
                "direction": {
                    "type": "string",
                    "enum": ["down", "up"],
                    "default": "down",
                    "description": "Scroll direction if action is 'scroll'."
                },
                "amount": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of scroll clicks if action is 'scroll'."
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key names to press in sequence for hotkeys (e.g. ['ctrl', 's'])."
                },
                "target": {
                    "type": "string",
                    "description": "Device target: 'device' (default host), 'android', 'macos', 'linux', or 'universal'."
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "get_element_details",
        "description": "Inspects a specific UI element by ID in full detail, returning its bounding box, center, role, OCR text, and dynamic moment.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "integer",
                    "description": "The element ID to inspect."
                }
            },
            "required": ["element_id"]
        }
    },
    {
        "name": "launch_app",
        "description": "Launches a desktop or mobile application across Windows, macOS, Linux, or Android, and guarantees its GUI window is initialized, visible, and ready before returning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "executable": {
                    "type": "string",
                    "description": "Executable name, package, or path (e.g. 'calc.exe', 'notepad.exe', 'com.android.settings')."
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Command line arguments to pass."
                },
                "expected_title": {
                    "type": "string",
                    "default": "",
                    "description": "Window title substring to wait for."
                },
                "timeout_s": {
                    "type": "number",
                    "default": 10.0,
                    "description": "Maximum seconds to wait for GUI window registration."
                },
                "target": {
                    "type": "string",
                    "description": "Device target: 'device' (default), 'android', 'macos', 'linux', or 'universal'."
                }
            },
            "required": ["executable"]
        }
    }
]


def send_response(response: Dict[str, Any]) -> None:
    """Writes a JSON-RPC message to stdout followed by a newline."""
    body = json.dumps(response, ensure_ascii=False)
    sys.stdout.write(body + "\n")
    sys.stdout.flush()


def run_server() -> None:
    """Main event loop for MCP stdio protocol."""
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    engine: Optional[VisionEngine] = None

    def get_engine() -> VisionEngine:
        nonlocal engine
        if engine is None:
            engine = VisionEngine.get_instance()
        return engine

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            send_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"}
            })
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        # MCP Lifecycle: Initialize
        if method == "initialize":
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": SERVER_INFO
                }
            })
            continue

        # MCP Notification: Initialized
        if method == "notifications/initialized":
            continue

        # MCP Ping
        if method == "ping":
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {}
            })
            continue

        # List Tools
        if method == "tools/list":
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": TOOLS
                }
            })
            continue

        # Call Tool
        if method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})

            try:
                if tool_name == "perceive_screen":
                    target = args.get("target", "device")
                    grouped = args.get("grouped", True)
                    interactive_only = args.get("interactive_only", True)
                    max_els = args.get("max_elements", 50)
                    app_name = args.get("app_name")
                    window_title = args.get("window_title")
                    pid = args.get("pid")
                    result_text = get_engine().perceive(
                        target=target,
                        grouped=grouped,
                        interactive_only=interactive_only,
                        max_elements=max_els,
                        app_name=app_name,
                        window_title=window_title,
                        pid=pid,
                    )
                    send_response({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": result_text}]
                        }
                    })

                elif tool_name == "track_screen":
                    timeout_s = float(args.get("timeout_s", 2.0))
                    target = args.get("target", "device")
                    delta = get_engine().track(timeout_s=timeout_s, target=target)
                    send_response({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": delta.to_markdown()}]
                        }
                    })

                elif tool_name == "interact_ui":
                    action = args.get("action")
                    res = get_engine().interact(
                        action=action,
                        element_id=args.get("element_id"),
                        x=args.get("x"),
                        y=args.get("y"),
                        text=args.get("text"),
                        press_enter=args.get("press_enter", False),
                        clear_first=args.get("clear_first", False),
                        direction=args.get("direction", "down"),
                        amount=args.get("amount", 3),
                        keys=args.get("keys"),
                        target=args.get("target")
                    )
                    send_response({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": res.to_markdown()}]
                        }
                    })

                elif tool_name == "get_element_details":
                    eid = int(args.get("element_id", 0))
                    eng = get_engine()
                    if eng.last_context:
                        el = eng.last_context.get_element(eid)
                        if el:
                            send_response({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "result": {
                                    "content": [{"type": "text", "text": json.dumps(el.to_dict(), indent=2)}]
                                }
                            })
                        else:
                            send_response({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "result": {
                                    "content": [{"type": "text", "text": f"Element #{eid} not found in current screen context."}]
                                }
                            })
                    else:
                        send_response({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "content": [{"type": "text", "text": "No screen context has been captured yet. Call perceive_screen first."}]
                            }
                        })

                elif tool_name == "launch_app":
                    from hue_vision.lifecycle import ProcessLifecycleManager
                    exe = args.get("executable")
                    cargs = args.get("args", [])
                    exp_title = args.get("expected_title", "")
                    t_out = float(args.get("timeout_s", 10.0))
                    target = args.get("target")
                    hwnd = ProcessLifecycleManager.launch_and_wait_gui(
                        executable=exe,
                        args=cargs,
                        expected_title=exp_title,
                        timeout_s=t_out,
                        target=target
                    )
                    send_response({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{
                                "type": "text",
                                "text": f"Application '{exe}' successfully launched. GUI Window ready (Handle: {hwnd})."
                            }]
                        }
                    })

                else:
                    send_response({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Method not found: {tool_name}"}
                    })

            except Exception as e:
                err_trace = traceback.format_exc()
                send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "isError": True,
                        "content": [{"type": "text", "text": f"Execution error: {e}\n{err_trace}"}]
                    }
                })
            continue

        # Unhandled method
        send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unhandled method: {method}"}
        })


if __name__ == "__main__":
    run_server()
