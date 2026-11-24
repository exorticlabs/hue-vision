# Hue Vision

> **Universal Live Multi-Device Display Understanding, Screen Context Synthesis & Causal Interaction Engine for AI Agents & LLMs.**

Hue Vision enables autonomous AI agents and language models to perceive, track, and causally interact with live screens across Windows, macOS, Linux, and Android devices.

---

## Key Capabilities

1. **Multi-Device Support**:
   - **Windows**: Native UIAutomation, Win32 GDI/DWM compositor, UWP/WinUI3 support.
   - **macOS**: Accessibility API, AppleScript, Quartz CoreGraphics.
   - **Linux**: X11 and Wayland display perception, xdotool, wmctrl, AT-SPI.
   - **Android**: Android Debug Bridge (ADB) screen streaming, ground-truth UIAutomator XML hierarchy extraction, and native touch/key event dispatch.
   - **Universal CV Fallback**: Morphological edge and contour detection for DOM-absent environments (VNC, remote desktops, games, canvas views).

2. **Semantic Screen Context**:
   - Synthesizes compact Markdown screen state with active window topology, actionable controls (`#1`, `#2`, ...), classified element types, semantic roles, and dynamic states.

3. **Continuous Delta Tracking**:
   - Monitors the display for visual delta mutations—alerting agents to newly rendered modal dialogs, dismissed popups, shifted focus, and state transitions.

4. **Causal Interaction Verification**:
   - Dispatches clicks, typing, hotkeys, and gestures, immediately captures post-action screen state, and causally verifies what changed as a result of the agent's action.

---

## Quick Start & Integration

### 1. Model Context Protocol (MCP)

Add Hue Vision to your MCP client configuration (`claude_desktop_config.json`, `mcp_config.json`, Cursor, Antigravity, or Goose):

```json
{
  "mcpServers": {
    "hue-vision": {
      "command": "python",
      "args": ["-m", "hue_vision.server"]
    }
  }
}
```

#### Exposed MCP Tools:
- `perceive_screen`: Returns structured Markdown context with active window details and actionable controls across desktop or Android devices.
- `track_screen`: Waits and reports live screen state mutations, window focus transitions, and newly rendered dialogs.
- `interact_ui`: Clicks elements by `#ID`, types into focused inputs, scrolls, gestures, and causally verifies visual changes.
- `get_element_details`: Deep inspects a specific control by its ID.
- `launch_app`: Spawns applications and verifies GUI window readiness before agent interaction.

---

### 2. Python SDK

```python
from hue_vision import VisionEngine

engine = VisionEngine.get_instance()

# 1. Perceive Desktop Screen Context
context_md = engine.perceive(target="device", interactive_only=True)
print(context_md)

# 2. Perceive Connected Android Device
# android_md = engine.perceive(target="android", interactive_only=True)

# 3. Causally Interact with Element #2
result = engine.interact(action="click", element_id=2)
print(result.causal_summary)

# 4. Observe Live Display Changes
delta = engine.track(timeout_s=3.0)
print(delta.summary)
```

---

### 3. Agent Framework Adapters

```python
# OpenAI Function Calling
from hue_vision.adapters.openai import get_openai_tools, handle_openai_tool_call
tools = get_openai_tools()

# Anthropic Claude Tool Use
from hue_vision.adapters.anthropic import get_anthropic_tools, handle_anthropic_tool_call
tools = get_anthropic_tools()
```

---

## Installation

```bash
# Core installation
pip install .

# With full neural model & desktop dependencies
pip install ".[all]"
```

---

## License
MIT License. Developed by Exortic Research Lab.
