"""
Hue Vision - Agent Framework Adapters
──────────────────────────────────────
Native tool calling and function calling integrations for OpenAI, Anthropic Claude,
and Model Context Protocol (MCP) compatible agents.
"""

from hue_vision.adapters.openai import get_openai_tools, handle_openai_tool_call
from hue_vision.adapters.anthropic import get_anthropic_tools, handle_anthropic_tool_call

__all__ = [
    "get_openai_tools",
    "handle_openai_tool_call",
    "get_anthropic_tools",
    "handle_anthropic_tool_call",
]
