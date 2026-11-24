"""
Tests for Hue Vision MCP server and OpenAI/Anthropic agent adapters.
"""

import unittest
import json
from hue_vision.server import TOOLS
from hue_vision.adapters.openai import get_openai_tools, handle_openai_tool_call
from hue_vision.adapters.anthropic import get_anthropic_tools, handle_anthropic_tool_call


class TestServerAndAdapters(unittest.TestCase):
    def test_mcp_tools_manifest(self):
        tool_names = [t["name"] for t in TOOLS]
        self.assertIn("perceive_screen", tool_names)
        self.assertIn("track_screen", tool_names)
        self.assertIn("interact_ui", tool_names)
        self.assertIn("get_element_details", tool_names)
        self.assertIn("launch_app", tool_names)

        # Check multi-device target option in perceive_screen
        perceive_tool = next(t for t in TOOLS if t["name"] == "perceive_screen")
        target_prop = perceive_tool["inputSchema"]["properties"]["target"]
        self.assertIn("android", target_prop["enum"])
        self.assertIn("macos", target_prop["enum"])
        self.assertIn("linux", target_prop["enum"])

    def test_openai_tools_schema(self):
        tools = get_openai_tools()
        self.assertEqual(len(tools), 3)
        fn_names = [t["function"]["name"] for t in tools]
        self.assertIn("perceive_screen", fn_names)
        self.assertIn("track_screen", fn_names)
        self.assertIn("interact_ui", fn_names)

    def test_anthropic_tools_schema(self):
        tools = get_anthropic_tools()
        self.assertEqual(len(tools), 3)
        names = [t["name"] for t in tools]
        self.assertIn("perceive_screen", names)
        self.assertIn("track_screen", names)
        self.assertIn("interact_ui", names)


if __name__ == "__main__":
    unittest.main()
