"""
Hue Vision - Command Line Interface
───────────────────────────────────
Run universal MCP server, capture formatted screen context, or track live display changes directly from terminal.

Commands:
    python -m hue_vision.cli serve       # Starts universal MCP stdio server
    python -m hue_vision.cli perceive    # Prints formatted Markdown screen context
    python -m hue_vision.cli track       # Waits and prints screen delta report
    python -m hue_vision.cli interact    # Performs causal action on screen
"""

import sys
import argparse
from hue_vision.engine import VisionEngine
from hue_vision.server import run_server


def main():
    parser = argparse.ArgumentParser(description="Hue Vision CLI - Universal Screen Understanding & Causal Interaction Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Serve command
    subparsers.add_parser("serve", help="Run MCP stdio server")

    # Perceive command
    p_parser = subparsers.add_parser("perceive", help="Capture and print screen context")
    p_parser.add_argument("--all", action="store_true", help="Include passive containers")
    p_parser.add_argument("--flat", action="store_true", help="Render flat table without groups")
    p_parser.add_argument("--max", type=int, default=50, help="Max elements to list")
    p_parser.add_argument("--target", type=str, default="device", help="Target: device, active_window, window, android, universal")
    p_parser.add_argument("--app", type=str, default=None, help="App name filter")

    # Track command
    t_parser = subparsers.add_parser("track", help="Observe and print screen delta")
    t_parser.add_argument("--timeout", type=float, default=3.0, help="Seconds to wait for changes")
    t_parser.add_argument("--target", type=str, default="device", help="Target device")

    # Interact command
    i_parser = subparsers.add_parser("interact", help="Execute UI interaction and verify delta")
    i_parser.add_argument("action", choices=["click", "double_click", "right_click", "type", "scroll", "hover"], help="Action to perform")
    i_parser.add_argument("--id", type=int, default=None, help="Element ID from perceive")
    i_parser.add_argument("--x", type=int, default=None, help="X coordinate")
    i_parser.add_argument("--y", type=int, default=None, help="Y coordinate")
    i_parser.add_argument("--text", type=str, default=None, help="Text to type")
    i_parser.add_argument("--target", type=str, default="device", help="Target device")

    args = parser.parse_args()

    if args.command == "serve":
        run_server()
    elif args.command == "perceive":
        engine = VisionEngine.get_instance()
        md = engine.perceive(
            target=args.target,
            grouped=not args.flat,
            interactive_only=not args.all,
            max_elements=args.max,
            app_name=args.app
        )
        print(md)
    elif args.command == "track":
        engine = VisionEngine.get_instance()
        print(f"[*] Observing live display changes (target: {args.target})...")
        delta = engine.track(timeout_s=args.timeout, target=args.target)
        print(delta.to_markdown())
    elif args.command == "interact":
        engine = VisionEngine.get_instance()
        result = engine.interact(
            action=args.action,
            element_id=args.id,
            x=args.x,
            y=args.y,
            text=args.text,
            target=args.target
        )
        print(result.to_markdown())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
