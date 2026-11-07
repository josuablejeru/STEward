#!/usr/bin/env python3
"""Claude Code PostToolUse hook: record benchmark-relevant file changes.

Reads the hook input JSON from stdin. Cheap: it never runs checks.
Configured in .claude/settings.json; see docs/benchmark-system.md.
"""

import sys

try:
    from steward.automation.hooks import main_post_tool_use
except ImportError as exc:  # pragma: no cover - environment problem
    print(f"STEward hook: cannot import steward ({exc}). Run `pixi install`.", file=sys.stderr)
    sys.exit(1)

sys.exit(main_post_tool_use())
