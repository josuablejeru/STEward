#!/usr/bin/env python3
"""Claude Code Stop hook: validate and benchmark recorded changes.

Reads the hook input JSON from stdin and prints the hook output JSON.
Configured in .claude/settings.json; see docs/benchmark-system.md.
"""

import sys

try:
    from steward.automation.hooks import main_stop
except ImportError as exc:  # pragma: no cover - environment problem
    print(f"STEward hook: cannot import steward ({exc}). Run `pixi install`.", file=sys.stderr)
    sys.exit(1)

sys.exit(main_stop())
