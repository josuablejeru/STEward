"""Command-line entry point for STEward."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from steward import __version__
from steward.environment import collect_environment_info

DISCLAIMER = (
    "STEward output is STE-oriented. It is not certified or approved ASD-STE100 compliance."
)


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="steward",
        description="Tools for STE-oriented technical documentation.",
        epilog=DISCLAIMER,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    info = subparsers.add_parser("info", help="Show version and runtime information.")
    info.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    info.set_defaults(handler=_cmd_info)

    return parser


def _cmd_info(args: argparse.Namespace) -> int:
    info = collect_environment_info()
    if args.json:
        print(json.dumps(info.to_dict(), indent=2, sort_keys=True))
    else:
        for key, value in info.to_dict().items():
            print(f"{key}: {value}")
        print()
        print(DISCLAIMER)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    result: int = handler(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
