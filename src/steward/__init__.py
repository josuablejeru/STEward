"""STEward: tooling for STE-oriented (ASD-STE100) technical documentation.

Output from this package is STE-oriented. It is not a certification of
compliance with ASD-STE100.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("steward")
except PackageNotFoundError:  # pragma: no cover - only when run from an uninstalled tree
    __version__ = "0.0.0+unknown"

STE_REFERENCE = "ASD-STE100 Issue 9 (January 2025)"
"""The STE standard issue this project targets as its human reference."""

__all__ = ["STE_REFERENCE", "__version__"]
