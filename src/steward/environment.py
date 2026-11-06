"""Collect runtime information for reproducibility and provenance records."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass

from steward import STE_REFERENCE, __version__


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """A snapshot of the runtime that produced an artifact."""

    steward_version: str
    python_version: str
    python_implementation: str
    platform: str
    ste_reference: str

    def to_dict(self) -> dict[str, str]:
        """Return the snapshot as a plain dictionary."""
        return asdict(self)


def collect_environment_info() -> EnvironmentInfo:
    """Return information about the current runtime.

    The result has no secrets, user names, or file paths.
    """
    return EnvironmentInfo(
        steward_version=__version__,
        python_version=platform.python_version(),
        python_implementation=sys.implementation.name,
        platform=platform.platform(terse=True),
        ste_reference=STE_REFERENCE,
    )
