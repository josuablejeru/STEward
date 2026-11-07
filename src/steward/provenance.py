"""Helpers that record where code and data came from."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_state(cwd: Path | None = None) -> tuple[str | None, bool | None]:
    """Return ``(commit, dirty)`` for the Git checkout, or ``(None, None)``.

    Never raises: provenance is best effort when Git is not available.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit or None, bool(status.strip())


def is_git_tracked_location(path: Path) -> bool:
    """Return True when ``path`` is inside a Git work tree and *not* ignored.

    Used to warn when generated data would be committed by mistake. Returns
    False when Git is not available or ``path`` is outside a work tree.
    """
    directory = path.parent if path.parent.exists() else Path.cwd()
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return False
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", str(path.resolve())],
            cwd=directory,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return ignored.returncode == 1
