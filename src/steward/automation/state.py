"""Local hook state in ``.steward/state/`` (git-ignored, safe to delete).

Files:

* ``hooks.json``: pending changes, validated fingerprints, outstanding work,
  and the last Stop-hook outcome.
* ``snapshot.json``: stat snapshot used to detect changes made by Bash.
* ``hook_log.jsonl``: one line per hook decision, for troubleshooting.
* ``workflow.lock``: prevents two validation runs at the same time.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from steward.automation.changes import ChangeCategory, Snapshot
from steward.datasets.jsonl import atomic_write_text

STATE_DIR = ".steward/state"
GUARD_ENV = "STEWARD_HOOK_ACTIVE"
MODE_ENV = "STEWARD_HOOKS"
MODES = ("auto", "light", "full", "off")
DISABLE_FILE = ".steward/hooks-disabled"


def state_dir(root: Path) -> Path:
    """Return (and create) the state directory."""
    path = root / STATE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_iso() -> str:
    """Current UTC time in ISO format."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@contextmanager
def file_lock(path: Path, *, blocking: bool) -> Iterator[bool]:
    """Hold an exclusive lock. Yields False when ``blocking`` is False and the lock is taken."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@dataclass
class HookState:
    """Persistent state shared by the PostToolUse and Stop hooks."""

    root: Path
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    validated: dict[str, str] = field(default_factory=dict)
    outstanding: list[str] = field(default_factory=list)
    last_stop: dict[str, Any] | None = None
    fingerprint_cache: dict[str, list[object]] = field(default_factory=dict)
    seen_runs: dict[str, int] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        """``hooks.json``."""
        return self.root / STATE_DIR / "hooks.json"

    @classmethod
    def load(cls, root: Path) -> HookState:
        """Load state; a missing or unreadable file gives an empty state."""
        path = root / STATE_DIR / "hooks.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls(root=root)
        if not isinstance(data, dict):
            return cls(root=root)
        return cls(
            root=root,
            pending=dict(data.get("pending") or {}),
            validated=dict(data.get("validated") or {}),
            outstanding=list(data.get("outstanding") or []),
            last_stop=data.get("last_stop"),
            fingerprint_cache=dict(data.get("fingerprint_cache") or {}),
            seen_runs=dict(data.get("seen_runs") or {}),
        )

    def save(self) -> None:
        """Write the state atomically."""
        state_dir(self.root)
        payload = {
            "pending": self.pending,
            "validated": self.validated,
            "outstanding": self.outstanding,
            "last_stop": self.last_stop,
            "fingerprint_cache": self.fingerprint_cache,
            "seen_runs": self.seen_runs,
        }
        atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def record(self, rel_path: str, category: ChangeCategory, tool: str) -> None:
        """Mark one path as changed and waiting for validation."""
        entry = self.pending.get(rel_path)
        stamp = now_iso()
        if entry is None:
            self.pending[rel_path] = {
                "category": category.value,
                "first_seen": stamp,
                "last_seen": stamp,
                "tools": [tool],
            }
        else:
            entry["last_seen"] = stamp
            if tool not in entry["tools"]:
                entry["tools"].append(tool)

    def categories(self) -> set[ChangeCategory]:
        """Categories with pending changes."""
        return {ChangeCategory(e["category"]) for e in self.pending.values()}

    def pending_summary(self) -> dict[str, list[str]]:
        """Pending paths grouped by category."""
        summary: dict[str, list[str]] = {}
        for path, entry in sorted(self.pending.items()):
            summary.setdefault(entry["category"], []).append(path)
        return summary

    def mode(self) -> str:
        """Hook mode: ``$STEWARD_HOOKS``, else ``off`` if the disable file exists, else ``auto``."""
        env = os.environ.get(MODE_ENV, "").strip().lower()
        if env in MODES:
            return env
        if (self.root / DISABLE_FILE).exists():
            return "off"
        return "auto"


def load_snapshot(root: Path) -> Snapshot | None:
    """The last stat snapshot, or None when there is none yet."""
    try:
        data = json.loads((root / STATE_DIR / "snapshot.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def save_snapshot(root: Path, snapshot: Snapshot) -> None:
    """Store a stat snapshot."""
    state_dir(root)
    atomic_write_text(root / STATE_DIR / "snapshot.json", json.dumps(snapshot, sort_keys=True))


def log(root: Path, event: str, **fields: Any) -> None:
    """Append one line to ``hook_log.jsonl`` (best effort)."""
    try:
        with (state_dir(root) / "hook_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"ts": now_iso(), "event": event, **fields}, default=str) + "\n"
            )
    except OSError:
        pass
