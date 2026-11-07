"""Claude Code hook handlers (``PostToolUse`` and ``Stop``).

Both read the hook input JSON from stdin. See ``docs/benchmark-system.md``
for setup and troubleshooting. Design:

* PostToolUse is cheap: it records changed paths and never runs checks.
* Stop runs the validation workflow once per task, only when relevant
  inputs changed, under a lock, with a reentrancy guard.
* A failure blocks the stop once (so Claude sees and fixes it). If the
  stop hook already blocked (``stop_hook_active``), it reports the failure
  without blocking again, and the changes stay pending.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from steward.automation.changes import (
    ChangeCategory,
    classify,
    diff_snapshots,
    relativize,
    take_snapshot,
)
from steward.automation.state import (
    GUARD_ENV,
    STATE_DIR,
    HookState,
    file_lock,
    load_snapshot,
    log,
    now_iso,
    save_snapshot,
)
from steward.automation.workflow import (
    CommandRunner,
    WorkflowReport,
    category_fingerprints,
    run_workflow,
    subprocess_runner,
)
from steward.evaluation.service import ProviderFactory, default_provider_factory

FILE_TOOLS = {
    "Write": "file_path",
    "Edit": "file_path",
    "MultiEdit": "file_path",
    "NotebookEdit": "notebook_path",
}


class HookInputError(ValueError):
    """The hook input is not valid JSON of the expected shape."""


@dataclass(frozen=True)
class HookResult:
    """What the hook prints and returns."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


def parse_input(text: str, expected_event: str) -> dict[str, Any]:
    """Parse and validate hook input."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise HookInputError(f"Hook input is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise HookInputError("Hook input must be a JSON object.")
    event = data.get("hook_event_name")
    if event != expected_event:
        raise HookInputError(f"Expected hook_event_name {expected_event!r}, got {event!r}.")
    if expected_event == "PostToolUse":
        if not isinstance(data.get("tool_name"), str):
            raise HookInputError("PostToolUse input has no 'tool_name' string.")
        if not isinstance(data.get("tool_input", {}), dict):
            raise HookInputError("PostToolUse 'tool_input' must be an object.")
    if expected_event == "Stop" and not isinstance(data.get("stop_hook_active", False), bool):
        raise HookInputError("Stop input 'stop_hook_active' must be a boolean.")
    return data


def project_root(data: dict[str, Any]) -> Path:
    """``$CLAUDE_PROJECT_DIR``, else the input ``cwd``, else the current directory."""
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env).resolve()
    cwd = data.get("cwd")
    return Path(cwd).resolve() if isinstance(cwd, str) and cwd else Path.cwd().resolve()


def _tool_failed(response: object) -> bool:
    if isinstance(response, dict):
        return response.get("success") is False or bool(response.get("is_error"))
    return False


def post_tool_use(data: dict[str, Any], root: Path) -> HookResult:
    """Record benchmark-relevant changes made by one tool call."""
    if os.environ.get(GUARD_ENV):
        return HookResult()
    state = HookState.load(root)
    if state.mode() == "off":
        return HookResult()
    tool = str(data["tool_name"])
    if _tool_failed(data.get("tool_response")):
        return HookResult()
    with file_lock(root / STATE_DIR / "state.lock", blocking=True):
        state = HookState.load(root)
        changed: list[str] = []
        if tool in FILE_TOOLS:
            rel = relativize(str(data.get("tool_input", {}).get(FILE_TOOLS[tool], "")), root)
            if rel is not None:
                changed = [rel]
            snapshot = load_snapshot(root)
            if snapshot is not None and rel is not None:
                # Keep the snapshot current so a later Bash diff does not report this file again.
                path = root / rel
                if path.is_file():
                    stat = path.stat()
                    snapshot[rel] = [stat.st_size, stat.st_mtime_ns]
                else:
                    snapshot.pop(rel, None)
                save_snapshot(root, snapshot)
        elif tool == "Bash":
            before = load_snapshot(root)
            after = take_snapshot(root)
            save_snapshot(root, after)
            changed = _git_changed(root) if before is None else diff_snapshots(before, after)
        recorded = []
        for rel in changed:
            category = classify(rel)
            if category is not None:
                state.record(rel, category, tool)
                recorded.append(rel)
        if recorded:
            state.save()
            log(root, "post_tool_use", tool=tool, recorded=recorded)
    return HookResult()


def _git_changed(root: Path) -> list[str]:
    """First-run fallback: files Git reports as changed or untracked."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted(
        {line[3:].split(" -> ")[-1].strip('"') for line in out.splitlines() if len(line) > 3}
    )


def _run_mtimes(root: Path) -> dict[str, int]:
    runs = root / "runs"
    if not runs.is_dir():
        return {}
    return {
        p.relative_to(root).as_posix(): p.stat().st_mtime_ns
        for p in sorted(runs.glob("*/training_summary.json"))
    }


def _unseen_training_runs(root: Path, state: HookState) -> list[str]:
    """Training summaries that are new or changed since the last successful Stop run."""
    return [rel for rel, mtime in _run_mtimes(root).items() if state.seen_runs.get(rel) != mtime]


def _format(report: WorkflowReport, categories: set[ChangeCategory]) -> str:
    lines = [f"STEward benchmark workflow ({', '.join(sorted(c.value for c in categories))}):"]
    for step in report.steps:
        lines.append(f"  [{'ok' if step.ok else 'FAILED'}] {step.name}")
        if not step.ok and step.detail:
            lines += [f"      {line}" for line in step.detail.splitlines()[:15]]
    for item in report.regressions:
        lines.append(f"  REGRESSION: {item}")
    for item in report.improvements:
        lines.append(f"  improvement: {item}")
    for item in report.outstanding:
        lines.append(f"  outstanding: {item}")
    if not report.steps:
        lines.append("  No checks were needed.")
    return "\n".join(lines)


def stop(
    data: dict[str, Any],
    root: Path,
    *,
    runner: CommandRunner | None = None,
    provider_factory: ProviderFactory | None = None,
) -> HookResult:
    """Run the workflow when relevant changes are pending."""
    if os.environ.get(GUARD_ENV):
        return HookResult()
    state = HookState.load(root)
    mode = state.mode()
    if mode == "off":
        return HookResult()
    new_runs = _unseen_training_runs(root, state)
    if new_runs:
        # Training may run outside Claude's file tools; a finished run is always picked up.
        with file_lock(root / STATE_DIR / "state.lock", blocking=True):
            state = HookState.load(root)
            for rel in new_runs:
                state.record(rel, ChangeCategory.MODEL, "training-scan")
            state.save()
    if not state.pending:
        return HookResult()
    already_blocked = bool(data.get("stop_hook_active", False))
    with file_lock(root / STATE_DIR / "workflow.lock", blocking=False) as acquired:
        if not acquired:
            msg = "STEward: another validation run is active; changes stay pending."
            log(root, "stop_skipped_locked")
            return HookResult(stdout=json.dumps({"systemMessage": msg}))
        state = HookState.load(root)
        pending = {p: ChangeCategory(e["category"]) for p, e in state.pending.items()}
        categories = set(pending.values())
        fingerprints = category_fingerprints(root, categories, state.fingerprint_cache)
        if all(state.validated.get(k) == v for k, v in fingerprints.items()):
            state.pending.clear()
            state.save()
            log(root, "stop_unchanged_inputs", categories=sorted(fingerprints))
            return HookResult()
        os.environ[GUARD_ENV] = "1"
        try:
            report = run_workflow(
                root,
                pending,
                mode=mode,
                runner=runner or subprocess_runner(),
                provider_factory=provider_factory or default_provider_factory(None),
                cache=state.fingerprint_cache,
            )
        except Exception as exc:
            report = WorkflowReport()
            from steward.automation.workflow import Step

            report.steps.append(Step("workflow", False, f"{type(exc).__name__}: {exc}"))
        finally:
            os.environ.pop(GUARD_ENV, None)
        message = _format(report, categories)
        state = HookState.load(root)
        state.outstanding = report.outstanding
        state.last_stop = {
            "finished_at": now_iso(),
            "result": "passed" if report.ok else "failed",
            "categories": sorted(c.value for c in categories),
            "failures": [s.name for s in report.failures()],
        }
        if report.ok:
            state.seen_runs.update(_run_mtimes(root))
            # Fingerprints after the workflow, so its own outputs do not look like new changes.
            state.validated.update(category_fingerprints(root, categories, state.fingerprint_cache))
            for path in pending:
                state.pending.pop(path, None)
        state.save()
        log(root, "stop_finished", ok=report.ok, steps=[(s.name, s.ok) for s in report.steps])
    if report.ok:
        return HookResult(stdout=json.dumps({"systemMessage": message}))
    if already_blocked:
        return HookResult(
            stdout=json.dumps(
                {
                    "systemMessage": message
                    + "\n  Validation is still FAILING. Changes stay pending; "
                    "see `pixi run benchmark-status`."
                }
            )
        )
    return HookResult(
        stdout=json.dumps(
            {
                "decision": "block",
                "reason": message
                + "\nFix these problems before you finish. If they cannot be fixed now, "
                "explain to the user what failed and why.",
            }
        )
    )


def _main(event: str, stdin: TextIO, handler: Any) -> int:
    try:
        data = parse_input(stdin.read(), event)
    except HookInputError as exc:
        print(f"STEward {event} hook: {exc}", file=sys.stderr)
        return 1
    root = project_root(data)
    try:
        result: HookResult = handler(data, root)
    except Exception as exc:
        log(root, f"{event}_error", error=repr(exc))
        print(f"STEward {event} hook error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.exit_code


def main_post_tool_use(stdin: TextIO = sys.stdin) -> int:
    """Entry point for ``scripts/hooks/post_tool_use.py``."""
    return _main("PostToolUse", stdin, post_tool_use)


def main_stop(stdin: TextIO = sys.stdin) -> int:
    """Entry point for ``scripts/hooks/stop.py``."""
    return _main("Stop", stdin, stop)
