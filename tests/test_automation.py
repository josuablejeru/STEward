import io
import json
import os
from pathlib import Path

import pytest

from bench_helpers import BenchFake, make_project
from steward.automation import hooks
from steward.automation.changes import ChangeCategory, classify, diff_snapshots, relativize
from steward.automation.hooks import HookInputError, main_stop, parse_input, post_tool_use, stop
from steward.automation.state import GUARD_ENV, STATE_DIR, HookState, file_lock
from steward.evaluation import hardware
from steward.evaluation.project import BenchmarkProject
from steward.evaluation.records import RecordStatus, ServingSpec, TrainingSummary


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(hardware, "gpu_info", lambda: None)
    monkeypatch.setattr(hardware, "gpu_memory_used_mib", lambda: None)
    monkeypatch.delenv(GUARD_ENV, raising=False)
    monkeypatch.delenv("STEWARD_HOOKS", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    monkeypatch.setattr("steward.evaluation.plots.render_all", _stub_render)
    return make_project(tmp_path)


def _stub_render(  # type: ignore[no-untyped-def]
    benchmarks, trainings, out_dir, formats=("svg", "png"), names=None, primary_key=None
):
    """Fast stand-in for figure rendering (real rendering is tested elsewhere)."""
    from steward.evaluation.plots import FIGURES

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names or FIGURES:
        for fmt in formats:
            path = out_dir / f"{name}.{fmt}"
            path.write_text(
                f"<svg>{name} {len(benchmarks)} {len(trainings)}</svg>", encoding="utf-8"
            )
            paths.append(path)
    return paths


class FakeRunner:
    def __init__(self, fail: set[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail or set()

    def __call__(self, cmd, cwd):  # type: ignore[no-untyped-def]
        self.calls.append(list(cmd))
        assert os.environ.get(GUARD_ENV) == "1"  # reentrancy guard is set for child processes
        return (1, f"{cmd[-1]} failed: E123") if cmd[-1] in self.fail else (0, "ok")


def post(root: Path, tool: str, **tool_input: object) -> None:
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": tool,
        "tool_input": tool_input,
        "tool_response": {"success": True},
        "cwd": str(root),
    }
    post_tool_use(payload, root)


def stop_input(active: bool = False) -> dict[str, object]:
    return {"hook_event_name": "Stop", "stop_hook_active": active}


# -- classification ---------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "category"),
    [
        ("data/raw/examples.jsonl", ChangeCategory.DATA),
        ("src/steward/datasets/schema.py", ChangeCategory.DATA),
        ("src/steward/generation/pipeline.py", ChangeCategory.GENERATION),
        ("prompts/ste_rewrite/v2.toml", ChangeCategory.GENERATION),
        ("prompts/benchmark_minimal/v1.toml", ChangeCategory.EVALUATION),
        ("configs/generation/default.toml", ChangeCategory.GENERATION),
        ("src/steward/training/sft.py", ChangeCategory.TRAINING),
        ("configs/training/lora.toml", ChangeCategory.TRAINING),
        ("src/steward/ste/rules.py", ChangeCategory.EVALUATION),
        ("benchmarks/golden/ste-golden-v0/items.jsonl", ChangeCategory.EVALUATION),
        ("benchmarks/thresholds.toml", ChangeCategory.EVALUATION),
        ("benchmarks/models.toml", ChangeCategory.MODEL),
        ("runs/r1/training_summary.json", ChangeCategory.MODEL),
        ("checkpoints/a/adapter.safetensors", ChangeCategory.MODEL),
        ("benchmarks/results/history.jsonl", None),
        ("benchmarks/figures/pareto_frontier.svg", None),
        ("docs/benchmarks.md", None),
        (".steward/state/hooks.json", None),
        ("src/steward/cli.py", None),
        ("README.md", None),
        ("src/steward/ste/__pycache__/rules.cpython-312.pyc", None),
    ],
)
def test_classify(path: str, category: ChangeCategory | None) -> None:
    assert classify(path) == category


def test_relativize(tmp_path: Path) -> None:
    assert relativize(str(tmp_path / "data" / "x.jsonl"), tmp_path) == "data/x.jsonl"
    assert relativize("data/x.jsonl", tmp_path) == "data/x.jsonl"
    assert relativize("/etc/passwd", tmp_path) is None
    assert relativize("", tmp_path) is None


def test_diff_snapshots() -> None:
    assert diff_snapshots({"a": [1, 1], "b": [1, 1]}, {"a": [1, 2], "c": [1, 1]}) == ["a", "b", "c"]


# -- input parsing -------------------------------------------------------------


def test_parse_input_validation() -> None:
    with pytest.raises(HookInputError, match="not JSON"):
        parse_input("{", "Stop")
    with pytest.raises(HookInputError, match="Expected hook_event_name"):
        parse_input('{"hook_event_name": "Stop"}', "PostToolUse")
    with pytest.raises(HookInputError, match="tool_name"):
        parse_input('{"hook_event_name": "PostToolUse"}', "PostToolUse")
    with pytest.raises(HookInputError, match="stop_hook_active"):
        parse_input('{"hook_event_name": "Stop", "stop_hook_active": "yes"}', "Stop")
    assert parse_input('{"hook_event_name": "Stop", "stop_hook_active": true}', "Stop")


def test_main_reports_bad_input(capsys: pytest.CaptureFixture[str]) -> None:
    assert main_stop(io.StringIO("garbage")) == 1
    assert "not JSON" in capsys.readouterr().err


# -- PostToolUse ------------------------------------------------------------------


def test_write_edit_are_recorded(root: Path) -> None:
    post(root, "Write", file_path=str(root / "src/steward/ste/rules.py"))
    post(root, "Edit", file_path=str(root / "src/steward/ste/rules.py"))
    post(root, "Write", file_path=str(root / "benchmarks/figures/x.svg"))
    post(root, "Write", file_path=str(root / "README.md"))
    state = HookState.load(root)
    assert list(state.pending) == ["src/steward/ste/rules.py"]
    assert state.pending["src/steward/ste/rules.py"]["tools"] == ["Write", "Edit"]


def test_failed_tool_is_ignored(root: Path) -> None:
    post_tool_use(
        {
            "hook_event_name": "PostToolUse",
            "tool_name": "Write",
            "tool_input": {"file_path": str(root / "configs/ste/terms.toml")},
            "tool_response": {"success": False},
        },
        root,
    )
    assert HookState.load(root).pending == {}


def test_bash_changes_are_detected_by_snapshot(root: Path) -> None:
    post(root, "Bash", command="ls")  # first call: baseline (no git repo -> nothing recorded)
    (root / "data").mkdir()
    (root / "data" / "new.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "benchmarks/results").mkdir(parents=True, exist_ok=True)
    (root / "benchmarks/results/history.jsonl").write_text("", encoding="utf-8")
    post(root, "Bash", command="python make_data.py")
    assert HookState.load(root).pending_summary() == {"data": ["data/new.jsonl"]}


def test_guard_and_off_mode_disable_recording(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GUARD_ENV, "1")
    post(root, "Write", file_path=str(root / "configs/ste/terms.toml"))
    monkeypatch.delenv(GUARD_ENV)
    monkeypatch.setenv("STEWARD_HOOKS", "off")
    post(root, "Write", file_path=str(root / "configs/ste/terms.toml"))
    assert HookState.load(root).pending == {}


# -- Stop --------------------------------------------------------------------------


def _out(result: hooks.HookResult) -> dict[str, object]:
    return json.loads(result.stdout) if result.stdout else {}


def test_stop_without_changes_does_nothing(root: Path) -> None:
    runner = FakeRunner()
    assert stop(stop_input(), root, runner=runner).stdout == ""
    assert runner.calls == []


def test_stop_runs_validation_and_clears_pending(root: Path) -> None:
    post(root, "Edit", file_path=str(root / "src/steward/ste/rules.py"))
    runner = FakeRunner()
    result = stop(
        stop_input(), root, runner=runner, provider_factory=lambda _: BenchFake().provider()
    )
    assert [c[-1] for c in runner.calls] == ["lint", "typecheck", "test"]
    message = str(_out(result)["systemMessage"])
    assert "[ok] pixi run test" in message
    assert "No current result for registered candidate tiny-base" in message
    assert (root / "docs/benchmarks.md").exists()
    state = HookState.load(root)
    assert state.pending == {} and state.last_stop and state.last_stop["result"] == "passed"
    # Same inputs again: nothing runs.
    post(root, "Edit", file_path=str(root / "src/steward/ste/rules.py"))
    runner2 = FakeRunner()
    assert stop(stop_input(), root, runner=runner2).stdout == ""
    assert runner2.calls == []


def test_stop_failure_blocks_once_then_reports(root: Path) -> None:
    post(root, "Edit", file_path=str(root / "src/steward/generation/pipeline.py"))
    (root / "src/steward/generation").mkdir(parents=True, exist_ok=True)
    (root / "src/steward/generation/pipeline.py").write_text("x = 1\n", encoding="utf-8")
    first = _out(stop(stop_input(), root, runner=FakeRunner({"test"})))
    assert first["decision"] == "block"
    assert "test failed: E123" in str(first["reason"])
    second = _out(stop(stop_input(active=True), root, runner=FakeRunner({"test"})))
    assert "decision" not in second
    assert "still FAILING" in str(second["systemMessage"])
    state = HookState.load(root)
    assert "src/steward/generation/pipeline.py" in state.pending
    assert state.last_stop and state.last_stop["result"] == "failed"


def test_stop_skips_when_locked(root: Path) -> None:
    post(root, "Edit", file_path=str(root / "src/steward/ste/rules.py"))
    with file_lock(root / STATE_DIR / "workflow.lock", blocking=False) as got:
        assert got
        result = _out(stop(stop_input(), root, runner=FakeRunner()))
    assert "another validation run" in str(result["systemMessage"])
    assert HookState.load(root).pending


def test_data_change_marks_retraining_without_scores(root: Path) -> None:
    (root / "data").mkdir()
    (root / "data/x.jsonl").write_text("", encoding="utf-8")
    post(root, "Write", file_path=str(root / "data/x.jsonl"))
    result = _out(stop(stop_input(), root, runner=FakeRunner()))
    message = str(result["systemMessage"])
    assert "[ok] dataset integrity" in message
    assert "Training and evaluation are required" in message
    assert BenchmarkProject.at(root).history() == []


def test_training_change_without_checkpoint_is_pending(root: Path) -> None:
    (root / "src/steward/training").mkdir(parents=True)
    (root / "src/steward/training/sft.py").write_text("x = 1\n", encoding="utf-8")
    post(root, "Write", file_path=str(root / "src/steward/training/sft.py"))
    message = str(_out(stop(stop_input(), root, runner=FakeRunner()))["systemMessage"])
    assert "no new checkpoint exists" in message
    assert BenchmarkProject.at(root).history() == []


def _summary(serving: ServingSpec | None) -> TrainingSummary:
    return TrainingSummary(
        run_id="run1",
        status=RecordStatus.COMPLETED,
        base_model="Example/Tiny-1B",
        checkpoint_id="ckpt-1",
        checkpoint_path="runs/run1/adapter",
        dataset_version="v1",
        dataset_fingerprint="f" * 64,
        config_sha256="c" * 64,
        parameter_count=1_200_000_000,
        hyperparameters={"lr": 2e-4},
        serving=serving,
    )


def _write_run(root: Path, summary: TrainingSummary) -> None:
    run = root / "runs/run1"
    run.mkdir(parents=True)
    (run / "training_summary.json").write_text(summary.model_dump_json(), encoding="utf-8")


def test_new_checkpoint_triggers_benchmark(root: Path) -> None:
    _write_run(root, _summary(ServingSpec(provider="ollama", model="tiny:1b")))
    post(root, "Bash", command="pixi run -e cuda train")
    fake = BenchFake()
    message = str(
        _out(
            stop(
                stop_input(), root, runner=FakeRunner(), provider_factory=lambda _: fake.provider()
            )
        )["systemMessage"]
    )
    assert "[ok] benchmark ft-run1" in message
    history = BenchmarkProject.at(root).history()
    assert [r.record_type for r in history] == ["training", "benchmark"]
    assert (root / "benchmarks/figures/training_loss.svg").exists()


def test_unavailable_model_stays_pending(root: Path) -> None:
    _write_run(root, _summary(ServingSpec(provider="ollama", model="tiny:1b")))
    post(root, "Bash", command="train")
    gone = BenchFake()
    gone.available = False
    message = str(
        _out(
            stop(
                stop_input(), root, runner=FakeRunner(), provider_factory=lambda _: gone.provider()
            )
        )["systemMessage"]
    )
    assert "Benchmark pending for ft-run1" in message
    assert [r.record_type for r in BenchmarkProject.at(root).history()] == ["training"]


def test_checkpoint_without_serving_spec_is_pending(root: Path) -> None:
    _write_run(root, _summary(None))
    post(root, "Bash", command="train")
    message = str(_out(stop(stop_input(), root, runner=FakeRunner()))["systemMessage"])
    assert "Training run run1 was recorded but cannot be benchmarked" in message
    assert "no serving spec" in message
    assert "pending" in message


def test_checkpoint_with_missing_adapter_is_not_benchmarked(root: Path) -> None:
    serving = ServingSpec(
        provider="transformers", model="Qwen/Qwen3-0.6B", adapter_path="runs/run1/adapter"
    )
    _write_run(root, _summary(serving))
    post(root, "Bash", command="train")
    bench = BenchFake()
    message = str(
        _out(
            stop(
                stop_input(), root, runner=FakeRunner(), provider_factory=lambda _: bench.provider()
            )
        )["systemMessage"]
    )
    assert "adapter_config.json" in message
    assert [r.record_type for r in BenchmarkProject.at(root).history()] == ["training"]


def test_light_mode_never_runs_models(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEWARD_HOOKS", "light")
    _write_run(root, _summary(ServingSpec(provider="ollama", model="tiny:1b")))
    post(root, "Bash", command="train")

    def no_provider(_):  # type: ignore[no-untyped-def]
        raise AssertionError("light mode must not call a model")

    message = str(
        _out(stop(stop_input(), root, runner=FakeRunner(), provider_factory=no_provider))[
            "systemMessage"
        ]
    )
    assert "light mode skips model runs" in message


def test_diagnostics_put_failures_first() -> None:
    from steward.automation.workflow import _diagnostics

    noise = "\n".join(f"INFO log line {i}" for i in range(100))
    out = _diagnostics(
        noise + "\nFAILED tests/test_x.py::test_y - AssertionError\n" + noise + "\n1 failed"
    )
    lines = out.splitlines()
    assert lines[0].startswith("FAILED tests/test_x.py::test_y")
    assert "1 failed" in out
    assert len(lines) <= 30
