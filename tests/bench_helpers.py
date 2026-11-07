"""Helpers for benchmark tests: a temporary project, a fake model, test records.

Records built here exist only inside tests. They are never written to the
repository's benchmark history.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from steward.datasets.schema import PromptRef, SamplingParameters
from steward.evaluation.history import finalize_record
from steward.evaluation.records import (
    BenchmarkRecord,
    CandidateRole,
    CandidateSpec,
    CodeRef,
    EvaluatorRef,
    Hardware,
    Metric,
    RecordStatus,
    SuiteRef,
)
from steward.generation.providers.ollama import OllamaProvider

REPO = Path(__file__).resolve().parents[1]

MODELS_TOML = """
[[candidates]]
id = "tiny-base"
role = "base"
provider = "ollama"
model = "tiny:1b"
base_model = "Example/Tiny-1B"
prompt = "benchmark_minimal@1"

[[candidates]]
id = "tiny-prompted"
role = "prompted"
provider = "ollama"
model = "tiny:1b"
base_model = "Example/Tiny-1B"
prompt = "ste_rewrite@1"
"""


def make_project(tmp_path: Path) -> Path:
    """Copy the committed benchmark inputs into a temporary project root."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "pixi.toml").write_text("[workspace]\nname = 'test'\n", encoding="utf-8")
    for rel in (
        "prompts",
        "configs",
        "benchmarks/golden",
        "src/steward/evaluation",
        "src/steward/ste",
    ):
        shutil.copytree(REPO / rel, root / rel, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO / "benchmarks/thresholds.toml", root / "benchmarks/thresholds.toml")
    (root / "benchmarks/models.toml").write_text(MODELS_TOML, encoding="utf-8")
    (root / "README.md").write_text("# Test project\n", encoding="utf-8")
    (root / "docs").mkdir()
    for doc in ("benchmark-system.md", "training.md", "generation.md", "dataset-quality.md"):
        shutil.copy(REPO / "docs" / doc, root / "docs" / doc)
    return root


class BenchFake:
    """Fake Ollama for benchmarks. By default it echoes the source text as the rewrite."""

    def __init__(self, model: str = "tiny:1b", parameter_size: str = "1.2B") -> None:
        self.model = model
        self.parameter_size = parameter_size
        self.calls: list[dict[str, Any]] = []
        self.queue: list[Any] = []
        self.available = True
        self.refuse_after: int | None = None
        self.rewrite: Callable[[str], str] = lambda source: source

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.0-test"})
        if request.url.path == "/api/ps":
            return httpx.Response(
                200, json={"models": [{"name": self.model, "size_vram": 2 * 1024**3}]}
            )
        if request.url.path == "/api/tags":
            models = [
                {
                    "name": self.model,
                    "model": self.model,
                    "digest": "e" * 64,
                    "size": 1_000_000_000,
                    "details": {
                        "parameter_size": self.parameter_size,
                        "quantization_level": "Q4_K_M",
                    },
                }
            ]
            return httpx.Response(200, json={"models": models if self.available else []})
        body = json.loads(request.content)
        self.calls.append(body)
        if self.refuse_after is not None and len(self.calls) > self.refuse_after:
            raise httpx.ConnectError("refused", request=request)
        if self.queue:
            content = self.queue.pop(0)
        else:
            user = body["messages"][1]["content"]
            source = re.search(r"<<<\n(.*)\n>>>", user, re.DOTALL)
            content = json.dumps({"ste_text": self.rewrite(source.group(1) if source else "")})
        return httpx.Response(
            200,
            json={
                "model": self.model,
                "message": {"role": "assistant", "content": content},
                "done": True,
                "done_reason": "stop",
                "eval_count": 50,
                "eval_duration": 500_000_000,
            },
        )

    def provider(self, **_: object) -> OllamaProvider:
        return OllamaProvider(base_url="http://fake", transport=httpx.MockTransport(self.handler))


T0 = datetime(2026, 11, 1, tzinfo=UTC)


def make_record(
    candidate_id: str = "cand",
    *,
    role: CandidateRole = CandidateRole.PROMPTED,
    params: int | None = 1_000_000_000,
    disk: int | None = 1_000_000_000,
    metrics: Mapping[str, float | None] | None = None,
    ci: dict[str, tuple[float, float]] | None = None,
    suite_version: int = 0,
    evaluator_version: str = "eval-0.1",
    status: RecordStatus = RecordStatus.COMPLETED,
    created: datetime | None = None,
    dataset_version: str | None = None,
    base_model: str = "Example/Tiny-1B",
    rule_counts: dict[str, int] | None = None,
) -> BenchmarkRecord:
    """Build a finalized test record with chosen metric values."""
    base = {
        "ste_rule_pass_rate": 0.5,
        "violations_per_100_words": 5.0,
        "grammar_violations_per_100_words": 2.0,
        "meaning_preservation": 0.97,
        "hallucination_rate": 0.0,
        "missing_information_rate": 0.05,
        "critical_failures": 0.0,
        "task_completion_rate": 1.0,
        "latency_mean_s": 1.0,
        "tokens_per_s": 100.0,
        "parameter_count": float(params) if params else None,
        "disk_size_bytes": float(disk) if disk else None,
    }
    base.update(metrics or {})
    ci = ci or {}
    metric_objs = (
        {
            k: Metric(
                value=v,
                unit="x",
                n=16,
                ci_low=ci.get(k, (None, None))[0],
                ci_high=ci.get(k, (None, None))[1],
            )
            for k, v in base.items()
        }
        if status is RecordStatus.COMPLETED
        else {}
    )
    data = {
        "status": status,
        "error": None if status is RecordStatus.COMPLETED else "provider failed",
        "code": CodeRef(
            steward_version="0.1.0", git_commit="abc", git_dirty=False, worktree_fingerprint="w"
        ),
        "hardware": Hardware(platform="test"),
        "candidate": CandidateSpec(
            candidate_id=candidate_id,
            role=role,
            provider="ollama",
            model=f"{candidate_id}:x",
            model_digest="d" * 64,
            base_model=base_model,
            parameter_count=params,
            disk_size_bytes=disk,
            prompt=PromptRef(template_id="t", version=1, sha256="0" * 64),
        ),
        "suite": SuiteRef(
            suite_id="ste-golden",
            version=suite_version,
            sha256="a" * 64,
            item_count=0,
            review_state="draft",
        ),
        "evaluator": EvaluatorRef(
            evaluator_version=evaluator_version,
            ruleset_version="r1",
            rules_config_sha256="b" * 64,
            code_fingerprint="c",
        ),
        "sampling": SamplingParameters(temperature=0.0, seed=1),
        "dataset_version": dataset_version,
        "metrics": metric_objs,
        "rule_counts": rule_counts or {"STE-GRAM-ING": 2},
    }
    record = finalize_record(BenchmarkRecord, data, created or T0)
    assert isinstance(record, BenchmarkRecord)
    return record


def later(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)
