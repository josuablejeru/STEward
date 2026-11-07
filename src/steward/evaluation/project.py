"""Paths and loaders for the benchmark system of one project checkout."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from steward.evaluation.history import benchmarks, read_history, trainings
from steward.evaluation.records import BenchmarkRecord, TrainingRecord
from steward.evaluation.status import current_key
from steward.evaluation.suite import (
    BenchmarkConfigError,
    CandidateRegistry,
    Suite,
    Thresholds,
    load_registry,
    load_suite,
    load_thresholds,
)
from steward.ste.rules import RuleConfig, load_rule_config

DEFAULT_SUITE = "benchmarks/golden/ste-golden-v0"


@dataclass(frozen=True)
class BenchmarkPaths:
    """All benchmark files, relative to the project root."""

    root: Path
    suite_dir: Path
    history: Path
    summary: Path
    figures: Path
    report: Path
    readme: Path
    registry: Path
    thresholds: Path
    terms: Path
    prompts: Path
    runs: Path

    @classmethod
    def for_root(cls, root: Path, suite: str = DEFAULT_SUITE) -> BenchmarkPaths:
        """Use the standard layout under ``root``."""
        return cls(
            root=root,
            suite_dir=root / suite,
            history=root / "benchmarks/results/history.jsonl",
            summary=root / "benchmarks/results/summary.json",
            figures=root / "benchmarks/figures",
            report=root / "docs/benchmarks.md",
            readme=root / "README.md",
            registry=root / "benchmarks/models.toml",
            thresholds=root / "benchmarks/thresholds.toml",
            terms=root / "configs/ste/terms.toml",
            prompts=root / "prompts",
            runs=root / "runs",
        )


class BenchmarkProject:
    """Lazy loaders for one checkout. Errors surface when a part is used."""

    def __init__(self, paths: BenchmarkPaths) -> None:
        """Wrap ``paths``."""
        self.paths = paths

    @classmethod
    def at(cls, root: Path) -> BenchmarkProject:
        """Create a project for ``root`` with the standard layout."""
        return cls(BenchmarkPaths.for_root(root))

    @cached_property
    def primary_suite_name(self) -> str:
        """Directory name of the primary suite (``primary_suite`` in models.toml)."""
        try:
            return self.registry.primary_suite
        except BenchmarkConfigError:
            return self.paths.suite_dir.name

    @cached_property
    def suite(self) -> Suite:
        """The primary frozen golden suite (model selection, report, README)."""
        return self.suite_named(self.primary_suite_name)

    def suite_named(self, name: str) -> Suite:
        """Load a suite by directory name (cached)."""
        cache: dict[str, Suite] = self.__dict__.setdefault("_suites", {})
        if name not in cache:
            directory = (
                self.paths.suite_dir
                if name == self.paths.suite_dir.name
                else self.paths.suite_dir.parent / name
            )
            cache[name] = load_suite(directory)
        return cache[name]

    def suite_names(self) -> list[str]:
        """The primary suite plus every suite a registered candidate uses."""
        names = [self.primary_suite_name]
        with contextlib.suppress(BenchmarkConfigError):
            names += [c.suite for c in self.registry.candidates]
        return list(dict.fromkeys(names))

    def key_for(self, name: str) -> str:
        """Comparability key that new results on suite ``name`` would get."""
        return current_key(self.suite_named(name), self.rules)

    @property
    def keys(self) -> dict[str, str]:
        """Suite name -> current comparability key, for every suite in use."""
        return {name: self.key_for(name) for name in self.suite_names()}

    @cached_property
    def rules(self) -> RuleConfig:
        """Rule configuration."""
        return load_rule_config(self.paths.terms)

    @cached_property
    def registry(self) -> CandidateRegistry:
        """Benchmark candidates."""
        return load_registry(self.paths.registry)

    @cached_property
    def thresholds(self) -> Thresholds:
        """Regression thresholds and selection requirements."""
        return load_thresholds(self.paths.thresholds)

    @cached_property
    def key(self) -> str:
        """Current comparability key."""
        return current_key(self.suite, self.rules)

    def history(self) -> list[BenchmarkRecord | TrainingRecord]:
        """Read the history (not cached: it can change during a run)."""
        return read_history(self.paths.history)

    def benchmarks(self) -> list[BenchmarkRecord]:
        """Benchmark records."""
        return benchmarks(self.history())

    def trainings(self) -> list[TrainingRecord]:
        """Training records."""
        return trainings(self.history())
