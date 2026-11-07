"""Prepare a supervised fine-tuning (SFT) set from accepted synthetic examples.

Steps (all deterministic for a fixed config and input):

1. Load accepted :class:`DatasetExample` records from the input files.
2. Re-run the current multitask checks. Exclude examples with a failed
   ``error`` check (lost or invented facts, lost safety labels, non-English
   targets) and, by default, targets that fail the heuristic STE rule check.
3. Remove exact and near duplicates (by source and by target).
4. Remove anything that matches a golden benchmark item (any suite).
5. Split by *scenario topic*, so the same subject never appears in both
   train and validation.
6. Derive STE identity examples from clean targets, and add out-of-domain
   examples from a hand-written list, both inside the same split as their
   group.
7. Down-sample to the configured task mix (optional) and write
   ``train.jsonl``, ``validation.jsonl``, and ``manifest.json``.

Every output record is still a *candidate*: nothing here is human review.
"""

from __future__ import annotations

import hashlib
import json
import random
import tomllib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from steward.datasets.jsonl import (
    DatasetFileError,
    atomic_write_text,
    fingerprint,
    normalize_for_dedup,
    read_jsonl,
    write_jsonl_atomic,
)
from steward.datasets.schema import (
    COMPLIANCE_NOTE,
    DatasetExample,
    ReviewState,
    SourceLanguage,
    TaskCategory,
    TaskType,
)
from steward.generation.config import ValidationSettings
from steward.generation.validation import MULTITASK_CHECKER_VERSION, validate_multitask
from steward.ste import preservation as pv
from steward.ste.rules import RuleConfig, RuleSeverity, check_text

SFT_SCHEMA_VERSION: Literal["1.0"] = "1.0"
PROCEDURAL = {TaskCategory.PROCEDURE, TaskCategory.TROUBLESHOOTING, TaskCategory.SAFETY_NOTICE}


class SftConfigError(Exception):
    """The SFT preparation config is missing or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OutOfDomainPrompt(_Strict):
    """A hand-written request that the STE-only model must refuse."""

    language: SourceLanguage
    text: str = Field(min_length=1)


class SftConfig(_Strict):
    """``configs/sft/*.toml``."""

    seed: int = 0
    prompt: str = "ste_only@1"
    validation_fraction: float = Field(default=0.12, gt=0, lt=0.5)
    mix: dict[TaskType, float]
    """Target shares of the in-domain tasks (identity included). Normalized."""
    enforce_mix: bool = True
    """Down-sample tasks with a surplus so the mix matches ``mix``."""
    out_of_domain_fraction: float = Field(default=0.05, ge=0, lt=0.5)
    """Out-of-domain examples, as a fraction of the in-domain examples."""
    require_rule_pass: bool = True
    """Exclude targets with an ``error`` from the heuristic STE rule check."""
    near_duplicate_threshold: float = Field(default=0.85, gt=0, le=1)
    out_of_domain: list[OutOfDomainPrompt] = Field(default_factory=list)
    validation: ValidationSettings = ValidationSettings()
    require_llm_review: bool = Field(default=False, exclude_if=lambda v: v is False)
    """Exclude examples without an independent LLM critique, or with a critical finding."""
    max_warning_caution_ratio: float | None = Field(
        default=None, gt=0, exclude_if=lambda v: v is None
    )
    """Down-sample WARNING-only targets until #WARNING <= ratio x #CAUTION targets."""
    prefer_focus: bool = Field(default=False, exclude_if=lambda v: v is False)
    """When a task is down-sampled, keep targeted (focus) examples first, and identity
    examples with labels, negations, or several lines first."""

    def sha256(self) -> str:
        """Stable hash of the effective config."""
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()


def load_sft_config(path: Path) -> SftConfig:
    """Load and validate an SFT preparation config."""
    try:
        return SftConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise SftConfigError(f"SFT config not found: {path}") from None
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise SftConfigError(f"Invalid SFT config {path}: {exc}") from exc


class SftRecord(_Strict):
    """One training or validation example for the STE-only student."""

    schema_version: Literal["1.0"] = SFT_SCHEMA_VERSION
    sft_id: str
    split: Literal["train", "validation"]
    task_type: TaskType
    task_category: TaskCategory | None
    source_language: SourceLanguage
    input: str = Field(min_length=1)
    target: str = Field(min_length=1)
    group: str = Field(description="Split group (scenario topic hash). Never crosses splits.")
    parent_example_id: str | None = None
    technical_entities: list[str] = Field(default_factory=list)
    safety_constraints: list[str] = Field(default_factory=list)
    teacher_model: str | None = None
    teacher_prompts: list[str] = Field(default_factory=list)
    generation_run_id: str | None = None
    checker_version: str
    warning_checks: list[str] = Field(default_factory=list)
    rule_errors: int = 0
    review_state: ReviewState = ReviewState.CANDIDATE
    compliance_note: str = COMPLIANCE_NOTE
    synthetic: Literal[True] = True
    focus: str | None = Field(default=None, exclude_if=lambda v: v is None)
    labels: list[str] | None = Field(default=None, exclude_if=lambda v: v is None)
    """WARNING/CAUTION/NOTE labels of the target."""
    llm_review: str | None = Field(default=None, exclude_if=lambda v: v is None)
    """Status of the independent LLM critique (passed, revised); an LLM judgement."""


class SftManifest(_Strict):
    """``manifest.json``: what went in, what came out, and why examples were dropped."""

    schema_version: Literal["1.0"] = SFT_SCHEMA_VERSION
    dataset_fingerprint: str
    config_sha256: str
    config: dict[str, object]
    prompt: str
    inputs: dict[str, str]
    """Input path -> SHA-256."""
    golden_suites: list[str]
    counts: dict[str, dict[str, int]]
    """split -> task -> count."""
    languages: dict[str, dict[str, int]]
    exclusions: dict[str, int]
    mix_target: dict[str, float]
    mix_actual: dict[str, float]
    groups: dict[str, list[str]]
    """split -> group IDs."""
    label: str
    note: str = (
        "Synthetic candidates filtered by automated heuristic checks only. Not human "
        "reviewed. Not ASD-STE100 compliance."
    )
    distributions: dict[str, dict[str, int]] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    """Label, focus, and LLM-review distributions of the selected examples."""


@dataclass(frozen=True, slots=True)
class PreparedSet:
    """Result of :func:`prepare`."""

    train: list[SftRecord]
    validation: list[SftRecord]
    manifest: SftManifest


def _priority(record: dict[str, object], task: TaskType) -> int:
    """Selection priority under ``prefer_focus`` (higher is kept first)."""
    if task is TaskType.STE_IDENTITY:
        target = str(record["target"])
        features = [
            bool(record.get("labels")),
            pv.negation_count(target, "en") > 0,
            target.count("\n") >= 2,
        ]
        return sum(features)
    return 1 if record.get("focus") else 0


def _balance_labels(
    kept: list[tuple[DatasetExample, list[str], int]],
    max_ratio: float,
    seed: int,
    exclusions: Counter[str],
) -> list[tuple[DatasetExample, list[str], int]]:
    """Drop WARNING-only targets until #WARNING <= max_ratio x #CAUTION."""
    labels = [pv.safety_labels(ex.ste_candidate, english_only=True) for ex, _, _ in kept]
    cautions = sum(1 for c in labels if c["CAUTION"])
    warnings = [i for i, c in enumerate(labels) if c["WARNING"]]
    allowed = int(max_ratio * cautions)
    if len(warnings) <= allowed:
        return kept
    warning_only = [i for i in warnings if not labels[i]["CAUTION"]]
    rng = random.Random(_hash(seed, "label-balance"))
    rng.shuffle(warning_only)
    drop = set(warning_only[: len(warnings) - allowed])
    exclusions["label_balance:warning_downsampled"] += len(drop)
    return [k for i, k in enumerate(kept) if i not in drop]


def _hash(*parts: object) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    tokens = normalize_for_dedup(text).split()
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NearDuplicateIndex:
    """Exact fingerprints plus word-trigram Jaccard similarity."""

    def __init__(self, threshold: float) -> None:
        """Create an empty index."""
        self.threshold = threshold
        self._exact: dict[str, str] = {}
        self._shingles: list[tuple[str, set[tuple[str, ...]]]] = []

    def add(self, key: str, text: str) -> None:
        """Index ``text`` under ``key``."""
        self._exact.setdefault(fingerprint(text), key)
        self._shingles.append((key, _shingles(text)))

    def find(self, text: str) -> str | None:
        """Return the key of an exact or near duplicate, or None."""
        exact = self._exact.get(fingerprint(text))
        if exact is not None:
            return exact
        grams = _shingles(text)
        for key, other in self._shingles:
            if _jaccard(grams, other) >= self.threshold:
                return key
        return None


def rule_errors(text: str, category: TaskCategory | None, rules: RuleConfig) -> int:
    """Number of ``error`` violations from the heuristic STE rule check."""
    procedural = category in PROCEDURAL if category else False
    return sum(
        1
        for v in check_text(text, procedural=procedural, config=rules)
        if v.severity is RuleSeverity.ERROR
    )


def _split_groups(
    groups: Sequence[str], fraction: float, seed: int, sizes: Counter[str]
) -> set[str]:
    """Pick validation groups by hash order until they hold ``fraction`` of the examples."""
    order = sorted(groups, key=lambda g: _hash(seed, "split", g))
    total = sum(sizes[g] for g in groups)
    chosen: set[str] = set()
    count = 0
    for group in order:
        if count >= fraction * total:
            break
        chosen.add(group)
        count += sizes[group]
    return chosen


def prepare(
    inputs: Iterable[Path],
    config: SftConfig,
    rules: RuleConfig,
    *,
    golden_texts: dict[str, list[str]],
    label: str = "experimental",
) -> PreparedSet:
    """Build the SFT train and validation sets. See the module docstring."""
    exclusions: Counter[str] = Counter()
    input_hashes: dict[str, str] = {}
    examples: list[DatasetExample] = []
    for path in sorted(inputs):
        try:
            examples += read_jsonl(path, DatasetExample)
        except DatasetFileError as exc:
            raise SftConfigError(str(exc)) from exc
        input_hashes[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()

    golden = NearDuplicateIndex(config.near_duplicate_threshold)
    for suite, texts in golden_texts.items():
        for text in texts:
            golden.add(suite, text)

    sources = NearDuplicateIndex(config.near_duplicate_threshold)
    targets = NearDuplicateIndex(config.near_duplicate_threshold)
    kept: list[tuple[DatasetExample, list[str], int]] = []
    for ex in sorted(examples, key=lambda e: (e.provenance.created_at, e.example_id)):
        if ex.task_type in {TaskType.STE_IDENTITY, TaskType.OUT_OF_DOMAIN}:
            exclusions["unsupported_input_task"] += 1
            continue
        report = validate_multitask(
            ex.task_type,
            ex.source_language,
            ex.task_category,
            ex.source_text,
            ex.ste_candidate,
            ex.technical_entities,
            config.validation,
            required_terms=ex.provenance.scenario.required_terms,
        )
        if not report.passed:
            failed = [
                c.check_id for c in report.checks if not c.passed and c.severity.value == "error"
            ]
            for check_id in failed:
                exclusions[f"failed:{check_id}"] += 1
            exclusions["failed_checks"] += 1
            continue
        errors = rule_errors(ex.ste_candidate, ex.task_category, rules)
        if config.require_rule_pass and errors:
            exclusions["ste_rule_errors"] += 1
            continue
        if config.require_llm_review:
            review = ex.llm_review
            if review is None or review.status == "error":
                exclusions["llm_review_missing"] += 1
                continue
            if review.critical:
                exclusions["llm_critical"] += 1
                for problem in review.critical:
                    exclusions[f"llm_critical:{problem.category}"] += 1
                continue
        if golden.find(ex.source_text) or golden.find(ex.ste_candidate):
            exclusions["golden_overlap"] += 1
            continue
        if sources.find(ex.source_text) or targets.find(ex.ste_candidate):
            exclusions["duplicate"] += 1
            continue
        sources.add(ex.example_id, ex.source_text)
        targets.add(ex.example_id, ex.ste_candidate)
        warnings = [c.check_id for c in report.checks if not c.passed]
        kept.append((ex, warnings, errors))

    if config.max_warning_caution_ratio is not None:
        kept = _balance_labels(kept, config.max_warning_caution_ratio, config.seed, exclusions)

    def group_of(ex: DatasetExample) -> str:
        return "topic-" + _hash("topic", ex.provenance.scenario.topic)[:12]

    def base_record(
        ex: DatasetExample, warnings: list[str], errors: int, task: TaskType, source: str
    ) -> dict[str, object]:
        return {
            "task_type": task,
            "task_category": ex.task_category,
            "source_language": "en" if task is TaskType.STE_IDENTITY else ex.source_language,
            "input": source,
            "target": ex.ste_candidate,
            "group": group_of(ex),
            "parent_example_id": ex.example_id,
            "technical_entities": ex.technical_entities,
            "safety_constraints": ex.safety_constraints,
            "teacher_model": ex.teacher.model,
            "teacher_prompts": [ex.prompt_version.source.ref, ex.prompt_version.rewrite.ref],
            "generation_run_id": ex.provenance.run_id,
            "checker_version": MULTITASK_CHECKER_VERSION,
            "warning_checks": warnings,
            "rule_errors": errors,
            "focus": ex.provenance.scenario.focus,
            "labels": sorted(pv.safety_labels(ex.ste_candidate, english_only=True)) or None,
            "llm_review": ex.llm_review.status if ex.llm_review else None,
        }

    candidates: dict[TaskType, list[dict[str, object]]] = {}
    for ex, warnings, errors in kept:
        candidates.setdefault(ex.task_type, []).append(
            base_record(ex, warnings, errors, ex.task_type, ex.source_text)
        )
        clean = not warnings and errors == 0 and ex.ste_candidate.strip()
        if clean and TaskType.STE_IDENTITY in config.mix:
            candidates.setdefault(TaskType.STE_IDENTITY, []).append(
                base_record(ex, warnings, errors, TaskType.STE_IDENTITY, ex.ste_candidate)
            )

    # Task mix: the largest total that every task can fill at its share.
    shares = {t: w for t, w in config.mix.items() if w > 0}
    total_weight = sum(shares.values())
    shares = {t: w / total_weight for t, w in shares.items()}
    rng = random.Random(config.seed)
    selected: list[dict[str, object]] = []
    if config.enforce_mix and shares:
        limit = min(len(candidates.get(t, [])) / s for t, s in shares.items())
        for task, share in shares.items():
            pool = sorted(candidates.get(task, []), key=lambda r: str(r["parent_example_id"]))
            rng.shuffle(pool)
            if config.prefer_focus:
                pool.sort(key=lambda r: -_priority(r, task))  # stable: shuffle order is kept
            take = int(limit * share)
            exclusions[f"mix_downsampled:{task.value}"] += len(pool) - take
            selected += pool[:take]
    else:
        for task in shares:
            selected += candidates.get(task, [])
    # An identity example and its parent share the group, so no leakage.

    sizes = Counter(str(r["group"]) for r in selected)
    val_groups = _split_groups(sorted(sizes), config.validation_fraction, config.seed, sizes)

    in_domain = len(selected)
    n_ood = round(config.out_of_domain_fraction * in_domain)
    ood_pool = sorted(config.out_of_domain, key=lambda p: _hash(config.seed, "ood", p.text))
    for prompt in ood_pool[:n_ood]:
        group = "ood-" + _hash("ood", prompt.text)[:12]
        selected.append(
            {
                "task_type": TaskType.OUT_OF_DOMAIN,
                "task_category": None,
                "source_language": prompt.language,
                "input": prompt.text,
                "target": pv.UNSUPPORTED_SENTINEL,
                "group": group,
                "checker_version": "handwritten",
            }
        )
        if _hash(config.seed, "ood-split", prompt.text)[0] in "01":
            val_groups.add(group)
    if n_ood > len(ood_pool):
        exclusions["ood_prompts_short"] = n_ood - len(ood_pool)

    train: list[SftRecord] = []
    validation: list[SftRecord] = []
    for data in sorted(selected, key=lambda r: _hash(r["task_type"], r["input"])):
        split: Literal["train", "validation"] = (
            "validation" if data["group"] in val_groups else "train"
        )
        record = SftRecord.model_validate(
            {
                **data,
                "split": split,
                "sft_id": "sft-" + _hash(data["task_type"], data["input"], data["target"])[:16],
            }
        )
        (validation if split == "validation" else train).append(record)

    counts: dict[str, dict[str, int]] = {}
    languages: dict[str, dict[str, int]] = {}
    for name, records in (("train", train), ("validation", validation)):
        counts[name] = dict(sorted(Counter(r.task_type.value for r in records).items()))
        languages[name] = dict(sorted(Counter(r.source_language for r in records).items()))
    everything = train + validation
    in_dom = [r for r in everything if r.task_type is not TaskType.OUT_OF_DOMAIN]
    mix_actual = {
        t.value: round(sum(1 for r in in_dom if r.task_type is t) / len(in_dom), 4)
        if in_dom
        else 0.0
        for t in shares
    }
    content = [r.model_dump(mode="json") for r in everything]
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    manifest = SftManifest(
        dataset_fingerprint=digest,
        config_sha256=config.sha256(),
        config=config.model_dump(mode="json"),
        prompt=config.prompt,
        inputs=input_hashes,
        golden_suites=sorted(golden_texts),
        counts=counts,
        languages=languages,
        exclusions=dict(sorted(exclusions.items())),
        mix_target={t.value: round(s, 4) for t, s in shares.items()},
        mix_actual=mix_actual,
        groups={
            "train": sorted({r.group for r in train}),
            "validation": sorted({r.group for r in validation}),
        },
        label=label,
        distributions={
            "labels": dict(
                sorted(Counter(lbl for r in in_dom for lbl in (r.labels or ["none"])).items())
            ),
            "focus": dict(sorted(Counter(r.focus or "none" for r in in_dom).items())),
            "llm_review": dict(sorted(Counter(r.llm_review or "none" for r in in_dom).items())),
            "task_language": dict(
                sorted(
                    Counter(f"{r.task_type.value}:{r.source_language}" for r in everything).items()
                )
            ),
        },
    )
    return PreparedSet(train=train, validation=validation, manifest=manifest)


def write_prepared(prepared: PreparedSet, out_dir: Path) -> dict[str, Path]:
    """Write ``train.jsonl``, ``validation.jsonl``, and ``manifest.json`` atomically."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "train.jsonl",
        "validation": out_dir / "validation.jsonl",
        "manifest": out_dir / "manifest.json",
    }
    write_jsonl_atomic(paths["train"], prepared.train)
    write_jsonl_atomic(paths["validation"], prepared.validation)
    atomic_write_text(
        paths["manifest"],
        json.dumps(prepared.manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
    )
    return paths


def load_prepared(out_dir: Path) -> PreparedSet:
    """Read a prepared set back and check its fingerprint."""
    manifest = SftManifest.model_validate_json((out_dir / "manifest.json").read_text("utf-8"))
    train = read_jsonl(out_dir / "train.jsonl", SftRecord)
    validation = read_jsonl(out_dir / "validation.jsonl", SftRecord)
    content = [r.model_dump(mode="json") for r in train + validation]
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if digest != manifest.dataset_fingerprint:
        raise SftConfigError(
            f"{out_dir}: content fingerprint {digest[:12]} does not match the manifest "
            f"({manifest.dataset_fingerprint[:12]}). Re-run `steward dataset prepare`."
        )
    leaked = {r.group for r in train} & {r.group for r in validation}
    if leaked:
        raise SftConfigError(f"{out_dir}: groups in both splits: {sorted(leaked)[:5]}")
    return PreparedSet(train=train, validation=validation, manifest=manifest)
