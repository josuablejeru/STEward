"""Deterministic scenario sampling.

The scenario for example ``index`` depends only on the config seed and the
index, so a run can be resumed or reproduced exactly.
"""

from __future__ import annotations

import hashlib
import random

from steward.datasets.schema import Scenario, SourceLanguage, TaskType
from steward.generation.config import GenerationConfig


def derive_seed(base_seed: int, *parts: object) -> int:
    """Return a stable 31-bit seed from ``base_seed`` and ``parts``."""
    key = ":".join(str(p) for p in (base_seed, *parts))
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF


def sample_scenario(config: GenerationConfig, index: int) -> Scenario:
    """Return the scenario for example number ``index``."""
    rng = random.Random(derive_seed(config.seed, "scenario", index))
    category = rng.choices(config.categories, weights=[c.weight for c in config.categories])[0]
    topic = rng.choice(config.topics)
    audience = rng.choice(config.audiences)
    sentence_count = rng.choice(config.sentence_counts)
    task = TaskType.PLAIN_TO_STE_REWRITE
    language: SourceLanguage = "en"
    if config.tasks is not None:
        # A separate stream, so adding tasks does not change the other choices.
        task_rng = random.Random(derive_seed(config.seed, "task", index))
        tasks = list(config.tasks.weights)
        task = task_rng.choices(tasks, weights=[config.tasks.weights[t] for t in tasks])[0]
        if task is TaskType.MULTILINGUAL_TO_STE:
            language = task_rng.choice(config.tasks.translation_languages)
        elif task is TaskType.REQUIREMENTS_TO_STE:
            language = task_rng.choice(config.tasks.requirement_languages)
    focus = focus_detail = None
    required: list[str] | None = None
    if config.focus is not None:
        focus, focus_detail, required = _sample_focus(config, index, task, language)
    return Scenario(
        focus=focus,
        focus_detail=focus_detail,
        required_terms=required,
        index=index,
        category=category.name,
        topic=topic,
        audience=audience,
        sentence_count=sentence_count,
        task_type=task,
        source_language=language,
    )


def _sample_focus(
    config: GenerationConfig, index: int, task: TaskType, language: str
) -> tuple[str | None, str | None, list[str] | None]:
    """Pick a focus that applies to ``task`` and ``language`` (own random stream)."""
    assert config.focus is not None
    rng = random.Random(derive_seed(config.seed, "focus", index))
    eligible = {
        name: spec
        for name, spec in config.focus.specs.items()
        if spec.weight > 0
        and (spec.tasks is None or task in spec.tasks)
        and (spec.options.get(language) or spec.options.get("any") or not spec.options)
    }
    names = [*eligible, "none"]
    weights = [eligible[n].weight for n in eligible] + [config.focus.none_weight]
    if not any(weights):
        return None, None, None
    name = rng.choices(names, weights=weights)[0]
    if name == "none":
        return None, None, None
    spec = eligible[name]
    options = spec.options.get(language) or spec.options.get("any") or []
    detail = rng.choice(options) if options else None
    required = None
    if detail is not None and spec.required_part is not None:
        parts = detail.split("|")
        if spec.required_part < len(parts):
            required = [parts[spec.required_part].strip()]
    return name, detail, required


def focus_text(template: str, detail: str | None, language_name: str) -> str:
    """Fill ``{o0}``, ``{o1}``, ... and ``{language}`` in a focus template."""
    parts = [p.strip() for p in (detail or "").split("|")] if detail else []
    values = {f"o{i}": part for i, part in enumerate(parts)}
    return template.strip().format(language=language_name, **values)
