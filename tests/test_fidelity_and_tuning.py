"""Milestone 4: fidelity checks, targeted generation, critique, SFT v3, sweeps, figures."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ste_helpers import ScriptedProvider, make_example
from steward.datasets.jsonl import write_jsonl_atomic
from steward.datasets.schema import CritiqueProblem, LlmReview, PromptRef, TaskType
from steward.datasets.sft import OutOfDomainPrompt, SftConfig, prepare
from steward.generation.config import ValidationSettings, load_config
from steward.generation.critique import review_example
from steward.generation.prompts import load_template
from steward.generation.scenarios import focus_text, sample_scenario
from steward.generation.validation import validate_multitask
from steward.inference.guard import GuardStatus, guard_output
from steward.ste import fidelity as fd
from steward.ste import preservation as pv
from steward.ste.rules import load_rule_config
from steward.training.config import TrainingConfigError, TrainSettings, load_training_config
from steward.training.sweep import (
    SweepConfig,
    TrialResult,
    load_sweep,
    rank_key,
    run_sweep,
    select,
    trial_config,
)
from steward.training.trainer import best_eval

REPO = Path(__file__).resolve().parents[1]
RULES = load_rule_config(REPO / "configs/ste/terms.toml")
SETTINGS = ValidationSettings()


# -- terminology, leakage, negation, labels, contradictions ---------------------------


@pytest.mark.parametrize(
    ("source", "output", "lang", "wrong"),
    [
        ("Die Mutter M-4 mit 12 N·m anziehen.", "Tighten the mother M-4 to 12 N·m.", "de", True),
        ("Die Mutter M-4 mit 12 N·m anziehen.", "Tighten nut M-4 to 12 N·m.", "de", False),
        (
            "Den Deckel mit einem 22-mm-Schlüssel lösen.",
            "Loosen the cover with a 22 mm key.",
            "de",
            True,
        ),
        (
            "Den Deckel mit einem 22-mm-Schlüssel lösen.",
            "Loosen the cover with a 22 mm wrench.",
            "de",
            False,
        ),
        ("Den Schlüssel des Schranks zurückgeben.", "Return the key of the cabinet.", "de", False),
        ("Ne pas toucher J7 à mains nues.", "Do not touch J7 while wearing gloves.", "fr", True),
        (
            "Ne pas toucher J7 à mains nues. Porter des gants.",
            "Do not touch J7 with bare hands. Wear gloves.",
            "fr",
            False,
        ),
        ("Scoateți panoul 21A.", "Remove panoul 21A.", "ro", True),
    ],
)
def test_terminology_fixtures(source: str, output: str, lang: str, wrong: bool) -> None:
    findings = fd.terminology_findings(source, output, lang)
    assert any(f.forbidden_found for f in findings) is wrong


def test_capitalized_mutter_only() -> None:
    # "Mutter" as a noun is case-sensitive; other words are not matched.
    assert fd.terminology_findings("Die mutterseelenallein Arbeit.", "mother", "de") == []


def test_source_language_leakage() -> None:
    assert "panoul" in fd.source_leakage("Deschideți panoul 312AR.", "Open panoul 312AR.", "ro")
    assert fd.source_leakage("Den Filter prüfen.", "Examine the filter.", "de") == []
    assert fd.source_leakage("x", "Remove the cover.", "en") == []


def test_two_part_negations_count_once() -> None:
    assert pv.negation_count("Ne jamais débrancher le connecteur.", "fr") == 1
    assert pv.negation_count("Nu deschideți niciodată capacul.", "ro") == 1
    assert pv.negation_count("Ne pas toucher. Il n'y a rien.", "fr") == 2
    assert pv.negation_count("Nettoyer les contacts.", "fr") == 0


def test_note_is_a_label_not_commentary() -> None:
    assert pv.safety_labels("NOTE: The value includes oil.\nHINWEIS: x")["NOTE"] == 2
    assert pv.contract_violations("NOTE: The torque value includes the lubricant.") == []
    assert pv.contract_violations("Note: I rewrote the text.") == ["meta_commentary"]


def test_contradiction() -> None:
    assert fd.contradictions("Open the valve.\nDo not open the valve.") == ["open the valve"]
    assert fd.contradictions("Open the valve.\nDo not open the door.") == []


def test_guard_rejects_known_m3_failures() -> None:
    cases = [
        (
            "Die Mutter M-4 mit 12 N·m anziehen.",
            "The mother M-4 with 12 N·m attached.",
            "de",
            "terminology_forbidden",
        ),
        (
            "Deconectați CB-45 înainte de a deschide panoul 312AR.",
            "Disconnect CB-45 before opening panoul 312AR.",
            "ro",
            "no_source_language_words",
        ),
        (
            "ATTENTION : Ne pas toucher J7 à mains nues.",
            "WARNING: Do not touch J7 while wearing gloves.",
            "fr",
            "safety_labels_preserved",
        ),
        (
            "Do not open the valve. Do not touch the pipe.",
            "Do not open the valve. Touch the pipe.",
            "en",
            "negation_preserved",
        ),
    ]
    for source, output, lang, check in cases:
        report = guard_output(source, output, RULES, source_language=lang)
        assert report.status is GuardStatus.REJECTED, check
        assert check in report.critical_failures, (check, report.critical_failures)
    assert report.terminology_sha256 and report.guard_version == "guard-0.3"


def test_dev_suite_references_pass_the_guard() -> None:
    from steward.evaluation.suite import load_suite

    suite = load_suite(REPO / "benchmarks/golden/ste-fidelity-dev-v0")
    assert len(suite.items) == 30
    for item in suite.items:
        if item.reference:
            report = guard_output(
                item.source_text,
                item.reference,
                RULES,
                source_language=item.source_language,
                category=item.category,
            )
            assert not report.critical_failures, (item.item_id, report.critical_failures)


# -- prelint-3 and targeted generation ----------------------------------------------


def test_prelint3_required_terms_and_exact_labels() -> None:
    ok = validate_multitask(
        TaskType.MULTILINGUAL_TO_STE,
        "de",
        make_example("a", "b").task_category,
        "VORSICHT: Die Mutter M-4 nicht lösen.",
        "CAUTION: Do not loosen nut M-4.",
        ["M-4"],
        SETTINGS,
        required_terms=["nut"],
        focus_source_term="Mutter",
    )
    assert ok.passed
    bad = validate_multitask(
        TaskType.MULTILINGUAL_TO_STE,
        "de",
        make_example("a", "b").task_category,
        "VORSICHT: Die Mutter M-4 nicht lösen.",
        "WARNING: Do not loosen part M-4.",
        ["M-4"],
        SETTINGS,
        required_terms=["nut"],
        focus_source_term="Schraube",
    )
    failed = {c.check_id for c in bad.checks if not c.passed}
    assert {"safety_labels_preserved", "required_terms_present", "focus_term_in_source"} <= failed


def test_focus_sampling_is_deterministic_and_respects_tasks() -> None:
    config = load_config(REPO / "configs/generation/fidelity.toml")
    a = [sample_scenario(config, i) for i in range(300)]
    b = [sample_scenario(config, i) for i in range(300)]
    assert a == b
    assert {s.focus for s in a} == {"terminology", "safety_label", "negation_condition"}
    for s in a:
        if s.focus == "terminology":
            assert s.source_language != "en" and s.required_terms
            assert s.task_type is not TaskType.PLAIN_TO_STE_REWRITE
    assert focus_text("Use '{o0}' as {o2} in {language}.", "Mutter|nut sense|nut", "German") == (
        "Use 'Mutter' as nut in German."
    )


def test_old_generation_config_hash_is_unchanged_by_the_focus_field() -> None:
    config = load_config(REPO / "configs/generation/multilingual.toml")
    assert config.focus is None
    assert "focus" not in config.model_dump(mode="json")


# -- independent critique and revision ----------------------------------------------


def _review_kwargs(critic: ScriptedProvider, reviser: ScriptedProvider | None) -> dict[str, Any]:
    from steward.datasets.schema import SamplingParameters

    return {
        "critic": critic,
        "critic_model": "critic:12b",
        "critique_template": load_template(REPO / "prompts", "fidelity_critique@1"),
        "reviser": reviser,
        "reviser_model": "teacher:8b" if reviser else None,
        "revise_template": load_template(REPO / "prompts", "ste_revise@1") if reviser else None,
        "rules": RULES,
        "settings": SETTINGS,
        "critic_sampling": SamplingParameters(temperature=0.0),
        "revise_sampling": SamplingParameters(temperature=0.2),
        "sleep": lambda _: None,
    }


BAD = json.dumps(
    {
        "problems": [
            {
                "category": "mistranslated_term",
                "severity": "critical",
                "source_span": "Mutter",
                "target_span": "mother",
                "explanation": "Mutter is a nut.",
            }
        ],
        "verdict": "fail",
    }
)
GOOD = '{"problems": [], "verdict": "pass"}'


def test_critique_revision_is_kept_when_it_passes_everything() -> None:
    ex = make_example(
        "Die Mutter M-4 entfernen.",
        "Remove the mother M-4.",
        task=TaskType.MULTILINGUAL_TO_STE,
        language="de",
        entities=["M-4"],
    )
    critic = ScriptedProvider(lambda user: BAD if "mother" in user else GOOD)
    reviser = ScriptedProvider(lambda _: '{"ste_text": "Remove nut M-4."}')
    out = review_example(ex, **_review_kwargs(critic, reviser))
    assert (
        out.ste_candidate == "Remove nut M-4."
        and out.pre_revision_candidate == "Remove the mother M-4."
    )
    assert out.llm_review is not None and out.llm_review.status == "revised"
    assert out.llm_review.method == "llm_judged" and out.llm_review.initial_problems


def test_critique_without_reviser_fails_and_keeps_text() -> None:
    ex = make_example(
        "Die Mutter M-4 entfernen.",
        "Remove the mother M-4.",
        task=TaskType.MULTILINGUAL_TO_STE,
        language="de",
    )
    out = review_example(ex, **_review_kwargs(ScriptedProvider(lambda _: BAD), None))
    assert out.ste_candidate == ex.ste_candidate
    assert (
        out.llm_review is not None and out.llm_review.status == "failed" and out.llm_review.critical
    )


def test_revision_that_breaks_a_check_is_discarded() -> None:
    ex = make_example(
        "Die 4 Muttern M-4 entfernen.",
        "Remove the 4 mothers M-4.",
        task=TaskType.MULTILINGUAL_TO_STE,
        language="de",
        entities=["M-4"],
    )
    critic = ScriptedProvider(lambda user: BAD)
    reviser = ScriptedProvider(lambda _: '{"ste_text": "Remove the nuts."}')  # loses 4 and M-4
    out = review_example(ex, **_review_kwargs(critic, reviser))
    assert out.ste_candidate == ex.ste_candidate and out.llm_review is not None
    assert out.llm_review.status == "failed"


def test_verdict_fail_without_details_is_critical() -> None:
    ex = make_example("Remove the cover.", "Remove the cover.")
    out = review_example(
        ex,
        **_review_kwargs(ScriptedProvider(lambda _: '{"problems": [], "verdict": "fail"}'), None),
    )
    assert out.llm_review is not None and out.llm_review.critical


# -- SFT v3 preparation ---------------------------------------------------------------


def _review(status: str, critical: bool = False) -> LlmReview:
    problems = (
        [CritiqueProblem(category="negation_changed", severity="critical")] if critical else []
    )
    return LlmReview(
        critic_provider="ollama",
        critic_model="critic",
        prompt=PromptRef(template_id="c", version=1, sha256="0" * 64),
        verdict="fail" if critical else "pass",
        problems=problems,
        status=status,  # type: ignore[arg-type]
        reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _sft_config(**kw: Any) -> SftConfig:
    data: dict[str, Any] = {
        "seed": 1,
        "mix": {"plain_to_ste_rewrite": 1.0},
        "validation_fraction": 0.3,
        "out_of_domain_fraction": 0.0,
        "out_of_domain": [OutOfDomainPrompt(language="en", text="Tell me a joke.")],
    }
    return SftConfig.model_validate(data | kw)


def test_prepare_requires_a_passed_llm_review(tmp_path: Path) -> None:
    exs = []
    for i in range(9):
        ex = make_example(
            f"The {i} bolts on panel P{i}A are removed.",
            f"Remove the {i} bolts from panel P{i}A.",
            topic=f"t{i % 3}",
            index=i,
            entities=[f"P{i}A"],
        )
        review = (
            None if i == 0 else _review("failed", critical=True) if i == 1 else _review("passed")
        )
        exs.append(ex.model_copy(update={"llm_review": review}))
    path = tmp_path / "in.jsonl"
    write_jsonl_atomic(path, exs)
    out = prepare([path], _sft_config(require_llm_review=True), RULES, golden_texts={})
    assert out.manifest.exclusions["llm_review_missing"] == 1
    assert out.manifest.exclusions["llm_critical:negation_changed"] == 1
    assert len(out.train) + len(out.validation) == 7
    assert {r.llm_review for r in out.train + out.validation} == {"passed"}


def test_label_balance_downsamples_warning_only_targets(tmp_path: Path) -> None:
    exs = []
    for i in range(10):
        label = "CAUTION" if i < 2 else "WARNING"
        exs.append(
            make_example(
                f"{label}: Do not touch panel P{i}A when it is hot.",
                f"{label}: Do not touch panel P{i}A when it is hot.",
                category=make_example("a", "b").task_category,
                topic=f"t{i}",
                index=i,
                entities=[f"P{i}A"],
            )
        )
    path = tmp_path / "in.jsonl"
    write_jsonl_atomic(path, exs)
    out = prepare(
        [path],
        _sft_config(max_warning_caution_ratio=1.5, enforce_mix=False),
        RULES,
        golden_texts={},
    )
    labels = [r.labels for r in out.train + out.validation]
    assert sum(1 for lbl in labels if lbl and "WARNING" in lbl) <= 3
    assert out.manifest.exclusions["label_balance:warning_downsampled"] == 5
    assert out.manifest.distributions and out.manifest.distributions["labels"]["CAUTION"] == 2


# -- training config, best checkpoint, sweep -----------------------------------------


def test_select_best_needs_aligned_steps_and_patience_needs_select_best() -> None:
    with pytest.raises(ValueError):
        TrainSettings(select_best=True, eval_steps=10, save_steps=15)
    with pytest.raises(ValueError):
        TrainSettings(early_stopping_patience=3)
    TrainSettings(select_best=True, eval_steps=10, save_steps=20, early_stopping_patience=3)


def test_best_eval_uses_validation_loss_only() -> None:
    history = [
        {"step": 10, "loss": 0.1, "eval_loss": 0.4},
        {"step": 20, "eval_loss": 0.3},
        {"step": 30, "loss": 0.01, "eval_loss": 0.35},
    ]
    assert best_eval(history) == (20, 0.3)
    assert best_eval([{"step": 5, "loss": 0.2}]) == (None, None)


def test_committed_sweep_has_the_three_trials() -> None:
    sweep = load_sweep(REPO / "configs/training/sweeps/m4-v0.2.toml")
    base = load_training_config(REPO / sweep.base_config)
    assert base.train.select_best and base.train.early_stopping_patience
    got = {t.name: trial_config(base, sweep, t) for t in sweep.trial}
    assert (got["A"].train.learning_rate, got["A"].lora.r, got["A"].train.epochs) == (2e-4, 16, 2)
    assert (got["B"].train.learning_rate, got["B"].lora.r, got["B"].train.epochs) == (1e-4, 16, 3)
    assert (got["C"].train.learning_rate, got["C"].lora.r, got["C"].train.epochs) == (1e-4, 32, 3)
    # Everything else stays the same across trials.
    for c in got.values():
        assert c.dataset_dir == base.dataset_dir and c.seed == base.seed
        assert c.train.per_device_batch_size == base.train.per_device_batch_size
        assert c.train.max_length == base.train.max_length


def test_invalid_trial_override_is_rejected() -> None:
    sweep = SweepConfig.model_validate(
        {
            "sweep_id": "s",
            "base_config": "x",
            "selection_suite": "d",
            "trial": [{"name": "Z", "train": {"learning_rate": -1}}],
        }
    )
    base = load_training_config(REPO / "configs/training/qwen3-0.6b-lora-v0.2.toml")
    with pytest.raises(TrainingConfigError):
        trial_config(base, sweep, sweep.trial[0])


def test_run_sweep_records_failures_and_continues(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    base = (REPO / "configs/training/qwen3-0.6b-lora-v0.2.toml").read_text(encoding="utf-8")
    (tmp_path / "configs/base.toml").write_text(base, encoding="utf-8")
    sweep = SweepConfig.model_validate(
        {
            "sweep_id": "s1",
            "base_config": "configs/base.toml",
            "selection_suite": "dev",
            "trial": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
        }
    )
    seen: list[str] = []

    def fake_train(config: Any) -> Path:
        seen.append(config.run_name)
        if config.run_name.endswith("-b"):
            raise RuntimeError("CUDA out of memory")
        run = tmp_path / "runs" / config.run_name
        run.mkdir(parents=True)
        summary = {
            "status": "completed",
            "hyperparameters": {"best_val_loss": 0.3, "best_step": 40, "global_steps": 80},
        }
        (run / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return Path(run)

    state = run_sweep(sweep, Path("sweep.toml"), root=tmp_path, train=fake_train)
    assert [t.status for t in state.trials] == ["completed", "failed", "completed"]
    assert "out of memory" in (state.trials[1].error or "")
    assert (tmp_path / "runs/sweeps/s1/sweep.json").is_file()
    # A second call skips completed trials and retries the failed one.
    seen.clear()
    run_sweep(sweep, Path("sweep.toml"), root=tmp_path, train=fake_train)
    assert len(seen) == 1 and seen[0].endswith("-b")


def test_selection_ranks_safety_before_loss() -> None:
    from steward.training.sweep import SweepState

    trials = [
        TrialResult(
            name="lowloss",
            status="completed",
            best_val_loss=0.10,
            dev_metrics={
                "critical_failures": 5,
                "technical_error_rate": 0.3,
                "meaning_preservation": 0.9,
            },
        ),
        TrialResult(
            name="safe",
            status="completed",
            best_val_loss=0.30,
            dev_metrics={
                "critical_failures": 2,
                "technical_error_rate": 0.4,
                "meaning_preservation": 0.8,
            },
        ),
        TrialResult(name="nodev", status="completed", best_val_loss=0.01),
    ]
    state = SweepState(
        sweep_id="s", sweep_file="f", base_config="b", selection_suite="d", trials=trials
    )
    assert select(state) == "safe"
    assert [t.rank for t in state.trials] == [2, 1, None]
    assert rank_key(trials[1]) < rank_key(trials[0])


# -- metrics and figures --------------------------------------------------------------


def test_fidelity_metrics_and_new_figures(tmp_path: Path) -> None:
    from steward.evaluation.plots import FIGURES, render_all
    from steward.evaluation.records import CandidateRole
    from steward.evaluation.runner import BenchmarkContext, run_candidate
    from steward.evaluation.suite import CandidateConfig, load_suite

    suite = load_suite(REPO / "benchmarks/golden/ste-fidelity-dev-v0")
    refs = {i.source_text: i.reference for i in suite.items}

    def wrong(source: str) -> str:
        ref = refs.get(source)
        if ref is None:
            return pv.UNSUPPORTED_SENTINEL
        return ref.replace("nut", "mother").replace("CAUTION", "WARNING")

    ctx = BenchmarkContext(root=REPO, suite=suite, rules=RULES, prompt_dir=REPO / "prompts")
    good_c = CandidateConfig(
        id="oracle",
        role=CandidateRole.BASE,
        provider="s",
        model="m",
        base_model="Qwen/Qwen3-0.6B",
        prompt="ste_only@1",
        suite="ste-fidelity-dev-v0",
    )
    good = run_candidate(
        ScriptedProvider(lambda s: refs.get(s) or pv.UNSUPPORTED_SENTINEL),
        good_c,
        ctx,
        sleep=lambda _: None,
    )
    bad = run_candidate(
        ScriptedProvider(wrong),
        good_c.model_copy(update={"id": "broken", "role": CandidateRole.FINETUNED}),
        ctx,
        sleep=lambda _: None,
    )
    g, b = good.metrics, bad.metrics
    assert g["technical_error_rate"].value == 0.0 and g["safety_label_preservation"].value == 1.0
    assert g["terminology_accuracy"].value == 1.0
    assert (b["technical_error_rate"].value or 0) > 0.2
    assert (b["safety_label_preservation"].value or 1) < 1 and (
        b["terminology_accuracy"].value or 1
    ) < 1
    assert b["critical_technical_errors"].value and b["critical_technical_errors"].value >= 5
    names = tuple(
        n
        for n in FIGURES
        if n
        in {
            "safety_label_preservation",
            "negation_preservation",
            "identifier_preservation",
            "version_comparison",
            "compliance_vs_critical_errors",
            "hyperparameter_trials",
            "error_distribution",
        }
    )
    assert len(names) == 7
    written = render_all(
        [good, bad], [], tmp_path, formats=("svg",), names=names, primary_key=good.comparability_key
    )
    assert len(written) == 7
