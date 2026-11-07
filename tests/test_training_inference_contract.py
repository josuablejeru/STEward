"""Order checks, the generate() API, model resolution, training summaries, and the CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ste_helpers import ScriptedProvider
from steward import cli
from steward.evaluation.records import RecordStatus, ServingSpec, TrainingSummary
from steward.evaluation.service import check_training_artifacts
from steward.inference.api import SteOnlyService
from steward.inference.guard import GUARD_VERSION, GuardStatus, guard_output
from steward.inference.models import ModelRefError, latest_run, resolve_model
from steward.ste import preservation as pv
from steward.ste.rules import RULESET_VERSION, load_rule_config
from steward.training.config import load_training_config
from steward.training.trainer import curve_from_history, final_losses

REPO = Path(__file__).resolve().parents[1]
RULES = load_rule_config(REPO / "configs/ste/terms.toml")
STEPS = "1. Set valve V-12 to 3 bar.\n2. Wait 10 min.\n3. Close valve V-14."


# -- procedural sequence ----------------------------------------------------------


def test_order_inversion_across_steps_is_found() -> None:
    swapped = "Wait 10 min.\nSet valve V-12 to 3 bar.\nClose valve V-14."
    assert ("V-12", "10") in pv.order_inversions(STEPS, swapped)
    assert pv.order_inversions(STEPS, STEPS) == []


def test_reordering_inside_one_sentence_is_allowed() -> None:
    source = "Die Schraube B-4 mit 25 Nm anziehen."
    assert pv.order_inversions(source, "Torque to 25 Nm the bolt B-4.", "de") == []


def test_structured_requirements_are_not_a_sequence() -> None:
    source = "Component: pump P-2\nPressure: 0 psi\nTool: kit CK-2"
    assert pv.is_structured(source)
    assert pv.order_inversions(source, "Connect kit CK-2. Make sure that P-2 is at 0 psi.") == []


def test_guard_rejects_changed_step_order() -> None:
    report = guard_output(STEPS, "Wait 10 min.\nSet valve V-12 to 3 bar.\nClose valve V-14.", RULES)
    assert report.status is GuardStatus.REJECTED
    assert "sequence_preserved" in report.critical_failures


# -- generate(text, source_language=None) -> result -----------------------------------


def test_generate_returns_text_and_separate_validation_metadata() -> None:
    good = "Set valve V-12 to 3 bar.\nWait 10 min.\nClose valve V-14."
    service = SteOnlyService(
        ScriptedProvider(lambda _: good), "base", RULES, prompt_dir=REPO / "prompts"
    )
    result = service.generate(STEPS, source_language="en")
    assert result.text == good
    assert result.status in {GuardStatus.ACCEPTED, GuardStatus.FLAGGED}
    assert result.failure_reason is None
    assert result.model_id is not None and result.model_id.startswith("base")
    assert result.validation_version == f"{GUARD_VERSION}+{RULESET_VERSION}"
    dumped = result.model_dump(mode="json")
    assert {"text", "status", "violations", "validation_version", "model_id"} <= set(dumped)
    assert "failure_reason" in dumped and "validation" in dumped


def test_generate_gives_a_failure_reason_and_no_text_when_rejected() -> None:
    service = SteOnlyService(
        ScriptedProvider(lambda _: "Wait 10 min."), "m", RULES, prompt_dir=REPO / "prompts"
    )
    result = service.generate(STEPS)
    assert result.status is GuardStatus.REJECTED and result.text is None
    assert result.failure_reason and "Numbers missing" in result.failure_reason


def test_generate_controlled_failure_reason() -> None:
    service = SteOnlyService(
        ScriptedProvider(lambda _: pv.UNSUPPORTED_SENTINEL), "m", RULES, prompt_dir=REPO / "prompts"
    )
    result = service.generate("Write a poem about the sea.")
    assert result.status is GuardStatus.UNSUPPORTED_INPUT and result.text is None
    assert result.failure_reason and "outside the supported" in result.failure_reason


# -- model references ---------------------------------------------------------------


def _summary(adapter: str | None = "runs/run1/adapter", **kw: object) -> TrainingSummary:
    data: dict[str, object] = {
        "run_id": "run1",
        "status": RecordStatus.COMPLETED,
        "base_model": "Qwen/Qwen3-0.6B",
        "base_revision": "rev1",
        "checkpoint_id": "run1:final",
        "checkpoint_path": adapter or "",
        "dataset_version": "d",
        "dataset_fingerprint": "d" * 64,
        "config_sha256": "c" * 64,
        "serving": ServingSpec(
            provider="transformers", model="Qwen/Qwen3-0.6B", revision="rev1", adapter_path=adapter
        ),
    }
    return TrainingSummary.model_validate(data | kw)


def _write_run(root: Path, summary: TrainingSummary, *, weights: bytes | None = b"w") -> Path:
    run = root / "runs/run1"
    (run / "adapter").mkdir(parents=True)
    (root / "pixi.toml").write_text("", encoding="utf-8")
    (run / "training_summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    (run / "adapter/adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-0.6B"}), encoding="utf-8"
    )
    if weights is not None:
        (run / "adapter/adapter_model.safetensors").write_bytes(weights)
    return run


def test_resolve_run_dir_adapter_dir_and_hub_id(tmp_path: Path) -> None:
    run = _write_run(tmp_path, _summary())
    by_run = resolve_model(str(run))
    assert by_run.model == "Qwen/Qwen3-0.6B" and by_run.revision == "rev1"
    assert by_run.adapter_path == tmp_path.resolve() / "runs/run1/adapter"
    by_adapter = resolve_model(str(run / "adapter"))
    assert by_adapter.adapter_path == run.resolve() / "adapter" and by_adapter.revision == "rev1"
    hub = resolve_model("Qwen/Qwen3-0.6B")
    assert hub.adapter_path is None and hub.model == "Qwen/Qwen3-0.6B"
    with pytest.raises(ModelRefError):
        resolve_model(str(tmp_path / "runs/missing"))


def _write_named_run(runs: Path, run_id: str, status: RecordStatus) -> Path:
    run = runs / run_id
    run.mkdir(parents=True)
    summary = _summary(adapter=f"runs/{run_id}/adapter", run_id=run_id, status=status)
    (run / "training_summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    return run


def test_latest_picks_newest_completed_run_by_timestamp(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    (tmp_path / "pixi.toml").write_text("", encoding="utf-8")
    _write_named_run(runs, "b-trial-20261107T010000Z", RecordStatus.COMPLETED)
    newest = _write_named_run(runs, "a-trial-20261107T020000Z", RecordStatus.COMPLETED)
    _write_named_run(runs, "c-trial-20261107T030000Z", RecordStatus.FAILED)
    (runs / "sweeps").mkdir()
    assert latest_run(runs) == newest
    ref = resolve_model("latest", runs_dir=runs)
    assert ref.run_dir == newest.resolve()
    assert ref.adapter_path == tmp_path.resolve() / "runs/a-trial-20261107T020000Z/adapter"


def test_latest_without_runs_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ModelRefError, match="No completed training run"):
        latest_run(tmp_path / "runs")


def test_checkpoint_validation(tmp_path: Path) -> None:
    from steward.evaluation.fingerprints import file_sha256

    run = _write_run(tmp_path, _summary())
    sha = file_sha256(run / "adapter/adapter_model.safetensors")
    assert check_training_artifacts(_summary(adapter_id=f"lora-{sha[:16]}"), tmp_path) == []
    assert "does not match" in check_training_artifacts(_summary(adapter_id="lora-0"), tmp_path)[0]
    failed = _summary(status=RecordStatus.FAILED, error="OOM")
    assert "failed" in check_training_artifacts(failed, tmp_path)[0]
    (run / "adapter/adapter_model.safetensors").unlink()
    assert any("missing" in p for p in check_training_artifacts(_summary(), tmp_path))


# -- training config and summary helpers ---------------------------------------------


def test_committed_training_config_matches_the_milestone_brief() -> None:
    config = load_training_config(REPO / "configs/training/qwen3-0.6b-lora.toml")
    assert config.base_model == "Qwen/Qwen3-0.6B" and config.base_revision
    assert (config.lora.r, config.lora.alpha, config.lora.dropout) == (16, 32, 0.05)
    t = config.train
    assert (t.learning_rate, t.epochs, t.per_device_batch_size) == (2e-4, 3, 4)
    assert (t.gradient_accumulation_steps, t.max_length, config.seed) == (4, 1024, 42)
    assert config.precision == "bf16" and config.eval_suite == "ste-multilingual-v1"
    assert set(config.lora.target_modules) >= {"q_proj", "v_proj", "down_proj"}


def test_curve_and_final_losses_from_trainer_history() -> None:
    history = [
        {"step": 5, "loss": 0.5, "learning_rate": 1e-4, "grad_norm": 0.9, "gpu_peak_mib": 900},
        {"step": 10, "eval_loss": 0.4, "epoch": 0.5},
        {"step": 10, "loss": 0.3, "gpu_allocated_mib": 800},
    ]
    curve = curve_from_history(history)
    assert [p.step for p in curve] == [5, 10]
    assert curve[0].gpu_peak_mib == 900 and curve[0].grad_norm == 0.9
    assert curve[1].val_loss == 0.4 and curve[1].train_loss == 0.3
    assert final_losses(history) == (0.3, 0.4)
    assert final_losses([]) == (None, None)


# -- CLI ---------------------------------------------------------------------------


def test_cli_has_evaluate_and_generate_input() -> None:
    parser = cli.build_parser()
    ev = parser.parse_args(["evaluate", "--model", "runs/x", "--baseline"])
    assert ev.model == "runs/x" and ev.baseline
    gen = parser.parse_args(["generate", "--model", "runs/x", "--input", "Remove the panel."])
    assert gen.input == "Remove the panel."


def test_generate_with_input_runs_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_infer(*args: object) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(cli, "_infer", fake_infer)
    code = cli.main(["generate", "--model", "runs/x", "--input", "Hallo", "--language", "de"])
    assert code == 0 and calls == [("runs/x", "Hallo", "de", None, False, False)]


# -- real chat template (CUDA environment; needs the cached Qwen3 tokenizer) -------------


def _qwen_tokenizer() -> Any:
    transformers = pytest.importorskip("transformers")
    config = load_training_config(REPO / "configs/training/qwen3-0.6b-lora.toml")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            config.base_model, revision=config.base_revision, local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen3 tokenizer is not in the local Hugging Face cache.")


def test_real_chat_template_masks_everything_but_the_answer() -> None:
    """With the real Qwen3 template, prompt tokens are an exact prefix of prompt+answer.

    That is what makes completion-only loss correct: masking the prompt-length prefix
    leaves exactly the STE answer and the end-of-turn token. TRL's own masking is
    checked on real data before every training run (``mask_check.json``).
    """
    from steward.datasets.sft import SftRecord
    from steward.generation.prompts import load_template
    from steward.training.data import check_completion_mask, to_prompt_completion

    tok = _qwen_tokenizer()
    template = load_template(REPO / "prompts", "ste_only@1")
    records = [
        SftRecord.model_construct(
            sft_id="t1", input="Den Deckel entfernen.", target="Remove the cover."
        ),
        SftRecord.model_construct(
            sft_id="t2",
            input="Componentă: pompa P-2\nPresiune: 0 psi",
            target="WARNING: Make sure that the pressure in pump P-2 is 0 psi.",
        ),
    ]
    for row in to_prompt_completion(records, template, tok):
        assert row["prompt"].rstrip().endswith("<think>\n\n</think>")  # thinking disabled
        assert "<think>" not in row["completion"] and row["completion"].endswith("<|im_end|>")
        prompt_ids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
        full_ids = tok(row["prompt"] + row["completion"], add_special_tokens=False)["input_ids"]
        assert full_ids[: len(prompt_ids)] == prompt_ids
        mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
        ok, text = check_completion_mask(full_ids, mask, row["completion"], tok.decode)
        assert ok, text
        assert "system" not in text and "user" not in text


@pytest.mark.gpu
def test_trained_adapter_reloads_and_generates() -> None:
    """Opt-in (pixi run -e cuda test-gpu): newest runs/<id>/adapter loads and writes text."""
    runs = sorted((REPO / "runs").glob("*/adapter/adapter_model.safetensors"))
    if not runs:
        pytest.skip("No trained adapter under runs/.")
    from steward.inference.transformers_provider import TransformersProvider

    ref = resolve_model(str(runs[-1].parent.parent))
    provider = TransformersProvider(revision=ref.revision, adapter_path=ref.adapter_path)
    service = SteOnlyService(provider, ref.model, RULES, prompt_dir=REPO / "prompts")
    try:
        result = service.generate("Remove the four screws and then take off the cover.")
    finally:
        provider.close()
    assert result.generation is not None and result.generation.raw_output.strip()
    assert result.model_id and "+" in result.model_id  # base revision + adapter hash
