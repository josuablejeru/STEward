from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ste_helpers import make_example
from steward.datasets.jsonl import write_jsonl_atomic
from steward.datasets.schema import TaskType
from steward.datasets.sft import (
    OutOfDomainPrompt,
    SftConfig,
    SftConfigError,
    load_prepared,
    load_sft_config,
    prepare,
    write_prepared,
)
from steward.generation.prompts import load_template
from steward.ste.preservation import UNSUPPORTED_SENTINEL
from steward.ste.rules import load_rule_config
from steward.training.data import check_completion_mask, to_prompt_completion

REPO = Path(__file__).resolve().parents[1]
RULES = load_rule_config(REPO / "configs/ste/terms.toml")


def _config(**kw: Any) -> SftConfig:
    data: dict[str, Any] = {
        "seed": 1,
        "mix": {
            "plain_to_ste_rewrite": 0.5,
            "multilingual_to_ste_translation": 0.5,
            "ste_identity": 0.2,
        },
        "validation_fraction": 0.3,
        "out_of_domain_fraction": 0.2,
        "out_of_domain": [
            OutOfDomainPrompt(language="en", text=f"Tell me a joke number {i}.") for i in range(5)
        ],
    }
    data.update(kw)
    return SftConfig.model_validate(data)


def _examples() -> list[Any]:
    exs = []
    for i in range(12):
        exs.append(
            make_example(
                f"The {i} bolts on panel P{i}A are to be removed when the cover is held.",
                f"Hold the cover.\nRemove the {i} bolts from panel P{i}A.",
                topic=f"topic-{i % 4}",
                index=i,
                entities=[f"P{i}A"],
            )
        )
        exs.append(
            make_example(
                f"Die {i} Schrauben am Deckel C{i}B nicht verlieren.",
                f"Do not lose the {i} screws of cover C{i}B.",
                task=TaskType.MULTILINGUAL_TO_STE,
                language="de",
                topic=f"topic-{i % 4}",
                index=100 + i,
                entities=[f"C{i}B"],
            )
        )
    # A bad one: the number is lost, so it must be excluded.
    exs.append(make_example("Remove the 7 screws.", "Remove the screws.", index=999))
    return exs


def _write(tmp_path: Path, examples: list[Any]) -> Path:
    path = tmp_path / "in.jsonl"
    write_jsonl_atomic(path, examples)
    return path


def test_prepare_split_mix_and_exclusions(tmp_path: Path) -> None:
    path = _write(tmp_path, _examples())
    prepared = prepare([path], _config(), RULES, golden_texts={})
    m = prepared.manifest
    assert m.exclusions["failed:numbers_preserved"] == 1
    # No group crosses the split.
    assert not {r.group for r in prepared.train} & {r.group for r in prepared.validation}
    everything = prepared.train + prepared.validation
    tasks = {r.task_type for r in everything}
    assert TaskType.STE_IDENTITY in tasks and TaskType.OUT_OF_DOMAIN in tasks
    for r in everything:
        if r.task_type is TaskType.STE_IDENTITY:
            assert r.input == r.target
        if r.task_type is TaskType.OUT_OF_DOMAIN:
            assert r.target == UNSUPPORTED_SENTINEL
        assert r.review_state.value == "candidate"
    assert abs(m.mix_actual["plain_to_ste_rewrite"] - m.mix_target["plain_to_ste_rewrite"]) < 0.1


def test_prepare_is_deterministic_and_roundtrips(tmp_path: Path) -> None:
    path = _write(tmp_path, _examples())
    a = prepare([path], _config(), RULES, golden_texts={})
    b = prepare([path], _config(), RULES, golden_texts={})
    assert a.manifest.dataset_fingerprint == b.manifest.dataset_fingerprint
    write_prepared(a, tmp_path / "out")
    loaded = load_prepared(tmp_path / "out")
    assert loaded.manifest.dataset_fingerprint == a.manifest.dataset_fingerprint


def test_tampered_prepared_set_is_detected(tmp_path: Path) -> None:
    path = _write(tmp_path, _examples())
    write_prepared(prepare([path], _config(), RULES, golden_texts={}), tmp_path / "out")
    train = tmp_path / "out/train.jsonl"
    train.write_text(train.read_text(encoding="utf-8").replace("bolts", "nuts"), encoding="utf-8")
    with pytest.raises(SftConfigError, match="fingerprint"):
        load_prepared(tmp_path / "out")


def test_golden_overlap_and_duplicates_are_removed(tmp_path: Path) -> None:
    exs = _examples()
    exs.append(make_example(exs[0].source_text + " ", exs[0].ste_candidate, index=500))
    path = _write(tmp_path, exs)
    golden = {"suite": [exs[2].source_text]}
    m = prepare([path], _config(), RULES, golden_texts=golden).manifest
    assert m.exclusions["golden_overlap"] == 1
    assert m.exclusions["duplicate"] == 1


def test_rule_errors_are_excluded_by_default(tmp_path: Path) -> None:
    bad = make_example(
        "The cover is to be removed.", "The cover is removed by the technician.", index=1
    )
    path = _write(tmp_path, [bad])
    m = prepare([path], _config(), RULES, golden_texts={}).manifest
    assert m.exclusions["ste_rule_errors"] == 1


def test_committed_sft_config_loads() -> None:
    config = load_sft_config(REPO / "configs/sft/default.toml")
    assert config.prompt == "ste_only@1"
    assert sum(config.mix.values()) == pytest.approx(1.0)
    assert len(config.out_of_domain) >= 20
    assert {p.language for p in config.out_of_domain} == {"en", "de", "fr", "ro"}


class FakeTokenizer:
    eos_token = "<|im_end|>"

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str:
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        body = "".join(f"<{m['role']}>{m['content']}" for m in conversation)
        return body + "<assistant><think>\n\n</think>\n\n"


def test_prompt_completion_rows_put_only_the_target_in_the_completion(tmp_path: Path) -> None:
    path = _write(tmp_path, _examples())
    prepared = prepare([path], _config(), RULES, golden_texts={})
    template = load_template(REPO / "prompts", "ste_only@1")
    rows = to_prompt_completion(prepared.train[:3], template, FakeTokenizer())
    for row, record in zip(rows, prepared.train[:3], strict=True):
        assert row["completion"] == record.target + "<|im_end|>"
        assert record.input in row["prompt"] and record.target not in row["prompt"]
        assert row["prompt"].endswith("<think>\n\n</think>\n\n")


def test_check_completion_mask() -> None:
    vocab = {1: "SYS ", 2: "USER ", 3: "Remove", 4: " it.", 5: "<|im_end|>"}

    def decode(ids: list[int]) -> str:
        return "".join(vocab[i] for i in ids)

    ok, text = check_completion_mask(
        [1, 2, 3, 4, 5], [0, 0, 1, 1, 1], "Remove it.<|im_end|>", decode
    )
    assert ok and text == "Remove it.<|im_end|>"
    bad, _ = check_completion_mask([1, 2, 3, 4, 5], [0, 1, 1, 1, 1], "Remove it.<|im_end|>", decode)
    assert not bad
