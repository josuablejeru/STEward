from pathlib import Path

import pytest

from steward.generation.config import ConfigError, GenerationConfig, load_config
from steward.generation.scenarios import derive_seed, sample_scenario


def test_default_config_is_valid(config: GenerationConfig) -> None:
    assert config.source_template == "source_passage@1"
    assert {c.name.value for c in config.categories} == {
        "procedure",
        "description",
        "safety_notice",
        "troubleshooting",
    }
    assert all("fictional" in topic for topic in config.topics)


def test_config_hash_is_stable_and_sensitive(config: GenerationConfig) -> None:
    assert config.sha256() == config.model_copy().sha256()
    assert config.sha256() != config.model_copy(update={"seed": 1}).sha256()


def test_invalid_config_reports_path(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('seed = "x"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"bad\.toml"):
        load_config(path)
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.toml")


def test_scenarios_are_deterministic(config: GenerationConfig) -> None:
    first = [sample_scenario(config, i) for i in range(20)]
    again = [sample_scenario(config, i) for i in range(20)]
    assert first == again
    assert len({(s.topic, s.category, s.audience) for s in first}) > 5
    other_seed = config.model_copy(update={"seed": config.seed + 1})
    assert [sample_scenario(other_seed, i) for i in range(20)] != first


def test_derive_seed_range() -> None:
    seeds = {derive_seed(1, "source", i) for i in range(100)}
    assert len(seeds) == 100
    assert all(0 <= s < 2**31 for s in seeds)
