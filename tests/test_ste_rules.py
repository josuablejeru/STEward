from pathlib import Path

from steward.ste.rules import RuleConfig, RuleSeverity, check_text, load_rule_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = load_rule_config(REPO / "configs/ste/terms.toml")


def ids(text: str, procedural: bool = True, config: RuleConfig = CONFIG) -> list[str]:
    return [v.rule_id for v in check_text(text, procedural=procedural, config=config)]


def test_clean_procedure_has_no_violations() -> None:
    assert ids("Remove the 4 bolts. Install the new O-ring. Do not drop the washers.") == []


def test_sentence_length_limits() -> None:
    long21 = " ".join(["word"] * 21) + "."
    assert ids(long21) == ["STE-SENT-LEN-PROC"]
    assert ids(long21, procedural=False) == []
    assert ids(" ".join(["word"] * 26) + ".", procedural=False) == ["STE-SENT-LEN-DESC"]


def test_ing_words_respect_allowlist() -> None:
    assert ids("Check the bearing and the housing.") == []
    assert ids("Continue removing the cover.") == ["STE-GRAM-ING"]


def test_contractions_are_errors() -> None:
    violations = check_text("Don't open it.", procedural=True, config=CONFIG)
    assert [(v.rule_id, v.severity) for v in violations] == [
        ("STE-GRAM-CONTRACTION", RuleSeverity.ERROR)
    ]


def test_passive_severity_depends_on_text_type() -> None:
    proc = check_text("The panel is removed by the mechanic.", procedural=True, config=CONFIG)
    desc = check_text("The valve is operated by a motor.", procedural=False, config=CONFIG)
    assert proc[0].severity is RuleSeverity.ERROR
    assert desc[0].severity is RuleSeverity.WARNING


def test_spans_point_at_text() -> None:
    text = "Open the valve. Continue removing the cover."
    (v,) = check_text(text, procedural=True, config=CONFIG)
    assert text[v.start : v.end] == "removing"


def test_config_hash_is_recorded() -> None:
    assert len(CONFIG.source_sha256) == 64
    assert "bearing" in CONFIG.ing_words_allowed
