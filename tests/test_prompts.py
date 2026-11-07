from pathlib import Path

import pytest

from steward.generation.prompts import PromptTemplateError, load_template, parse_ref


@pytest.mark.parametrize("ref", ["source_passage@1", "ste_rewrite@1"])
def test_shipped_templates_load(prompt_dir: Path, ref: str) -> None:
    template = load_template(prompt_dir, ref)
    assert f"{template.id}@{template.version}" == ref
    assert len(template.sha256) == 64
    assert template.ref.ref == ref


def test_rewrite_template_renders(prompt_dir: Path) -> None:
    template = load_template(prompt_dir, "ste_rewrite@1")
    rendered = template.render(category="procedure", source_text="Open the valve.")
    assert "<<<\nOpen the valve.\n>>>" in rendered.user
    assert "Simplified Technical English" in rendered.system


def test_render_requires_exact_variables(prompt_dir: Path) -> None:
    template = load_template(prompt_dir, "ste_rewrite@1")
    with pytest.raises(PromptTemplateError, match="missing"):
        template.render(category="procedure")
    with pytest.raises(PromptTemplateError, match="unexpected"):
        template.render(category="x", source_text="y", extra="z")


def test_dollar_signs_in_values_are_not_expanded(prompt_dir: Path) -> None:
    template = load_template(prompt_dir, "ste_rewrite@1")
    rendered = template.render(category="procedure", source_text="Cost is $source_text.")
    assert "Cost is $source_text." in rendered.user


def _write(tmp_path: Path, body: str, name: str = "demo", version: int = 1) -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"v{version}.toml").write_text(body, encoding="utf-8")
    return tmp_path


def test_undeclared_placeholder_is_an_error(tmp_path: Path) -> None:
    body = 'id="demo"\nversion=1\ndescription="d"\nvariables=["a"]\nsystem="s"\nuser="$a $b"\n'
    with pytest.raises(PromptTemplateError, match="do not match"):
        load_template(_write(tmp_path, body), "demo@1")


def test_version_in_file_must_match_path(tmp_path: Path) -> None:
    body = 'id="demo"\nversion=2\ndescription="d"\nvariables=[]\nsystem="s"\nuser="u"\n'
    with pytest.raises(PromptTemplateError, match="expected demo@1"):
        load_template(_write(tmp_path, body), "demo@1")


def test_missing_template(tmp_path: Path) -> None:
    with pytest.raises(PromptTemplateError, match="not found"):
        load_template(tmp_path, "nothing@1")


@pytest.mark.parametrize("bad", ["demo", "demo@0", "Demo@1", "demo@v1"])
def test_invalid_refs(bad: str) -> None:
    with pytest.raises(PromptTemplateError):
        parse_ref(bad)
