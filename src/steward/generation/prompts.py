"""Versioned prompt templates stored as TOML files.

Layout: ``prompts/<template_id>/v<version>.toml``. Each file has::

    id = "ste_rewrite"
    version = 1
    description = "..."
    variables = ["category", "source_text"]
    system = \"\"\"...\"\"\"
    user = \"\"\"... $source_text ...\"\"\"

Templates use :class:`string.Template` (``$name`` placeholders; ``$$`` for
a literal dollar sign). A published version must never change: create a
new version file instead. The SHA-256 of the file is stored on every record.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path
from string import Template

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from steward.datasets.schema import PromptRef

_REF_RE = re.compile(r"^(?P<id>[a-z][a-z0-9_]*)@(?P<version>[1-9][0-9]*)$")


class PromptTemplateError(Exception):
    """Raised when a template is missing, invalid, or rendered incorrectly."""


class RenderedPrompt(BaseModel):
    """System and user messages after substitution."""

    model_config = ConfigDict(frozen=True)

    system: str
    user: str


class PromptTemplate(BaseModel):
    """One versioned prompt template."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: int = Field(ge=1)
    description: str
    variables: list[str]
    system: str
    user: str
    sha256: str = Field(default="", exclude=True)

    @model_validator(mode="after")
    def _check_placeholders(self) -> PromptTemplate:
        used: set[str] = set()
        for name, text in (("system", self.system), ("user", self.user)):
            template = Template(text)
            if not template.is_valid():
                raise ValueError(f"{name} text has an invalid '$' placeholder")
            used.update(template.get_identifiers())
        declared = set(self.variables)
        if used != declared:
            raise ValueError(
                f"declared variables {sorted(declared)} do not match placeholders {sorted(used)}"
            )
        return self

    @property
    def ref(self) -> PromptRef:
        """Return the reference stored on dataset records."""
        return PromptRef(template_id=self.id, version=self.version, sha256=self.sha256)

    def render(self, **values: str) -> RenderedPrompt:
        """Substitute ``values``. Every declared variable must be given, and no others."""
        missing = set(self.variables) - values.keys()
        extra = values.keys() - set(self.variables)
        if missing or extra:
            raise PromptTemplateError(
                f"{self.id}@{self.version}: missing variables {sorted(missing)}, "
                f"unexpected variables {sorted(extra)}"
            )
        return RenderedPrompt(
            system=Template(self.system).substitute(values),
            user=Template(self.user).substitute(values),
        )


def parse_ref(ref: str) -> tuple[str, int]:
    """Split ``"template_id@version"`` into its parts."""
    match = _REF_RE.match(ref)
    if match is None:
        raise PromptTemplateError(f"Invalid template reference {ref!r}; expected 'name@version'.")
    return match["id"], int(match["version"])


def load_template(prompt_dir: Path, ref: str) -> PromptTemplate:
    """Load and validate the template named by ``ref`` from ``prompt_dir``."""
    template_id, version = parse_ref(ref)
    path = prompt_dir / template_id / f"v{version}.toml"
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise PromptTemplateError(f"Prompt template {ref!r} not found at {path}.") from None
    try:
        data = tomllib.loads(raw.decode("utf-8"))
        template = PromptTemplate.model_validate(
            {**data, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise PromptTemplateError(f"Invalid prompt template {path}: {exc}") from exc
    if (template.id, template.version) != (template_id, version):
        raise PromptTemplateError(
            f"{path} declares {template.id}@{template.version}, expected {ref}."
        )
    return template
