"""Cross-language technical-fidelity checks shared by data validation and the output guard.

* :func:`terminology_findings`: context-dependent term fixtures
  (``configs/ste/terminology.toml``). A *forbidden* rendering in the output is a
  deterministic finding; a missing *expected* rendering is only heuristic.
* :func:`source_leakage`: words left in the source language. Uses ``wordfreq``
  word frequencies: a word that is rare in English but common in the source
  language is reported. Heuristic.
* :func:`negation_shortfall`: fewer negations in the output than in the source.
  Heuristic (word counts, not syntax).
* :func:`contradictions`: the output gives the same instruction positively and
  negatively. Deterministic string comparison after normalization.
* :func:`missing_actions`: English-source command verbs that do not occur in the
  output. Heuristic; only for English input.

None of these checks understands meaning. They catch frequent, known error types.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from steward.ste import preservation as pv

DEFAULT_TERMINOLOGY = Path(__file__).resolve().parents[3] / "configs/ste/terminology.toml"
_WORD_RE = re.compile(r"[^\W\d_][^\W\d_'’-]*", re.UNICODE)  # noqa: RUF001


@dataclass(frozen=True)
class TermFixture:
    """One context-dependent source term."""

    id: str
    language: str
    pattern: re.Pattern[str]
    expected: tuple[str, ...]
    forbidden: tuple[str, ...]
    exclusive: bool = False
    note: str = ""


@dataclass(frozen=True)
class Terminology:
    """The loaded fixture set."""

    terms: tuple[TermFixture, ...]
    sha256: str
    source: str = ""


@dataclass(frozen=True)
class TermFinding:
    """A fixture that matched the source."""

    term_id: str
    source_match: str
    forbidden_found: list[str] = field(default_factory=list)
    expected_found: bool = True


def load_terminology(path: Path | None = None) -> Terminology:
    """Load the terminology fixtures (default: ``configs/ste/terminology.toml``)."""
    return _load(str(path or DEFAULT_TERMINOLOGY))


@lru_cache(maxsize=8)
def _load(path: str) -> Terminology:
    file = Path(path)
    if not file.is_file():
        return Terminology(terms=(), sha256="none", source=path)
    raw = file.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    terms = []
    for entry in data.get("term", []):
        flags = 0 if entry.get("case_sensitive") else re.IGNORECASE
        terms.append(
            TermFixture(
                id=entry["id"],
                language=entry["language"],
                pattern=re.compile(entry["pattern"], flags),
                expected=tuple(e.casefold() for e in entry.get("expected", [])),
                forbidden=tuple(f.casefold() for f in entry.get("forbidden", [])),
                exclusive=bool(entry.get("exclusive", False)),
                note=entry.get("note", ""),
            )
        )
    return Terminology(tuple(terms), hashlib.sha256(raw).hexdigest(), path)


def contains_phrase(text: str, phrase: str) -> bool:
    """``phrase`` occurs in ``text`` as whole words (case-insensitive)."""
    return re.search(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", text, re.IGNORECASE) is not None


def terminology_findings(
    source: str, output: str, language: str, terminology: Terminology | None = None
) -> list[TermFinding]:
    """Fixtures that match ``source`` and what the output does with them."""
    terminology = terminology or load_terminology()
    findings: list[TermFinding] = []
    covered: list[tuple[int, int]] = []
    for term in terminology.terms:
        if term.language != language:
            continue
        matches = [
            m
            for m in term.pattern.finditer(source)
            if not any(a <= m.start() < b for a, b in covered)
        ]
        if not matches:
            continue
        if term.exclusive:
            covered += [(m.start(), m.end()) for m in matches]
        findings.append(
            TermFinding(
                term_id=term.id,
                source_match=matches[0].group(0),
                forbidden_found=[f for f in term.forbidden if contains_phrase(output, f)],
                expected_found=not term.expected
                or any(contains_phrase(output, e) for e in term.expected),
            )
        )
    return findings


def source_leakage(source: str, output: str, language: str) -> list[str]:
    """Output words that look like untranslated source-language words.

    A word counts when it is alphabetic, at least 4 letters, not an identifier,
    rare in English (Zipf < 1.8), and common in the source language (Zipf >= 2.5).
    Also counted: output words that occur in the source and contain letters outside
    plain ASCII (for example ``panoul`` written as ``panoul`` or ``închideți``).
    """
    if language not in {"de", "fr", "ro"} or not output.strip():
        return []
    from wordfreq import zipf_frequency

    ids = {i.casefold() for i in pv.identifiers(output)}
    source_words = {w.casefold() for w in _WORD_RE.findall(source)}
    leaked: dict[str, None] = {}
    for raw in _WORD_RE.findall(output):
        word = raw.casefold().strip("'’-")  # noqa: RUF001
        if len(word) < 4 or word in ids:
            continue
        non_ascii = any(ord(c) > 127 for c in word)
        if non_ascii and word in source_words:
            leaked.setdefault(word, None)
            continue
        if zipf_frequency(word, "en") < 1.8 and zipf_frequency(word, language) >= 2.5:
            leaked.setdefault(word, None)
    return list(leaked)


def negation_shortfall(source: str, output: str, language: str) -> tuple[int, int]:
    """``(source_negations, output_negations)`` when the output has fewer, else ``(0, 0)``."""
    src = pv.negation_count(source, language)
    out = pv.negation_count(output, "en")
    return (src, out) if out < src else (0, 0)


_NEG_PREFIX_RE = re.compile(r"^(?:do\s+not|don't|never|you\s+must\s+not|must\s+not)\s+", re.I)


def _instructions(text: str) -> list[tuple[bool, str]]:
    from steward.ste.sentences import split_sentences

    found = []
    for sentence in split_sentences(text):
        s = re.sub(r"^(?:WARNING|CAUTION|NOTE)\s*:\s*", "", sentence.strip())
        s = pv.normalize_text(s)
        negated = bool(_NEG_PREFIX_RE.match(s))
        core = _NEG_PREFIX_RE.sub("", s).strip(" .")
        if len(core.split()) >= 2:
            found.append((negated, core))
    return found


def contradictions(output: str) -> list[str]:
    """Instructions that occur both as a command and as a prohibition."""
    items = _instructions(output)
    positive = {core for neg, core in items if not neg}
    negative = {core for neg, core in items if neg}
    return sorted(positive & negative)


def missing_actions(source: str, output: str, language: str) -> list[str]:
    """Command verbs of an English source that the output does not contain (heuristic)."""
    if language != "en":
        return []
    from steward.inference.guard import COMMAND_VERBS

    src = {w.casefold() for w in _WORD_RE.findall(source)}
    out = {w.casefold() for w in _WORD_RE.findall(output)}

    def forms(verb: str) -> set[str]:
        stem = verb[:-1] if verb.endswith("e") else verb
        return {verb, verb + "s", stem + "ed", stem + "ing", verb + "d"}

    missing = []
    for verb in sorted(COMMAND_VERBS):
        if len(verb) < 4 or "-" in verb:
            continue
        variants = {verb, *_EQUIVALENT_VERBS.get(verb, ())}
        out_forms = set().union(*(forms(v) for v in variants)) | {
            w for v in variants if " " in v for w in [v.split()[0]]
        }
        if forms(verb) & src and not out_forms & out:
            missing.append(verb)
    return missing


# STE-oriented writers often replace these verbs; any equivalent counts as present.
_EQUIVALENT_VERBS: dict[str, tuple[str, ...]] = {
    "check": ("examine", "make", "verify", "inspect", "measure", "test"),
    "inspect": ("examine", "check"),
    "examine": ("check", "inspect"),
    "disconnect": ("remove", "release", "disengage"),
    "remove": ("disconnect", "release", "take"),
    "release": ("remove", "disconnect", "loosen"),
    "loosen": ("release", "remove"),
    "tighten": ("torque",),
    "torque": ("tighten",),
    "attach": ("install", "connect", "put"),
    "install": ("attach", "put", "connect"),
    "energize": ("apply", "connect"),
    "de-energize": ("remove", "disconnect"),
    "drain": ("remove",),
    "fill": ("add", "put"),
    "wait": ("let",),
}
