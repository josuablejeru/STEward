"""Cross-language meaning-preservation and output-contract checks.

These helpers compare a source text (English, German, French, or Romanian)
with an English STE-oriented output. All checks are approximate:

* Language identification uses stopword and diacritic counts. Short texts
  and texts that are mostly part numbers can be misclassified.
* Negation detection uses word lists. It cannot tell which statement a
  negation belongs to, and it misses negation expressed by other words
  (for example "prevent" or "avoid").
* Measurement and identifier checks compare literal tokens. A value that
  is rewritten in another form (for example "two" instead of "2", or
  "0.5" instead of ".5") counts as missing.

They are used for data filtering, the inference output guard, and the
benchmark. They never edit text.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Literal

SourceLanguage = Literal["en", "de", "fr", "ro"]
LANGUAGES: tuple[SourceLanguage, ...] = ("en", "de", "fr", "ro")
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "ro": "Romanian",
}

UNSUPPORTED_SENTINEL = "STEWARD_UNSUPPORTED_INPUT"
"""The exact text the STE-only model returns for out-of-domain input."""

# -- language identification ----------------------------------------------------

_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        [
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "in",
            "on",
            "at",
            "for",
            "with",
            "is",
            "are",
            "be",
            "do",
            "not",
            "this",
            "that",
            "it",
            "from",
            "by",
            "as",
            "if",
            "then",
            "when",
            "before",
            "after",
            "you",
            "your",
            "must",
            "make",
            "sure",
            "until",
            "all",
            "each",
        ]
    ),
    "de": frozenset(
        [
            "der",
            "die",
            "das",
            "den",
            "dem",
            "des",
            "und",
            "oder",
            "ist",
            "sind",
            "nicht",
            "ein",
            "eine",
            "einen",
            "einem",
            "mit",
            "von",
            "zu",
            "im",
            "auf",
            "für",
            "bei",
            "wenn",
            "dann",
            "vor",
            "nach",
            "sie",
            "ihr",
            "muss",
            "müssen",
            "werden",
            "wird",
            "auch",
            "alle",
            "kein",
            "keine",
            "bis",
            "aus",
            "dass",
            "sich",
        ]
    ),
    "fr": frozenset(
        [
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "et",
            "ou",
            "est",
            "sont",
            "ne",
            "pas",
            "de",
            "du",
            "au",
            "aux",
            "avec",
            "pour",
            "dans",
            "sur",
            "par",
            "si",
            "avant",
            "après",
            "vous",
            "votre",
            "doit",
            "doivent",
            "être",
            "que",
            "qui",
            "il",
            "elle",
            "ce",
            "cette",
            "tous",
            "toutes",
            "jusqu",
        ]
    ),
    "ro": frozenset(
        [
            "și",
            "sau",
            "este",
            "sunt",
            "nu",
            "un",
            "o",
            "de",
            "la",
            "cu",
            "pentru",
            "în",
            "pe",
            "din",
            "dacă",
            "înainte",
            "după",
            "că",
            "care",
            "trebuie",
            "toate",
            "toți",
            "acest",
            "această",
            "se",
            "să",
            "ale",
            "al",
            "ai",
            "până",
            "fi",
        ]
    ),
}
_DIACRITICS: dict[str, frozenset[str]] = {
    "de": frozenset("äöüß"),
    "fr": frozenset("éèêëàâçùûôîœ"),
    "ro": frozenset("ăâîșşțţ"),
}
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)  # noqa: RUF001


@dataclass(frozen=True, slots=True)
class LanguageGuess:
    """Best-guess language of a text with per-language stopword shares."""

    language: str
    confidence: float
    scores: dict[str, float]


def detect_language(text: str) -> LanguageGuess:
    """Guess whether ``text`` is English, German, French, or Romanian.

    Returns ``"unknown"`` when the text has too few words to decide.
    """
    tokens = [t.casefold() for t in _TOKEN_RE.findall(text)]
    if len(tokens) < 3:
        return LanguageGuess("unknown", 0.0, {})
    counts = Counter(tokens)
    total = len(tokens)
    scores: dict[str, float] = {}
    lowered = text.casefold()
    for lang, stops in _STOPWORDS.items():
        share = sum(n for t, n in counts.items() if t in stops) / total
        marks = _DIACRITICS.get(lang, frozenset())
        diacritic_share = sum(lowered.count(ch) for ch in marks) / max(total, 1)
        scores[lang] = round(share + 0.5 * diacritic_share, 4)
    best = max(scores, key=lambda k: scores[k])
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - ordered[1]
    if ordered[0] < 0.05:
        return LanguageGuess("unknown", 0.0, scores)
    return LanguageGuess(best, round(min(1.0, margin / max(ordered[0], 1e-9)), 3), scores)


_FOREIGN_CHARS = frozenset("äöüßăâîșşțţéèêëàçùûôœ")


def foreign_characters(text: str) -> int:
    """Count letters typical of German, French, or Romanian (rare in English text)."""
    return sum(1 for ch in text.casefold() if ch in _FOREIGN_CHARS)


def is_english(text: str) -> bool:
    """True when ``text`` is classified as English (or too short to classify)."""
    guess = detect_language(text)
    return guess.language in {"en", "unknown"} and foreign_characters(text) < 2


# -- safety labels --------------------------------------------------------------

SAFETY_LABELS: dict[str, dict[str, tuple[str, ...]]] = {
    "WARNING": {
        "en": ("WARNING",),
        "de": ("WARNUNG",),
        "fr": ("AVERTISSEMENT",),
        "ro": ("AVERTISMENT", "AVERTIZARE"),
    },
    "CAUTION": {
        "en": ("CAUTION",),
        "de": ("VORSICHT", "ACHTUNG"),
        "fr": ("ATTENTION", "MISE EN GARDE"),
        "ro": ("ATENȚIE", "ATENTIE", "ATENŢIE"),
    },
    "NOTE": {
        "en": ("NOTE",),
        "de": ("HINWEIS",),
        "fr": ("REMARQUE", "NOTE"),
        "ro": ("NOTĂ", "NOTA"),
    },
}


def _label_re(words: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(w) for w in words)
    return re.compile(rf"(?<![\w])(?:{alternatives})(?![\w])")


_LABEL_RES = {
    kind: _label_re(tuple(w for ws in by_lang.values() for w in ws))
    for kind, by_lang in SAFETY_LABELS.items()
}
_ENGLISH_LABEL_RES = {kind: _label_re(by_lang["en"]) for kind, by_lang in SAFETY_LABELS.items()}


def safety_labels(text: str, *, english_only: bool = False) -> Counter[str]:
    """Count WARNING, CAUTION, and NOTE labels (upper case) in any supported language."""
    patterns = _ENGLISH_LABEL_RES if english_only else _LABEL_RES
    return Counter({kind: n for kind, rx in patterns.items() if (n := len(rx.findall(text)))})


# -- negation -------------------------------------------------------------------

_NEGATION_WORDS: dict[str, str] = {
    "en": r"not|no|never|none|nothing|nobody|without|cannot",
    "de": r"nicht|kein|keine|keinen|keinem|keiner|keines|nie|niemals|nichts|ohne",
    "fr": r"ne|jamais|aucun|aucune|sans|rien|non",
    "ro": r"nu|niciodată|niciodata|fără|fara|niciun|nicio|nimic|nici",
}
_NEGATION_RES = {
    lang: re.compile(
        rf"(?<![\w])(?:{words})(?![\w])" + (r"|(?<![\w])n['’]" if lang == "fr" else ""),  # noqa: RUF001
        re.I,
    )
    for lang, words in _NEGATION_WORDS.items()
}
_EN_CONTRACTION_RE = re.compile(r"\w+n't\b", re.I)


# French "ne ... jamais/aucun/rien" and Romanian "nu ... niciodată/niciun/nimic" are one
# negation; the second word is not counted when it follows the particle in the same clause.
_NEG_PARTICLES = {
    "fr": re.compile(r"(?<![\w])(?:ne(?![\w])|n['’])", re.I),  # noqa: RUF001
    "ro": re.compile(r"(?<![\w])nu(?![\w])", re.I),
}
_NEG_COMPANIONS = {
    "fr": re.compile(r"(?<![\w])(?:jamais|aucun|aucune|rien)(?![\w])", re.I),
    "ro": re.compile(r"(?<![\w])(?:niciodată|niciodata|niciun|nicio|nimic|nici)(?![\w])", re.I),
}
_CLAUSE_END = re.compile(r"[.;:!?,\n]")


def negation_count(text: str, language: str = "en") -> int:
    """Count negations for ``language`` (falls back to all languages).

    French and Romanian two-part negations count once.
    """
    if language == "en":
        return len(_NEGATION_RES["en"].findall(text)) + len(_EN_CONTRACTION_RE.findall(text))
    pattern = _NEGATION_RES.get(language)
    if pattern is None:
        return sum(len(rx.findall(text)) for rx in _NEGATION_RES.values())
    total = len(pattern.findall(text))
    if language in _NEG_PARTICLES:
        particles = [m.end() for m in _NEG_PARTICLES[language].finditer(text)]
        for m in _NEG_COMPANIONS[language].finditer(text):
            before = [p for p in particles if p <= m.start()]
            if before and not _CLAUSE_END.search(text, before[-1], m.start()):
                total -= 1
    return total


# -- numbers, measurements, identifiers -----------------------------------------

_UNIT_ALIASES: dict[str, str] = {
    # time
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "sekunde": "s",
    "sekunden": "s",
    "seconde": "s",
    "secondes": "s",
    "secundă": "s",
    "secunde": "s",
    "min": "min",
    "minute": "min",
    "minutes": "min",
    "minuten": "min",
    "minut": "min",
    "minutele": "min",
    "h": "h",
    "hour": "h",
    "hours": "h",
    "stunde": "h",
    "stunden": "h",
    "heure": "h",
    "heures": "h",
    "oră": "h",
    "ore": "h",
    # length, torque, pressure, temperature, volume, mass, electrical
    "mm": "mm",
    "cm": "cm",
    "m": "m",
    "in": "in",
    "nm": "N·m",
    "n·m": "N·m",
    "n.m": "N·m",
    "lbf.in": "lbf·in",
    "lbf·in": "lbf·in",
    "lbf-in": "lbf·in",
    "in-lb": "lbf·in",
    "psi": "psi",
    "bar": "bar",
    "kpa": "kPa",
    "mbar": "mbar",
    "°c": "°C",
    "°f": "°F",
    "l": "L",
    "ml": "mL",
    "kg": "kg",
    "g": "g",
    "v": "V",
    "vdc": "V DC",
    "vac": "V AC",
    "a": "A",
    "ma": "mA",
    "ohm": "ohm",
    "ohms": "ohm",
    "ω": "ohm",
    "kω": "kohm",
    "mω": "Mohm",
    "%": "%",
    "rpm": "rpm",
    "hz": "Hz",
}
_NUMBER_RE = re.compile(r"(?<![\w.,])[-+]?\d+(?:[.,  ]\d+)*(?![\w])")
_ATTACHED_RE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)([A-Za-z°%Ωω]+)(?![\w])")
_MEASURE_RE = re.compile(
    r"(?<![\w.,])(?P<num>\d+(?:[.,  ]\d+)*)\s?(?P<unit>°\s?[CFcf]|%|[A-Za-zΩω·.\-ăâîșțéèêüöä]+)"
)
_LIST_MARKER_RE = re.compile(
    r"^\s*[-*•]?\s*(?:(?:step|schritt|étape|etape|pas|pasul)\s*)?"
    r"(?:\(?\d{1,2}[.):]|[a-zA-Z][.)])\s+",
    re.MULTILINE | re.IGNORECASE,
)
_IDENTIFIER_RE = re.compile(
    r"(?<![\w-])(?=[A-Z0-9/.\-]*\d)(?=[A-Z0-9/.\-]*[A-Z])[A-Z0-9][A-Z0-9/.\-]*[A-Z0-9](?![\w-])"
)
_SECTION_RE = re.compile(r"(?<![\w.])\d{2}-\d{2}(?:-\d{2})?(?![\w])")


def normalize_number(token: str, language: str = "en") -> str:
    """Normalize decimal and thousands separators so values compare across languages."""
    token = token.replace(" ", "").replace(" ", "").lstrip("+")
    if language == "en":
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", token):
            return token.replace(",", "")
        return token.replace(",", ".") if re.fullmatch(r"\d+,\d{1,2}", token) else token
    # de/fr/ro: comma is the decimal separator; dot or space groups thousands.
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", token):
        token = token.replace(".", "")
    return token.replace(",", ".")


def _strip_markers(text: str) -> str:
    return _LIST_MARKER_RE.sub("", text)


def numbers(text: str, language: str = "en") -> list[str]:
    """Numeric values in ``text``.

    List markers (``1.``, ``Step 2:``) and digits inside identifier codes
    (``FQ-2``, ``412AL``) are ignored: identifiers are checked separately.
    """
    cleaned = _IDENTIFIER_RE.sub(" ", _strip_markers(text))
    values = []
    for match in _NUMBER_RE.finditer(cleaned):
        raw = match.group(0).strip()
        if re.fullmatch(r"\d+(?:[  ]\d{3})+", raw) and language == "en":
            values.extend(raw.split())
            continue
        values.append(normalize_number(raw, language))
    for match in _ATTACHED_RE.finditer(cleaned):
        if match.group(2).casefold() in _UNIT_ALIASES:
            values.append(normalize_number(match.group(1), language))
    return values


def measurements(text: str, language: str = "en") -> list[tuple[str, str]]:
    """``(value, unit)`` pairs where the unit is a known unit word or symbol."""
    found = []
    for match in _MEASURE_RE.finditer(_strip_markers(text)):
        unit = match.group("unit").casefold().replace(" ", "").rstrip(".")
        canonical = _UNIT_ALIASES.get(unit)
        if canonical is not None:
            found.append((normalize_number(match.group("num"), language), canonical))
    return found


def identifiers(text: str) -> list[str]:
    """Upper-case alphanumeric codes such as ``P/N 4471-A2`` parts, ``HYD-12``, or ``412AL``."""
    found = [m.group(0).rstrip(".-/") for m in _IDENTIFIER_RE.finditer(text)]
    found += _SECTION_RE.findall(text)
    seen: dict[str, None] = {}
    for item in found:
        seen.setdefault(item, None)
    return list(seen)


_SEGMENT_END_RE = re.compile(r"[.!?;:](?=\s)|\n")


def fact_anchors(text: str, language: str = "en") -> list[tuple[str, int]]:
    """Numbers and identifier codes in text order, with their segment index.

    A segment is a sentence, clause after ``;``/``:``, or line. The result is
    used to check that facts keep their order across steps.
    """
    stripped = _strip_markers(text)
    bounds = [m.end() for m in _SEGMENT_END_RE.finditer(stripped)]

    def segment(pos: int) -> int:
        return sum(1 for b in bounds if b <= pos)

    found: list[tuple[int, str]] = []
    for m in _IDENTIFIER_RE.finditer(stripped):
        found.append((m.start(), "id:" + m.group(0).rstrip(".-/")))
    blanked = _IDENTIFIER_RE.sub(lambda m: " " * len(m.group(0)), stripped)
    for m in _NUMBER_RE.finditer(blanked):
        found.append((m.start(), "n:" + normalize_number(m.group(0).strip(), language)))
    return [(token, segment(pos)) for pos, token in sorted(found)]


_FIELD_RE = re.compile(r"^\s*[^\W\d][\w /()-]{0,40}:\s*\S")


def is_structured(text: str) -> bool:
    """True when most lines are ``Label: value`` fields (requirements input)."""
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) >= 2 and sum(1 for line in lines if _FIELD_RE.match(line)) >= 0.6 * len(lines)


def order_inversions(source: str, output: str, language: str = "en") -> list[tuple[str, str]]:
    """Pairs of facts from different source segments that appear in reverse order.

    Only facts that occur exactly once in the source and once in the output are
    compared, so repeated values cannot cause false alarms. Facts in the same
    source sentence may change order (translation often reorders a sentence).
    Structured ``key: value`` input is a set of facts, not a sequence, and is
    not checked.
    """
    if is_structured(source):
        return []
    src = fact_anchors(source, language)
    out = fact_anchors(output, "en")
    src_count = Counter(t for t, _ in src)
    out_pos: dict[str, int] = {}
    out_count = Counter(t for t, _ in out)
    for i, (t, _) in enumerate(out):
        out_pos.setdefault(t, i)
    unique = [(t, seg) for t, seg in src if src_count[t] == 1 and out_count[t] == 1]
    inverted = []
    for i, (a, seg_a) in enumerate(unique):
        for b, seg_b in unique[i + 1 :]:
            if seg_a < seg_b and out_pos[a] > out_pos[b]:
                inverted.append((a.split(":", 1)[1], b.split(":", 1)[1]))
    return inverted


def multiset_missing(expected: list[str], actual: list[str]) -> tuple[list[str], list[str]]:
    """Return ``(missing, extra)`` for two lists compared as multisets."""
    remaining = Counter(actual)
    missing = []
    for item in expected:
        if remaining[item] > 0:
            remaining[item] -= 1
        else:
            missing.append(item)
    extra = [item for item, n in remaining.items() for _ in range(n)]
    return missing, extra


# -- output contract --------------------------------------------------------------

_COMMENTARY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("reasoning_trace", re.compile(r"</?think>|<\|im_start\|>|<\|im_end\|>", re.I)),
    ("markdown_fence", re.compile(r"```")),
    ("markdown_heading", re.compile(r"^\s{0,3}#{1,6}\s", re.M)),
    ("markdown_emphasis", re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__")),
    (
        "assistant_preamble",
        re.compile(
            r"^\s*(?:sure|certainly|of course|okay|ok|here is|here's|here are|below is|"
            r"the (?:rewritten|following|revised|translated)|i (?:have|will|would|can|'ve)|"
            r"as an ai|hello|hi\b|hier ist|voici|iată|iata)",
            re.I,
        ),
    ),
    (
        "meta_commentary",
        # "NOTE:" in upper case is an STE label, not commentary; "Note:" is commentary.
        re.compile(
            r"^\s*(?:(?!NOTE\b)(?i:note)|(?i:explanation|changes made|rationale|translation|"
            r"rewritten text|ste version|output))\s*:",
            re.M,
        ),
    ),
    (
        "closing_remark",
        re.compile(r"(?:let me know|i hope this helps|feel free to|if you need)", re.I),
    ),
)


def contract_violations(output: str) -> list[str]:
    """IDs of output-contract problems: commentary, markdown, or reasoning traces."""
    return [name for name, rx in _COMMENTARY_PATTERNS if rx.search(output)]


def is_unsupported_sentinel(output: str) -> bool:
    """True when the model returned the controlled-failure sentinel."""
    return output.strip().strip(".").strip() == UNSUPPORTED_SENTINEL


def strip_reasoning(output: str) -> str:
    """Remove an *empty* ``<think></think>`` block that some chat templates emit."""
    return re.sub(r"^\s*<think>\s*</think>\s*", "", output)


def normalize_text(text: str) -> str:
    """Lower case, NFC, collapsed whitespace: for unchanged-text comparisons."""
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()
